"""
Eval driver: compile the kernel, run each workload, check correctness,
measure speedup against torch.compile baseline. Designed to be called by
`eval_solution.sh`.

Exit code 0 always (even on INCORRECT workloads) — the agent parses the
textual output to decide. Use exit code 2 only for driver-level failures
(import error, CUDA unavailable).

Output format:
  [workload <name>] shape=(M, D) dtype=<dtype> variant=<variant>
    status: CORRECT | INCORRECT | COMPILE_ERROR | RUNTIME_ERROR | TIMEOUT
    correctness: max_abs_err=<f>  rel_err=<f>  tol_abs=<f>  tol_rel=<f>
    torch_compile: <ms> ms
    custom:        <ms> ms
    speedup: <s>x

  === SUMMARY ===
  workloads: <n_total>  correct: <n_correct>
  speedup_geomean: <s>x
  speedup_min: <s>x
  speedup_max: <s>x
  all_correct: <true|false>
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import torch

# Make sibling modules importable when script is run from anywhere
_HERE = Path(__file__).resolve().parent
_KIT = _HERE.parent
sys.path.insert(0, str(_KIT))
sys.path.insert(0, str(_HERE))

from reference import dispatch_reference  # noqa: E402
from workloads import DTYPE_MAP, Workload, select  # noqa: E402

TORCH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _load_custom_kernel(binding_path: Path):
    """Load the agent's binding.py and return the dispatch callable.
    The binding module must expose `rmsnorm(x, w, eps, variant, r=None, b=None)`.
    """
    spec = importlib.util.spec_from_file_location("agent_binding", binding_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load binding at {binding_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "rmsnorm"):
        raise RuntimeError("binding.py must export `rmsnorm(x, w, eps, variant, r=None, b=None)`")
    return module.rmsnorm


def _make_inputs(w: Workload, device: torch.device):
    dtype = TORCH_DTYPES[w.dtype]
    torch.manual_seed(0)
    x = torch.randn(w.M, w.D, device=device, dtype=dtype) * 0.5
    weight = torch.randn(w.D, device=device, dtype=dtype) * 0.1 + 1.0
    r = torch.randn(w.M, w.D, device=device, dtype=dtype) * 0.5 if w.variant == "residual" else None
    b = torch.randn(w.D, device=device, dtype=dtype) * 0.1 if w.variant == "affine" else None
    return x, weight, r, b


def _correctness(out: torch.Tensor, ref: torch.Tensor, w: Workload) -> tuple[bool, float, float]:
    out_f = out.to(torch.float32)
    ref_f = ref.to(torch.float32)
    diff = (out_f - ref_f).abs()
    max_abs = diff.max().item()
    denom = ref_f.abs().clamp_min(1e-8)
    max_rel = (diff / denom).max().item()
    ok = (max_abs <= w.abs_tol) and (max_rel <= w.rel_tol)
    return ok, max_abs, max_rel


def _time_callable(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    # One-shot event timing averaged over iters avoids launch-overhead artifacts
    start_ev.record()
    for _ in range(iters):
        fn()
    end_ev.record()
    torch.cuda.synchronize()
    total_ms = start_ev.elapsed_time(end_ev)
    return total_ms / iters


_COMPILED_CACHE: dict[tuple[str, str], Callable] = {}


def _torch_compile_baseline(variant: str, dtype_key: str):
    """Return a torch.compile-d function for the given variant/dtype.
    Cached across workloads with the same (variant, dtype).
    """
    key = (variant, dtype_key)
    if key in _COMPILED_CACHE:
        return _COMPILED_CACHE[key]

    def _plain(x, w, eps):
        return dispatch_reference("plain", x, w, eps)

    def _residual(x, r, w, eps):
        return dispatch_reference("residual", x, r, w, eps)

    def _affine(x, w, b, eps):
        return dispatch_reference("affine", x, w, b, eps)

    fn = {"plain": _plain, "residual": _residual, "affine": _affine}[variant]
    compiled = torch.compile(fn, mode="reduce-overhead", dynamic=False, fullgraph=True)
    _COMPILED_CACHE[key] = compiled
    return compiled


def _run_torch_compile(variant: str, x, w_tensor, r, b, eps: float, dtype_key: str):
    fn = _torch_compile_baseline(variant, dtype_key)
    if variant == "plain":
        return lambda: fn(x, w_tensor, eps)
    if variant == "residual":
        return lambda: fn(x, r, w_tensor, eps)
    if variant == "affine":
        return lambda: fn(x, w_tensor, b, eps)
    raise AssertionError(variant)


def _run_custom(custom_fn, variant: str, x, w_tensor, r, b, eps: float):
    if variant == "plain":
        return lambda: custom_fn(x, w_tensor, eps, variant)
    if variant == "residual":
        return lambda: custom_fn(x, w_tensor, eps, variant, r=r)
    if variant == "affine":
        return lambda: custom_fn(x, w_tensor, eps, variant, b=b)
    raise AssertionError(variant)


def evaluate_workload(
    w: Workload,
    custom_fn: Callable,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict:
    result = {
        "name": w.name,
        "shape": (w.M, w.D),
        "dtype": w.dtype,
        "variant": w.variant,
        "status": "UNKNOWN",
        "max_abs_err": float("nan"),
        "max_rel_err": float("nan"),
        "abs_tol": w.abs_tol,
        "rel_tol": w.rel_tol,
        "torch_compile_ms": float("nan"),
        "custom_ms": float("nan"),
        "speedup": float("nan"),
        "error": "",
    }

    try:
        x, weight, r, b = _make_inputs(w, device)
    except Exception as e:
        result["status"] = "RUNTIME_ERROR"
        result["error"] = f"input generation: {e}"
        return result

    # Reference output (for correctness)
    try:
        ref = dispatch_reference(
            w.variant,
            *([x, r, weight] if w.variant == "residual" else ([x, weight, b] if w.variant == "affine" else [x, weight])),
            eps=w.eps,
        )
    except Exception as e:
        result["status"] = "RUNTIME_ERROR"
        result["error"] = f"reference computation: {e}"
        return result

    # Custom kernel run
    try:
        kwargs = {}
        if w.variant == "residual":
            kwargs["r"] = r
        if w.variant == "affine":
            kwargs["b"] = b
        out = custom_fn(x, weight, w.eps, w.variant, **kwargs)
        torch.cuda.synchronize()
    except Exception as e:
        result["status"] = "RUNTIME_ERROR"
        result["error"] = f"custom kernel crashed: {e}"
        return result

    if out.shape != ref.shape:
        result["status"] = "INCORRECT"
        result["error"] = f"shape mismatch: got {tuple(out.shape)} expected {tuple(ref.shape)}"
        return result

    ok, mae, mre = _correctness(out, ref, w)
    result["max_abs_err"] = mae
    result["max_rel_err"] = mre
    if not ok:
        result["status"] = "INCORRECT"
        return result

    # Timing: torch.compile baseline + custom
    try:
        tc_fn = _run_torch_compile(w.variant, x, weight, r, b, w.eps, w.dtype)
        cu_fn = _run_custom(custom_fn, w.variant, x, weight, r, b, w.eps)

        tc_ms = _time_callable(tc_fn, warmup=warmup, iters=iters)
        cu_ms = _time_callable(cu_fn, warmup=warmup, iters=iters)
    except Exception as e:
        result["status"] = "RUNTIME_ERROR"
        result["error"] = f"timing failed: {e}\n{traceback.format_exc()}"
        return result

    result["torch_compile_ms"] = tc_ms
    result["custom_ms"] = cu_ms
    result["speedup"] = tc_ms / cu_ms if cu_ms > 0 else float("inf")
    result["status"] = "CORRECT"
    return result


def print_result(r: dict) -> None:
    print(f"[workload {r['name']}] shape={r['shape']} dtype={r['dtype']} variant={r['variant']}")
    print(f"  status: {r['status']}")
    if r["status"] in ("CORRECT", "INCORRECT"):
        print(
            f"  correctness: max_abs_err={r['max_abs_err']:.3e}  "
            f"rel_err={r['max_rel_err']:.3e}  "
            f"tol_abs={r['abs_tol']:.1e}  tol_rel={r['rel_tol']:.1e}"
        )
    if r["status"] == "CORRECT":
        print(f"  torch_compile: {r['torch_compile_ms']:.4f} ms")
        print(f"  custom:        {r['custom_ms']:.4f} ms")
        print(f"  speedup: {r['speedup']:.3f}x")
    if r["error"]:
        print(f"  error: {r['error']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True, type=Path, help="Path to binding.py")
    parser.add_argument("--workload-set", default=os.environ.get("WORKLOAD_SET", "baseline"),
                        help="Comma-separated tags to select workloads, or 'all'")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("DRIVER_ERROR: torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # Import the agent's kernel
    try:
        custom_fn = _load_custom_kernel(args.binding)
    except Exception:
        print(f"COMPILE_ERROR: failed to load {args.binding}", file=sys.stderr)
        traceback.print_exc()
        for w in select(args.workload_set):
            print_result({**{k: None for k in []}, "name": w.name, "shape": (w.M, w.D),
                          "dtype": w.dtype, "variant": w.variant, "status": "COMPILE_ERROR",
                          "max_abs_err": float("nan"), "max_rel_err": float("nan"),
                          "abs_tol": w.abs_tol, "rel_tol": w.rel_tol,
                          "torch_compile_ms": float("nan"), "custom_ms": float("nan"),
                          "speedup": float("nan"),
                          "error": "see stderr traceback"})
        print("\n=== SUMMARY ===")
        print("workloads: 0  correct: 0")
        print("speedup_geomean: nan")
        print("all_correct: false")
        return 0

    workloads = select(args.workload_set)
    if not workloads:
        print(f"DRIVER_ERROR: no workloads matched set='{args.workload_set}'", file=sys.stderr)
        return 2

    results: list[dict] = []
    for w in workloads:
        t0 = time.time()
        r = evaluate_workload(w, custom_fn, device, args.warmup, args.iters)
        r["wall_ms"] = (time.time() - t0) * 1000
        print_result(r)
        print()
        results.append(r)

    # Summary
    correct = [r for r in results if r["status"] == "CORRECT"]
    n_total, n_correct = len(results), len(correct)
    speedups = [r["speedup"] for r in correct if math.isfinite(r["speedup"])]
    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        mn, mx = min(speedups), max(speedups)
    else:
        geo = mn = mx = float("nan")

    print("=== SUMMARY ===")
    print(f"workloads: {n_total}  correct: {n_correct}")
    print(f"speedup_geomean: {geo:.3f}x")
    print(f"speedup_min: {mn:.3f}x")
    print(f"speedup_max: {mx:.3f}x")
    print(f"all_correct: {'true' if n_correct == n_total else 'false'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
