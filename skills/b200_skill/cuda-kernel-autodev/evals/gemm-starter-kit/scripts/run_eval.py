"""
GEMM eval driver: compile, run each workload, check correctness, measure
speedup against torch.compile baseline.

Workload.D is interpreted as N (output feature dim). Workload.K is the
reduction dim. Shapes: X=[M,K], W=[N,K], bias=[N], Y=[M,N].
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

_HERE = Path(__file__).resolve().parent
_KIT = _HERE.parent
sys.path.insert(0, str(_KIT))
sys.path.insert(0, str(_HERE))

from reference import dispatch_reference  # noqa: E402
from workloads import DTYPE_MAP, Workload, select  # noqa: E402

TORCH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _load_custom_kernel(binding_path: Path):
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
    M, N, K = w.M, w.D, w.K
    # Scale to keep matmul output within a reasonable numerical range for bf16
    scale_x = 1.0 / math.sqrt(K)
    x = (torch.randn(M, K, device=device, dtype=dtype) * scale_x).contiguous()
    weight = (torch.randn(N, K, device=device, dtype=dtype) * 0.5).contiguous()
    bias = (torch.randn(N, device=device, dtype=dtype) * 0.1).contiguous() if w.variant == "fused" else None
    return x, weight, None, bias


def _correctness(out: torch.Tensor, ref: torch.Tensor, w: Workload) -> tuple[bool, float, float]:
    """Combined-tolerance check: |out - ref| <= atol + rtol * |ref|.
    Matches torch.testing.assert_close semantics. Tensor-core-friendly because
    it doesn't fail on per-element rel-error when ref is near zero — the atol
    term carries the slack in that region."""
    out_f = out.to(torch.float32)
    ref_f = ref.to(torch.float32)
    diff = (out_f - ref_f).abs()
    # Use the combined check (torch.testing style) rather than enforcing each separately.
    threshold = w.abs_tol + w.rel_tol * ref_f.abs()
    violated = diff > threshold
    ok = not violated.any().item()
    max_abs = diff.max().item()
    # max_rel is reported for visibility but NOT used as a pass/fail gate here
    denom = ref_f.abs().clamp_min(1e-8)
    max_rel = (diff / denom).max().item()
    return ok, max_abs, max_rel


def _time_callable(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        fn()
    end_ev.record()
    torch.cuda.synchronize()
    return start_ev.elapsed_time(end_ev) / iters


_COMPILED_CACHE: dict[tuple[str, str], Callable] = {}


def _torch_compile_baseline(variant: str, dtype_key: str):
    key = (variant, dtype_key)
    if key in _COMPILED_CACHE:
        return _COMPILED_CACHE[key]

    def _plain(x, w):
        return dispatch_reference("plain", x, w)

    def _fused(x, w, b):
        return dispatch_reference("fused", x, w, b)

    fn = {"plain": _plain, "fused": _fused}[variant]
    compiled = torch.compile(fn, mode="reduce-overhead", dynamic=False, fullgraph=True)
    _COMPILED_CACHE[key] = compiled
    return compiled


def _run_torch_compile(variant: str, x, weight, bias):
    fn = _torch_compile_baseline(variant, str(x.dtype))
    if variant == "plain":
        return lambda: fn(x, weight)
    if variant == "fused":
        return lambda: fn(x, weight, bias)
    raise AssertionError(variant)


def _run_custom(custom_fn, variant: str, x, weight, bias):
    if variant == "plain":
        return lambda: custom_fn(x, weight, 0.0, "plain")
    if variant == "fused":
        return lambda: custom_fn(x, weight, 0.0, "fused", b=bias)
    raise AssertionError(variant)


def evaluate_workload(
    w: Workload, custom_fn: Callable, device: torch.device, warmup: int, iters: int
) -> dict:
    result = {
        "name": w.name, "shape": (w.M, w.D, w.K), "dtype": w.dtype, "variant": w.variant,
        "status": "UNKNOWN", "max_abs_err": float("nan"), "max_rel_err": float("nan"),
        "abs_tol": w.abs_tol, "rel_tol": w.rel_tol,
        "torch_compile_ms": float("nan"), "custom_ms": float("nan"),
        "speedup": float("nan"), "error": "",
    }

    try:
        x, weight, _r, bias = _make_inputs(w, device)
    except Exception as e:
        result["status"] = "RUNTIME_ERROR"; result["error"] = f"input gen: {e}"; return result

    try:
        if w.variant == "plain":
            ref = dispatch_reference("plain", x, weight)
        else:
            ref = dispatch_reference("fused", x, weight, bias)
    except Exception as e:
        result["status"] = "RUNTIME_ERROR"; result["error"] = f"ref: {e}"; return result

    try:
        kwargs = {}
        if w.variant == "fused":
            kwargs["b"] = bias
        out = custom_fn(x, weight, 0.0, w.variant, **kwargs)
        torch.cuda.synchronize()
    except Exception as e:
        result["status"] = "RUNTIME_ERROR"; result["error"] = f"custom: {e}"; return result

    if out.shape != ref.shape:
        result["status"] = "INCORRECT"
        result["error"] = f"shape: got {tuple(out.shape)} exp {tuple(ref.shape)}"
        return result

    ok, mae, mre = _correctness(out, ref, w)
    result["max_abs_err"] = mae; result["max_rel_err"] = mre
    if not ok:
        result["status"] = "INCORRECT"; return result

    try:
        tc_fn = _run_torch_compile(w.variant, x, weight, bias)
        cu_fn = _run_custom(custom_fn, w.variant, x, weight, bias)
        tc_ms = _time_callable(tc_fn, warmup=warmup, iters=iters)
        cu_ms = _time_callable(cu_fn, warmup=warmup, iters=iters)
    except Exception as e:
        result["status"] = "RUNTIME_ERROR"
        result["error"] = f"timing: {e}\n{traceback.format_exc()}"
        return result

    result["torch_compile_ms"] = tc_ms
    result["custom_ms"] = cu_ms
    result["speedup"] = tc_ms / cu_ms if cu_ms > 0 else float("inf")
    result["status"] = "CORRECT"
    return result


def print_result(r: dict) -> None:
    M, N, K = r["shape"]
    print(f"[workload {r['name']}] M={M} N={N} K={K} dtype={r['dtype']} variant={r['variant']}")
    print(f"  status: {r['status']}")
    if r["status"] in ("CORRECT", "INCORRECT"):
        print(f"  correctness: max_abs_err={r['max_abs_err']:.3e}  rel_err={r['max_rel_err']:.3e}  "
              f"tol_abs={r['abs_tol']:.1e}  tol_rel={r['rel_tol']:.1e}")
    if r["status"] == "CORRECT":
        print(f"  torch_compile: {r['torch_compile_ms']:.4f} ms")
        print(f"  custom:        {r['custom_ms']:.4f} ms")
        print(f"  speedup: {r['speedup']:.3f}x")
    if r["error"]:
        print(f"  error: {r['error']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--workload-set", default=os.environ.get("WORKLOAD_SET", "baseline"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("DRIVER_ERROR: CUDA unavailable", file=sys.stderr); return 2

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    try:
        custom_fn = _load_custom_kernel(args.binding)
    except Exception:
        print("COMPILE_ERROR", file=sys.stderr); traceback.print_exc()
        print("\n=== SUMMARY ==="); print("all_correct: false"); print("speedup_geomean: nanx"); return 0

    workloads = select(args.workload_set)
    if not workloads:
        print(f"DRIVER_ERROR: no workloads match '{args.workload_set}'", file=sys.stderr); return 2

    results = []
    for w in workloads:
        t0 = time.time()
        r = evaluate_workload(w, custom_fn, device, args.warmup, args.iters)
        r["wall_ms"] = (time.time() - t0) * 1000
        print_result(r); print()
        results.append(r)

    correct = [r for r in results if r["status"] == "CORRECT"]
    speedups = [r["speedup"] for r in correct if math.isfinite(r["speedup"])]
    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        mn, mx = min(speedups), max(speedups)
    else:
        geo = mn = mx = float("nan")

    print("=== SUMMARY ===")
    print(f"workloads: {len(results)}  correct: {len(correct)}")
    print(f"speedup_geomean: {geo:.3f}x")
    print(f"speedup_min: {mn:.3f}x")
    print(f"speedup_max: {mx:.3f}x")
    print(f"all_correct: {'true' if len(correct) == len(results) else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
