# RMS Norm eval harness

Test harness for the `cuda-kernel-autodev` skill (and its siblings). Ten prompts
run a CUDA kernel optimization loop on RMS norm against a `torch.compile` baseline
on B200.

**You run this in a shell with working CUDA access** (the harness needs `nvidia-smi`
and `torch.cuda.is_available() == True`). Drop output files back into the
`eval-<N>/*` directories for the skill-creator's review viewer to pick up.

## Layout

```
evals/
├── README.md                       ← this file
├── evals.json                      ← 10 test prompts + metadata
├── rmsnorm-starter-kit/            ← naive correct baseline (scenarios 1-8)
├── rmsnorm-starter-kit-broken/     ← intentional bug (scenario 9)
└── rmsnorm-starter-kit-good/       ← already-decent baseline (scenario 10)

Each starter kit:
  solution/cuda/kernel.cu           ← the file the agent edits
  solution/cuda/binding.py          ← JIT-compiles kernel.cu via torch cpp_extension
  reference.py                      ← torch RMS norm reference
  scripts/eval_solution.sh          ← the agent invokes this
  scripts/run_eval.py               ← eval driver (correctness + timing)
  scripts/workloads.py              ← workload definitions (shape/dtype/variant)
  notes/perf_log.md                 ← initial log stub
  plan/  profile/  results/         ← empty; populated by agent
```

## How the eval works

1. Agent edits `solution/cuda/kernel.cu` (and `binding.py` if the signature changes).
2. Agent runs `bash scripts/eval_solution.sh`.
3. The script selects workloads by `WORKLOAD_SET` env var (default: `baseline`).
4. For each workload:
   - JIT-compiles the agent's kernel (rebuild on `kernel.cu` edit, cached otherwise).
   - Runs reference + custom kernel, checks correctness (max_abs_err vs abs_tol, max_rel_err vs rel_tol).
   - Times custom kernel via `torch.cuda.Event`, 10 warmup + 50 iters.
   - Times `torch.compile(reference, mode='reduce-overhead')`, same warmup/iters.
   - Reports speedup = `torch_compile_ms / custom_ms`.
5. Final `=== SUMMARY ===` line prints `speedup_geomean` and `all_correct`.

The agent's **KEEP threshold is `speedup_geomean` improved ≥ 1% over the previous baseline**. The **submission bar** is `all_correct == true` AND `speedup_geomean >= 1.0x` (beats `torch.compile`).

## Running one test case

Suppose you want to run scenario #3 (large batch) with the skill loaded:

```bash
# 1) Copy the starter kit to a workspace (so the agent has its own history)
cp -r /home/lihongbin/code_agent/.claude/skills/cuda-kernel-autodev/evals/rmsnorm-starter-kit \
      /tmp/rms_eval_3_with_skill
cd /tmp/rms_eval_3_with_skill

# 2) Initialize git (the workflow uses commits for revert)
git init -q && git add -A && git commit -q -m "v1: naive baseline"

# 3) Confirm the harness works as-is (should print v1 perf)
CUDA_VISIBLE_DEVICES=0 WORKLOAD_SET=large_batch bash scripts/eval_solution.sh \
    > results/v1_baseline.txt 2>&1
grep -E "speedup_geomean|all_correct" results/v1_baseline.txt

# 4) Launch the agent with the skill directory loaded, pointing at this workspace
#    Use the prompt from evals.json eval id=3. The agent should edit kernel.cu
#    and loop for at most 12 experiments.

# 5) After the run, collect:
#    - solution/cuda/kernel.cu   (final kernel)
#    - notes/perf_log.md         (experiment log with chart)
#    - results/all_eval_v*.txt   (per-experiment eval output)
#    - notes/submission.md       (final summary)
```

## Running with-skill vs without-skill (for benchmark comparison)

Skill-creator wants a baseline. For each of the 10 evals, run twice:

- **with_skill**: agent has `/home/lihongbin/code_agent/.claude/skills/` accessible
- **without_skill**: agent gets the same prompt but no skill directory (baseline)

Put outputs under:
```
<workspace>/iteration-1/eval-<N>/with_skill/outputs/
<workspace>/iteration-1/eval-<N>/without_skill/outputs/
```

Save at minimum: `kernel.cu`, `perf_log.md`, `submission.md`, `results/*.txt`. The viewer renders whatever it finds.

## Parallelizing across GPUs

You have 8x B200. To run all 10 evals in parallel, dispatch per-GPU:

```bash
for i in 1 2 3 4 5 6 7 8; do
  gpu=$((i-1))
  (CUDA_VISIBLE_DEVICES=$gpu run_single_eval.sh eval_$i with_skill &)
done
wait
# Then the remaining 2 on whichever GPUs finish first
```

Each eval is independent — starter kits are separate directories, JIT-compile caches
are per-workspace. Make sure each subagent pins its own `CUDA_VISIBLE_DEVICES`.

## Troubleshooting

**JIT compile fails.** `ninja` must be installed (`pip install ninja`). Also ensure
`nvcc` and the CUDA toolkit version match PyTorch's build. Error messages go to
the python traceback — read the whole thing.

**`torch.cuda.is_available() == False` despite having a B200.** Your shell may be
in a sandbox that doesn't expose NVML. Run outside the sandbox, or with an env
that grants `/dev/nvidia*` + NVML access. Claude Code's Bash tool may need a
specific config or flag to see NVML.

**Speedup < 0.5x across the board.** torch.compile may have fused more than you
think (e.g., it's comparing against a fused RMSNorm+ResidualAdd baseline while you
ship a plain kernel). Inspect `torch.compile`'s generated code via
`TORCH_LOGS=+dynamo,+inductor` to understand what you're competing against.

**`speedup_geomean: nan`.** All workloads failed correctness. Check the per-workload
`INCORRECT` lines for max_abs_err / max_rel_err to see how far off you are.

**The `-gencode=compute_100,code=sm_100` flag fails.** Older PyTorch / CUDA
toolkits don't know about SM100. Set `RMS_ARCH=sm_90a` or `sm_90` as a fallback
(you lose B200-specific paths but the code still runs).

## Hardware context — B200 spec sheet

- SM100, compute capability 10.0
- 148 SMs
- 228 KB opt-in shared memory per SM (256 KB unified pool)
- 65536 regs/SM, 255 regs/thread max
- 8 TB/s HBM3e, 192 GB
- 126 MB L2 cache
- Tensor cores: WGMMA (fp8/bf16/fp16/tf32), MXFP
- Async copy: `cp.async`, TMA, `cp.async.bulk`

This file is also referenced from the `cuda-kernel-autodev` skill's
`references/hardware-discovery.md`.

## What to report back

After running all 10 evals, send me:

1. For each eval: the final `grep -E "speedup_geomean|all_correct"` line (10 lines total, with-skill and without-skill variants).
2. For evals where the behavior was interesting (big differential, surprising failures), the corresponding `notes/perf_log.md`.
3. Any error cases where the harness itself failed (so I can fix the scaffolding).

I'll grade with-skill vs without-skill quantitatively on the JSON results, load the
transcripts into skill-creator's review viewer for you to review qualitatively,
and we'll iterate from there.
