#!/bin/bash
# eval_solution.sh — run the agent's kernel against torch.compile on the
# selected workload set, writing human-readable output to stdout.
#
# Usage:
#   bash scripts/eval_solution.sh                  # default WORKLOAD_SET=baseline
#   WORKLOAD_SET=small_batch bash scripts/eval_solution.sh
#   WORKLOAD_SET=all bash scripts/eval_solution.sh

set -u

KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINDING="${KIT_ROOT}/solution/cuda/binding.py"

if [ ! -f "${BINDING}" ]; then
  echo "ERROR: binding not found at ${BINDING}" >&2
  exit 2
fi

: "${WORKLOAD_SET:=baseline}"
: "${WARMUP:=10}"
: "${ITERS:=50}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

echo "# Eval started at $(date -Iseconds)"
echo "# WORKLOAD_SET=${WORKLOAD_SET}  WARMUP=${WARMUP}  ITERS=${ITERS}  GPU=${CUDA_VISIBLE_DEVICES}"
echo

python "${KIT_ROOT}/scripts/run_eval.py" \
  --binding "${BINDING}" \
  --workload-set "${WORKLOAD_SET}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}"

status=$?
echo
echo "# Eval finished at $(date -Iseconds), exit=${status}"
exit ${status}
