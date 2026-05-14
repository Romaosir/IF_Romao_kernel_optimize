# Control flow techniques

Warp divergence (threads in a warp taking different branches) serializes execution. These techniques replace branches with predicated / masked math so all threads in a warp execute the same instructions.

---

## Branchless masking with `-INFINITY`

**What it does.** Replaces `if (valid) compute else skip` with `compute_with_sentinel_on_invalid`. For softmax-style kernels, the sentinel is `-INFINITY` — after `exp`, it becomes zero and contributes nothing.

**When it helps.** Sparse attention with padded indices, attention with causal/local masking, any kernel where a fraction of elements are "invalid" and would otherwise be skipped.

**When it hurts.** If the "invalid" fraction is very high and the unneeded compute is expensive, you waste work. Rare in practice — warp divergence penalty usually dominates.

**How.**

```cuda
// BRANCHY — warp may divergence if some lanes have idx >= 0 and some have idx < 0
float logit;
if (idx >= 0) {
    logit = compute_dot(q_reg, k_cache + idx * HEAD_DIM);
} else {
    continue;  // diverges
}

// BRANCHLESS — all lanes do the same work; invalid lanes get -INFINITY
int valid_idx = max(idx, 0);  // clamp to a safe address
float logit = compute_dot(q_reg, k_cache + valid_idx * HEAD_DIM);
logit = (idx >= 0) ? logit : -INFINITY;
```

After `exp2f(logit - max)`, the `-INFINITY` becomes 0, contributing nothing to the running sum or output accumulator.

**Safety.** `max(idx, 0)` prevents reading from negative addresses, which would segfault. Use 0 or any other known-valid address; the result is discarded via the `-INFINITY` mask.

**Field note.** V3 v6–v7 of the DSA run replaced `if (idx >= 0)` branches with the `-INFINITY` pattern and got +8%. Became a standard pattern in every subsequent variant.

---

## Predication with `?:`

**What it does.** Uses ternary expressions that compile to predicated instructions. Both sides are evaluated; the predicate selects.

**When it helps.** Short, cheap computations where the branch cost outweighs doing the extra work.

**When it hurts.** Expensive sides of the ternary — you're paying full cost for both branches, just avoiding divergence.

**How.**

```cuda
// Warp may diverge
int value;
if (cond) {
    value = expensive_call();
} else {
    value = other_expensive_call();
}

// If cond varies within warp: diverges.
// If both calls are cheap: predicate.
int value = cond ? cheap_call() : other_cheap_call();

// Masked-store pattern
if (threadIdx.x < num_valid) out[idx] = value;
// becomes:
int safe_idx = min(idx, out_size - 1);  // avoid OOB
out[safe_idx] = (threadIdx.x < num_valid) ? value : out[safe_idx];  // re-write same value when invalid
```

PTX has `@P` predicate prefix on most instructions — the compiler emits these when it spots patterns that reduce to predication.

---

## Warp divergence avoidance — structural patterns

**What it does.** Designs the kernel so branches split on boundaries that align with warp boundaries (threadIdx.x / 32). Then within a warp, all 32 lanes take the same branch and nothing diverges.

**When it helps.** Any kernel with per-thread conditional work. Aligning the condition to `(threadIdx.x >> 5)` or `blockIdx.y` instead of `threadIdx.x` eliminates intra-warp divergence.

**How.**

```cuda
// BAD — condition is threadIdx.x % 2, diverges every other lane
if (threadIdx.x % 2 == 0) { ... } else { ... }

// GOOD — condition is (threadIdx.x / 32) % 2, aligns with warp
int warp_id = threadIdx.x >> 5;
if ((warp_id & 1) == 0) { ... } else { ... }

// BETTER — hoist the branch to blockIdx.y or a template parameter
template<bool EVEN_PATH>
__global__ void kernel(...) {
    if constexpr (EVEN_PATH) { ... } else { ... }
}
```

Template-based specialization (`if constexpr`) eliminates the branch entirely at compile time.

---

## Hoisting invariants

**What it does.** Moves loop-invariant computation out of the inner loop.

**When it helps.** Every inner loop, every time. The compiler does some of this automatically, but not all — especially anything involving integer-pointer arithmetic or address computation.

**How.**

```cuda
// BAD — recomputes base address every iteration
for (int k = 0; k < num_kv; k++) {
    float v = k_cache[(page_idx * PAGE_SIZE + k) * HEAD_DIM + lane * 4];
    ...
}

// GOOD — hoist invariant base
const bf16* k_base = k_cache + page_idx * PAGE_SIZE * HEAD_DIM + lane * 4;
for (int k = 0; k < num_kv; k++) {
    float v = k_base[k * HEAD_DIM];
    ...
}

// Also: hoist index arithmetic, scale factors, masks
const float sm_scale_log2e = sm_scale * 1.4426950408889634f;   // once
for (int k = 0; k < num_kv; k++) {
    float logit = dot(...) * sm_scale_log2e;  // not recomputed
}
```

**Diagnostic.** If NCU reports "Instructions" count much higher than the algorithm implies, or "integer ALU pipe" showing up in stall reasons, the compiler is redoing address math. Hoist explicitly.

---

## Switching branches into lookup or arithmetic

**What it does.** Replaces an `if/else` ladder with a lookup table or an arithmetic expression.

**When it helps.** Branch chains with 3+ cases, especially when the cases select a constant value.

**How.**

```cuda
// BAD
int stride;
if (dtype == 0) stride = 4;
else if (dtype == 1) stride = 2;
else if (dtype == 2) stride = 1;

// GOOD — lookup
__constant__ int dtype_stride[4] = {4, 2, 1, 1};
int stride = dtype_stride[dtype];

// ALSO GOOD — arithmetic, if the mapping is regular
int stride = 4 >> dtype;
```

**When it hurts.** Only if the cases have different control-flow depth. For shallow constant selection, always prefer arithmetic or tables.

---

## Quick reference

| Symptom | Fix |
|---|---|
| Warp divergence in inner loop | Predication or `-INFINITY` masking |
| If-chain selecting constants | Lookup table |
| Branches splitting within a warp | Hoist to warp or block granularity |
| NCU shows high integer-ALU usage | Hoist address arithmetic |
| Specialized behaviors at boundaries | Template on `if constexpr` |
