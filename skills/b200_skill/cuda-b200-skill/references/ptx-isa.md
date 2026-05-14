# Using the local PTX ISA docs

The local PTX docs are the authoritative fast path for instruction-level questions.

## Use them for

- PTX instruction syntax
- operand order
- fragment/register layout
- memory ordering semantics
- WGMMA details
- `tcgen05.*`
- TMA / tensor map details
- `mbarrier`
- swizzle and layout legality
- special registers and addressing rules

## Search patterns

```bash
grep -R "wgmma" references/ptx-docs/9-instruction-set/
grep -R "tcgen05" references/ptx-docs/9-instruction-set/
grep -R "mbarrier" references/ptx-docs/
grep -R "swizzle" references/ptx-docs/
grep -R "Tensor Memory" references/ptx-docs/9-instruction-set/
```

## When to consult PTX docs before answering

Always check the PTX docs when the answer depends on:
- exact operand order
- exact layout descriptor meaning
- barrier or wait semantics
- legal shape/type combinations
- architecture-specific instruction forms
- memory-consistency details

## Good usage pattern

1. search the instruction name
2. locate the exact form
3. verify operand and type constraints
4. verify architecture notes and target requirements
5. only then propose inline PTX or interpret compiler output
