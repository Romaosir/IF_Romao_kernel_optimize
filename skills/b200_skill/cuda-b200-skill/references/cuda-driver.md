# Using the local CUDA Driver API docs

Use Driver API references only when lower-level control is required.

## Use them for

- `CUresult`
- context creation and control
- module loading
- manual PTX/CUBIN handling
- virtual memory APIs
- lower-level integration work

## Search patterns

```bash
grep -R "cuModuleLoad" references/cuda-driver-docs/
grep -R "cuCtxCreate" references/cuda-driver-docs/
grep -R "cuMemMap" references/cuda-driver-docs/
grep -R "CUDA_ERROR_" references/cuda-driver-docs/
```

## Typical questions

- How do I load a PTX string manually?
- What does this `CUDA_ERROR_*` code mean?
- How does Driver API virtual memory work?

## Guidance

For ordinary kernel authoring, do not default to Driver API.
Start with Runtime API unless explicit lower-level control is required.
