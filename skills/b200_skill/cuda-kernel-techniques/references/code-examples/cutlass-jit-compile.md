# CUTLASS JIT Compilation Pattern

Compile CUTLASS source at runtime and `dlopen` the resulting `.so`. This lets you vendor CUTLASS into your project without requiring a system install, and works around version drift across machines.

## Build command

```
nvcc -std=c++17 \
     -gencode arch=compute_100a,code=sm_100a \
     -O2 \
     --shared -Xcompiler -fPIC \
     <source.cu> \
     -o /tmp/<name>.so \
     -lcuda
```

Critical flags:
- `-gencode arch=compute_100a,code=sm_100a` — without the `_a`, tcgen05 and some SM100-specific features are silently dropped
- `-Xcompiler -fPIC` — required for `--shared`
- `-lcuda` — link CUDA driver API (needed for `cuMemcpy*`, `cuStreamSynchronize`, etc.)

## Runtime load pattern

Observed in production MoE kernel:

```cpp
typedef int (*TcgenGemmFn)(CutlassBwArgs*, cudaStream_t);
typedef int (*TcgenSetupFn)(int N, int K);
typedef void (*TcgenSetRowsFn)(int total_rows);

static TcgenGemmFn g_tcgen05_gemm    = nullptr;
static TcgenSetupFn g_tcgen05_setup  = nullptr;
static TcgenSetRowsFn g_tcgen05_set_rows = nullptr;
static TcgenGemmFn g_tcgen05_gemm2   = nullptr;   // separate symbol for GEMM2
static TcgenSetupFn g_tcgen05_setup2 = nullptr;

static void load_tcgen05_so() {
    static bool tried = false;
    if (tried) return;
    tried = true;

    const char* path = "/tmp/libtcgen05_moe_gemm.so";

    // Build if missing
    if (access(path, F_OK) != 0) {
        fprintf(stderr, "[tcgen05] .so not found at %s, building...\n", path);
        int ret = system(
            "nvcc -std=c++17 "
            "-gencode arch=compute_100a,code=sm_100a "
            "-O2 --shared -Xcompiler -fPIC "
            "/tmp/tcgen05_fp8_moe.cu "
            "-o /tmp/libtcgen05_moe_gemm.so "
            "-lcuda 2>/dev/null");
        if (ret != 0) {
            fprintf(stderr, "[tcgen05] Build failed\n");
            return;
        }
    }

    // Load
    void* lib = dlopen(path, RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "[tcgen05] dlopen failed: %s\n", dlerror());
        return;
    }

    g_tcgen05_gemm     = (TcgenGemmFn)    dlsym(lib, "tcgen05_grouped_gemm");
    g_tcgen05_setup    = (TcgenSetupFn)   dlsym(lib, "tcgen05_setup_tma");
    g_tcgen05_set_rows = (TcgenSetRowsFn) dlsym(lib, "tcgen05_set_total_rows");
    g_tcgen05_gemm2    = (TcgenGemmFn)    dlsym(lib, "tcgen05_grouped_gemm2");
    g_tcgen05_setup2   = (TcgenSetupFn)   dlsym(lib, "tcgen05_setup_tma2");

    if (g_tcgen05_gemm && g_tcgen05_setup) {
        fprintf(stderr, "[tcgen05] Loaded successfully (gemm2=%p)\n",
                (void*)g_tcgen05_gemm2);
    } else {
        fprintf(stderr, "[tcgen05] dlsym failed\n");
        g_tcgen05_gemm = nullptr;
        g_tcgen05_setup = nullptr;
    }
}
```

## cuBLAS dynamic loading (parallel pattern)

When you need cuBLAS as a fallback path (for T-dependent dispatch), load it the same way rather than linking at build time:

```cpp
static void* g_cublas_lib = nullptr;

typedef cublasStatus_t (*fn_cublasCreate)(cublasHandle_t*);
typedef cublasStatus_t (*fn_cublasGemmEx)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, cudaDataType, int,
    const void*, cudaDataType, int,
    const void*,
    void*, cudaDataType, int,
    cublasComputeType_t, cublasGemmAlgo_t);

static fn_cublasCreate p_cublasCreate = nullptr;
static fn_cublasGemmEx p_cublasGemmEx = nullptr;
// ... more function pointers

static void ensure_cublas_loaded() {
    if (g_cublas_lib) return;

    g_cublas_lib = dlopen("libcublas.so",  RTLD_NOW | RTLD_GLOBAL);
    if (!g_cublas_lib)
        g_cublas_lib = dlopen("libcublas.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!g_cublas_lib) {
        fprintf(stderr, "Failed to load libcublas: %s\n", dlerror());
        return;
    }

    p_cublasCreate  = (fn_cublasCreate)  dlsym(g_cublas_lib, "cublasCreate_v2");
    p_cublasGemmEx  = (fn_cublasGemmEx)  dlsym(g_cublas_lib, "cublasGemmEx");
    // ...
}
```

Benefits:
- Build doesn't require a cuBLAS install
- Your binary won't fail to load on machines with a different cuBLAS version
- You control which of `.so.12`, `.so.13`, etc. to prefer

## Static compile with embedded source (+7% gain)

If you want to eliminate the first-run compile cost entirely, embed the CUTLASS source as a raw string and compile at module init:

```cpp
static const char kCutlassSrc[] = R"CUTLASS_SRC(
// <<< paste full CUTLASS grouped GEMM source here, ~500 lines >>>
)CUTLASS_SRC";

static void build_cutlass_so_static() {
    const char* src_path = "/tmp/cutlass_src.cu";
    const char* out_path = "/tmp/libcutlass_bw_gemm.so";

    // Write source
    FILE* f = fopen(src_path, "w");
    fwrite(kCutlassSrc, 1, sizeof(kCutlassSrc) - 1, f);
    fclose(f);

    // Compile
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "nvcc -std=c++17 -gencode arch=compute_100a,code=sm_100a "
        "-O2 --shared -Xcompiler -fPIC "
        "-I/path/to/vendored/cutlass/include "
        "%s -o %s -lcuda 2>&1",
        src_path, out_path);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "[cutlass] build failed (ret=%d)\n", ret);
    }
}
```

Call `build_cutlass_so_static()` at library init (e.g., in a `__attribute__((constructor))`). The first benchmark invocation then hits the warm-cache path, eliminating the cold-vs-warm discrepancy.

Observed gain from this change alone: **+7%**.

## Pitfalls

| Pitfall | Result |
|---|---|
| Forgot `_a` suffix (`compute_100` instead of `compute_100a`) | tcgen05 instructions silently dropped; kernel runs slower than intended |
| Missing `-lcuda` | Symbol lookup failures at `dlopen` for any CUDA driver API |
| `.so` from a different machine with different GPU | May load but produce wrong results or crash |
| Raw string with unescaped `"` or `\` | Compile error; use carefully or use separate `.cu` file on disk |
| `.so` rebuilt mid-benchmark | Measurement noise — make sure first invocation happens at warmup |

## Debugging a failed load

```cpp
void* lib = dlopen(path, RTLD_NOW);
if (!lib) {
    fprintf(stderr, "[error] dlopen %s failed: %s\n", path, dlerror());
    // Check if file exists: access(path, F_OK) == 0
    // Check if file has correct architecture: file /tmp/xxx.so
    // Check missing symbols: ldd /tmp/xxx.so
    return -1;
}
void* sym = dlsym(lib, "symbol_name");
if (!sym) {
    fprintf(stderr, "[error] dlsym %s failed: %s\n", "symbol_name", dlerror());
    // Check: nm -D /tmp/xxx.so | grep symbol_name
    // Most common: C++ name mangling — wrap exports in extern "C"
    return -2;
}
```

Always `extern "C"` your exported symbols — otherwise C++ mangling makes `dlsym` lookup fragile:

```cpp
extern "C" int tcgen05_grouped_gemm(CutlassBwArgs* a, cudaStream_t stream) {
    // ...
}
```
