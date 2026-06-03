# Learning ML Compilers

Working through the [MLC (Machine Learning Compilation) course](https://mlc.ai/chapter_introduction/index.html) step by step.

> **Note:** The course is fairly old and the setup instructions are outdated. If you're following along, here's what actually works:
>
> - Install via uv, not plain pip. The course tells you to run `python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu` but with uv you need to configure the find-links source in `pyproject.toml` (see below).
> - `pytest` is required as a dependency even if you're not running tests — tvm imports it internally and will crash without it.
> - Pin your Python version to `>=3.11,<3.13` — the package doesn't support 3.13 yet and uv will fail to resolve without this constraint.

## How TVM Works

TVM is an ML compiler: it takes a model and compiles it to run efficiently on a specific hardware target (CPU, GPU, phone, custom chip). The same model can be compiled for different targets without rewriting anything.

```
YOUR MODEL (PyTorch / numpy / etc.)
        ↓
   IRModule — TVM's internal representation, holds two things:
   ├── Relax  (@R.function)   — the computational graph; describes how layers connect
   └── TensorIR  (@T.prim_func) — the individual ops; describes the actual loops and math
        ↓
   OPTIMIZE — two ways:
   ├── Manual:    sch.split() / reorder() / vectorize() / parallel() / ...
   └── Automatic: TVM searches thousands of variants and picks the fastest (Chapter 4)
        ↓
   tvm.compile(mod, target="llvm")  →  machine code for your hardware
        ↓
   VirtualMachine / rt_lib  →  run on device
```

**Component quick reference:**

| Component | What it is |
|---|---|
| `IRModule` | Container that holds the whole program (Relax + TensorIR functions) |
| `@R.function` (Relax) | High-level graph — describes model structure like `forward()` in PyTorch |
| `@T.prim_func` (TensorIR) | Low-level op — explicit loops, buffers, and math |
| `TVMScript` | Python-like syntax for writing TensorIR/Relax by hand |
| `Tensor Expression (TE)` | Declarative shorthand that auto-generates TensorIR loops |
| `Schedule` | Wraps an IRModule and lets you restructure loops without changing the result |
| `tvm.compile` | Converts the optimized IRModule into runnable machine code |
| `VirtualMachine` | Executes compiled Relax programs on a device |

**Why TVM exists:** there are many ML frameworks (PyTorch, JAX, TensorFlow) and many hardware targets (CPU, GPU, TPU, phones). Every combination would need manual optimization — TVM automates this so a model written once can run fast anywhere.

## Chapter 2 Exercises

### Writing TensorIR

Every TensorIR function has three parts: buffers (the data), loops (iteration), and compute statements (the math). Axes are annotated as spatial (`S`) or reduce (`R`) — spatial axes are output dimensions that can be parallelized, reduce axes are accumulation dimensions that cannot.

```python
vi, vj, vk = T.axis.remap("SSR", [i, j, k])  # j loops = spatial, k = reduce
with T.init():
    Y[vi, vj] = 0       # runs once per (i,j) before the k reduction
Y[vi, vj] += A[vi, vk] * B[vk, vj]
```

### Schedule Primitives

| Primitive | What it does |
|---|---|
| `split(loop, factors=[a, b])` | Break one loop into two (outer × inner) |
| `reorder(l1, l2, ...)` | Change the order of loops |
| `parallel(loop)` | Run this loop across CPU cores |
| `vectorize(loop)` | Use SIMD instructions for this loop |
| `unroll(loop)` | Eliminate loop overhead by inlining iterations |
| `reverse_compute_at(block, loop)` | Move a consumer block inside a producer's loop |
| `decompose_reduction(block, loop)` | Separate init (zeroing) from update (accumulation) |

### `get_loops` always returns all surrounding loops

```python
n, i, j, k = sch.get_loops(Y)
# returns every loop from outermost → innermost that wraps the block
```

After `reverse_compute_at(C, j0)`, calling `get_loops(C)` returns `[n, i, j0, ax0]` — all loops now surrounding C, not just its own.

### Ordering rules that matter

1. `parallel` must come **before** `decompose_reduction` — TVM needs the whole reduction block intact to verify the parallel is safe.
2. `reorder` must come **before** `reverse_compute_at` — the consumer block needs to see the final loop order when being moved.
3. `split` must come **before** `reorder` — the new loops need to exist before you can reorder them.

### Why each transformation helps performance

```
split j (chunks of 8)   → right size for SIMD vectorization
split k (chunks of 4)   → right size for loop unrolling
reorder (j1 innermost)  → cache-friendly memory access
parallel (n)            → spreads batches across CPU cores
vectorize (j1)          → processes 8 elements per SIMD instruction
unroll (k1)             → removes 4-iteration loop overhead
reverse_compute_at      → relu reads Y while it's still in cache
decompose_reduction     → clean separation of init and update phases
```

**Core insight:** The same computation expressed with different loop structure produces very different performance. TensorIR makes the loop structure explicit so the compiler — or you — can transform it to match what the hardware is good at (multiple cores, SIMD units, cache hierarchy).

---

## Chapter 3 — End-to-End Model Execution

### New abstractions introduced

**Relax (`@R.function`)** — the high-level orchestrator. Describes the full model as a computational graph connecting primitive functions together. Lives alongside TensorIR prim_funcs in the same IRModule.

**`call_dps_packed`** — bridges DPS (Destination Passing Style) primitive functions into the Relax graph. DPS functions write into an output buffer rather than returning a value — `call_dps_packed` allocates that buffer and makes the result a trackable value in the graph.

```python
lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
```

**`R.dataflow()` block** — marks a region of pure computation (no side effects). TVM can freely optimize inside this block. `R.output()` declares what leaves it.

**`T.handle` + `T.match_buffer`** — used instead of `T.Buffer` when shapes are dynamic (not known until runtime). The sizes are declared as `T.int64()` variables and resolved at runtime.

**`BindParams`** — bakes model weights into the IRModule. Before: `vm["main"](data, w0, b0, w1, b1)`. After: `vm["main"](data)`. Useful for inference since weights don't change after training.

**External library calls (`env.linear`, `env.relu`)** — register PyTorch (or any) functions under a TVM global name. The Relax graph calls them identically to TensorIR prim_funcs. Zero-copy via DLPack: `torch.from_dlpack(tvm_tensor)` shares memory between frameworks — no data duplication.

### The MLC theme

```
same computation → multiple abstractions
numpy → TensorIR → Relax → compiled code

MLC = transform between abstractions
    = get best performance on target hardware
```

The key flexibility: TensorIR and external libraries can be mixed in the same module. You can write and optimize specific ops in TensorIR, and delegate others to PyTorch/cuDNN — all connected through the same Relax graph.

---

## Chapter 4 — Automatic Program Optimization

### The problem Chapter 4 solves

In Chapters 2 and 3, we manually chose transformations — split `j` by 4, reorder, vectorize. But how do we know 4 is the best factor? What if 8 is faster? Trying every combination by hand is impossible. Chapter 4 lets TVM search automatically.

### Deterministic vs Stochastic schedule

A **deterministic schedule** always applies the same transformation:
```python
j_0, j_1 = sch.split(j, factors=[None, 4])  # always 4, same result every time
```

A **stochastic schedule** uses `sample_perfect_tile` to randomly pick valid split factors:
```python
j_factors = sch.sample_perfect_tile(loop=j, n=2)  # could be [4,32], [8,16], [16,8], ...
j_0, j_1 = sch.split(loop=j, factors=j_factors)
```

The stochastic schedule function doesn't describe one program — it describes an entire **search space** of valid programs. Every call to it produces a different variant.

### Symbolic variables and the trace

`sample_perfect_tile` returns **symbolic variables** (`tvm.tir.expr.Var`), not real numbers. The actual chosen values are stored in the trace's `decision` field:

```python
# trace shows:
v4, v5 = sch.sample_perfect_tile(..., decision=[8, 16])  # 8 and 16 were chosen this run
```

This lets TVM replay any past choice exactly (reproducible), or search over many different choices systematically.

### Random search

The simplest search: call the stochastic schedule repeatedly, compile each variant, benchmark it, keep the fastest:
```python
for i in range(num_trials):
    sch = stochastic_schedule_mm(tvm.s_tir.Schedule(mod))  # random variant
    result = f_timer(a, b, c).mean                          # benchmark it
    if result < best_result: best_sch = sch                 # keep best
```

### MetaSchedule `tune_tir` — smarter than random search

`tune_tir` does the same thing but with three improvements:

| Feature | What it does |
|---|---|
| Parallel benchmarking | Compiles and runs many variants simultaneously |
| XGBoost cost model | ML model predicts performance — skips bad variants without running them |
| Evolutionary search | Learns from results; good variant → try similar ones next round |

```python
database = ms.tune_tir(
    mod=MyModule,
    target=tvm.target.Target({"kind": "llvm", "num-cores": 1}),
    max_trials_global=64,   # total search budget
    num_trials_per_iter=64, # per round (more rounds = smarter)
    space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
    work_dir="./tune_tmp",
)
```

### Auto-scheduling (no manual space)

If you omit the `space=` parameter, TVM analyzes the loop structure itself and generates the search space automatically — looking at spatial/reduce axes, data access patterns, and trying splits on all loops, parallelization, vectorization, and unrolling. The resulting structure is deeper than what we wrote manually (e.g. `SSRSRS` tiling — spatial/reduce loops interleaved across multiple levels) and produces better results.

### Plugging the optimized op back into the full model

```python
# 1. Extract linear0, rename to "main" for tuning
mod_linear = tvm.IRModule.from_expr(
    MyModuleMixture["linear0"].with_attr("global_symbol", "main")
)

# 2. Tune
database = ms.tune_tir(mod=mod_linear, ...)
sch = ms.tir_integration.compile_tir(database, mod_linear, target)

# 3. Rename back and swap into the full module
new_func = sch.mod["main"].with_attr("global_symbol", "linear0")
gv = MyModuleWithParams2.get_global_var("linear0")
MyModuleWithParams2.update_func(gv, new_func)
```

### The full MLC picture

```
Chapter 2: ABSTRACTIONS  — how to represent computation (TensorIR, Relax)
Chapter 3: END-TO-END    — how to connect ops into a full model
Chapter 4: OPTIMIZATION  — how to find the fastest implementation automatically

write simple code → TVM auto-tunes → fast deployment on any hardware
```

---

## Code

- `tensorIR.py` — Chapter 2 notes and examples (coded by me, annotated by Claude)
- `tensorIR_EX.py` — Chapter 2 exercises: element-wise add, broadcasting, 2D convolution, bmm_relu transformation (coded by me, annotated by Claude)
- `end_2_end.py` — Chapter 3 end-to-end MLP on FashionMNIST (coded by me, annotated by Claude)
- `automatic_program_optimization.py` — Chapter 4 automatic optimization with meta_schedule (coded by me, annotated by Claude)


