# L06 — Mapping: Partitioning

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze (MIT EECS)
> **Lecture date:** February 18, 2026 · **Slides:** 76 · **Source:** [`Lecture/L06-Mapping-Partitioning.pdf`](../../Lecture/L06-Mapping-Partitioning.pdf)
>
> *This is a conceptual walkthrough that reconstructs the lecture's narrative from the slides. It is organized by idea, not slide-by-slide. Each section cites the slide range it draws from so you can follow along with the original deck.*

---

## TL;DR

Partitioning is the mapping decision that **splits a tensor's index space into smaller tiles**, adding a new rank for each dimension that is split. It serves two orthogonal goals: **(1) temporal reuse** — keeping a working set small enough to fit in a fast buffer and be reused across loop iterations, and **(2) spatial parallelism** — distributing independent subsets of work across multiple Processing Elements simultaneously. The lecture develops the concept step by step — from partitioning a 1-D vector, to a 2-D matrix, to a full partitioned matrix-vector computation — and then demonstrates the power of these ideas through two substantial case studies: a **distributed matrix-matrix multiply** (based on the ThunderKittens algorithm) and **three partitioning strategies for Transformer attention**. The central algebraic fact is that partitioning a rank adds a new subscript level, the loop nest grows accordingly, and the choice of which loops are temporal (`for`) versus spatial (`spatial-for`) determines *which data sits where and for how long*.

---

## Learning Objectives

After this lecture you should be able to:

- State the **two objectives of partitioning**: controlling reuse distance for buffer retention and enabling parallel computation.
- Explain why **partitioning always adds a rank**: splitting index `i` into `(i1, i0)` where `i = i1 × I0 + i0`.
- Trace how the loop nest transforms when a tensor is partitioned (new outer loops, unchanged inner loops, tighter working sets).
- Distinguish **temporal partitioning** (`for` loops) from **spatial partitioning** (`spatial-for` / `parallel_for`).
- Analyze the **distributed matrix-multiply** case study (ThunderKittens): how A, B, and Z are tiled across G PEs, how delayed reduction is introduced, and how the Einsums chain together.
- Compare the **three Attention partitioning strategies** — Tensor Parallel, Head Parallel, and Data Parallel — in terms of which rank is split and what is run in parallel.

---

## Chapter 1 — What Partitioning Is and Why It Exists

> *Slides: L06-2 … L06-8*

### The problem partitioning solves

Without partitioning, a matrix-vector multiply traverses the *entire* A tensor for every output element. The reuse distance for any element of A — the number of accesses between two uses of the same value — grows as the problem size grows. When the reuse distance exceeds a buffer's capacity, values must be evicted and reloaded from a slower level of the hierarchy, incurring the high-cost DRAM accesses that L01 established as the dominant energy term (≈ 200× an ALU operation).

![Unpartitioned matrix-vector multiply — for each output, the entire A tensor is traversed](../../assets/L06/L06-p04-unpartitioned-matvec.png)

The slide shows the unpartitioned loop nest for `Zm = Ak,m × Bk`:

```python
for m in range(M):
    for k in range(K):
        Z[m] += A[k, m] * B[k]
```

For each value of `m`, the inner `k` loop traverses all `K` columns of A. If A does not fit in the local buffer, every column is re-fetched from DRAM on each outer-loop iteration.

Slide 3 names the two objectives that partitioning addresses:

1. **Reduce reuse distance** so a tensor value can be retained in a fast buffer across the accesses that need it.
2. **Identify separate sets of data** that can be computed in parallel by different PEs.

![Objectives of partitioning — reduce reuse distance and enable parallelism](../../assets/L06/L06-p03-objectives.png)

### Partitioning always adds a rank

The fundamental algebraic move of partitioning is to replace a single index with a pair of indices. For a 1-D tensor indexed by `i`:

```
i  →  (i1, i0)   where  i = i1 × I0 + i0
```

The original range `I` factors as `I = I1 × I0`. The tensor `A_i` becomes `A_{i1,i0}`. This is not a different tensor — it holds the same values, addressed differently.

![Partitioning a 1-D vector — splitting rank I into (I1, I0)](../../assets/L06/L06-p05-partitioning-vector.png)

> **Why it matters:** Every rank introduced by partitioning generates a new loop in the loop nest. The *outer* loop controls which tile (partition) is active; the *inner* loop iterates within the tile. This structural addition is what creates the hierarchy of temporal and spatial loops the course uses to reason about data movement.

### Partitioning a matrix — two dimensions, two rank splits

When a 2-D tensor `A_{k,m}` is partitioned in both dimensions, both `k` and `m` get split:

```
k → (k1, k0)   where  k = k1 × K0 + k0
m → (m1, m0)   where  m = m1 × M0 + m0
```

The tensor becomes `A_{k1,k0,m1,m0}`. Visually, the matrix is divided into a grid of tiles, where each tile has shape `K0 × M0`.

![Partitioning a 2-D matrix — tiled into (K1,K0) × (M1,M0) blocks](../../assets/L06/L06-p06-partitioning-matrix.png)

Note that the tile boundaries are what constrain the working set. When the outer loops `(m1, k1)` are fixed — that is, for a particular choice of tile — the computation touches only the `K0 × M0` sub-block of A and the corresponding `K0` elements of B.

### The partitioned loop nest

After partitioning both A and B in the matrix-vector example, the Einsum becomes:

```
Z_{m1,m0} = A_{k1,k0,m1,m0} × B_{k1,k0}
```

and the loop nest becomes:

```python
for m1 in range(M1):
    for k1 in range(K1):
        for m0 in range(M0):
            for k0 in range(K0):
                Z[m1, m0] += A[k1, k0, m1, m0] * B[k1, k0]
```

The computation now stays within an `M0 × K0` tile of A and a `K0`-element row of B for the duration of the inner `(m0, k0)` loops. If the tile fits in the on-chip buffer, those values are reused at SRAM cost (6×) rather than DRAM cost (200×).

![Partitioned matrix-vector computation — inner loops stay within the M0×K0 tile](../../assets/L06/L06-p07-partitioned-matvec-computation.png)

### Spatial partitioning: parallel_for

Everything described so far is **temporal partitioning**: the outer loops execute sequentially and the inner loops exploit reuse within a tile. The second use of partitioning is **spatial**: assigning different partitions to different PEs and running them concurrently.

The notation for spatial execution is `spatial-for` (also written `parallel_for`):

```python
spatial-for i1 in range(I1):      # runs on I1 distinct PEs in parallel
    for i0 in range(I0):
        Z[i1, i0] = A[i1, i0] * B[i1, i0]
```

In this element-wise multiply example, `i1` partitions the output into `I1` independent sub-problems, each assigned to one PE. There is no data dependency between different `i1` values, so they can run truly in parallel.

![Partitioning for parallelism — spatial-for i1 distributes work across PEs](../../assets/L06/L06-p08-partitioning-for-parallelism.png)

> **Why it matters:** The same rank split that produces an outer temporal loop (for reuse) can instead be made spatial. The choice between `for` and `spatial-for` is the central design knob that determines the accelerator's PE utilization and its data-movement pattern. Many realistic mappings mix both: some ranks are temporal (outer loops feeding buffers), others are spatial (parallel PE assignment).

---

## Chapter 2 — Case Study: Distributed Matrix-Matrix Multiply

> *Slides: L06-9 … L06-18*

### Motivation and setup

The first extended case study applies partitioning to **dense matrix-matrix multiplication** in a multi-PE (multi-GPU) setting, following the algorithm from [Spector et al., *ThunderKittens*, ICLR 2025]. The base computation is:

```
Z_{m,n} = A_{k,m} × B_{k,n}
```

The constraint is that no single PE can hold the full A, B, or Z — the tensors must be distributed across `G` PEs such that each PE holds and operates on only its local portion.

![Distributed matrix-multiply overview — partitioned Einsum across G PEs](../../assets/L06/L06-p11-distributed-matmul-overview.png)

### The partitioning strategy

Both `k` and `m` are split:

```
k → (k1, k0)   K = K1 × K0,  G = K1
m → (m1, m0)   M = M1 × M0
```

Setting `G = K1` means each of the `G` PEs is responsible for a distinct `k1` slice of the computation — a distinct partition of the `k` dimension. The resulting Einsum on partitioned tensors is:

```
Z_{m1,m0,n} = A_{k1,k0,m1,m0} × B_{k1,k0,n}
```

This is verified to be mathematically equivalent to the original by flattening the tuple indices and substituting back:

```
Z_{(m1×M0+m0),n} = A_{(k1×K0+k0),(m1×M0+m0)} × B_{(k1×K0+k0),n}
→  Z_{m,n} = A_{k,m} × B_{k,n}   ✓
```

### Distributed tensor shapes

Each PE `g` (corresponding to a `k1` value) receives:

- **Distributed A** (`AD_{g,k0,m1,m0}`): G matrices, each of shape `K0 × (M1×M0)`. Rows are indexed by flattened `(m1,m0)` pairs; columns by `k0`.
- **Distributed B** (`BD_{g,k0,n}`): G matrices, each of shape `K0 × N`.
- **Distributed Z** (`ZD_{g,m0,n}`): G matrices, each of shape `M0 × N`, indexed by `m1`.

![Distributed A tensor — G slices of shape K0 × (M1×M0)](../../assets/L06/L06-p14-distributed-A.png)

### Delayed reduction

A key challenge is that after each PE computes its local `Z_{m1,m0,n}` using its `k1` slice, the partial results across all PEs must be **reduced** (summed). A naive approach would require collecting all partial sums at one place; the ThunderKittens approach instead uses **delayed reduction** to restructure the computation:

1. **Move `k1` to the left-hand side** of the Einsum, deferring the sum over `k1`:
   ```
   ZT_{k1,m1,m0,n} = A_{k1,k0,m1,m0} × B_{k1,k0,n}
   ```
   Now `ZT` has an explicit `k1` dimension — each PE produces one `k1` slice of `ZT`.

2. **Do the reduction** as a separate step:
   ```
   Z_{m1,m0,n} = sum over k1 of ZT_{k1,m1,m0,n}
   ```

![Delayed reduction — deferring the k1 sum to a separate Einsum](../../assets/L06/L06-p12-delayed-reduction.png)

The payoff is that the reduction can be overlapped with computation and structured as efficient all-reduce patterns across PEs, rather than forcing a synchronization barrier after every local multiply.

### Full Einsum chain

The complete algorithm consists of three stages expressed as Einsums:

**Distribution** (loading partitioned tensors from global storage into each PE's local store):

```
AD_{g,k0,m1,m0} = A_{g,m1,k0,m0}   (reshape/permute)
BD_{g,k0,n}     = B_{g,k0,n}        (select k1=g slice)
```

**Main computation** (local matrix multiply + partial accumulation):

```
ZL_{g,m1,m0,n}  = AD_{g,k0,m1,m0} × BD_{g,k0,n}
ZD_{g,m0,n}     = ZL_{h,g,m0,n}     (reduce over h — the inter-PE reduction)
```

**Finalization** (assemble final Z from the distributed result):

```
ZM_{g,m0,n} = ZD_{g,m0,n}
```

![Full Einsum chain for distributed matrix multiply](../../assets/L06/L06-p17-distributed-einsums.png)

> **Why it matters:** This case study shows that partitioning is not just a software loop-ordering trick — it directly determines how data is distributed across a parallel machine, how communication (the reduction) is structured, and what the working set of each PE is. Expressing the algorithm as a chain of Einsums makes each stage's data movement and reduction explicit, enabling systematic analysis with tools like Timeloop.

---

## Chapter 3 — Case Study: Partitioned Attention in Transformers

> *Slides: L06-19 … L06-22*

### Why partition Attention?

Transformer attention involves multiple large matrix multiplications chained together (Q/K/V projections, QK^T, softmax, AV contraction, output projection). These are memory-intensive and naturally admit partitioned execution across many PEs. The lecture presents **three distinct partitioning strategies**, each splitting a different rank:

### Strategy 1 — Tensor Parallel (partition the batch dimension B)

The batch index `b` is split: `b → (b1, b0)`.

All matrix multiplications that carry `b` as a free index are replicated with the split index. Since the `b1` partitions are independent (no batch element shares data with another in the forward pass), `b1` can be run with `spatial-for` — assigning each `b1` slice to a distinct PE (or group of PEs).

![Attention — Tensor Parallel: partitioning batch dimension B](../../assets/L06/L06-p20-attention-tensor-parallel.png)

The weight matrices `W_I, W_Q, W_K, W_V, W_Z` are **replicated** across all `B1` PEs (since they are independent of `b`). Each PE holds a full copy of the weights but only a `B0`-sized slice of the activations. Tensor Parallel is attractive when the batch size is large relative to the weight size.

### Strategy 2 — Head Parallel (partition the head dimension H)

The head index `h` is split: `h → (h1, h0)`.

Different attention heads are independent of one another; each head has its own Q, K, V, and output projections. Splitting `h` and running `h1` with `spatial-for` assigns different groups of heads to different PEs. Each PE holds a full copy of the input `I_{b,m,d}` but only its subset of head weights and activations.

![Attention — Head Parallel: partitioning attention head dimension H](../../assets/L06/L06-p21-attention-head-parallel.png)

Head Parallel is the most common partitioning for attention in current multi-GPU systems (e.g., tensor parallelism in Megatron-LM uses exactly this strategy). The output concatenation `C_{b,p,(h1×H0+h0)×F+f}` recombines the per-head outputs.

### Strategy 3 — Data Parallel (partition both H and D simultaneously)

The most aggressive strategy splits both the head dimension `h → (h1, h0)` *and* the model-dimension `d → (d1, d0)`. Running `h1` and `d1` spatially achieves higher parallelism at the cost of requiring communication to combine the partial `d` sums (since `d` is a contraction index in the Q/K/V projections).

All three strategies illustrate the same pattern: **choose a rank that partitions the computation into independent (or nearly independent) sub-problems, split that rank, and annotate the outer loop as spatial**.

> **Why it matters:** Modern large-scale inference and training are built on exactly these partitioning choices. Knowing which rank to split — and whether it introduces a reduction dependency (like `k` in matmul) or is a free rank (like `b` or `h`) — determines whether the parallel execution requires an all-reduce or is embarrassingly parallel. Architects must reason about this at design time.

---

## Chapter 4 — Partitioning in the Loop-Nest Framework

> *Slides: L06-2 … L06-8, reviewed in context of the two case studies*

### Summary of the algebraic rule

Every partitioning decision follows one template:

| Before partitioning | After partitioning |
|---|---|
| Index `i`, range `I` | Indices `(i1, i0)`, ranges `(I1, I0)`, with `I = I1 × I0` |
| Tensor rank `i` | Two ranks `(i1, i0)` in the Einsum |
| One loop | Two loops: outer over `i1`, inner over `i0` |

Multiple independent dimensions can be partitioned simultaneously; the result is a multiplication of all the splits in the Einsum.

### Temporal vs. spatial loops

Once a rank is split, the outer loop `i1` can be:

- **Temporal** (`for i1 in range(I1)`): the iterations run sequentially on the *same* PE. The benefit is that the inner tile (`i0`) fits in the local buffer, and the buffer contents are *reused* across inner-loop iterations before being evicted.
- **Spatial** (`spatial-for i1 in range(I1)`): the iterations are assigned to *different* PEs and run concurrently. Each PE holds its own `i0`-sized working set and executes the inner loop independently.

A single Einsum may have multiple partitioned ranks, some temporal and some spatial. The full mapping is the combination of all such choices across all ranks.

### Connection to L05 (Mapping — Dataflows)

L05 introduced the loop-nest representation of DNN computation and described **dataflows** as the *order* of loop execution and the *placement* of data in the memory hierarchy. L06 completes the Mapping picture by adding **partitioning**: which dimension bounds are tiled, what the tile shapes are, and which loops are spatial. Together, loop order + tile shapes + spatial/temporal annotation = the complete mapping specification that tools like Timeloop evaluate.

> **Why it matters:** Partitioning is the mechanism through which the abstract algorithm is brought into correspondence with the finite resources of the hardware — finite PE count, finite buffer capacity, finite bandwidth. Correctly dimensioning the tiles is the primary lever for hitting the arithmetic intensity threshold where the hardware is compute-bound rather than memory-bandwidth-bound.

---

## Standalone Study Guide

### What to master before moving on

- Treat partitioning as a rank split: `i -> (i1, i0)`, with the original coordinate recoverable from the pair.
- Distinguish temporal partitioning for reuse from spatial partitioning for parallelism.
- Explain why partitioning a reduction rank introduces a reduction across partitions.
- Compare tensor, head, and data parallelism for attention by which rank is split.

### Self-check questions

1. What new loops appear when `k` and `m` are split in a matrix multiply?
2. Why is splitting a free rank usually easier than splitting a reduction rank?
3. In delayed reduction, what value is made explicit on the left-hand side before the final reduction?

### Exercises

1. Partition `Z[m] = A[k,m] * B[k]` by splitting both `k` and `m`, then write the new loop nest.
2. Pick one attention partitioning strategy and list which tensors are replicated, sharded, or reduced.
3. For a four-PE system, propose a partition of matrix multiply and identify the communication step.

### Common traps

- Treating tiling and parallelization as the same thing. They can use the same split but different loop annotations.
- Forgetting that partitioning changes the rank structure seen by the mapper.
- Assuming all parallel strategies are communication-free. Reduction-rank splits require communication.

---

## Key Terms

| Term | Gloss |
|---|---|
| **Partitioning** | Splitting a tensor's index range into tiles; each split adds one new rank to the Einsum. |
| **Tiling** | Synonym for temporal partitioning; divides the iteration space into fixed-size blocks to improve data locality. |
| **Rank** | A dimension (subscript) of a tensor in Einsum notation. Partitioning adds ranks. |
| **Tile / Partition** | The sub-block of a tensor that fits in one level of the buffer hierarchy or is assigned to one PE. |
| **Reuse distance** | The number of intervening accesses between two uses of the same data value; reduced by tiling. |
| **Temporal partitioning** | Using a `for` loop over the outer (tile) index; sequential execution on one PE to achieve data reuse within the inner tile. |
| **Spatial partitioning** | Using a `spatial-for` / `parallel_for` over the outer (tile) index; different tile assigned to each PE for concurrent execution. |
| **Working set** | The set of data values actively used in the inner loops; tiling shrinks the working set to fit in fast local buffers. |
| **Delayed reduction** | Restructuring a computation so that a summation (reduction) over a partitioned index is deferred to a separate, later Einsum step, enabling overlap with computation. |
| **Distributed matrix multiply** | A partition-based parallel algorithm (here: ThunderKittens) where G PEs each hold `1/G` of the k-dimension and then reduce partial results. |
| **Tensor Parallel** | Attention partitioning strategy that splits the batch dimension B across PEs; weights are replicated. |
| **Head Parallel** | Attention partitioning strategy that splits the head dimension H across PEs; different heads run on different PEs. |
| **Data Parallel** | Attention partitioning strategy that splits both H and D; achieves higher PE count but requires reduction over D. |
| **spatial-for / parallel_for** | Loop annotation indicating that iterations of that loop are executed simultaneously on different PEs (spatial, not temporal, execution). |
| **Einsum** | The tensor contraction notation used throughout the course; partitioning is expressed by adding subscripts to existing Einsums. |
| **ThunderKittens** | A high-performance GPU matrix-multiply kernel [Spector et al., ICLR 2025] used as a case study for distributed partitioned matrix multiply. |

---

## Takeaways

- **Partitioning always adds a rank.** Splitting index `i` into `(i1, i0)` introduces a new subscript level in the Einsum and a new loop in the loop nest. Every partition decision has this form.
- **Temporal partitioning controls reuse distance.** By tiling, the inner loops can retain their working set in a fast on-chip buffer (SRAM ≈ 6×) rather than repeatedly fetching from DRAM (≈ 200×).
- **Spatial partitioning enables parallelism.** Annotating an outer loop as `spatial-for` assigns each partition to a distinct PE. Free-rank partitions (like batch `b` or head `h`) are embarrassingly parallel. Contraction-rank partitions (like `k`) require a reduction step.
- **The two objectives are orthogonal but often combined.** A realistic accelerator mapping typically has both spatial loops (for PE parallelism) and temporal loops (for buffer reuse) in the same loop nest.
- **Delayed reduction decouples computation from synchronization.** By making the `k1` summation a separate Einsum step, the ThunderKittens algorithm allows each PE to compute its full local tile without waiting, then reduces asynchronously.
- **Attention admits multiple partitioning strategies.** Tensor Parallel (split B), Head Parallel (split H), and Data Parallel (split H+D) represent different trade-offs in replication vs. communication cost. Head Parallel is the most common in deployed multi-GPU inference.

---

## Connections to Later Lectures

- **Completes the Mapping layer (with L05).** L05 covered loop ordering and dataflows (which loops are innermost, where data is placed). L06 adds partitioning (tile sizes and spatial assignments). Together they fully specify the Mapping node of the TeAAL Pyramid of Concerns.
- **Sparse architectures (L07–L10)** extend these same partitioning ideas to *non-uniform* tile sizes — tiles that skip zero elements — but the rank-splitting algebra remains identical.
- **Memory hierarchy and bandwidth** (discussed in L04 and throughout): correctly sizing partitions to fit in each level of the buffer hierarchy (RF → local SRAM → Global Buffer → DRAM) is the reason tiling matters; it was established in L01's energy-cost table.
- **Distributed inference systems**: the Tensor Parallel, Head Parallel, and Data Parallel strategies introduced here are exactly the same strategies used in real distributed inference frameworks (Megatron-LM, DeepSpeed, TensorRT-LLM), making this lecture directly applicable to system-level DNN deployment.

---

## Appendix — Slide-to-Section Map

| Slides | Section |
|---|---|
| L06-1 | Title |
| L06-2 | Chapter 1 — Partitioning section header |
| L06-3 | Ch.1 — Objectives of partitioning |
| L06-4 … L06-5 | Ch.1 — Unpartitioned matrix-vector; partitioning a vector |
| L06-6 … L06-7 | Ch.1 — Partitioning a matrix; partitioned matrix-vector computation |
| L06-8 | Ch.1 — Partitioning for parallelism (spatial-for) |
| L06-9 | Ch.2 — Distributed Matrix Multiply section header |
| L06-10 … L06-11 | Ch.2 — Objective and overview |
| L06-12 | Ch.2 — Delayed reduction |
| L06-13 … L06-16 | Ch.2 — Distributed tensor shapes (AD, BD, ZD) |
| L06-17 … L06-18 | Ch.2 — Full Einsum chain; approach summary |
| L06-19 | Ch.3 — Partitioned Attention section header |
| L06-20 | Ch.3 — Attention: Tensor Parallel |
| L06-21 | Ch.3 — Attention: Head Parallel |
| L06-22 | Ch.3 — Attention: Data Parallel |
| L06-23 … L06-76 | Animation frames (loop-nest visualization and execution trace) |
