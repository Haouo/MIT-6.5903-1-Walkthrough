# L09 — Sparse Architectures, Part 2

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze (MIT EECS)
> **Lecture date:** March 4, 2026 · **Slides:** 138 · **Source:** [`Lecture/L09-Sparse_Architectures-2.pdf`](../../Lecture/L09-Sparse_Architectures-2.pdf)
>
> *This is a conceptual walkthrough that reconstructs the lecture's narrative from the slides. It is organized by idea, not slide-by-slide. The deck is animation-heavy — many slides are intermediate build states of the same figure; each section below cites the slide range it synthesizes and shows the most-complete version of each diagram.*

---

## TL;DR

Sparsity is pervasive in real DNN workloads, but extracting the resulting speedup and energy savings requires two things working together: a **compressed tensor representation** that skips over zeros in storage, and **hardware that can traverse and intersect those representations efficiently**. This lecture builds the entire formal vocabulary for doing so — fibertrees, fiber representations (uncompressed, coordinate/payload list, CSR/CSC), concordant vs. discordant traversal, coordinate projection, and fiber intersection — then applies that vocabulary to a progression of increasingly sophisticated sparse-CONV loop nests (output-stationary, weight-stationary, input-stationary, and two-sparse combined), culminating in three real accelerators: **Cambricon-X**, **SCNN**, and **ISOSceles**.

---

## Learning Objectives

After this lecture you should be able to:

- Define the **fibertree abstraction** and explain how ranks, fibers, coordinates, and payloads represent any tensor — dense or sparse.
- Name the principal **fiber representations** (uncompressed array, coordinate/payload list, bitmask, CSR, CSC, COO) and state the time complexity of `getPayload()` and `getNext()` for each.
- Distinguish **concordant** (stride-aligned) from **discordant** (random-access) traversal and explain why discordant traversal is expensive.
- Write loop nests for **output-stationary, weight-stationary, and input-stationary** convolutions that exploit weight sparsity, input sparsity, and both simultaneously.
- Explain **coordinate projection** and **fiber intersection** as the two key primitives for combining sparse operands.
- Describe the hardware dataflow and microarchitecture of **Cambricon-X**, **SCNN**, and **ISOSceles**, and state the sparsity they exploit and the PE utilization gains they achieve.

---

## Chapter 1 — Why Sparsity, and the Fibertree Abstraction

> *Slides: L09-1 … L09-17*

### Sparsity is everywhere

Real DNN tensors are far from dense. The opening slide (from the ExTensor paper, MICRO 2019) shows a representative cross-section of problems — natural-language processing, graph analytics, recommendation systems, scientific simulation, and image classification — all of which consume **sparse tensors** as their primary input or intermediate state.

![Sparse tensors appear across many problem domains](../../assets/L09/L09-p02-sparse-motivation.png)

A companion slide (from the SCNN paper, ISCA 2017) drills into AlexNet specifically: **input-activation density** (fraction of non-zero values) after ReLU is well below 1.0 for all convolutional layers, and **weight density** drops across the network too. Crucially, the work (number of multiply-accumulates) is proportional to the product of both densities — so simultaneously exploiting weight and activation sparsity can in principle reduce work to `density_weight × density_activation × dense_work`. The hardware challenge is realizing that potential in practice.

The lecture organizes three orthogonal tools for handling sparsity:

![The three sparsity levers: Gating, Format, and Skipping](../../assets/L09/L09-p05-sparsity-aspects.png)

- **Gating:** execute the same loop schedule as the dense case but suppress reads, multiplies, and writes whenever a zero operand is detected. *Saves energy, not time.*
- **Format:** compress the tensor representation to omit zeros from storage, reducing memory footprint and the bandwidth needed to stream data. *Saves storage and bandwidth.*
- **Skipping:** restructure the loop traversal itself to iterate only over non-zero coordinates, eliminating the wasted cycles entirely. *Saves both time and energy.*

This lecture focuses heavily on **Format** and **Skipping**, since they enable the most substantial speedup.

### The fibertree abstraction

To reason about any tensor representation in a unified way, the course introduces the **fibertree** (fiber-tree) abstraction. A tensor of *n* ranks is represented as a tree with *n* levels:

- Each **rank** (dimension) corresponds to a level of the tree.
- Each node at a given level holds a **coordinate** and a **payload**. For non-leaf levels, the payload is a reference to a child fiber. For leaf levels, the payload is the numeric value.
- A **fiber** is the ordered list of (coordinate, payload) tuples at one node's children.

![Fibertree abstraction: ranks, coordinates, fibers, and payloads](../../assets/L09/L09-p10-fibertree-abstraction.png)

In the dense case (all elements non-zero), every valid coordinate appears at every level. In the sparse case, only non-zero coordinates appear — rows (or higher-rank slices) that are entirely zero simply vanish, shrinking the tree.

Two operations on a fiber drive all tensor traversal:

- **`getPayload(coordinate)`** — random access to the payload at a given coordinate.
- **`getNext()`** — sequential (concordant) iteration to the next coordinate in traversal order.

The efficiency of each operation depends heavily on the chosen fiber representation, introduced next.

> **Why it matters:** The fibertree abstraction separates *what* a tensor represents from *how* it is stored. Any hardware that implements `getPayload` and `getNext` correctly can exploit sparsity regardless of the specific storage format, enabling a clean architectural interface between the sparse-format layer and the compute layer.

---

## Chapter 2 — Tensor Representations and Traversal Efficiency

> *Slides: L09-18 … L09-51*

### Fiber representation choices

A fiber can be stored in several ways, each with different trade-offs in space and access time:

| Representation | Description | `getPayload` cost | `getNext` (concordant) cost |
|---|---|---|---|
| **Uncompressed (U)** | Dense array; position = coordinate | O(1) | O(1) |
| **Run-length encoded (R)** | (run of zeros, non-zero value) pairs | O(n) linear scan | O(1) |
| **Coordinate/payload list (C)** | Sorted list of (coordinate, value) pairs | O(log n) binary search | O(1) |
| **Hash table (Hf/Hr)** | Hash from coordinate to payload | O(1) amortized | O(n) (poor locality) |
| **Coordinate bitmask** | One bit per coordinate, 1 = non-zero | O(popcount) | O(1) |

For **concordant traversal** (iterating in storage order), the coordinate/payload list is as efficient as the uncompressed array. For **discordant traversal** (random access by arbitrary coordinate), only the uncompressed array and hash table offer O(1) lookup. The bitmask format also allows cheap non-zero checking even without compression, as used in Eyeriss gating.

### Compressed Sparse Row (CSR) — a concrete implementation

The most widely used 2-D format is **CSR** (Compressed Sparse Row), which in fibertree notation is written `Tensor<U,C>(H,W)`:

- **H rank is uncompressed (U):** the row index equals its position, so no coordinate metadata is needed. A *segment array* of length H+1 stores the starting position and length of each row's fiber in the payload array.
- **W rank is compressed (C):** only non-zero column coordinates are stored, in a *coordinate array*. The corresponding values appear at the same position in a *value array*.

![CSR: Tensor<U,C>(H,W) — segment array, coordinate array, value array](../../assets/L09/L09-p31-csr-representation.png)

Swapping rank order gives **CSC** (Compressed Sparse Column): `Tensor<U,C>(W,H)`. The two formats have identical memory footprint for the same data, but their **concordant traversal orders differ** — CSR is efficient for row-major scans, CSC for column-major scans. Forcing a traversal to go in the opposite direction from the storage order is a **discordant traversal** and can require O(log n) or worse random access per element.

Merging two ranks into a single flat coordinate produces **COO** (Coordinate List): `Tensor<C2>(H,W)`, which stores (H,W) coordinate tuples directly. More merge options and their notation (`U2`, `R2`, `H2`) allow other trade-offs.

### Merging, splitting, and specifying sparsity patterns

Slides 52–71 make an important move: sparsity is no longer just "some coordinates are missing." It becomes a **formal specification over ranks** in the fibertree. That is what lets the same notation describe unstructured pruning, channel pruning, sub-kernel sparsity, Nvidia-style 2:4 sparsity, and hierarchical structured sparsity.

**Merging ranks** combines two or more tensor ranks into one storage rank. For example, `Tensor<C2>(H,W)` stores a flat list of `(h,w)` coordinate tuples, while `Tensor<U2>(H,W)` stores a flattened dense array whose original coordinates can be recovered with division and modulo arithmetic. Merging is useful when the hardware wants to traverse a multi-dimensional object as one linear fiber.

**Splitting fibers** goes the other direction: a single rank is divided into multiple ranks. There are two distinct choices:

- **Coordinate-space splitting:** split by coordinate ranges, e.g., coordinates `0–7`, `8–15`, and so on. This preserves geometric locality and makes the meaning of each tile easy to reason about, but sparse tiles may have very different occupancies.
- **Position-space splitting:** split by stored positions, i.e., by non-zero count. This balances work across PEs because each split gets about the same number of stored elements, but the coordinate ranges become irregular.

The specification examples show how pruning patterns can be written as transformations on the rank order:

- **Channel-based sparsity:** keep the `C` rank explicit and apply an unstructured rule to channel fibers, yielding a pattern like `C unstructured -> R -> S`.
- **Sub-kernel sparsity:** flatten the spatial filter ranks `R` and `S` into `RS`, then apply a structured rule such as "2 of 4" within that flattened rank.
- **Fully unstructured sparsity:** flatten `C`, `R`, and `S` into one `CRS` rank, then apply an unstructured rule over the full filter volume.
- **2:4 sparsity:** reorder and partition the channel rank into outer and inner ranks (`C1`, `C0`), then apply the 2-of-4 rule at the innermost rank where hardware can decode it cheaply.
- **Hierarchical Structured Sparsity (HSS):** partition a rank into several nested ranks and apply different `G:H` rules at different levels, such as a coarse 3:4 rule outside a fine 2:4 rule.

> **Why it matters:** These rank transformations are the bridge between model pruning rules and hardware format design. A pruning method is hardware-friendly only if its rank order, flattening, and split points make the remaining non-zero coordinates cheap to encode, traverse, and distribute across PEs.

### Hardware for concordant traversal

The figure below shows the hardware structure for a 2-D concordant traversal (e.g., scanning a CSR tensor in row-major order). Two stages of position generators (Pgen) and coordinate/payload extractors operate in a pipeline: the H-rank stage produces the fiber boundaries, and the W-rank stage sequences through the coordinates within each fiber.

![Hardware for 2-D concordant traversal of a sparse tensor](../../assets/L09/L09-p48-2d-concordant-traversal.png)

Each rank in the hardware directly implements the `getNext()` iterator. For uncompressed ranks, the position generator is a simple counter. For coordinate/payload list ranks, it is a pointer that advances through the stored list. The outputs at every rank are a (coordinate, payload) pair — a clean, representation-agnostic interface.

> **Why it matters:** Concordant traversal in a coordinate/payload list has the same O(1) per-element cost as an uncompressed array, while only reading non-zero elements. This is the key enabler for **time savings** from sparsity. Discordant traversal destroys that advantage, which is why the design of loop nests — and which operand is iterated vs. looked up — matters so much.

---

## Chapter 3 — Exploiting Sparse Weights in Convolution

> *Slides: L09-72 … L09-96*

### The convolution loop nest and coordinate arithmetic

The 1-D convolution `O[q] += I[w] * F[s]` subject to the index constraint `w = q + s` is the running example throughout this portion of the lecture. Dense implementations iterate over all (q, s) pairs; the two inner indices are always coordinates and positions simultaneously (uncompressed tensors).

When filters are sparse (many zero weights), the natural approach is to compress F into a coordinate/payload list and traverse it concordantly. The choice of **which loop is outer** then determines the dataflow:

**Output-stationary with sparse weights:**
```
for q in [0, Q):
    for (s, f_val) in f:        # concordant traversal of compressed filter
        w = q + s               # coordinate projection
        o[q] += i[w] * f_val   # look up input by computed coordinate
```
This traversal is concordant for F. The computed `w = q+s` is a **coordinate projection** — the coordinate of the required input activation is derived from the output and weight coordinates. Input activation `i` is kept uncompressed so that `i[w]` is O(1).

![Output-stationary with sparse weights: compressed filter traversal vs. dense](../../assets/L09/L09-p83-output-stationary-sparse-weights.png)

The slide compares the compressed-filter schedule (right) against the uncompressed schedule (left): with 2-of-5 weights non-zero, the compressed version performs only 2 iterations of the inner loop instead of 5 — a direct speedup proportional to weight density.

**Weight-stationary with sparse weights:** loop order is reversed (`for (s,f_val) in f: for q in [0,Q):`), keeping each weight stationary while sweeping over all outputs. The projection `w = q+s` still applies. Both dataflows achieve the same work reduction; the difference is which buffer holds the stationary data.

### Hardware microarchitecture

The hardware for the output-stationary sparse-weight schedule has three data streams:

1. **Filter fiber** — a Pgen + Coord/Payload stage that sequences through non-zero (s, f_val) pairs.
2. **Input activation array** — an uncompressed array indexed by the computed `w = q + s`.
3. **Partial sum array** — indexed by `q`; accumulates results with a MAC unit.

A coordinate generator (Cgen) computes `w` and `q` from the traversal state. The Cambricon-X accelerator (Zhang et al., Micro 2016) uses exactly this structure: metadata stored alongside each weight identifies which input activations are needed, and the PE loads those activations by their computed address rather than streaming a dense window.

> **Why it matters:** Sparse weight traversal is the simplest form of skipping. If weights are 50% sparse, the inner loop runs half as many times, halving both time and energy for the filter and output reads. The critical hardware requirement is a fast coordinate projection unit and an uncompressed input buffer that can be randomly accessed by the computed coordinate.

---

## Chapter 4 — Exploiting Sparse Inputs, and Both Simultaneously

> *Slides: L09-97 … L09-128*

### Sparse inputs: the sliding-window problem

When input activations are sparse (many post-ReLU zeros) but weights are dense, the dual approach is to compress `i` and iterate concordantly over non-zero inputs. For a weight-stationary dataflow:

```
for s in [0, S):
    for (w, i_val) in i if s <= w < Q+s:  # windowed concordant traversal
        q = w – s
        o[q] += i_val * f[s]
```

The `if s <= w < Q+s` constraint is a **sparse sliding window**: for each weight position `s`, only the input coordinates `w` that fall within the valid convolution window contribute to any output. As the slides animate with increasing `q`, the active window slides through the input fiber, picking up non-zero values and projecting them to the corresponding output coordinates. The CNVLUTIN accelerator exploits exactly this structure: it compresses zero activations, and for each non-zero input it looks up the appropriate weight and accumulates into the appropriate output.

For an output-stationary dataflow with sparse inputs:
```
for q in [0, Q):
    for (w, i_val) in i if q <= w < q+S:
        s = w – q
        o[q] += i_val * f[s]   # look up weight by computed s
```
Each output `q` iterates over the (sparse) inputs within its window and looks up the weight by the derived `s = w - q`. This requires random access to weights (uncompressed filter recommended).

### Two-sparse: projection followed by intersection

When **both** weights and inputs are sparse, the maximum work reduction requires iterating only over (w, s) pairs where both `i[w] ≠ 0` and `f[s] ≠ 0`. An output-stationary formulation is:

```
for q in [0,Q):
    for (s, (f_val, i_val)) in f.project(+q) & i:
        o[q] += i_val * f_val
```

This uses two primitives chained together:

1. **Coordinate projection** (`f.project(+q)`): shift the coordinate of every non-zero weight by `+q` to compute the required `w` for each `s`. This produces a new (virtual) fiber in the `w` coordinate space.
2. **Fiber intersection** (`&`): retain only coordinates present in *both* the projected filter fiber and the input activation fiber.

![Fiber intersection: only coordinates present in both fibers are kept](../../assets/L09/L09-p127-fiber-intersection.png)

The slide shows that intersecting two fibers with coordinates {2,4,7,8} and {1,2,5,8} yields {2,8} — only the common coordinates. The multiply-accumulate is performed only for those pairs. For a dense kernel (all 8 coordinates present in one fiber), intersection degenerates to a simple lookup; for a sparse kernel and sparse activations, it achieves the product-of-densities work reduction.

The coordinate bitmask and uncompressed representations make intersection particularly fast (a bitwise AND), at the cost of storing metadata for all coordinates, even zeros. The coordinate/payload list requires a merge-sort-style pass.

The hardware for the output-stationary two-sparse schedule feeds both the projected filter fiber and the input fiber into a shared **Intersection** unit, which emits matched (s, f_val, i_val) triples to the MAC unit.

> **Why it matters:** Fiber intersection is the key primitive that achieves **multiplicative** work reduction — if weight density is `d_W` and input density `d_I`, the work scales as `d_W × d_I`. Without intersection, the best you can do is exploit only one operand's sparsity, leaving work proportional to the denser of the two. Achieving full two-sparse reduction requires explicit coordinate matching hardware.

---

## Chapter 5 — SCNN: Cartesian-Product Sparse Acceleration

> *Slides: L09-112 … L09-124*

### The SCNN input-stationary dataflow

SCNN (Sparse CNN, Parashar et al., ISCA 2017) uses an **input-stationary** dataflow that places the inner loop over weights:

```
for (w, i_val) in i:
    for (s, f_val) in f if w-Q <= s < w:
        q = w – s
        o[q] += i_val * f_val
```

For each non-zero input `i[w]`, SCNN iterates over all non-zero weights `f[s]` that could interact with it (those within the valid window). This exploits both input and weight sparsity simultaneously — the outer loop skips zero inputs, the inner loop skips zero weights.

### All-to-all multiplication and the scatter network

To maximize PE utilization, SCNN tiles both the input and weight fibers by position, then performs a **Cartesian product** (all-to-all) multiplication of the non-zero elements within each tile:

![SCNN: Cartesian product of non-zero inputs and non-zero weights](../../assets/L09/L09-p117-scnn-cartesian-product.png)

If a tile has 4 non-zero inputs {i₁, i₂, i₃, i₄} and 4 non-zero weights {w₁, w₂, w₃, w₄}, the PE produces 16 partial products in one shot, using a 4×4 multiplier array. The coordinate of each partial product determines which output accumulator it belongs to via `q = w – s`; products are then routed by a **scatter network** to the correct output partial-sum buffer.

The SCNN PE microarchitecture reflects this: a densely packed frontend stores compressed weights and activations (with metadata), feeds them into an all-to-all multiplier array, and a scatter network routes products to the output-stationary backend accumulator bank.

SCNN's measured results show that at realistic CNN sparsity levels, **latency scales roughly linearly with joint density** (the product of weight and activation density), and **energy per non-zero multiply drops significantly** compared to a dense baseline — confirming that the Cartesian product + scatter approach effectively converts joint sparsity into proportional savings.

> **Why it matters:** SCNN demonstrates that input-stationary dataflow with Cartesian-product multiplication achieves full two-sparse work reduction in hardware. The scatter network is the hardware cost of that generality — partial products land at scattered output addresses, requiring a flexible routing fabric rather than a simple accumulate-in-place structure.

---

## Chapter 6 — ISOSceles: IS-OS Pipelined Dataflow

> *Slides: L09-129 … L09-138*

### The IS-OS two-step computation

ISOSceles (Yang et al., HPCA 2023) introduces an **IS-OS (Input-Stationary / Output-Stationary) pipelined dataflow** designed to combine the strengths of both dataflows while avoiding their individual weaknesses. The key observation is that the convolution Einsum

```
O[n,p,q,m] = I[n,h,w,c] × F[m,c,r,s]   (with h=p+r, w=q+s)
```

can be split into two steps by introducing an intermediate tensor T:

**Step 1 (IS frontend):**
```
T[h,w,r,s] = I[h,w,c] × F[m,c,r,s]
```
This step iterates input-stationarily over non-zero inputs `(h,w)`, intersects with non-zero filters `(c,r,s)`, and accumulates into T indexed by `(h, w-s, r)`. The `w-s` projection maps each (input, weight) pair to the appropriate position in T.

**Step 2 (OS backend):**
```
O[p,q] = T[h,w,r,s]   (with h=p+r, q=w-s)
```
This step reads T in an output-stationary fashion, accumulating partial results into final outputs.

The challenge is that T is written in IS order (indexed by input coordinates `h,w`) but read in OS order (indexed by output coordinates `p,q`). This is a **discordant traversal** of T. ISOSceles solves it by **rank-swizzling** T's storage layout between the two steps — reindexing the stored tensor from `(H,R,Q)` to `(Q,R,H)` — so that the OS backend can traverse T concordantly.

### The pipelined microarchitecture

![ISOSceles IS-OS pipeline: IS frontend → small T buffer → OS backend](../../assets/L09/L09-p136-isosceles-pipeline.png)

The ISOSceles pipeline has three stages:

1. **IS frontend** — processes an input wavefront, computes all partial products involving the current batch of non-zero inputs, and writes results into a small intermediate tensor T.
2. **Small T buffer** — holds the rank-swizzled intermediate results between the two pipeline stages. Keeping T small is critical; ISOSceles tiles the computation so T fits in a local buffer.
3. **OS backend** — reads T concordantly and accumulates results into the output wavefront.

The pipeline delivers a measured **7.5× latency speedup** over a dense baseline on a sparse CNN benchmark, with a **1.7× energy improvement** as well. These results appear on the final slide:

![ISOSceles speedup: 7.5× latency, 1.7× energy improvement](../../assets/L09/L09-p138-isosceles-speedup.png)

The IS frontend exploits input sparsity (skips zero activations); the OS backend exploits output sparsity (skips zero partial sums); and the intersection in Step 1 further reduces work when both operands are sparse. The rank-swizzle overhead is a one-time reindexing of T, which is small compared to the total compute saved.

> **Why it matters:** ISOSceles shows that a single accelerator can exploit both input and weight sparsity through a carefully designed two-stage pipeline, achieving better PE utilization than either pure IS or pure OS alone. The rank-swizzle is a software/hardware co-design trick that converts a discordant access pattern into a concordant one — a concrete example of the Format and Binding layers of the TeAAL pyramid working together.

---

## Standalone Study Guide

### What to master before moving on

- Use the fibertree abstraction to describe dense and sparse tensors uniformly.
- Explain `getPayload()` versus `getNext()` and why concordant traversal is cheap.
- Distinguish coordinate projection from fiber intersection.
- Trace sparse-weight, sparse-input, and two-sparse convolution loop nests.
- Compare Cambricon-X, CNVLUTIN, SCNN, and ISOSceles by which sparsity they exploit and what hardware they add.

### Self-check questions

1. Why does CSR make row-major traversal cheap but column-major traversal expensive?
2. When both operands are sparse, why is projection alone insufficient?
3. What problem does rank swizzling solve in ISOSceles?

### Exercises

1. Draw a fibertree for a 3x3 matrix with three non-zero values, then label coordinates, positions, fibers, and payloads.
2. For an output-stationary sparse-weight convolution, compute the projected input coordinate `w` for several `(q,s)` pairs.
3. Given two sorted sparse fibers, manually perform their intersection and list the emitted matched payloads.

### Common traps

- Confusing coordinate with position. Coordinate is the mathematical index; position is where the item sits in storage.
- Assuming compressed storage implies cheap random access. Many compressed formats are cheap only for concordant traversal.
- Treating SCNN and ISOSceles as equivalent because both exploit two-sparse work. Their output-routing strategies are different.

---

## Key Terms

| Term | Gloss |
|---|---|
| **Fibertree** | A tree-based abstraction for tensors: each level is a rank, each node holds a (coordinate, payload) pair, each child list is a fiber. |
| **Fiber** | An ordered list of (coordinate, payload) tuples at one level of the fibertree. |
| **Coordinate** | The index identifying an element within a rank (equivalent to the mathematical index). |
| **Position** | The physical storage location (offset) in memory; equals coordinate only for uncompressed ranks. |
| **Payload** | The data stored at a fiber node: a reference to the next-level fiber (for non-leaf ranks) or a numeric value (for leaf ranks). |
| **`getPayload(c)`** | Random-access lookup of the payload at coordinate `c` in a fiber. |
| **`getNext()`** | Iterator advancing to the next (coordinate, payload) pair in a fiber (concordant traversal). |
| **Uncompressed (U)** | Fiber representation where every coordinate is stored and position = coordinate; O(1) for both `getPayload` and `getNext`. |
| **Coordinate/payload list (C)** | Compressed fiber storing only non-zero (coordinate, value) pairs; O(log n) `getPayload`, O(1) concordant `getNext`. |
| **CSR** | Compressed Sparse Row: `Tensor<U,C>(H,W)` — uncompressed H rank, compressed W rank. |
| **CSC** | Compressed Sparse Column: `Tensor<U,C>(W,H)` — rank order swapped from CSR. |
| **COO** | Coordinate List: `Tensor<C2>(H,W)` — two ranks merged into one flat list of (H,W) coordinate tuples. |
| **Concordant traversal** | Iterating through a fiber in storage order (cheap: O(1) per step with a coordinate/payload list). |
| **Discordant traversal** | Accessing a fiber out of storage order, e.g., looking up arbitrary coordinates (expensive: O(log n) with a C list). |
| **Coordinate projection** | Deriving one operand's required coordinate from another's coordinate via arithmetic (e.g., `w = q + s`). |
| **Fiber intersection** | Retaining only coordinate positions present in both of two fibers; the hardware primitive for two-sparse work reduction. |
| **Fiber split (position space)** | Dividing a fiber into equal-sized chunks by position (non-zero count) for parallel processing. |
| **Gating** | Suppressing reads and computes when an operand is zero; saves energy but not time. |
| **Skipping** | Restructuring the loop to iterate only over non-zero coordinates; saves both time and energy. |
| **Cambricon-X** | Sparse accelerator (Zhang et al., Micro 2016) exploiting weight sparsity via metadata-guided input loading. |
| **SCNN** | Sparse CNN accelerator (Parashar et al., ISCA 2017) using input-stationary Cartesian-product multiplication and a scatter network for two-sparse reduction. |
| **ISOSceles** | IS-OS pipelined accelerator (Yang et al., HPCA 2023) combining an IS frontend and OS backend with rank-swizzled intermediate tensor T; 7.5× latency speedup. |
| **Rank swizzle** | Reordering (transposing) the rank order of a tensor to convert a discordant traversal into a concordant one. |

---

## Takeaways

- **Fibertrees unify all tensor representations** under one abstraction: fibertrees with uncompressed, coordinate/payload, run-length, hash, or merged (CSR/CSC/COO) rank implementations all implement the same `getPayload`/`getNext` interface; hardware can be designed against the abstraction and then specialized to the representation.
- **Concordant traversal is cheap; discordant is not.** The design of a sparse loop nest must ensure that the operand being iterated concordantly is compressed and the one being looked up is either uncompressed (O(1) random access) or small.
- **Coordinate projection** is the arithmetic that maps between index spaces across operands in a convolution (e.g., `w = q+s`); it is cheap but must be explicitly computed by hardware.
- **Fiber intersection** achieves multiplicative (product-of-densities) work reduction when both operands are sparse; it is the key primitive distinguishing "exploit one sparse operand" from "exploit both."
- **Three real accelerators** embody three points on the design space: Cambricon-X (weight sparsity only, weight-stationary), SCNN (both sparse, input-stationary with Cartesian product), ISOSceles (both sparse, IS-OS pipelined with rank swizzle, best PE utilization).
- **Data layout (Format) and loop order (Mapping) are tightly coupled** for sparse accelerators: changing which rank is compressed or which loop is outer can flip a concordant traversal to a discordant one, turning an O(1) per-step operation into O(log n). These are the Format and Mapping layers of the TeAAL pyramid in action.

---

## Connections to Later Lectures

- **L08 (Sparse Architectures 1):** introduced the motivation for sparsity and basic gating/format concepts; this lecture (L09) extends those foundations to the full fibertree abstraction and loop-nest design for skipping.
- **L10 (Sparse Architectures 3):** continues the sparse-accelerator story with further designs and the TeAAL framework for formally specifying sparse tensor algebra hardware, closing the three-lecture arc.
- **Einsum formalism (L04 / L07):** the `Z[m,n] = A[m,k] × B[k,n]` and convolution Einsum notation used throughout this lecture was introduced earlier; here it is extended with intersection (`&`) and projection (`.project()`) operators for sparse operands.
- **Fiber representations and the Format layer:** the CSR/CSC/COO/bitmask formats introduced here are the concrete instances of the **Format** layer of the TeAAL Pyramid of Concerns first seen in L01.

---

## Appendix — Slide-to-Section Map

| Slides | Section |
|---|---|
| L09-1 | Title |
| L09-2 … L09-5 | Ch.1 — Motivation: sparse tensors everywhere; three sparsity levers |
| L09-6 … L09-17 | Ch.1 — Fibertree abstraction: ranks, coordinates, payloads, fibers |
| L09-18 … L09-35 | Ch.2 — Fiber representation choices; CSR/CSC/COO notation |
| L09-36 … L09-51 | Ch.2 — Traversal efficiency: concordant vs. discordant; hardware datapath |
| L09-52 … L09-71 | Ch.2 — Merging/splitting ranks; sparsity specifications (channel, sub-kernel, 2:4, HSS); Einsum review |
| L09-72 … L09-96 | Ch.3 — Sparse weights in CONV: output-stationary and weight-stationary loop nests; Cambricon-X |
| L09-97 … L09-111 | Ch.4 — Sparse inputs: sliding-window traversal; CNVLUTIN |
| L09-112 … L09-128 | Ch.4 — Two-sparse: projection + intersection; output-stationary hardware |
| L09-112 … L09-124 | Ch.5 — SCNN: Cartesian product, scatter network, latency/energy vs. density |
| L09-129 … L09-138 | Ch.6 — ISOSceles: IS-OS two-step dataflow, rank swizzle, 7.5× speedup |
