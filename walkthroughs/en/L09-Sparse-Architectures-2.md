# L09 — Sparse Architectures, Part 2

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer and Vivienne Sze
> **Lecture date:** March 4, 2026
> **Primary source:** [`Lecture/L09-Sparse_Architectures-2.pdf`](../../Lecture/L09-Sparse_Architectures-2.pdf)

This chapter reconstructs the lecture narration from the public slides and local papers. The original deck is animation-heavy; this walkthrough collapses repeated build slides into self-contained explanations.

---

## TL;DR

Lecture 08 introduced the sparse hardware vocabulary: gating, skipping, format, metadata, and structured sparsity. Lecture 09 turns that vocabulary into a working method for designing sparse convolution accelerators.

The main idea is the **fibertree abstraction**. A tensor is represented as ranks, fibers, coordinates, and payloads. Once we view sparse tensors this way, sparse acceleration becomes a question of which fiber operation is cheap:

- `getNext()` is cheap when traversal is **concordant**, meaning the loop order matches the storage order.
- `getPayload(coordinate)` can be expensive in compressed formats because it is random access.
- **Projection** maps coordinates between tensors, e.g., \(w=q+s\) in convolution.
- **Intersection** keeps only coordinates that are present in both sparse operands.

The lecture then applies these primitives to sparse convolution: sparse weights, sparse inputs, and both operands sparse. It closes with SCNN and ISOSceles, showing two very different ways to exploit the same mathematical sparsity.

## What Problem This Lecture Solves

Lecture 08 showed that skipping requires knowing where the nonzeros are. Lecture 09 asks the next question: once a tensor is compressed, how does the hardware traverse it while still computing the correct DNN operation?

The problem is not simply "store sparse tensors." The problem is:

1. Represent sparse tensors in a way that exposes coordinates.
2. Match loop order to representation order.
3. Use coordinate projection to connect convolution operands.
4. Use intersection when multiple operands are sparse.
5. Choose a dataflow that balances reuse, skipping, routing, and utilization.

This is the bridge from sparse format design to sparse accelerator microarchitecture.

## Why This Lecture Matters

In dense convolution, changing loop order mainly changes reuse. In sparse convolution, changing loop order can change whether the inner loop is a cheap iterator or a random metadata lookup. That is a much sharper constraint.

For example, a coordinate/payload list is excellent when the loop says "visit the next stored nonzero." It is much less convenient when the loop says "tell me whether coordinate 137 exists." This is why sparse architecture design needs a formal vocabulary for traversal.

Hardware implications include:

- **Bandwidth:** compressed formats read fewer values only when traversal is concordant.
- **Latency:** skipping is possible only if the next useful coordinate can be produced quickly.
- **Area:** coordinate generators, position generators, intersection units, and scatter networks become first-class hardware blocks.
- **Utilization:** sparse work is irregular; splitting by position can balance work but may destroy geometric regularity.
- **Correctness:** projection such as \(w=q+s\) and \(q=w-s\) must route products to the right output coordinate.

## Prerequisites and Mental Model

You should remember:

- Lecture 08's distinction among gating, skipping, and format.
- 1-D convolution: \(O[q] = \sum_s I[q+s]F[s]\).
- Tensor ranks and coordinates from the earlier Einsum lectures.
- Dataflow: which loop is stationary, which loop streams, and which loop is parallelized.

Mental model:

> Sparse convolution is dense convolution plus explicit coordinate bookkeeping.

The MAC is still simple. The difficult part is producing the right pairs of nonzero payloads and sending each product to the correct partial sum.

## Learning Objectives

After this lecture, you should be able to:

- Define rank, coordinate, point, fiber, payload, and fibertree.
- Explain the difference between coordinate and position.
- Compare uncompressed arrays, coordinate/payload lists, bitmasks, RLE, hash tables, CSR, CSC, and COO.
- Explain `getPayload()` and `getNext()` and why their costs differ by representation.
- Distinguish concordant, partially discordant, and discordant traversal.
- Use \(w=q+s\), \(q=w-s\), and \(s=w-q\) as coordinate projections in 1-D convolution.
- Write loop nests for sparse weights, sparse inputs, and two-sparse convolution.
- Explain how Cnvlutin exploits activation sparsity.
- Explain how SCNN uses input-stationary Cartesian-product sparse multiplication and why it needs scatter.
- Explain what ISOSceles tries to solve with an IS-OS pipeline and rank swizzling.

## Main Textbook-Style Narrative

### 1. Sparse tensors need a representation-independent abstraction

The lecture begins with a simple tensor vocabulary:

- A **rank** is a tensor dimension.
- A **coordinate** is an index along a rank.
- A **point** is a tuple of coordinates, one per rank.
- A **payload** is either a scalar value or a pointer/reference to a lower-rank fiber.
- A **fiber** is an ordered set of coordinate/payload pairs at one rank.
- A **fibertree** is a tree whose levels are ranks and whose edges carry fibers.

For a dense \(3 \times 3\) matrix, every row and every column coordinate exists. For a sparse matrix, all-zero subtrees can be omitted. A point such as \((2,1)\) is found by asking the H-rank fiber for coordinate 2, then asking the W-rank child fiber for coordinate 1.

The abstraction matters because it separates semantics from layout. The tensor has mathematical coordinates; the implementation chooses how to store those coordinates and payloads.

### 2. Coordinate is not position

A **coordinate** is the mathematical index. A **position** is the physical offset in storage.

In an uncompressed vector, coordinate equals position. In a compressed coordinate/payload list, they differ:

```text
Dense vector coordinates:   0  1  2  3  4  5
Dense values:               0  0  c  d  0  f

Compressed positions:       0  1  2
Stored coordinates:         2  3  5
Stored payloads:            c  d  f
```

Coordinate 5 is stored at position 2. This distinction becomes crucial when a loop variable such as \(s\) is a mathematical coordinate, but a hardware pointer is a storage position.

### 3. Fiber operations: `getPayload` and `getNext`

The lecture uses two operations:

- `getPayload(coordinate)`: return the payload at a requested coordinate, if present.
- `getNext()`: return the next coordinate/payload pair in traversal order.

Their costs depend on representation:

| Representation | `getPayload(c)` | Concordant `getNext()` | Hardware intuition |
|---|---|---|---|
| Uncompressed array | \(O(1)\) | \(O(1)\) | Address = coordinate |
| Coordinate/payload list | \(O(\log n)\) with binary search, or scan | \(O(1)\) | Pointer increments are cheap; random lookup is not |
| RLE | \(O(n)\) scan/accumulate | \(O(1)\) | Good streaming, poor random access |
| Hash table | \(O(1)\) average | Poor locality for ordered traversal | Uses hashing and extra references |
| Bitmask | Test bit; payload lookup may need popcount | Often cheap | Good for presence checks |

This table explains the lecture's repeated warning: compressed storage does not automatically mean efficient sparse computation. It is efficient only for the traversal patterns it supports.

### 4. Concordant and discordant traversal

A traversal is **concordant** when the loop visits coordinates in the same order the representation stores them. A row-major scan over CSR is concordant. A column-major scan over CSR is discordant because each column lookup jumps across row fibers.

CSR and CSC have the same nonzeros and similar storage cost, but opposite natural traversal orders:

- CSR, written as `Tensor<U,C>(H,W)`, has uncompressed H rank and compressed W fibers. It is naturally row-major.
- CSC, written as `Tensor<U,C>(W,H)`, swaps rank order and is naturally column-major.

Hardware meaning:

- Concordant traversal can be implemented with counters, pointers, and sequential SRAM reads.
- Discordant traversal may require binary search, hash lookup, decompression, or a format conversion.

This is the first major bridge from Lecture 09 back to mapping: loop order must be selected together with format.

### 5. CSR as a concrete fibertree implementation

For a sparse matrix with nonzeros:

```text
row 0: (col 0 -> a), (col 2 -> c)
row 1: empty
row 2: (col 0 -> g), (col 1 -> h)
```

CSR stores:

```text
segment array:    [0, 2, 2, 4]
coordinate array: [0, 2, 0, 1]
value array:      [a, c, g, h]
```

The segment array tells where each row's W fiber begins and ends. Row 1 has start and end both equal to 2, so it is empty. This is a concrete implementation of a fibertree: the H rank is uncompressed, and each W-rank fiber is compressed.

### 6. Rank transformations: merge, split, swizzle

Sparse accelerators often transform ranks:

- **Merge/flatten:** combine ranks, e.g., \((H,W)\) into a single coordinate tuple. COO is a coordinate-list version of this idea.
- **Split by coordinate space:** divide a rank into fixed coordinate ranges. This preserves geometry but can give uneven nonzero counts.
- **Split by position space:** divide a rank by stored nonzero count. This balances work but creates irregular coordinate ranges.
- **Swizzle:** reorder ranks, e.g., changing a tensor's rank order from \([H,R,Q]\) to \([Q,R,H]\).

These transformations are not mathematical changes to the DNN operation. They are layout and scheduling choices that make a sparse dataflow more implementable.

### 7. Convolution as coordinate projection

The 1-D convolution in the lecture is:

\[
O[q] = \sum_s I[q+s]F[s].
\]

The relation \(w=q+s\) is a **coordinate projection**: given output coordinate \(q\) and filter coordinate \(s\), it tells which input coordinate \(w\) is needed.

Projection has three common forms:

- Output-stationary view: \(w=q+s\).
- Weight-stationary input view: \(q=w-s\).
- Output-stationary sparse-input view: \(s=w-q\).

The arithmetic is small, but it is not optional. Without this coordinate generator, the hardware cannot route the product to the correct partial sum.

### 8. Exploiting sparse weights

If weights are sparse and inputs are dense or randomly accessible, the natural schedule is to iterate the compressed filter concordantly:

```text
for q in [0, Q):
    for (s, f_val) in f:
        w = q + s
        o[q] += i[w] * f_val
```

This is output-stationary: \(o[q]\) remains the accumulation target while the filter nonzeros stream. The filter traversal is cheap because it uses `getNext()`. The input \(i[w]\) should be uncompressed or otherwise cheap to index.

A weight-stationary version reverses the outer loops:

```text
for (s, f_val) in f:
    for q in [0, Q):
        w = q + s
        o[q] += i[w] * f_val
```

Both exploit weight sparsity. They differ in reuse: one keeps the output stationary, the other keeps the weight stationary.

Hardware blocks:

- A position generator for the compressed filter fiber.
- A coordinate generator computing \(w=q+s\).
- A random-access input buffer.
- A partial-sum buffer indexed by \(q\).

Cambricon-X is slide-cited as an accelerator that follows this weight-sparsity style: metadata associated with weights guides activation access.

### 9. Exploiting sparse inputs

If inputs are sparse and weights are dense or randomly accessible, iterate nonzero inputs. For weight-stationary sparse inputs:

```text
for s in [0, S):
    for (w, i_val) in i if s <= w < Q+s:
        q = w - s
        o[q] += i_val * f[s]
```

The condition \(s \le w < Q+s\) is a sparse sliding window. Each weight \(s\) only interacts with input coordinates \(w\) that produce legal output coordinates \(q\).

For output-stationary sparse inputs:

```text
for q in [0, Q):
    for (w, i_val) in i if q <= w < q+S:
        s = w - q
        o[q] += i_val * f[s]
```

This visits only nonzero inputs in the active window, then looks up the corresponding weight. The sparse input traversal can be cheap; the weight lookup must also be cheap, so weights are often uncompressed for this view.

Cnvlutin is slide-cited and paper-verified as an activation-sparsity design. Its central idea is to remove zero-valued neurons from the input stream and encode nonzero neurons with offsets so lanes can skip zeros while still indexing the correct weights.

### 10. Exploiting sparse weights and sparse inputs

When both operands are sparse, projection alone is insufficient. You must project one operand's coordinates into the other's coordinate space and then intersect.

Output-stationary two-sparse convolution can be written:

```text
for q in [0, Q):
    for (s, (f_val, i_val)) in f.project(+q) & i:
        o[q] += i_val * f_val
```

Read this carefully:

1. `f.project(+q)` converts each filter coordinate \(s\) into input coordinate \(w=s+q\).
2. `& i` intersects the projected filter fiber with the input fiber.
3. Only coordinates present in both operands produce a multiply.

Example:

```text
filter nonzeros s:        0, 2, 5, 6
for q = 2, projected w:   2, 4, 7, 8
input nonzeros w:         1, 2, 5, 8
intersection:             2, 8
```

Only projected coordinates 2 and 8 have matching input activations. The hardware emits two products, not four.

### 11. SCNN: Cartesian-product sparse convolution

SCNN uses an input-stationary Cartesian-product dataflow. A simplified 1-D view is:

```text
for (w, i_val) in i:
    for (s, f_val) in f if w-Q <= s < w:
        q = w - s
        o[q] += i_val * f_val
```

SCNN tiles input activations and weights by position and multiplies groups of nonzeros against each other. If a tile supplies \(I\) nonzero activations and \(F\) nonzero weights, a PE can form up to \(I \times F\) products.

The catch is output routing. The product's output coordinate is computed from \(q=w-s\), so products from the same Cartesian product tile scatter to different output locations. SCNN therefore needs a scatter network and a dense accumulation backend.

This is a beautiful sparse-design tradeoff:

- The compressed frontend keeps zeros away from the multipliers.
- The all-to-all multiplier array exposes many useful products.
- The scatter/accumulator backend pays the price of irregular output coordinates.

### 12. ISOSceles and the IS-OS pipeline

The slides end with ISOSceles, an IS-OS dataflow. The idea is to split convolution through an intermediate tensor \(T\):

\[
T[\cdot] = I[\cdot]F[\cdot],
\qquad
O[\cdot] = \text{reduce}(T[\cdot]).
\]

The exact slide notation uses substitutions such as \(h=p+r\) and \(q=w-s\). The conceptual point is:

1. An input-stationary frontend processes sparse input wavefronts and creates partial results in \(T\).
2. An output-stationary backend reads \(T\) and accumulates final outputs.
3. The intermediate \(T\) is written in one rank order and read in another, so ISOSceles uses rank swizzling to make the backend traversal concordant.

This is the same lesson as CSR vs. CSC, now inside a pipeline: if a tensor is produced in one order and consumed in another, a rank transformation may be cheaper than repeated discordant lookups.

Quantitative ISOSceles results in this chapter are slide-derived because the local ISOSceles PDF was not part of the Worker B input.

## Worked Examples

### Example 1: CSR lookup vs. traversal

Using:

```text
segment array:    [0, 2, 2, 4]
coordinate array: [0, 2, 0, 1]
value array:      [a, c, g, h]
```

Concordant row traversal:

- Row 0 uses positions 0 to 1: \((0,a),(2,c)\).
- Row 1 uses positions 2 to 1: empty.
- Row 2 uses positions 2 to 3: \((0,g),(1,h)\).

This is cheap because each W fiber is read sequentially. Asking "what is column 1 in every row?" is not as cheap in CSR, because each row's compressed coordinate list may need a search.

### Example 2: Sparse-weight convolution

Let \(f\) have nonzeros \((s=0,f_0=8)\) and \((s=2,f_2=6)\). Let \(Q=3\). Output-stationary sparse-weight traversal emits:

- For \(q=0\): use \(w=0\) and \(w=2\).
- For \(q=1\): use \(w=1\) and \(w=3\).
- For \(q=2\): use \(w=2\) and \(w=4\).

Dense traversal with \(S=3\) would perform 9 filter positions. Sparse traversal performs 6 because only two of three filter coordinates exist.

### Example 3: Projection plus intersection

Let \(q=1\), filter nonzeros \(s=\{0,3,4\}\), input nonzeros \(w=\{1,2,5\}\).

Project filter by \(+q\): \(w=\{1,4,5\}\). Intersect with input: \(\{1,5\}\). Only \(s=0\) and \(s=4\) produce products. The \(s=3\) nonzero weight is real, but for this output coordinate it wants \(w=4\), where the input is zero.

## Key Equations and How to Read Them

### 1-D convolution

\[
O[q] = \sum_s I[q+s]F[s].
\]

The equation says that output coordinate \(q\) accumulates products from filter coordinate \(s\) and input coordinate \(w=q+s\).

### Projection equations

\[
w=q+s,\qquad q=w-s,\qquad s=w-q.
\]

These are the same convolution relation solved for different coordinates. Which one appears in hardware depends on dataflow.

### Ideal two-sparse work

\[
N_\text{work}\approx d_I d_F N_\text{dense}.
\]

This approximation assumes independent sparse positions. It explains why exploiting both input and weight sparsity can be multiplicative. It is a teaching model, not a guaranteed workload statistic.

## Hardware Implications

- **Fibertree hardware:** each rank can be implemented with a position generator and a coordinate/payload extractor.
- **CSR/CSC:** the rank order determines which traversal is cheap. Format conversion or rank swizzling may be required for a different consumer.
- **Sparse weights:** compressed filter traversal is cheap, but input access becomes projected random access.
- **Sparse inputs:** input traversal is cheap, but weight lookup and sliding-window restriction become important.
- **Two-sparse:** intersection hardware determines whether the accelerator can actually realize product-of-densities work reduction.
- **SCNN:** all-to-all multiplication improves useful work exposure, but scatter routing and dense accumulators cost area and energy.
- **ISOSceles:** intermediate tensors can decouple IS and OS advantages, but rank swizzling and buffering become explicit costs.

## Common Misconceptions

### Misconception: A compressed tensor is always faster to traverse.

Only concordant traversal is naturally fast. Random lookup into compressed metadata can be slower than dense access.

### Misconception: Coordinate and position are interchangeable.

They are equal only in uncompressed ranks. In compressed ranks, coordinate is the mathematical index and position is the storage offset.

### Misconception: Projection is the same as intersection.

Projection changes coordinate spaces. Intersection removes coordinates not present in both operands. Two-sparse convolution needs both.

### Misconception: SCNN's Cartesian product means it multiplies everything densely.

SCNN forms Cartesian products of compressed nonzero groups, not dense tensors. It avoids zero operands but still must scatter products to irregular output coordinates.

### Misconception: Rank swizzling is just a software transpose.

In sparse hardware, rank swizzling is a design choice that can trade one-time reordering/buffering for cheaper repeated traversal.

## Connections to Previous and Later Lectures

- **L04 Einsums:** the convolution equations are Einsums with index arithmetic.
- **L05-L06 mapping/dataflow:** stationarity and loop order determine which tensor is traversed and which tensor is looked up.
- **L08 sparse architectures:** gating, skipping, metadata, format, and intersection are the vocabulary used here.
- **L10 sparse architectures part 3:** TeAAL formalizes these choices as mapped Einsum cascades, formats, bindings, and architecture specifications.
- **Lab 4/SparseLoop:** the cost of gating, skipping, and sparse formats can be modeled only when traversal and format are specified together.

## Paper Bridge: TeAAL

### Bibliographic identity

- **Title:** TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators
- **Authors:** N. Nayak et al.
- **Year / venue:** MICRO 2023
- **Local PDF:** `papers/TeAAL.pdf`

### Problem addressed

TeAAL provides a precise way to describe sparse tensor accelerators whose behavior depends on mapped Einsums, fibertree formats, rank transformations, and real sparse inputs.

### Core idea

The paper uses fibertrees as the tensor abstraction, mapped Einsums as the computation/mapping abstraction, and transformations such as flattening, partitioning, and swizzling to express sparse orchestration.

### Relevance to this lecture

Lecture 09's fibertree vocabulary, rank transformations, concordant traversal, and IS-OS rank swizzling are exactly the ideas TeAAL formalizes.

### Key claims used in this chapter

- TeAAL Section 2.1 defines ranks, coordinates, points, fibers, payloads, and fibertrees, and notes that sparse fibertrees omit empty payloads.
- Section 2.2 states that Einsums specify computation but do not specify iteration order.
- Section 2.3 explains mapping attributes such as loop order, partitioning, and work scheduling, and connects sparse compression to load imbalance and memory footprint variation.
- Section 3.2 describes rank flattening, partitioning, sorting/merging, and rank swizzling as content-preserving transformations.

### What students should remember

1. Fibertrees are not just diagrams; they are an interface for sparse traversal.
2. Mapping determines whether `getNext()` or `getPayload()` dominates.
3. Rank transformations are architectural tools for making sparse traversal feasible.

### Limitations and assumptions

TeAAL models accelerators; it does not by itself decide which accelerator is best. This chapter uses it to ground the terminology.

## Paper Bridge: Cnvlutin

### Bibliographic identity

- **Title:** Cnvlutin: Ineffectual-Neuron-Free Deep Neural Network Computing
- **Authors:** J. Albericio et al.
- **Year / venue:** ISCA 2016
- **Local PDF:** `papers/L18_Cnvlutin_Albericio_ISCA2016.pdf`

### Problem addressed

Cnvlutin targets zero-valued input neurons in convolutional layers. Baseline wide-lane DNN accelerators process neurons in lockstep, so a zero neuron can waste multiplier slots and cycles.

### Core idea

Cnvlutin stores nonzero input neurons in a Zero-Free Neuron Array format with offsets. The offsets let each nonzero neuron find the correct synapse/weight location while lanes proceed independently, skipping zero neurons without changing the DNN result.

### Relevance to this lecture

Cnvlutin is the concrete paper bridge for sparse-input traversal. It shows why skipping activation zeros requires both a format and a dispatch mechanism, not just an if-statement.

### Key claims used in this chapter

- The abstract identifies zero-operand multiplications as intrinsically ineffectual and presents Cnvlutin as value-based acceleration that removes them.
- Section III describes decoupling neuron lanes and using an encoded input format to remove zero-valued neurons from the critical path.
- Section IV-B describes the Zero-Free Neuron Array format, including nonzero value/offset pairs grouped into bricks.
- Section V reports speedup and energy improvements over the paper's baseline; this chapter does not reuse those exact numbers except as paper context.

### What students should remember

1. Activation sparsity is runtime data-dependent.
2. Offsets are coordinate metadata; they let compressed activations still address the right weights.
3. Lane decoupling is a utilization solution to irregular nonzero counts.

### Limitations and assumptions

Cnvlutin focuses on activation sparsity in convolutional layers and builds on a DaDianNao-like baseline. It does not exploit pruned weight sparsity in the same way SCNN does.

## Paper Bridge: SCNN

### Bibliographic identity

- **Title:** SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks
- **Authors:** A. Parashar et al.
- **Year / venue:** ISCA 2017
- **Local PDF:** `papers/L17_SCNN_Parashar_ISCA2017.pdf`

### Problem addressed

SCNN targets the combined sparsity of pruned weights and ReLU activations in CNN inference, while trying to keep both weights and activations compressed through the computation.

### Core idea

The PT-IS-CP-sparse dataflow is planar-tiled, input-stationary, and Cartesian-product based. It feeds compressed nonzero weight and activation groups to an all-to-all multiplier array, computes output coordinates from metadata, and scatters products into dense accumulators.

### Relevance to this lecture

SCNN is the main concrete example of two-sparse convolution. It demonstrates how projection and multiplication can be efficient while output accumulation becomes irregular.

### Key claims used in this chapter

- The abstract states that SCNN exploits zero-valued weights from pruning and zero-valued activations from ReLU.
- Section II motivates multiplicative work reduction from the product of weight and activation densities.
- Section III defines the PT-IS-CP-sparse dataflow and its input-stationary Cartesian-product structure.
- Section IV describes the PE architecture with compressed buffers, \(F \times I\) multipliers, coordinate handling, and scatter accumulation.
- Section VIII summarizes that SCNN keeps both weights and activations compressed and delivers only nonzero operands to multipliers.

### What students should remember

1. SCNN's work reduction comes from both operands being sparse.
2. Cartesian-product multiplication is a way to keep multipliers busy with nonzero groups.
3. Sparse output routing is the price of that freedom.

### Limitations and assumptions

The design is specialized to CNN-style convolution and relies on the paper's compressed block organization and accumulator banking. Its quantitative results should not be generalized without checking workload and baseline.

## Paper Bridge: Eyeriss v2

### Bibliographic identity

- **Title:** Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices
- **Authors:** Y.-H. Chen et al.
- **Year / venue:** JETCAS 2019
- **Local PDF:** `papers/L17_EyerissV2_Chen_JETCAS2019.pdf`

### Problem addressed

Eyeriss v2 addresses compact and sparse DNNs whose shapes make dense reuse and PE utilization difficult.

### Core idea

It uses a hierarchical mesh NoC and sparse PE architecture. In sparse PE mode, activations and weights use CSC-like compressed streams, with address and count metadata that allow zeros to be skipped in the compressed domain.

### Relevance to this lecture

Eyeriss v2 is a concrete implementation of the Lecture 08 to Lecture 09 transition: original Eyeriss gated zero activations; Eyeriss v2 uses compressed formats and skipping to improve throughput.

### Key claims used in this chapter

- Section IV explicitly distinguishes original Eyeriss gating from Eyeriss v2 sparse processing that skips zeros for throughput.
- Section IV describes CSC compressed data, count/address metadata, and sparse PE pipeline support.
- Section V reports implementation results and discusses workload imbalance and cases where sparse compression may not create skippable cycles.

### What students should remember

1. Sparse support changes the PE pipeline, not only the memory format.
2. A flexible NoC matters because sparse and compact models stress bandwidth/reuse differently.
3. Workload imbalance remains a limiting factor.

### Limitations and assumptions

The paper's reported improvements depend on implementation, workloads, and baselines. This chapter uses Eyeriss v2 for architectural mechanisms, not universal performance constants.

## Standalone Study Guide

Study this lecture in five passes:

1. Draw a fibertree for a tiny sparse matrix.
2. Label coordinate vs. position for every stored value.
3. For each representation, ask whether `getPayload` or `getNext` is cheap.
4. Rewrite \(O[q]=\sum_s I[q+s]F[s]\) in output-stationary, weight-stationary, and input-stationary forms.
5. Explain SCNN and ISOSceles as two different answers to "where do sparse products accumulate?"

## Self-Check Questions

1. Why does CSR make row traversal cheap but column traversal expensive?
2. What is the difference between coordinate and position in a compressed fiber?
3. Why is `getNext()` cheap for coordinate/payload lists but `getPayload()` may be expensive?
4. In sparse-weight convolution, why should the input often be uncompressed?
5. In sparse-input convolution, what does the sliding-window condition do?
6. Why does two-sparse convolution require projection and intersection?
7. Why does SCNN need a scatter network?
8. What problem does rank swizzling solve in the IS-OS pipeline?

## Exercises

1. **Fibertree:** Draw the fibertree and CSR arrays for a \(4 \times 4\) matrix with nonzeros at \((0,1)\), \((2,0)\), and \((2,3)\).
2. **Traversal:** For the same matrix, list the operations needed to traverse by row and by column if the format is CSR.
3. **Projection:** Let \(Q=4\), \(S=3\), and filter nonzeros \(s=\{0,2\}\). List all projected \(w=q+s\) coordinates for each \(q\).
4. **Intersection:** For \(q=2\), filter nonzeros \(s=\{0,1,4\}\), and input nonzeros \(w=\{2,3,5,7\}\), compute projection plus intersection.
5. **Design tradeoff:** Compare Cnvlutin and SCNN. Which sparsity does each exploit, and what extra hardware does each need?
6. **Paper reading:** In SCNN Section III, identify why input stationarity is attractive and why accumulation becomes harder.

## Key Terms

| Term | Definition |
|---|---|
| **Rank** | A tensor dimension, represented as one level of a fibertree. |
| **Coordinate** | Mathematical index within a rank. |
| **Point** | Tuple of coordinates identifying a tensor element. |
| **Position** | Physical offset in a storage array; equals coordinate only in uncompressed ranks. |
| **Payload** | Either a scalar value or a pointer/reference to a lower-rank fiber. |
| **Fiber** | Ordered set of coordinate/payload pairs at one rank. |
| **Fibertree** | Tree representation of a tensor where ranks are levels and fibers connect levels. |
| **`getPayload(c)`** | Random lookup of the payload at coordinate \(c\). |
| **`getNext()`** | Iterator returning the next coordinate/payload pair in traversal order. |
| **CSR** | Compressed Sparse Row; `Tensor<U,C>(H,W)`, efficient for row-major traversal. |
| **CSC** | Compressed Sparse Column; `Tensor<U,C>(W,H)`, efficient for column-major traversal. |
| **COO** | Coordinate list format storing merged coordinate tuples. |
| **Concordant traversal** | Traversal order matches storage/rank order, enabling cheap sequential reads. |
| **Discordant traversal** | Traversal order conflicts with storage/rank order, often requiring random lookup. |
| **Projection** | Coordinate arithmetic mapping one tensor's coordinate space into another, e.g., \(w=q+s\). |
| **Intersection** | Operation that emits only coordinates present in both sparse operands. |
| **Position-space split** | Splitting a fiber by stored nonzero count to balance work. |
| **Coordinate-space split** | Splitting a fiber by coordinate ranges to preserve geometry. |
| **Rank swizzle** | Reordering tensor ranks to make a later traversal concordant. |
| **Cnvlutin** | Activation-sparsity accelerator using zero-free neuron encoding and lane decoupling. |
| **SCNN** | Sparse CNN accelerator using input-stationary Cartesian-product sparse multiplication. |
| **ISOSceles** | Slide-cited IS-OS sparse convolution dataflow using an intermediate tensor and rank swizzling. |

## Takeaways

- Fibertrees give a common language for dense and sparse tensor layouts.
- Coordinate and position must be separated whenever a rank is compressed.
- Sparse formats are traversal-dependent: concordant `getNext()` can be cheap while discordant `getPayload()` can be expensive.
- Convolution sparsity requires coordinate projection; two-sparse convolution additionally requires intersection.
- Cnvlutin, SCNN, Eyeriss v2, and ISOSceles occupy different points in the design space: activation skipping, two-sparse Cartesian products, compressed-domain sparse PEs, and IS-OS pipelining.
- Sparse dataflow design is a three-way negotiation among reuse, skipping, and output routing.

## Connections

- **L08:** introduced SAFs and metadata tradeoffs; L09 shows how metadata is traversed.
- **L10:** continues toward formal sparse accelerator specification and TeAAL-style modeling.
- **Mapping lectures:** stationarity determines whether a tensor is streamed, looked up, or accumulated.
- **Einsum lectures:** projection and reduction are direct consequences of convolution index expressions.
- **Paper bridges:** TeAAL provides the abstraction, Cnvlutin shows activation skipping, SCNN shows two-sparse Cartesian products, and Eyeriss v2 shows compressed-domain sparse PE design.

## Appendix — Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L09-1 to L09-5 | Motivation and SAF recap | Expanded as TL;DR, problem, and why it matters |
| L09-6 to L09-18 | Tensor terminology and fibertree | Expanded with coordinate/position explanation |
| L09-19 to L09-35 | Fiber representations, CSR/CSC | Rewritten as representation and storage examples |
| L09-36 to L09-51 | Traversal efficiency | Expanded with `getPayload`/`getNext` cost model |
| L09-52 to L09-60 | Merge, split, sparsity specs, HSS | Integrated into rank transformations |
| L09-61 to L09-71 | Einsum review and convolution projection | Rewritten as coordinate projection |
| L09-72 to L09-96 | Sparse weights in convolution | Expanded with loop nests and hardware blocks |
| L09-97 to L09-111 | Sparse inputs and Cnvlutin | Expanded with sparse sliding window |
| L09-112 to L09-128 | Sparse inputs and weights, SCNN, intersection | Expanded with projection-plus-intersection example |
| L09-129 to L09-138 | ISOSceles IS-OS dataflow | Slide-derived explanation; no local ISOSceles PDF in Worker B input |

## Source Notes

- Fibertree terminology and traversal ordering follow Lecture 09 slides 6-51 and TeAAL Sections 2.1-2.3.
- Rank merge/split/swizzle discussion follows Lecture 09 slides 51-60 and TeAAL Section 3.2.
- Convolution projection and loop nests follow Lecture 09 slides 72-128.
- Cnvlutin discussion uses `papers/L18_Cnvlutin_Albericio_ISCA2016.pdf`, especially Sections III, IV-B, and V.
- SCNN discussion uses `papers/L17_SCNN_Parashar_ISCA2017.pdf`, especially Sections II-IV and VIII.
- Eyeriss v2 discussion uses `papers/L17_EyerissV2_Chen_JETCAS2019.pdf`, especially Sections IV-V.
- Cambricon-X and ISOSceles are discussed from Lecture 09 slide anchors only; local PDFs were not included in the Worker B input.
- `papers/L08_FastAlgorithmsWinograd_Lavin_2015.pdf` was inspected but not used substantively because Winograd convolution is not central to Lecture 09's sparse traversal narrative.

## Uncertainty Notes

- This chapter reconstructs the likely lecture narration from slides and papers; live lecture emphasis may differ.
- ISOSceles quantitative claims are slide-derived only.
- Existing `assets/L09/` image files may be copyright-sensitive, but asset cleanup is outside Worker B's requested write scope.
