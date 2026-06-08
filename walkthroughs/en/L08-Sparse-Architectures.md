# L08 — Sparse Architectures

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze (MIT EECS)
> **Lecture date:** March 2, 2026 · **Slides:** 96 · **Source:** [`Lecture/L08 - Sparse Architectures.pdf`](../../Lecture/L08%20-%20Sparse%20Architectures.pdf)
>
> *This is a conceptual walkthrough that reconstructs the lecture's narrative from the slides. It is organized by idea, not slide-by-slide. Each section cites the slide range it draws from so you can follow along with the original deck.*

---

## TL;DR

Modern DNNs are **sparse**: many of their weights and activations are zero, so roughly half (or more) of all multiply-accumulate operations are *ineffectual* — they contribute nothing to the result. The question is not whether to exploit sparsity, but *how* and at *what cost*. This lecture introduces three hardware strategies — **gating**, **skipping**, and **compressed-format representation** — collectively called **Sparse Acceleration Features (SAFs)**. It works out *exactly* how many reads and cycles each strategy saves on a concrete 1-D dot-product example, derives the tradeoffs between different **representation formats** (Uncompressed, Bitmask, Coordinate Payload, Run-Length Encoding), then moves on to **structured sparsity**, **Hierarchical Structured Sparsity (HSS)**, and finally the thorny problem of **tiling sparse tensors** — where overbooking strategies such as **Tailors + Swiftiles** are needed to maximize buffer utilization. Throughout, concrete accelerator designs (Eyeriss, SCNN, ExTensor, HighLight) illustrate how academic theory has been put into silicon.

---

## Learning Objectives

After this lecture you should be able to:

- Explain the two root sources of sparsity in DNNs (**activation sparsity** from ReLU and **weight sparsity** from pruning) and define *effectual* vs. *ineffectual* operations.
- Distinguish **gating** from **skipping** and state precisely what each saves (energy only vs. energy and time).
- Explain **single-sided** vs. **dual-sided** intersection and derive the read/cycle counts for each case.
- Name the four canonical representation formats (Uncompressed, Bitmask, Coordinate Payload, Run-Length Encoding) and compare their **compression efficiency** vs. **access efficiency** across the density spectrum.
- Contrast **unstructured** and **structured** sparsity, and explain how **G:H sparsity** (e.g., NVIDIA's 2:4) and **Hierarchical Structured Sparsity (HSS)** extend the sparsity-degree spectrum while keeping hardware simple.
- Describe the **uniform-occupancy vs. uniform-shape** tiling dilemma and explain how **overbooking with Tailors + Swiftiles** resolves it for ExTensor-style sparse accelerators.

---

## Chapter 1 — Why Sparsity Matters (and Why It's Hard)

> *Slides: L08-1 … L08-8*

### Two sources of zeros

Sparsity arises from two largely independent phenomena:

- **Activation (input) sparsity** — the ReLU nonlinearity clamps every negative value to zero, often leaving 50–80% of feature-map values at zero. Correlation in the input data and structure in certain representations (e.g., graph adjacency matrices) can drive activation sparsity even higher.
- **Weight sparsity** — network pruning (Han et al., NeurIPS 2015) removes small-magnitude weights, producing models where 50–90% of parameters are structurally zero without significant accuracy loss.

The two types compound: a pruned weight multiplied by a zero activation is doubly ineffectual.

### The arithmetic of ineffectual operations

The lecture formalizes two counts:

- **Total operations** = effectual operations + **ineffectual operations**
- **Total operations performed** = effectual operations + **unexploited** ineffectual operations

Any operation involving a zero is ineffectual: `anything × 0 = 0` and `anything + 0 = anything`. These contribute nothing to the final output but consume storage bandwidth, PE cycles, and energy if the hardware does not detect and suppress them. The goal of sparse acceleration is to drive *unexploited ineffectual operations* toward zero — but doing so is not free.

### The irregularity problem

Exploiting sparsity makes processing **irregular**. Non-zero counts vary within and across tensors, causing:

- **Variation in cycles** — different tiles finish at different times, leaving PEs idle.
- **Variable storage demand** — compressed tiles have different sizes, complicating buffer management.
- **Unknown non-zero locations** — the hardware must discover or pre-compute where the non-zeros reside.

The irregularity problem motivates every design choice in this lecture.

![Sparse Acceleration Feature (SAF) taxonomy — gating, skipping, format](../../assets/L08/L08-p07-saf-overview.png)

> **Why it matters:** Both training-side (pruning) and inference-side (ReLU) produce zero-rich tensors. A hardware design that ignores this is wasting at least half its memory bandwidth and compute cycles on useless work — which, given that DRAM access costs ~200× an ALU operation, translates directly into energy and time wasted.

---

## Chapter 2 — Gating vs. Skipping: The Two Modes of Sparse Acceleration

> *Slides: L08-9 … L08-22*

### The running example

The lecture uses a 1-D dot product of two 6-element vectors to build intuition:

```
A = [ 0  0  c  d  0  f ]
B = [ g  h  0  j  k  l ]
Z = A · B = c·0 + d·j + f·l = dj + fl
```

Total operations = 6; effectual operations = 2; ineffectual operations = 4.

![The 1-D dot product example — 6 total, 2 effectual, 4 ineffectual operations](../../assets/L08/L08-p09-dot-product-example.png)

### Finding the intersection

To know which operations are effectual, the hardware must determine which positions in A and B are *simultaneously* non-zero — this is the **intersection** problem. Two orthogonal design axes:

1. **Single-sided vs. dual-sided**: Does the hardware check one operand (the *leader*) or both?
   - *Single-sided (leader → follower)*: Scan the leader for non-zeros; fetch the follower only when the leader is non-zero.
   - *Dual-sided*: Scan both operands and compute the intersection of their non-zero coordinate sets.

2. **Gating vs. skipping**: What does the hardware *do* with zero-valued leader reads?
   - **Gating**: The cycle still runs; the hardware simply suppresses the memory read of the follower and the multiply, staying idle — saving **energy but not cycles**.
   - **Skipping**: The hardware jumps to the *next non-zero coordinate* in the leader (or the intersection), eliminating the cycle entirely — saving **both energy and time**.

### The cycle/read accounting

The lecture works through all four combinations on the A-B example and tabulates the results:

| Strategy | Cycles | A reads | B reads | Operations performed |
|---|---|---|---|---|
| None (baseline) | 6 | 6 | 6 | 6 |
| Gating (Gate B ← A) | 6 | 6 | 3 | 3 |
| Skipping (Skip B ← A) | 3 | 3 | 3 | 3 |
| Dual-sided skipping (A ⋂ B) | 2 | 2 | 2 | 2 |

Gating reduces reads and compute operations but leaves the cycle count unchanged, because the hardware must *check* the leader in every cycle even when it is zero. Skipping requires that the next non-zero coordinate be available *before* the cycle starts — which is exactly the job of a **representation format**.

![Impact of different SAF approaches — cycle, read, and compute counts](../../assets/L08/L08-p12-impact-approaches-detail.png)

### Gating vs. Skipping — practical implications

Because gating leaves cycles unchanged, the hardware can discover zeros **in real time** (no pre-computation needed). Skipping requires that the coordinate of the next non-zero be available *a priori* — this is why the lecture immediately turns to compressed tensor formats.

Table of SAFs used in real accelerators (slide 11):

| Accelerator | SAF strategy |
|---|---|
| Eyeriss [JSSC 2017] | Single-sided gating: Gate W ← I, Gate O ← I |
| Eyeriss v2 [JETCAS 2019] | Single-sided skipping: Skip W ← I, Skip O ← I & W |
| SCNN [ISCA 2017] | Single-sided skipping: Skip W ← I, Skip O ← I & W |
| ExTensor [MICRO 2019] | Dual-sided skipping: Skip A ⋂ B, Skip Z ← A & B |
| DSTC [ISCA 2021] | Dual-sided skipping: Skip A ⋂ B, Skip Z ← A & B |

![SAF taxonomy table — which accelerators use which strategies](../../assets/L08/L08-p11-saf-table.png)

### Dual-sided intersection mechanics

Dual-sided skipping requires finding *matching* non-zero coordinates in both A and B simultaneously — a harder problem. The lecture describes a serial merge-style scan: compare the current leading coordinate in each operand; advance the one at the smaller coordinate; repeat until a match is found or one list is exhausted. ExTensor [MICRO 2019] improves on this with a **binary search** of remaining coordinates, which is especially effective when the next matching coordinate is far away. A design choice is a **maximum iteration count**: if no match is found within that bound, emit an idle cycle and continue.

![Gating: dual-sided access pattern — both operands scanned sequentially](../../assets/L08/L08-p17-gating-dual-sided.png)

> **Why it matters:** Gating is easy but only saves energy; skipping saves both energy and time and can deliver a speedup equal to the *effectual density*, but it requires compressed metadata to locate non-zeros. The compression format choice is therefore load-bearing for the overall system.

---

## Chapter 3 — Representation Formats: Compressing the Sparse Tensor

> *Slides: L08-23 … L08-48*

### Two goals: compression and access

The **Format** layer of the TeAAL Pyramid serves two purposes simultaneously:

1. **Compression efficiency** — reduce the bits required to store a tensor. Fewer bits in memory means either a smaller (cheaper, lower-energy) buffer or a larger tile for the same buffer size, increasing data reuse and reducing DRAM traffic.
2. **Access efficiency** — make it fast and cheap to locate the next non-zero value. This is what enables *skipping*.

These two goals are in tension: formats that compress aggressively (e.g., Coordinate Payload at high sparsity) store fewer metadata bits per non-zero but require more computation to locate the next non-zero when the density changes. The lecture evaluates four canonical formats.

### The four formats

![The four canonical representation formats: Uncompressed, Bitmask, Coordinate Payload, Run-Length Encoding](../../assets/L08/L08-p27-compression-formats.png)

**Uncompressed (U):** Store every value, including zeros. No metadata. Compression ratio = 1 always.
- Best for: dense data.

**Bitmask (B):** One bit per coordinate indicates zero (0) or non-zero (1); non-zero values stored separately.
- Metadata overhead: 1 bit per coordinate in the full vector.
- Best for: moderate sparsity. Maximum compression ratio is 1 / (bits per value) — e.g., for 8-bit values, 1/8 of uncompressed at extreme sparsity.
- Finding next non-zero: scan the bitmask sequentially (1-bit reads per coordinate).

**Coordinate Payload (CP):** Each non-zero is paired with its coordinate index stored as an n-bit field.
- Metadata overhead: n bits per non-zero, where n = ⌈log₂(vector length)⌉.
- Best for: very high sparsity.
- Finding next non-zero: direct — just read the next (coordinate, value) pair. No computation needed.
- Caveat: at low sparsity, CP is *larger* than uncompressed (metadata per non-zero grows faster than the savings).

**Run-Length Encoding (RLE):** Instead of the coordinate, store the *run length* (number of zeros between consecutive non-zeros) as an r-bit field.
- Metadata overhead: r bits per run segment. When runs exceed 2^r − 1, multiple metadata entries are needed.
- Best for: very high sparsity with long runs of zeros (structured data).
- Finding next non-zero: accumulate run lengths — requires an adder and a running counter. When r = 1, RLE degenerates to Bitmask.
- Design choice: value of r is chosen based on the expected run-length distribution.

### Compression efficiency across the density spectrum

The lecture provides a worked example (K=16, 8-bit values):

| Format | 1 non-zero (6.25% density) | 8 non-zeros (50%) | 16 non-zeros (100%) |
|---|---|---|---|
| Uncompressed | 128 bits | 128 bits | 128 bits |
| Bitmask (1 bit/coord) | 24 bits | 80 bits | 144 bits |
| CP (4 bits/coord) | 12 bits | 96 bits | 192 bits |
| RLE (4 bits/run) | 12 bits | 96 bits | 192 bits |

Key insight: **Bitmask is worse than Uncompressed at 100% density** (because the metadata itself adds overhead). **CP and RLE are worse than Bitmask at 50% density**. There is no universally superior format — the right choice depends on the expected density of the tensor.

![Compression efficiency table — formats vs. density for K=16, 8-bit values](../../assets/L08/L08-p34-compression-efficiency-table.png)

An important practical point (slide 35): for *unstructured* sparsity, the metadata to encode coordinates accounts for approximately **half of the total compressed storage** (from Han, ICLR 2016). Minimizing metadata overhead is therefore as important as minimizing zero-value storage.

### Eyeriss: Run-Length Encoding in practice

Eyeriss (Chen, ISSCC 2016) implements RLE on the off-chip DRAM link for both input activations and output feature maps. The encoding uses 5-bit run-length fields and 16-bit values, transmitted over a 64-bit wide DRAM bus. Results show that RLE compression achieves **1.2× to 1.9× reduction in DRAM access volume** across AlexNet's convolutional layers — within 5–10% of the theoretical entropy limit.

![Eyeriss RLE compression — diagram and DRAM access reduction results](../../assets/L08/L08-p33-eyeriss-rle.png)

### Access efficiency and the "concordant traversal" requirement

For skipping, the hardware must find the *coordinate of the next non-zero* quickly. The access-efficiency summary:

| Format | Steps to find next non-zero | Scales with |
|---|---|---|
| Uncompressed | Compare every coordinate until non-zero found | Vector size |
| Bitmask | Read 1-bit metadata at each coordinate | Vector size |
| RLE | Accumulate run lengths | Number of runs |
| CP | Read next (coordinate, value) pair directly | Number of non-zeros |

An important caveat: all of the above assumes **concordant traversal** — i.e., the hardware traverses the tensor in the same order the data is compressed. If the dataflow requires a different traversal order (*discordant traversal*), the compressed data must be re-indexed or decompressed, incurring significant extra cost. The choice of representation format is therefore coupled to the dataflow (loop order).

> **Why it matters:** Compression is not free. The metadata cost of encoding coordinates can negate much of the benefit. The right format matches both the expected density distribution and the hardware's traversal order — and that coupling propagates all the way up to the Mapping layer of the TeAAL pyramid.

---

## Chapter 4 — Structured Sparsity and Hierarchical Structured Sparsity (HSS)

> *Slides: L08-49 … L08-66*

### The tradeoff: flexibility vs. hardware simplicity

**Unstructured sparsity** allows non-zeros to appear at any coordinate. This maximizes model-design flexibility (and typically accuracy), but the hardware cost is high: finding non-zeros requires searching the full coordinate range, and metadata overhead is large.

**Structured sparsity** constrains where non-zeros can appear, reducing the search space and metadata overhead at the cost of model-design flexibility (and potentially accuracy). The lecture illustrates the granularity spectrum:

![Granularity spectrum of sparsity — weight pruning to channel pruning](../../assets/L08/L08-p51-granularity-sparsity.png)

From fine-grained to coarse-grained: *weight pruning* → *filter pruning* → *row pruning* → *channel pruning*. As granularity coarsens, hardware becomes simpler (non-zero locations are predictable), but the model has fewer degrees of freedom for maintaining accuracy.

### G:H structured sparsity (NVIDIA Sparse Tensor Core)

NVIDIA's 2:4 structured sparsity (also called the **Sparse Tensor Core** pattern) enforces that in every contiguous group of 4 values, exactly 2 must be non-zero (50% sparsity exactly). This is the canonical **G:H** pattern with G=2, H=4.

The hardware benefit is substantial: the non-zero locations within each H-element group can be encoded with just ⌈log₂(C(H,G))⌉ bits of metadata — for 2:4, only 2 bits per pair of non-zeros. The metadata overhead is minimal compared to unstructured sparsity, enabling extremely efficient skipping.

![NVIDIA 2:4 Sparse Tensor Core — per-row structured sparsity pattern](../../assets/L08/L08-p52-structured-2-4.png)

The limitation: the design locks in 50% sparsity. If the model is only 30% sparse, no speedup; if it is 80% sparse, the hardware still runs at 50% utilization. A single G:H value is inflexible.

### The inflexibility problem

Modern DNNs use a wide mix of operations:
- **Pruning** (Han, NeurIPS 2015) → sparse weights, variable degree.
- **Activation functions** (ReLU, etc.) → sparse or dense activations depending on input.
- **Attention modules** (Vaswani, NeurIPS 2017) → dense or variable-sparse attention maps.
- **Depth-wise separable layers** (Howard, CVPR 2017) → dense weights, fewer parameters.

A single G:H value cannot efficiently serve all of these. The naïve solution — support multiple G:H ratios (2:4, 2:6, 2:8, …) in hardware — does not scale: the hardware complexity grows roughly linearly with the number of ratios supported.

### Hierarchical Structured Sparsity (HSS)

Wu et al. [MICRO 2023] introduce **HSS** as a composable solution. Instead of designing hardware for each G:H ratio separately, HSS **composes simple G:H patterns hierarchically**:

An N-rank HSS pattern applies the G:H rule at each of N nested granularities. For example, the **3:4 → 2:4** pattern:
- **Rank 1 (outer)**: Select 3 non-empty blocks out of every 4 blocks.
- **Rank 0 (inner)**: Within each block, keep 2 non-zero values out of 4 elements (standard 2:4).

The effective sparsity of the 3:4→2:4 pattern is 1 − (3/4)(2/4) = 1 − 6/16 = 62.5%.

![Hierarchical Structured Sparsity — composing 3:4 and 2:4 patterns](../../assets/L08/L08-p58-hss-pattern.png)

Because sparsity fractions multiply, a 2-rank HSS with m rank-1 options and n rank-0 options covers m×n distinct sparsity degrees — far more than m+n ratios would. A concrete example from slide 64: combining rank-1 options {4:4, 4:5, 4:6, 4:7} with rank-0 options {4:4, 2:4, 1:4} yields **12 distinct sparsity degrees** spanning 0% to 86%.

![HSS sparsity degrees — 12 degrees from two 3-option ranks](../../assets/L08/L08-p64-hss-sparsity-degrees.png)

The key hardware benefit: **the hardware only needs to implement simple G:H acceleration at each rank independently** — the hierarchical composition is a representation/format choice, not an additional hardware mode. This keeps the sparsity-acceleration overhead low.

### HighLight: HSS in silicon

The **HighLight** accelerator [Wu, MICRO 2023] implements HSS on a 16×16 PE array with a two-level skipping hierarchy:
- **Rank-1 acceleration** reduces storage requirements and energy consumption by skipping entire empty blocks.
- **Rank-0 acceleration** (inside the PE) reduces latency and energy by skipping zero elements within a block.

The compressed representation uses an HSS-based format that is simple enough for hardware to decode efficiently. HighLight is evaluated against baselines (STC and DSTC) on ResNet50 and Transformer-Big across a range of pruning sparsity degrees, reaching the **accuracy–energy-delay product Pareto frontier** — meaning it achieves the best tradeoff between model accuracy and hardware efficiency across the full sparsity spectrum.

![HighLight accelerator architecture — 16×16 PE array with hierarchical skipping](../../assets/L08/L08-p65-highlight-accelerator.png)

> **Why it matters:** The sparsity degree in deployed models is not fixed — it varies by layer, by pruning method, and by input data. Hardware that can exploit *any* sparsity degree efficiently (rather than locking to a single G:H ratio) is essential for practical deployment across diverse models and workloads.

---

## Chapter 5 — Tiling Sparse Tensors: The Overbooking Problem

> *Slides: L08-67 … L08-94*

### Why tiling is hard with sparsity

A fundamental principle from earlier lectures: to maximize energy efficiency, you want to choose the **largest tile that fits in the on-chip buffer**, because larger tiles enable more data reuse and reduce DRAM traffic. With dense tensors, tile size is determined by the tensor dimensions and a simple capacity constraint.

With sparse tensors, the *number of non-zeros* in a tile (its **occupancy**) varies unpredictably. Two tiles of the same shape can have wildly different numbers of non-zeros. Sparsity varies not just between different workloads but within a single tensor, as illustrated for graph computing, scientific simulations, and recommendation systems (slide 68).

The challenge: if you size the tile to fit the **maximum-occupancy** tile, most tiles will be only partially full — wasting buffer space and reducing effective tile size (and thus data reuse).

### The two imperfect tiling strategies

**Uniform occupancy** (tile size chosen to equalize non-zero counts):
- Low occupancy variation — all tiles have about the same number of non-zeros.
- Problem: the non-uniform shape means the companion operand is hard to tile (irregular addressing).

**Uniform shape** (tile size chosen by tensor dimensions, independent of sparsity):
- Easy to tile both operands simultaneously (regular addressing).
- Problem: high occupancy variation — some tiles are dense, some nearly empty. If you size the buffer for the worst-case (maximum-occupancy) tile, average utilization is very low.

Neither strategy is satisfactory in isolation.

### Overbooking: the airline seat analogy

The lecture introduces **overbooking** as the solution, using the airline seat analogy:
- Airlines overbook flights because on average not all ticketed passengers show up. The expected number of passengers who actually board is close to the number of seats.
- Analogously: if tiles are overbooked (nominally larger than the buffer), *on average* the number of non-zeros that actually land in the buffer will be close to the buffer capacity — because most tiles are sparse.

When a tile's non-zeros exceed the buffer capacity (**"bumped data"**), the excess is **streamed** directly to compute without being buffered (losing the reuse benefit for those values, but not blocking progress).

![Overbooking concept — tile occupancy vs. buffer capacity, with bumped data](../../assets/L08/L08-p84-overbooking-concept.png)

### Tailors: handling bumped data

The **Tailors** mechanism [Xue, MICRO 2023] manages the two data streams:
- **Unbumped data**: loaded into the buffer normally and reused across multiple passes of the traversal loop.
- **Bumped data**: streamed from DRAM in a separate pass — this data loses its reuse opportunity but the traversal continues unblocked.

The traversal order for bumped data is adjusted to maintain as much reuse as possible given the streaming constraint.

### Swiftiles: predicting tile occupancy

To determine *how much* to overbook, the hardware needs to estimate the occupancy distribution of tiles. A full traversal of the tensor (to count all non-zeros in every tile) is too expensive. **Swiftiles** [Xue, MICRO 2023] uses **random sampling**: sample a small fraction of tiles, build an approximate occupancy distribution, then scale it to the buffer size. The target is to set the overbook ratio so that a chosen percentile (e.g., 90th) of tiles fits in the buffer.

Evaluation on ExTensor [Hegde, MICRO 2019]:
- **ExTensor-Naive**: no sparsity-aware tiling; tiles are sized as if dense (worst case). 90th-percentile occupancy is only 6% of buffer; tiles are vastly undersized.
- **ExTensor-Overbooking** (Tailors + Swiftiles, 90th-percentile target): **52.7× speedup** and **22.5× energy efficiency improvement** over ExTensor-Naive.
- **ExTensor-Prescient** (oracle: knows exact tile occupancy in advance): Overbooking achieves **2.3× speedup** and **2.5× energy efficiency** over even this ideal baseline, because the Swiftiles prediction is good enough that most tiles fit — and the benefit of larger tiles (more reuse) outweighs the cost of occasional bumped-data streaming.

> **Why it matters:** Choosing the right tile size for sparse data is as important as choosing the right compression format. A naïvely sized tile wastes nearly all the buffer capacity; overbooking with lightweight occupancy prediction recovers most of the theoretical maximum efficiency — without requiring a full pre-scan of the data.

---

## Chapter 6 — Interplay with Dataflow and Summary

> *Slides: L08-94 … L08-96*

### Dataflow affects sparsity exploitation

The lecture closes by connecting sparsity back to the **Mapping** layer of the TeAAL pyramid. The loop order (dataflow) must be aligned with the storage order of the representation format to enable **concordant traversal** — i.e., the hardware accesses non-zeros in the order they are stored in memory. Discordant traversal requires random access into the metadata, which is expensive.

Additional dataflow considerations:
- **Increasing stationarity** for a data type (moving its loop to the outermost/spatial position) amortizes the per-access cost of metadata decoding over more computations.
- **Parallelism and workload balance**: when loops are parallelized across PEs (spatial_for), sparsity causes *workload imbalance* — some PEs get dense tiles and run for many cycles, while others finish quickly. The choice of which loop to parallelize should account for expected sparsity variation.

These interactions are explored in Lab 4 (SparseLoop tool) and the Final Project (Chapter 8.3 of the textbook, *Efficient Processing of Deep Neural Networks*, Sze & Emer).

### Summary of the lecture

The lecture closes with a precise statement of the challenges and costs:

**Irregularity** from sparsity causes: underutilization of buffers and PEs; workload imbalance across the PE array; random data access patterns.

**Overheads** that must not exceed sparsity benefits: storage for coordinate metadata; intersection logic for checking operand zero-ness.

Every design in this lecture — Eyeriss, SCNN, ExTensor, HighLight, Swiftiles — represents a different point on the tradeoff curve between **exploiting sparsity** and **paying for the hardware complexity** needed to find, compress, and intersect sparse data.

> **Why it matters:** Sparsity is not a minor optimization — it is a first-order determinant of energy efficiency and throughput in production DNN accelerators. But the hardware cost of exploiting it correctly (intersection logic, metadata decoders, tiling controllers) requires careful co-design across Format, Mapping, and Architecture layers of the pyramid. The next two lectures (L09–L10) will examine specific sparse accelerator architectures in more depth.

---

## Standalone Study Guide

### What to master before moving on

- Distinguish gating, skipping, and compressed-format representation as separate sparse acceleration features.
- Compare uncompressed, bitmask, coordinate-payload, and run-length formats by compression and access efficiency.
- Explain why structured sparsity lowers metadata and decoder cost.
- Explain the sparse tiling problem: occupancy varies, so fixed-shape dense-style tiles waste buffer capacity.
- Describe Tailors and Swiftiles as an overbooking strategy for sparse tiles.

### Self-check questions

1. Which sparse acceleration feature saves energy but not cycles?
2. Why can a compressed format be larger than uncompressed storage at high density?
3. What is the difference between a bumped and unbumped tile in Tailors?

### Exercises

1. For a length-16 vector with four non-zeros, encode it using bitmask and coordinate-payload formats. Count metadata bits separately from payload bits.
2. Pick a 2:4 sparse group and calculate how many bits are needed to encode the non-zero positions.
3. Explain why choosing a tile size from maximum occupancy can destroy reuse for the average tile.

### Common traps

- Treating compression ratio as the only format metric. Access efficiency can dominate runtime.
- Assuming structured sparsity is always better. It may reduce model flexibility or accuracy.
- Ignoring workload balance: skipping can make different PEs finish at different times.

---

## Key Terms

| Term | Gloss |
|---|---|
| **Effectual operation** | An operation where neither operand is zero; it contributes to the output. |
| **Ineffectual operation** | An operation involving at least one zero operand; `anything × 0 = 0`. |
| **Gating** | A SAF that suppresses the memory read and multiply for a zero operand, saving energy but not cycles. The cycle still occurs. |
| **Skipping** | A SAF that eliminates the entire cycle for zero or non-intersecting operand pairs, saving both energy and time. Requires pre-computed non-zero coordinates. |
| **SAF (Sparse Acceleration Feature)** | Collective term for gating, skipping, and compressed-format strategies (Wu, MICRO 2022). |
| **Single-sided intersection** | One operand (the *leader*) drives the access pattern; the other (*follower*) is fetched only when the leader is non-zero. |
| **Dual-sided intersection** | Both operands' non-zero coordinate sets are searched to find matching pairs; only matching non-zeros are processed. |
| **Representation format** | How a sparse tensor is encoded in memory, including both values and coordinate metadata. |
| **Uncompressed (U)** | Every value stored, including zeros. No metadata overhead; works best for dense data. |
| **Bitmask (B)** | One bit per coordinate; works best for moderate sparsity. |
| **Coordinate Payload (CP)** | Explicit coordinate stored per non-zero; best for high sparsity; access is direct. |
| **Run-Length Encoding (RLE)** | Count of zeros between non-zeros; best for high sparsity with long zero runs; requires accumulation to find next coordinate. |
| **Compression efficiency** | Ratio of compressed representation size to uncompressed size; depends on density and metadata overhead. |
| **Access efficiency** | Computational cost of locating the next non-zero from the metadata; determines skipping hardware complexity. |
| **Concordant traversal** | Accessing non-zeros in the same order they are stored — the efficient default. |
| **Discordant traversal** | Accessing data in a different order from storage — expensive; requires random access into metadata. |
| **Structured sparsity** | Sparsity patterns constrained so non-zeros appear only at predictable positions; reduces hardware complexity. |
| **Unstructured sparsity** | No constraint on non-zero locations; maximum model flexibility but higher hardware cost. |
| **G:H sparsity** | Exactly G non-zeros in every contiguous group of H values. NVIDIA's 2:4 is the canonical example. |
| **HSS (Hierarchical Structured Sparsity)** | Composing multiple G:H patterns at nested granularities to cover a broad sparsity-degree spectrum with simple hardware (Wu, MICRO 2023). |
| **Tile occupancy** | The number of non-zeros in a given tile; varies unpredictably with unstructured sparsity. |
| **Overbooking** | Nominating tiles larger than the buffer capacity; works because most tiles are sparse and their actual occupancy fits on average. |
| **Tailors** | Hardware mechanism that handles "bumped" (overflow) tile data by streaming it rather than buffering it, without stalling compute [Xue, MICRO 2023]. |
| **Swiftiles** | Lightweight tiling algorithm using random sampling to predict tile occupancy distribution and set the overbook ratio [Xue, MICRO 2023]. |
| **Workload imbalance** | Variation in non-zero counts across tiles assigned to different PEs, causing some PEs to finish earlier and idle. |

---

## Takeaways

- Every DNN inference operation can be classified as **effectual** or **ineffectual**; exploiting ineffectual operations is the entire goal of sparse acceleration.
- **Gating** saves energy only; **skipping** saves energy and time, but requires pre-computed non-zero location metadata — making the **representation format** a first-class architectural concern.
- The four canonical formats (Uncompressed, Bitmask, CP, RLE) each have a density range where they are optimal; no single format dominates across all sparsity levels.
- **Structured sparsity** (G:H, HSS) reduces hardware complexity at the cost of model flexibility; **HSS** extends coverage to a broad sparsity-degree spectrum by composing simple patterns hierarchically.
- **Tiling sparse tensors** is fundamentally different from tiling dense tensors: the tile occupancy varies unpredictably, and naïve worst-case sizing wastes almost all buffer capacity. **Overbooking with Tailors + Swiftiles** recovers 52.7× speedup vs. naive tiling in the ExTensor example.
- Sparsity, dataflow (loop order), and representation format are tightly coupled: the format must match the traversal order, and the parallelism choice must account for workload imbalance.

---

## Connections to Later Lectures

- **L07 (Sparsity)** — the preceding lecture established *why* and *how much* sparsity exists in DNNs; L08 picks up by asking *how* hardware exploits it.
- **L09–L10 (Sparse Architectures II & III)** — dive into specific sparse accelerator architectures in greater detail, covering designs such as SCNN, ExTensor, and more advanced intersection hardware.
- **Format layer (TeAAL Pyramid, L01)** — the representation formats introduced here are the concrete realization of the *Format* layer first mentioned in the introductory lecture.
- **Mapping layer (L05–L06, Dataflows)** — the concordant/discordant traversal discussion here shows that the choice of dataflow (loop order) cannot be made independently of the choice of sparse representation format.
- **Lab 4 (SparseLoop)** — uses the SparseLoop tool (sparseloop.mit.edu) to evaluate SAF strategies and compression formats on real workloads; the lab directly operationalizes the theory from this lecture.
- **Textbook** — Sections 8.2 (Compression) and 8.3 (Sparse Dataflows) of *Efficient Processing of Deep Neural Networks*, Sze & Emer.

---

## Appendix — Slide-to-Section Map

| Slides | Section |
|---|---|
| L08-1 | Title |
| L08-2 … L08-8 | Ch.1 — Why Sparsity Matters (sources, arithmetic, irregularity) |
| L08-9 … L08-22 | Ch.2 — Gating vs. Skipping (intersection, single/dual-sided, accounting) |
| L08-23 … L08-48 | Ch.3 — Representation Formats (U, B, CP, RLE; compression/access efficiency; Eyeriss RLE) |
| L08-49 … L08-66 | Ch.4 — Structured Sparsity & HSS (G:H, 2:4 STC, HSS, HighLight) |
| L08-67 … L08-93 | Ch.5 — Tiling Sparse Tensors (uniform occupancy/shape dilemma, overbooking, Tailors, Swiftiles, ExTensor evaluation) |
| L08-94 … L08-96 | Ch.6 — Dataflow interplay, summary, recommended reading |
