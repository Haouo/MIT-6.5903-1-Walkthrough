# L08 — Sparse Architectures

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer and Vivienne Sze
> **Lecture date:** March 2, 2026
> **Primary source:** [`Lecture/L08 - Sparse Architectures.pdf`](../../Lecture/L08%20-%20Sparse%20Architectures.pdf)

This chapter reconstructs the lecture narration from the public slides and local papers. It avoids copied slide or paper figures; diagrams are described textually or redrawn as small original examples.

---

## TL;DR

Sparsity is attractive because zero operands create **ineffectual work**: compute, reads, writes, and interconnect traffic that cannot change the output. The hard part is that zeros are not just "missing values." They create irregular control flow, variable tile occupancy, metadata, intersection logic, and load imbalance.

Lecture 08 introduces three sparse acceleration features:

- **Gating:** detect a zero and idle the relevant hardware for that cycle. This saves energy, but the dense schedule still consumes the cycle.
- **Skipping:** move directly to the next useful coordinate. This saves energy and time, but requires metadata and traversal hardware.
- **Format:** encode sparse tensors so zeros are not stored or moved. This saves capacity and bandwidth, but metadata can become the bottleneck.

The lecture then expands from unstructured formats to structured sparsity, hierarchical structured sparsity (HSS), and sparse tiling. The architectural lesson is blunt: sparsity only helps when the saved work exceeds the cost of finding, representing, routing, and balancing the nonzeros.

## What Problem This Lecture Solves

Dense DNN accelerators assume regular arrays: every loop iteration fetches operands, performs a MAC, updates a partial sum, and advances to the next coordinate. Sparse tensors break that assumption. If an activation or weight is zero, the corresponding multiplication is unnecessary, but the accelerator must know that before spending energy or time.

This lecture solves the first half of the sparse architecture problem:

1. How do we define which operations are useful?
2. How can hardware avoid reading or computing zero operands?
3. How should sparse data be represented?
4. Why does structured sparsity simplify hardware?
5. Why does tiling become difficult when nonzero counts vary across tiles?

Lecture 09 will use these ideas to build concrete sparse convolution dataflows.

## Why This Lecture Matters

Sparse acceleration is not a small optimization sitting after the dataflow design. It changes the meaning of memory layout, loop order, PE utilization, and tile sizing. A dense dataflow asks, "Which operand should stay near the PE?" A sparse dataflow also asks, "Which operand can reveal the next useful coordinate cheaply?"

For hardware architects, sparsity affects:

- **Energy:** fewer value reads, fewer metadata-dependent reads, fewer MACs, and fewer partial-sum updates.
- **Latency:** skipping can reduce cycle count, but gating cannot.
- **Bandwidth:** compression can reduce DRAM and SRAM traffic, but metadata must also move.
- **Area:** sparse decoders, intersection units, metadata buffers, and flexible routers consume silicon.
- **Utilization:** nonzero counts vary, so PEs may receive unequal work.
- **Programmability:** the same mathematical tensor may need different formats depending on the loop order.

Source note: these motivations follow Lecture 08 slides 2-7 and 25-26; the broader statement about sparse accelerator modeling is consistent with TeAAL Section 2.3.

## Prerequisites and Mental Model

You should be comfortable with:

- A MAC computing \(o \leftarrow o + a \times b\).
- A dot product \(Z = \sum_k A_k B_k\).
- Density \(d\), the fraction of stored coordinates that are nonzero.
- Dataflow, meaning the loop order and placement of weights, activations, and partial sums across memory and PEs.

The central mental model is:

> A sparse accelerator is a dense accelerator plus a **coordinate machine**.

The dense part still multiplies values. The coordinate machine decides which coordinates should be visited, where their payloads live, and which output coordinate receives the result.

## Learning Objectives

After this lecture, you should be able to:

- Define effectual and ineffectual operations.
- Distinguish total operations from total operations performed.
- Explain gating, skipping, and format as separate sparse acceleration features.
- Compare single-sided and dual-sided intersection.
- Compute small read/cycle counts for a sparse dot product.
- Compare uncompressed, bitmask, coordinate-payload, and run-length formats by compression efficiency and access efficiency.
- Explain why metadata can dominate unstructured sparse storage.
- Explain the flexibility/efficiency tradeoff in structured sparsity.
- Compute the effective density of a hierarchical structured sparsity pattern.
- Explain sparse tiling, tile occupancy, overbooking, Tailors, and Swiftiles at the conceptual level.
- Connect sparse formats to later fibertree and sparse dataflow lectures.

## Main Textbook-Style Narrative

### 1. From zeros to ineffectual work

Lecture 08 begins with two sources of sparsity:

- **Activation sparsity:** ReLU, input correlations, graph-like representations, and other input-dependent effects can produce zeros in activations.
- **Weight sparsity:** pruning can set trained weights to zero.

The useful distinction is not "zero vs. nonzero" alone, but **effectual vs. ineffectual operation**.

For a multiplication, an operation is effectual only if both operands are nonzero. If either operand is zero, \(a \times b = 0\), so the multiplication does not change the partial sum. For an addition, adding zero is also ineffectual because \(x + 0 = x\).

Lecture 08 uses two counts:

- \(N_\text{total} = N_\text{effectual} + N_\text{ineffectual}\).
- \(N_\text{performed} = N_\text{effectual} + N_\text{unexploited ineffectual}\).

The goal is not merely to have sparse tensors. The goal is to reduce \(N_\text{unexploited ineffectual}\) without making each remaining operation so expensive that the accelerator loses.

### 2. Irregularity is the price of exploiting sparsity

Dense tensors are regular: the \(k\)-th loop iteration usually maps to the \(k\)-th stored value. Sparse tensors break this:

- The number of nonzeros can vary across vectors, rows, channels, tiles, and layers.
- The positions of nonzeros may be unknown until metadata is decoded.
- A PE assigned to a dense sparse tile may run longer than a PE assigned to an empty sparse tile.
- A compressed tensor may require metadata reads before value reads.

This is why Lecture 08 says exploiting sparsity makes processing irregular. The accelerator saves work only after paying for metadata, decoding, coordinate arithmetic, intersection, and scheduling.

### 3. Sparse Acceleration Features: gating, skipping, format

The lecture organizes sparse hardware mechanisms as **Sparse Acceleration Features (SAFs)**:

| SAF | What it does | Saves energy? | Saves cycles? | Needs sparse metadata before the cycle? |
|---|---|---:|---:|---:|
| Gating | Detect a zero and idle some hardware for that cycle | Yes | No | Usually no |
| Skipping | Jump directly to useful nonzero coordinates | Yes | Yes | Yes |
| Format | Store and move only useful payloads plus metadata | Yes | Sometimes | Yes, for skipping |

The key transition is: gating can discover zeros during the dense schedule; skipping must already know where to jump. That is why representation format becomes an architectural concern instead of a software detail.

### 4. The dot-product example

Use the lecture's 1-D dot product:

```text
A = [0, 0, c, d, 0, f]
B = [g, h, 0, j, k, l]
Z = A dot B = d*j + f*l
```

There are six algorithmic multiply positions, but only coordinates \(3\) and \(5\) are effectual. The nonzero coordinate sets are \(A_\text{nz} = \{2,3,5\}\) and \(B_\text{nz} = \{0,1,3,4,5\}\). Their intersection is \(\{3,5\}\).

The strategies differ:

| Strategy | Cycles | A value reads | B value reads | Computes performed | Explanation |
|---|---:|---:|---:|---:|---|
| Dense baseline | 6 | 6 | 6 | 6 | Visit every coordinate |
| Gate \(B \leftarrow A\) | 6 | 6 | 3 | 3 | A is leader; B is read only when A is nonzero |
| Skip \(B \leftarrow A\) | 3 | 3 | 3 | 3 | Visit A's nonzero coordinates only |
| Dual-sided skip \(A \cap B\) | at least 2 | 2 | 2 | 2 | Visit only coordinates present in both operands |

Source note: the counts are slide-derived from Lecture 08 slides 9-20. The table excludes metadata reads, matching the slide note.

The important misconception is that "skip A's zeros" is already optimal. It is not. If A is nonzero at coordinate 2 but B is zero, the operation \(c \times 0\) is still ineffectual. Full work reduction requires finding the intersection of both operands.

### 5. Single-sided vs. dual-sided intersection

**Single-sided intersection** chooses a leader. If A is the leader, the hardware visits A's nonzero coordinates and then asks for B at those same coordinates. This is simpler, but its benefit depends on whether the leader is a good predictor of useful work.

**Dual-sided intersection** treats both operands as sparse lists and emits only matching coordinates. A simple merge-style intersection compares the current coordinate from each list:

```text
A coordinates: 2, 3, 5
B coordinates: 0, 1, 3, 4, 5

compare 2 and 0 -> advance B
compare 2 and 1 -> advance B
compare 2 and 3 -> advance A
compare 3 and 3 -> emit 3
compare 5 and 4 -> advance B
compare 5 and 5 -> emit 5
```

Dual-sided skipping is powerful, but the number of comparisons is data-dependent. Hardware often bounds the number of metadata steps per cycle; if no match is found quickly, the PE may idle. Lecture 08 notes that ExTensor uses binary search over remaining coordinates, which can be effective when the next match is far away.

### 6. Format: compression efficiency and access efficiency

A sparse format must answer two questions:

1. **Compression efficiency:** How many bits are needed relative to dense storage?
2. **Access efficiency:** How cheaply can the hardware find the next nonzero coordinate or test whether a coordinate is present?

The four lecture formats are:

| Format | Metadata idea | Good density regime | Access behavior |
|---|---|---|---|
| Uncompressed | No metadata; store every value | Dense | Direct coordinate access, but no compression |
| Bitmask | One bit per coordinate | Moderate sparsity | Scan bits or perform bit operations |
| Coordinate payload | Store coordinate per nonzero | High sparsity | Direct next-nonzero access |
| Run-length encoding | Store zeros between nonzeros | High sparsity with long zero runs | Accumulate run lengths |

For vector \(A=[0,0,c,d,0,f]\) with 8-bit values:

- Uncompressed uses \(6 \times 8 = 48\) bits.
- Bitmask uses \(6 \times 1 + 3 \times 8 = 30\) bits.
- Coordinate payload with 3-bit coordinates uses \(3 \times 3 + 3 \times 8 = 33\) bits.
- RLE with 3-bit runs uses \(3 \times 3 + 3 \times 8 = 33\) bits.

For a length-16 vector with 8-bit values and 4-bit coordinates:

| Nonzeros | Density | Uncompressed | Bitmask | Coordinate payload | RLE with 4-bit runs |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.25% | 128 bits | 24 bits | 12 bits | 12 bits |
| 8 | 50% | 128 bits | 80 bits | 96 bits | 96 bits |
| 16 | 100% | 128 bits | 144 bits | 192 bits | 192 bits |

The lesson is not "compressed is better." At 100% density, every compressed format in this example is worse than uncompressed because metadata adds overhead.

### 7. The hardware meaning of metadata

Metadata is not passive. It changes the datapath.

- A bitmask needs bit reads and often popcount or bit-scan logic.
- Coordinate payload needs coordinate storage, coordinate comparison, and sometimes random lookup in the other operand.
- RLE needs accumulation state to reconstruct absolute coordinates.
- Dual-sided skipping needs an intersection unit.

Slide 35 cites Han et al. to make a practical point: for unstructured sparsity, index metadata can account for roughly half of storage. Treat this as a warning, not as a universal constant; it depends on value precision, coordinate width, and sparsity pattern.

### 8. Structured sparsity

Unstructured sparsity allows nonzeros anywhere. It is flexible for model design, but expensive for hardware because each nonzero may need coordinate metadata.

**Structured sparsity** restricts the legal positions of nonzeros. The hardware benefit is that the search space is smaller and the metadata can be compact.

The common \(G:H\) pattern means: in every group of \(H\) values, exactly \(G\) values are nonzero. NVIDIA's 2:4 pattern is the lecture's canonical example: exactly two nonzeros per group of four, i.e., 50% density.

The benefit is simple decode. The limitation is also simple: one \(G:H\) ratio supports one sparsity degree. If a layer wants 30% sparsity or 80% sparsity, fixed 2:4 hardware cannot translate that smoothly into proportional savings.

### 9. Hierarchical Structured Sparsity

Hierarchical Structured Sparsity (HSS) composes simple \(G:H\) rules at multiple nested granularities. Suppose we use a two-level pattern \(3:4 \rightarrow 2:4\):

- Outer rule: keep 3 non-empty blocks out of 4.
- Inner rule: within each surviving block, keep 2 values out of 4.

The resulting density is \((3/4)(2/4)=3/8=37.5\%\), so the sparsity is \(62.5\%\).

Lecture 08's HSS example combines outer options \(\{4:4,4:5,4:6,4:7\}\) with inner options \(\{4:4,2:4,1:4\}\), yielding \(4 \times 3 = 12\) possible sparsity degrees. The hardware does not need twelve unrelated decoders; it composes simple per-rank decoders.

This is a format-level trick with architecture-level consequences: the model can choose more sparsity degrees while the hardware remains closer to a small set of simple structured primitives.

### 10. Sparse tiling and overbooking

Dense tiling asks: "What rectangular tile fits in the buffer?" Sparse tiling asks: "How many nonzeros will this tile contain?"

Two same-shape sparse tiles can have very different occupancies. If the buffer is sized for the densest tile, most average tiles waste capacity. If tiles are split by equal nonzero count, the coordinate ranges become irregular, making it hard to tile the other operand.

The lecture contrasts:

- **Uniform occupancy:** balanced nonzero count, irregular shape.
- **Uniform shape:** regular shape, variable nonzero count.

**Overbooking** chooses a larger nominal tile than the buffer could hold in the worst case, betting that most tiles are sparse enough to fit. When a tile has too many nonzeros, the overflow is **bumped data**. Tailors streams bumped data instead of buffering it. Swiftiles estimates how much to overbook using random sampling of tile occupancy.

The hardware implication is subtle: overbooking increases average tile size, improving reuse, while accepting occasional streamed overflow. It is a controlled violation of worst-case dense tiling assumptions.

## Worked Examples

### Example 1: Density and expected useful work

If weight density is \(d_W=0.4\) and activation density is \(d_A=0.5\), and if nonzero locations are independent for this toy estimate, dual-sided skipping would visit roughly \(d_W d_A = 0.2\) of dense multiply positions.

For a dense loop with 1000 multiplications, the idealized effectual count is \(1000 \times 0.2 = 200\). A single-sided scheme using only activations as leader would visit \(1000 \times 0.5 = 500\) positions. It saves work, but still performs many multiplications where the corresponding weight is zero.

Teaching interpretation: independence is a simplifying assumption for the example. Real DNN sparsity can be correlated by layer, channel, and input.

### Example 2: Metadata can defeat compression

Suppose values are 8 bits, the vector length is 16, and coordinates need 4 bits. Coordinate payload size is \(n_\text{nz}(8+4)\). Dense size is \(16 \times 8 = 128\) bits.

Coordinate payload is smaller than dense only when \(12n_\text{nz} < 128\), or \(n_\text{nz} < 10.67\). If the vector has 11 or more nonzeros, coordinate payload is larger than dense.

Hardware meaning: if a layer has high density, sparse decoding burns area and bandwidth to move metadata that does not remove much work.

### Example 3: Effective sparsity in HSS

For \(4:6 \rightarrow 1:4\):

- Outer density is \(4/6\).
- Inner density is \(1/4\).
- Effective density is \((4/6)(1/4)=1/6\).
- Effective sparsity is \(1-1/6=5/6\approx 83.3\%\).

The point is not that this exact pattern is always best. The point is that nested ratios multiply, giving a compact way to cover many sparsity degrees.

## Key Equations and How to Read Them

### Effectual work

\[
N_\text{performed}=N_\text{effectual}+N_\text{unexploited ineffectual}.
\]

Read this as an accounting identity. Sparse hardware improves performance only by reducing the second term, and only if the overhead per performed operation remains reasonable.

### Ideal independent two-sided work

\[
N_\text{effectual}\approx d_A d_B N_\text{dense}.
\]

This is a teaching approximation. It says that if operand A is nonzero with probability \(d_A\) and operand B is nonzero with probability \(d_B\), both are nonzero with probability \(d_A d_B\). It clarifies why dual-sided intersection can be much better than exploiting only one sparse operand.

### HSS density

\[
d_\text{HSS}=\prod_i \frac{G_i}{H_i}.
\]

Each hierarchy level keeps \(G_i\) entries out of \(H_i\). The kept fractions multiply because a value must survive all levels.

## Hardware Implications

- **Gating:** needs zero detection and enable signals; it preserves dense timing, which makes control simple but leaves latency unchanged.
- **Skipping:** needs metadata that can produce next coordinates early enough; it changes cycle count and can create pipeline bubbles.
- **Dual-sided intersection:** needs coordinate comparison/search logic; performance depends on sparsity pattern and metadata format.
- **Compression:** saves capacity and bandwidth only when metadata is smaller than omitted zero payloads.
- **Structured sparsity:** reduces decoder complexity and metadata width but constrains the model.
- **HSS:** expands the sparsity-degree menu without requiring a fully independent decoder for every degree.
- **Sparse tiling:** buffer sizing must consider occupancy distributions, not just tensor shape.
- **Parallel sparse execution:** PEs can become imbalanced because different tiles contain different numbers of nonzeros.

## Common Misconceptions

### Misconception: Sparsity automatically gives speedup.

Sparsity gives an opportunity. Speedup requires skipping cycles. Gating can reduce energy without reducing latency.

### Misconception: The sparsest format is always best.

A format with excellent compression can have poor access efficiency. If the hardware must scan, accumulate, or randomly probe metadata for every useful value, the saved payload bits may not translate into throughput.

### Misconception: Single-sided skipping is equivalent to dual-sided skipping.

Single-sided skipping avoids zeros in the leader only. If the follower is zero at a leader nonzero coordinate, the operation is still ineffectual.

### Misconception: Structured sparsity is simply worse model quality for easier hardware.

That tradeoff exists, but structured sparsity is also a way to make savings predictable. HSS tries to recover flexibility by composing simple structures.

### Misconception: Sparse tiles should be sized by worst case.

Worst-case sizing can waste most buffer capacity when occupancy is highly variable. Overbooking is valuable precisely because average occupancy can be far below maximum occupancy.

## Connections to Previous and Later Lectures

- **L05-L06 mapping/dataflows:** sparse formats only work well when loop order matches storage order. This is the same mapping concern, now with metadata in the loop.
- **L07 sparsity/pruning:** pruning creates weight sparsity; L08 asks what hardware must do to exploit it.
- **L09 sparse architectures part 2:** fibertrees, CSR/CSC, projection, and intersection formalize the ideas introduced here.
- **L10 sparse architectures part 3:** TeAAL and sparse accelerator specifications give a language for describing these design choices.
- **Lab 4/SparseLoop:** the lecture's SAF language becomes a modeling vocabulary for analyzing sparse accelerator tradeoffs.

## Paper Bridge: TeAAL

### Bibliographic identity

- **Title:** TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators
- **Authors:** N. Nayak et al.
- **Year / venue:** MICRO 2023
- **Used in lecture(s):** L01, L08, L09, L10
- **Local PDF:** `papers/TeAAL.pdf`

### Problem addressed

TeAAL addresses the difficulty of specifying and comparing sparse tensor accelerators. Sparse accelerators differ not only in PE arrays, but also in loop order, tensor formats, partitioning, rank transformations, and sparse orchestration. Informal descriptions make these differences hard to compare.

### Core idea

The paper represents sparse accelerators using cascades of mapped Einsums plus content-preserving transformations on fibertrees. It separates computation from mapping and format, which matches Lecture 08's separation of skipping, format, and dataflow.

### Relevance to this lecture

Lecture 08 uses TeAAL's concern stack to explain why **Format** and **Mapping** are separate but coupled. A compressed format is useful only if the mapping traverses it concordantly. Sparse tiling is also a mapping/format problem because partitioning by shape and partitioning by occupancy expose different tradeoffs.

### Key claims used in this chapter

- Sparse tensors are naturally represented as fibertrees with missing coordinate/payload pairs; see TeAAL Section 2.1.
- Einsums specify computation but not iteration order; mapping chooses loop order and affects locality and load balance; see Section 2.2 and Section 2.3.
- Sparse tensors are typically compressed to remove zero elements, but sparse execution can introduce memory footprint variation, transfer imbalance, and compute load imbalance; see Section 2.3.
- Rank flattening, rank partitioning, and rank swizzling capture common sparse data orchestration behaviors; see Section 3.2.

### What students should remember

1. Sparse architecture is not just a PE microarchitecture problem.
2. Format, mapping, binding, and architecture must be specified together.
3. Fibertree transformations are a precise way to talk about sparse layout and tiling.
4. Load imbalance is a first-class sparse design issue.

### Limitations and assumptions

TeAAL is a modeling/specification framework, not a single accelerator design. This chapter uses it as conceptual support, not as proof that a specific SAF is optimal.

### Suggested insertion points

Use TeAAL when explaining why format choice is coupled to traversal order, why sparse tiling needs occupancy-aware partitioning, and why later lectures introduce fibertrees.

## Paper Bridge: SCNN

### Bibliographic identity

- **Title:** SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks
- **Authors:** A. Parashar et al.
- **Year / venue:** ISCA 2017
- **Local PDF:** `papers/L17_SCNN_Parashar_ISCA2017.pdf`

### Problem addressed

SCNN asks how to exploit both pruned weights and ReLU-induced activation sparsity in convolutional layers while keeping activations and weights compressed through most of the computation.

### Core idea

SCNN uses a planar-tiled input-stationary Cartesian-product sparse dataflow. It delivers vectors of nonzero weights and nonzero activations to a multiplier array, computes their Cartesian product, and scatters products to output accumulators.

### Relevance to this lecture

Lecture 08 uses SCNN as an example of single-sided skipping and as evidence that sparsity benefits are not free. SCNN's scatter network, compressed buffers, and metadata handling are exactly the overheads that the lecture warns about.

### Key claims used in this chapter

- The abstract states SCNN exploits zero-valued weights from pruning and zero-valued activations from ReLU, while using compressed encoding to reduce transfers and storage.
- Section II reports that typical layers can reduce work by a factor of 4 and up to a factor of 10 under the paper's measured density products.
- Section III introduces the PT-IS-CP-sparse dataflow and explains why input-stationary Cartesian-product computation matches sparse weights and activations.
- Section IV describes the PE with compressed storage, all-to-all multiplication, and scatter accumulation.
- The conclusion states SCNN uses both weight and activation sparsity and becomes more efficient than dense architectures when weights and activations are each below roughly 85% density.

### What students should remember

1. Dual-sparse work reduction requires both values and coordinates.
2. Cartesian products increase useful multiply opportunities but scatter output addresses.
3. Compression saves bandwidth only when decoders and routers can keep up.

### Limitations and assumptions

SCNN targets CNN inference and depends on compressed sparse blocks and sufficient nonzero work per PE. It is not a universal sparse tensor accelerator.

### Suggested insertion points

Use SCNN when discussing why avoiding zero work requires extra routing and metadata machinery, and as a preview of Lecture 09's sparse convolution dataflows.

## Paper Bridge: Eyeriss v2

### Bibliographic identity

- **Title:** Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices
- **Authors:** Y.-H. Chen et al.
- **Year / venue:** JETCAS 2019
- **Local PDF:** `papers/L17_EyerissV2_Chen_JETCAS2019.pdf`

### Problem addressed

Eyeriss v2 addresses compact and sparse DNNs whose layer shapes and sparsity patterns vary. The goal is to keep throughput and energy efficiency high when dense reuse assumptions no longer hold.

### Core idea

It combines a hierarchical mesh NoC with sparse PE support. The sparse PE stores activations and weights in a CSC-like compressed format, skips zeros directly in the compressed domain, and uses SIMD support to recover utilization.

### Relevance to this lecture

Eyeriss v2 demonstrates the distinction between gating and skipping. Original Eyeriss used gating for zero activations; Eyeriss v2 keeps data compressed on-chip and skips zeros to improve throughput.

### Key claims used in this chapter

- Section IV states original Eyeriss exploited input-activation zeros by gating logic/data accesses, while Eyeriss v2 skips zeros to improve throughput as well as energy.
- Section IV describes CSC encoding for both activations and weights and notes that compressed-domain processing can skip zeros without spending extra cycles.
- Section V reports large improvements for sparse AlexNet and sparse MobileNet under the paper's evaluation setup, while also noting workload imbalance and layer-shape limitations.

### What students should remember

1. Moving from gating to skipping changes the PE pipeline and storage format.
2. Sparse support adds control and storage overhead.
3. Workload imbalance remains even in a carefully designed sparse accelerator.

### Limitations and assumptions

The quantitative results are tied to Eyeriss v2's 65 nm implementation, benchmark models, batch size, and comparison baselines. This chapter uses them as evidence of design tradeoffs, not as universal speedup claims.

### Suggested insertion points

Use Eyeriss v2 in the gating/skipping distinction and in the discussion of compressed formats with throughput impact.

## Paper Bridge: The State of Sparsity in Deep Neural Networks

### Bibliographic identity

- **Title:** The State of Sparsity in Deep Neural Networks
- **Authors:** D. Blalock et al.
- **Year / venue:** MLSys 2020
- **Local PDF:** `papers/L16_StateOfPruning_Blalock_MLSys2020.pdf`

### Problem addressed

The paper surveys pruning methods and shows that pruning research often suffers from inconsistent comparisons and metrics.

### Core idea

It distinguishes unstructured pruning from structured pruning and argues that pruning should be evaluated as an efficiency/quality tradeoff curve rather than a single compression number.

### Relevance to this lecture

Lecture 08's structured sparsity section depends on the model-side fact that sparsity patterns are design choices. A hardware-friendly pattern can simplify metadata and traversal, but may change the model's accuracy/efficiency frontier.

### Key claims used in this chapter

- Section 2 defines pruning as producing a model with masked or removed parameters and distinguishes unstructured and structured pruning.
- Section 2 emphasizes the tradeoff between model efficiency and quality.
- The paper's literature review warns that reported compression and speedup metrics are not interchangeable.

### What students should remember

1. Sparsity is produced by model decisions, not only hardware decisions.
2. Structured sparsity is valuable only if the model can tolerate the constraint.
3. Theoretical speedup and realized hardware speedup are different metrics.

### Limitations and assumptions

The paper is about pruning evaluation, not sparse accelerator design. This chapter uses it to contextualize why hardware-friendly sparsity patterns must be evaluated with accuracy tradeoffs.

## Standalone Study Guide

Read the lecture in this order:

1. Reproduce the dot-product accounting table without looking.
2. Explain why skipping needs format metadata but gating does not.
3. For each sparse format, ask: "How do I find the next nonzero?"
4. Compute one HSS density by multiplying kept fractions.
5. Explain overbooking as an average-case buffer-utilization strategy.

## Self-Check Questions

1. Why does gating save energy but not cycles?
2. In the dot-product example, why does \(A\)-leader skipping still perform three computes when only two are effectual?
3. Why can bitmask compression be worse than uncompressed storage at 100% density?
4. What hardware state does RLE need that coordinate payload does not?
5. Why does a fixed 2:4 accelerator fail to exploit arbitrary 80% sparsity?
6. How does HSS produce more sparsity degrees than a flat list of \(G:H\) modes?
7. Why does sparse tiling make "largest tile that fits" difficult?
8. What is bumped data in overbooking?

## Exercises

1. **Format calculation:** For a length-32 vector with six nonzeros, 8-bit values, and 5-bit coordinates, compute uncompressed, bitmask, coordinate-payload, and 5-bit RLE storage sizes.
2. **Intersection trace:** Intersect \(A=\{1,4,9,10\}\) and \(B=\{0,4,5,10,11\}\) using a merge-style algorithm. Count metadata comparisons.
3. **Leader choice:** Suppose \(A\) has density 0.2 and \(B\) has density 0.8. Which operand should be the leader for single-sided skipping, and why?
4. **HSS design:** Choose two \(G:H\) levels that produce 75% sparsity. Explain the hardware and model-flexibility tradeoff.
5. **Tiling reasoning:** Describe a sparse tensor distribution where uniform-shape tiling wastes buffer capacity, then describe how overbooking changes the average tile size.
6. **Paper bridge:** Use SCNN to explain why a sparse accelerator may need a scatter network even when it performs fewer multiplications.

## Key Terms

| Term | Definition |
|---|---|
| **Activation sparsity** | Zeros in activations, often input-dependent; hardware must detect or encode them at runtime. |
| **Weight sparsity** | Zeros in trained weights, often produced by pruning; can often be known before inference. |
| **Effectual operation** | An operation that can change the output, e.g., multiplying two nonzero operands. |
| **Ineffectual operation** | An operation involving a zero operand or zero addend that cannot affect the final value. |
| **Gating** | Suppressing reads or compute in a cycle after detecting a zero; saves energy but not time. |
| **Skipping** | Advancing directly to useful coordinates; saves time and energy but requires metadata and traversal logic. |
| **Format** | The representation of values and coordinates in memory. |
| **Metadata** | Non-payload information such as bitmasks, coordinates, run lengths, segment pointers, or offsets. |
| **Single-sided intersection** | One operand's nonzeros drive traversal; the other operand is checked or fetched as follower. |
| **Dual-sided intersection** | Both operands' coordinate streams are intersected so only matching nonzero coordinates are processed. |
| **Bitmask** | A format with one bit per coordinate indicating whether the payload is nonzero. |
| **Coordinate payload** | A format storing each nonzero value with its coordinate. |
| **Run-length encoding** | A format storing the number of zeros before each nonzero. |
| **Structured sparsity** | Sparsity constrained to a predictable pattern, reducing metadata and decoder cost. |
| **\(G:H\) sparsity** | Exactly \(G\) nonzeros in every group of \(H\) values. |
| **HSS** | Hierarchical Structured Sparsity; nested \(G:H\) patterns whose densities multiply. |
| **Tile occupancy** | Number of nonzeros in a sparse tile. |
| **Overbooking** | Choosing nominal tiles larger than worst-case buffer capacity because most sparse tiles fit on average. |
| **Bumped data** | Nonzeros that overflow an overbooked buffer and must be streamed instead of reused from the buffer. |
| **Workload imbalance** | Unequal work across PEs caused by different nonzero counts. |

## Takeaways

- Sparse acceleration is an accounting problem: reduce unexploited ineffectual work without letting metadata/control overhead dominate.
- Gating, skipping, and format are separate design levers; skipping is the only one that directly reduces cycles.
- Format choice must be judged by compression efficiency and access efficiency.
- Structured sparsity trades model flexibility for predictable metadata and decoder cost; HSS composes simple structures to regain degree flexibility.
- Sparse tiling is occupancy-driven rather than shape-only; overbooking improves average buffer use while managing overflow.

## Appendix — Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L08-1 | Title | Administrative |
| L08-2 to L08-8 | What problem, TL;DR, SAF overview | Expanded with definitions and irregularity explanation |
| L08-9 to L08-22 | Dot-product example; gating vs. skipping | Rewritten as worked examples and accounting table |
| L08-23 to L08-48 | Representation formats | Expanded with bit-count calculations and access-efficiency discussion |
| L08-49 to L08-66 | Structured sparsity and HSS | Expanded with \(G:H\) and HSS density equations |
| L08-67 to L08-93 | Sparse tiling and overbooking | Rewritten as buffer-utilization narrative |
| L08-94 to L08-96 | Dataflow interplay, summary, reading | Integrated into hardware implications, connections, and source notes |

## Source Notes

- Lecture ordering and SAF definitions follow Lecture 08 slides 2-7.
- Dot-product counts follow Lecture 08 slides 9-22.
- Format bit-count examples follow Lecture 08 slides 27-36 and 37-47.
- Structured sparsity and HSS follow Lecture 08 slides 50-66.
- Sparse tiling, Tailors, and Swiftiles follow Lecture 08 slides 68-93. The local PDFs for Tailors/Swiftiles and HighLight were not provided in the Worker B input, so paper-specific claims for those works are kept slide-anchored.
- TeAAL discussion uses `papers/TeAAL.pdf`, especially Sections 2.1, 2.2, 2.3, and 3.2.
- SCNN discussion uses `papers/L17_SCNN_Parashar_ISCA2017.pdf`, especially Sections II-IV and VIII.
- Eyeriss v2 discussion uses `papers/L17_EyerissV2_Chen_JETCAS2019.pdf`, especially Sections IV-V.
- Pruning context uses `papers/L16_StateOfPruning_Blalock_MLSys2020.pdf`, especially Sections 2 and 3.

## Uncertainty Notes

- This chapter reconstructs the likely spoken explanation from slides and papers; the live lecture may have emphasized examples differently.
- The chapter avoids embedded slide images. Existing files under `assets/L08/` may still be copyright-sensitive, but they are outside Worker B's requested write scope.
- Quantitative claims from slides are cited as slide-derived unless independently checked against local PDFs.
