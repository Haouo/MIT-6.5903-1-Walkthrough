# L06 - Mapping: Partitioning

> **Course:** 6.5930/1 - Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze (MIT EECS)
> **Lecture date:** February 18, 2026 · **Slides:** 76 · **Source:** [`Lecture/L06-Mapping-Partitioning.pdf`](../../Lecture/L06-Mapping-Partitioning.pdf)
>
> This chapter reconstructs the teaching layer behind the slides. The slide deck supplies the sequence and notation; the paper bridge supplies external technical context; the explanations, examples, and hardware interpretations are written as a self-study companion.

---

## TL;DR

Partitioning is the mapping decision that splits a tensor index into an outer tile index and an inner in-tile index. If an index $i$ with extent $I$ is split into $(i_1,i_0)$, then $I = I_1 I_0$ and $i = i_1 I_0 + i_0$. This simple algebraic move has two hardware consequences. First, it creates smaller working sets, so data can be retained in a fast buffer instead of repeatedly fetched from a distant memory. Second, it exposes independent partitions that can be assigned to different processing elements (PEs). The difficult part is not the notation; it is choosing which ranks to split, which split levels are temporal, which are spatial, and where reductions or communication become unavoidable.

L06 builds on L05's dataflow discussion. L05 asked, "In what order do loops execute, and where do weights, activations, and partial sums live?" L06 adds, "How large is each loop tile, and which tile loops run in parallel?" Together, loop order, tile shape, and spatial assignment define a mapping.

---

## What Problem This Lecture Solves

The naive view of mapping says: write the loop nest, then run it. Hardware rarely has that luxury. A DNN layer may have far more tensor data than can fit near the PEs, and a modern accelerator may contain many PEs that must be kept busy. If the loop nest is not partitioned, two bad things happen.

First, reuse appears too late. Suppose a value of $B_k$ is useful for many output elements $Z_m$. If the computation walks through a huge matrix before returning to that value, $B_k$ may have been evicted from the local buffer. The arithmetic is still correct, but energy is wasted reloading values.

Second, parallelism remains implicit. A tensor expression may contain thousands of independent output elements, but hardware needs a concrete rule saying which PE owns which subset. Partitioning supplies that rule by turning one index into multiple levels and allowing some outer levels to become spatial loops.

Source note: the two stated objectives, reducing reuse distance and identifying independent data sets for parallelism, are directly from L06 slides 3 and 8.

---

## Why This Lecture Matters

Partitioning is where abstract tensor algebra becomes a machine schedule. The same matrix multiplication can be mapped as a small local tile, a large streaming tile, or a distributed computation across many devices. These choices affect:

- **Energy:** whether data comes from RF/SRAM/global buffer/DRAM.
- **Bandwidth:** whether the memory system can feed the PEs.
- **Latency:** whether work runs sequentially or across many PEs.
- **Utilization:** whether PEs receive balanced work.
- **Communication:** whether a split rank requires a reduction across partitions.
- **Programmability:** whether the mapping can be expressed systematically in tools such as Timeloop or TeAAL.

A useful mental model is that partitioning moves a computation along a Roofline plot. Better tiling can increase operational intensity by reducing bytes moved per operation; more spatial partitioning can expose compute throughput, but only if the memory and communication system can sustain it.

---

## Prerequisites and Mental Model

You should be comfortable with four ideas from earlier lectures.

**Einsum notation:** repeated indices are summed; free indices remain in the output. For example, $Z_m = \sum_k A_{k,m}B_k$ is a matrix-vector multiply.

**Loop nests:** an einsum can be implemented by loops over free and reduction indices. The loop order controls locality and accumulation.

**Memory hierarchy:** values close to the PE are cheaper to access than values in DRAM. The exact energy ratios are technology-dependent, but the qualitative ordering is stable: local reuse matters.

**Dataflow:** a dataflow is a storage and scheduling policy for weights, activations, and partial sums. Partitioning is one of the main knobs that creates a dataflow.

Teaching interpretation: think of partitioning as putting graph paper over a tensor. The grid cells are tiles. A temporal tile says "process this cell, then the next cell, on the same hardware." A spatial tile says "give these cells to different PEs at the same time."

---

## Learning Objectives

After this lecture, you should be able to:

- Explain why partitioning an index always adds a rank.
- Convert a simple tensor expression from unpartitioned to partitioned notation.
- Distinguish temporal partitioning from spatial partitioning.
- Explain how partition size changes reuse distance and working-set size.
- Identify when splitting a reduction rank creates a cross-partition reduction.
- Trace the distributed matrix multiply example from partitioned tensors to delayed reduction.
- Compare tensor, head, and data parallel partitioning for attention.
- Use Roofline reasoning to explain why better tiling can improve attainable performance.

---

## Main Textbook-Style Narrative

### 1. Partitioning Adds Structure Without Changing the Math

The most important rule in L06 is:

$$
i \rightarrow (i_1,i_0), \qquad i = i_1 I_0 + i_0, \qquad I = I_1 I_0.
$$

The tensor values do not change. Only their addresses change. A vector $A_i$ becomes $A_{i_1,i_0}$. If $I=16$ and $I_0=4$, then $I_1=4$. The original element $A_{11}$ is now $A_{2,3}$ because $11 = 2 \cdot 4 + 3$.

This is why the slide says partitioning always adds a rank. The original index $i$ has been represented by two indices. A mapper now has two loops to place, order, and possibly distribute.

Common misconception: partitioning is not merely slicing a tensor for prettier storage. It changes the loop structure that the hardware sees. That loop structure is what determines locality, communication, and PE assignment.

Source note: the vector and matrix index identities are directly based on L06 slides 5-6.

### 2. Temporal Partitioning Reduces Reuse Distance

Consider the matrix-vector multiply from the slides:

$$
Z_m = \sum_k A_{k,m}B_k.
$$

The unpartitioned loop nest is:

```python
for m in range(M):
    for k in range(K):
        Z[m] += A[k, m] * B[k]
```

For each output $m$, the computation walks across all $k$. If $K$ is large, the active row/column working set may not fit near the PE. Values that should be reused later may be evicted before reuse occurs.

Now split both $m$ and $k$:

$$
m = m_1 M_0 + m_0, \qquad k = k_1 K_0 + k_0.
$$

The same computation becomes:

$$
Z_{m_1,m_0} = \sum_{k_1,k_0} A_{k_1,k_0,m_1,m_0} B_{k_1,k_0}.
$$

One possible loop nest is:

```python
for m1 in range(M1):
    for k1 in range(K1):
        for m0 in range(M0):
            for k0 in range(K0):
                Z[m1, m0] += A[k1, k0, m1, m0] * B[k1, k0]
```

When $m_1$ and $k_1$ are fixed, the inner loops touch a tile of $A$ with shape $K_0 \times M_0$, a tile of $B$ with $K_0$ values, and a tile of $Z$ with $M_0$ partial sums. If that working set fits in a local buffer, the hardware can reuse those values before moving to the next tile.

Worked example: let $M=8$, $K=8$, $M_0=2$, and $K_0=4$. For fixed $(m_1,k_1)$, the inner work uses $2 \times 4 = 8$ values of $A$, $4$ values of $B$, and $2$ partial sums of $Z$. Without tiling, one sweep for a single $m$ conceptually interacts with all $8$ values of $B$ and one full column of $A$. The partitioned version gives the mapper a concrete local working set of $8+4+2=14$ tensor values. If a PE-local buffer can hold those values, the tile can run with little traffic to higher memory levels.

Hardware implication: temporal partitioning is mainly a locality tool. It does not automatically add more PEs. It improves energy and bandwidth pressure by making reuse happen soon enough to be captured.

### 3. Spatial Partitioning Turns Tile Loops Into Parallel Work

The same split can be used spatially. For an elementwise multiply,

$$
Z_i = A_iB_i,
$$

split $i$ into $(i_1,i_0)$:

$$
Z_{i_1,i_0} = A_{i_1,i_0}B_{i_1,i_0}.
$$

If there are $I_1$ PEs, the outer loop can be spatial:

```python
spatial_for i1 in range(I1):
    for i0 in range(I0):
        Z[i1, i0] = A[i1, i0] * B[i1, i0]
```

The reason this is easy is that $i_1$ is a free output partition. Different $i_1$ values write different output elements and do not need to communicate.

This yields a useful rule:

- Splitting a **free rank** often creates independent output partitions.
- Splitting a **reduction rank** creates partial sums that must eventually be combined.

This rule is a hinge for the rest of the lecture.

### 4. Distributed Matrix Multiply: Partitioning Creates Communication

The matrix multiply case study begins with:

$$
Z_{m,n} = \sum_k A_{k,m}B_{k,n}.
$$

The slides state that the implementation is based on the ThunderKittens distributed matrix multiply algorithm, but the local repository does not include that paper PDF. In this chapter, ThunderKittens-specific details are treated as slide-derived anchors, while the general partitioning explanation is teaching interpretation.

The partitioned expression is:

$$
Z_{m_1,m_0,n} = \sum_{k_1,k_0} A_{k_1,k_0,m_1,m_0}B_{k_1,k_0,n}.
$$

Flattening shows that this is still the same matrix multiply:

$$
Z_{m_1M_0+m_0,n}
= \sum_{k_1,k_0} A_{k_1K_0+k_0,m_1M_0+m_0}B_{k_1K_0+k_0,n}.
$$

If $k_1$ is distributed across $G$ PEs, each PE owns only part of the reduction dimension. PE $g$ can compute the contribution from its local $k_1=g$ slice, but it cannot independently produce the final $Z$, because the final value needs the sum across all $k_1$ partitions.

That is the core communication lesson: splitting a reduction rank gives parallel work, but the result is partial. Parallelism was purchased with a reduction.

### 5. Delayed Reduction Makes the Dependency Explicit

The slide introduces a delayed-reduction variant. Instead of immediately reducing over $k_1$, put $k_1$ on the left-hand side:

$$
ZT_{k_1,m_1,m_0,n}
= \sum_{k_0} A_{k_1,k_0,m_1,m_0}B_{k_1,k_0,n}.
$$

Then reduce later:

$$
Z_{m_1,m_0,n} = \sum_{k_1} ZT_{k_1,m_1,m_0,n}.
$$

This is not a mathematical trick to change the answer. It is a scheduling trick to change when communication happens. Each PE can compute its local $ZT$ slice using local data, then a separate communication/reduction stage combines the slices.

Hardware implication: delayed reduction can improve utilization because PEs spend longer doing local matrix multiply before synchronizing. It can also make communication explicit enough for the compiler or runtime to choose an all-reduce, reduce-scatter, or other collective pattern. The tradeoff is that $ZT$ is larger than $Z$ because it carries the extra $k_1$ rank until reduction.

### 6. Distributed Tensor Shapes Are Mapping Contracts

Slides 13-17 introduce distributed tensors such as $AD_{g,k_0,m_1,m_0}$, $BD_{g,k_0,n}$, and $ZD_{g,m_0,n}$. These are not merely renamed arrays. They are contracts between the mapping and the machine.

- $AD_{g,k_0,m_1,m_0}$ says PE/group $g$ has the $k_1=g$ slice of $A$.
- $BD_{g,k_0,n}$ says the same PE/group has the matching $k_1=g$ slice of $B$.
- $ZD_{g,m_0,n}$ says an output partition is held locally before or after communication, depending on the stage.

A student should read each distributed tensor name by asking three questions:

1. Which rank chooses the PE?
2. Which ranks are local loops inside the PE?
3. Which missing ranks imply replication, reduction, or later assembly?

### 7. Attention Partitioning: Three Ways to Split Transformer Work

The attention section applies the same rank-splitting logic to transformer equations. The lecture source presents three strategies.

**Tensor parallel in the slides:** split the batch dimension $b \rightarrow (b_1,b_0)$ and run $b_1$ in parallel. Since different batch elements are independent in the forward pass, this is communication-light for the attention core. The cost is weight replication: each PE group needs the projection weights.

**Head parallel:** split the attention head dimension $h \rightarrow (h_1,h_0)$ and run $h_1$ in parallel. Heads are mostly independent until concatenation and output projection. This is attractive because transformer attention already decomposes computation across heads.

**Data/model-dimension parallel in the slides:** split both $h$ and $d$, then run $h_1$ and $d_1$ in parallel. This exposes more parallelism, but splitting $d$ touches projection reductions, so partial results may require communication.

The precise tensor names in slides 20-22 are dense, but the design question is simple: is the split rank a free rank, a naturally independent structural rank, or a contraction rank?

Worked example: suppose a transformer layer has $H=8$ heads and $4$ PE groups. A head-parallel split can choose $H_1=4$, $H_0=2$. Each PE group computes two heads. The attention score and value contraction for those heads can run locally. Later, the outputs must be concatenated into the model dimension and consumed by the output projection. If instead the model dimension $d$ is split across the PE groups, the Q/K/V projections create partial sums over $d$, so a reduction or collective communication step appears.

Connection to L04: L04 introduced attention as einsums over query, key, value, sequence, and head dimensions. L06 asks which of those ranks should be split for hardware.

---

## Worked Examples

### Example 1: Recovering the Original Coordinate

Let $I=12$, $I_0=3$, and $I_1=4$. The original index is $i=i_1I_0+i_0$.

- $(i_1,i_0)=(0,2)$ maps to $i=2$.
- $(i_1,i_0)=(2,1)$ maps to $i=7$.
- $(i_1,i_0)=(3,2)$ maps to $i=11$.

Hardware meaning: $i_1$ can name the tile or PE assignment; $i_0$ names the position inside that tile.

### Example 2: Free-Rank Split vs. Reduction-Rank Split

For $Z_m=\sum_k A_{k,m}B_k$:

- Splitting $m$ creates separate output tiles. Different PEs can own different $m_1$ values, because they write different $Z$ elements.
- Splitting $k$ creates partial sums. Different PEs can own different $k_1$ values, but they all contribute to the same final $Z_m$ values, so a reduction is needed.

Hardware meaning: both splits expose parallelism, but only one is communication-free.

### Example 3: Roofline Reading for a Tile

Operational intensity is:

$$
\text{operational intensity}=\frac{\text{operations}}{\text{bytes moved from the chosen memory level}}.
$$

Suppose a tiny tile performs $128$ MACs. If the implementation moves $512$ bytes from DRAM, its operational intensity is $128/512=0.25$ MAC/byte. If better temporal partitioning lets the same $128$ MACs move only $128$ bytes from DRAM, the operational intensity becomes $1$ MAC/byte. The arithmetic count is unchanged; the mapping changed the memory traffic.

Source note: this example is original. The definition and performance-bound interpretation are based on Williams, Waterman, and Patterson, "Roofline," Communications of the ACM 2009, Roofline Model section and Figure 1.

---

## Key Equations and How to Read Them

### Partition Identity

$$
i = i_1 I_0 + i_0,\qquad I=I_1I_0.
$$

$i_1$ selects the tile. $i_0$ selects the coordinate inside the tile. Hardware architects care because $i_1$ can become a temporal tile loop or a spatial PE-assignment loop.

### Partitioned Matrix-Vector Multiply

$$
Z_{m_1,m_0}
= \sum_{k_1,k_0} A_{k_1,k_0,m_1,m_0}B_{k_1,k_0}.
$$

The free ranks $(m_1,m_0)$ identify output elements. The repeated ranks $(k_1,k_0)$ identify the reduction. If $k_1$ is spatial, final output correctness requires summing across $k_1$ partitions.

### Delayed Reduction

$$
ZT_{k_1,m_1,m_0,n}
= \sum_{k_0} A_{k_1,k_0,m_1,m_0}B_{k_1,k_0,n},
\qquad
Z_{m_1,m_0,n}=\sum_{k_1}ZT_{k_1,m_1,m_0,n}.
$$

The temporary tensor $ZT$ preserves the partitioned reduction rank. That makes the communication boundary visible.

### Roofline Bound

$$
\text{attainable performance}
=\min(\text{peak compute},\ \text{peak bandwidth}\times\text{operational intensity}).
$$

The equation says that a mapping can be compute-bound or bandwidth-bound. Partitioning can help when it reduces bytes moved for the same operations, thereby increasing operational intensity.

---

## Hardware Implications

**Buffer sizing:** tile dimensions must be chosen so the active weights, activations, and partial sums fit in the intended buffer level.

**Bandwidth:** if a temporal tile is too large, values spill to higher memory. If it is too small, overhead and poor reuse may dominate.

**PE utilization:** spatial partitioning exposes parallel work, but only if the partitions have enough work and similar cost.

**Reduction cost:** splitting a reduction rank creates communication. The mapping must account for collective bandwidth and synchronization.

**Area:** more local storage can support larger tiles, but it consumes area and may lower clock frequency or PE count.

**Correctness:** partitioning must preserve the original flattened coordinate mapping and must not lose reductions.

**Programmability:** explicit rank splits make mappings easier to specify in formal tools, because tile shape and spatial assignment become first-class objects.

---

## Common Misconceptions

### Misconception: Partitioning and parallelism are the same thing.

Partitioning creates extra rank levels. Parallelism appears only when one of those levels is assigned spatially. A tiled loop can still run sequentially on one PE.

### Misconception: Splitting any rank is communication-free.

Splitting a free output rank is usually easy. Splitting a reduction rank creates partial sums that must be combined.

### Misconception: Smaller tiles are always better.

Smaller tiles may fit in local memory, but they can reduce reuse, increase loop overhead, and underutilize PEs. Tile size is a tradeoff, not a monotonic good.

### Misconception: Tensor names like $AD$ and $BD$ are bookkeeping only.

Distributed tensor names encode ownership. They tell us which PE holds which data and which communication step will be needed.

---

## Connections to Previous and Later Lectures

**L01-L03:** memory hierarchy and operational intensity explain why partitioning matters. The computation may be mathematically identical, but the energy and bandwidth can differ dramatically.

**L04:** attention is introduced as tensor algebra. L06 uses the same ranks to reason about parallelism in transformer execution.

**L05:** loop order and dataflow describe how data moves through a PE array. L06 adds tile shape and spatial assignment.

**L07-L10:** sparse architectures still need partitioning, but sparsity makes tile work uneven and introduces metadata. Load balance becomes harder.

**L13:** calculating motion turns these qualitative mapping choices into explicit counts of reads, writes, and transfers.

---

## Paper Bridge: Roofline

### Bibliographic identity

- **Title:** "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
- **Authors:** Samuel Williams, Andrew Waterman, and David Patterson
- **Year / venue:** Communications of the ACM, 2009
- **Used in lecture(s):** Supports L01/L02 roofline material and L06's partitioning-as-locality discussion.

### Problem addressed

The paper addresses the difficulty of understanding performance on multicore systems whose compute capability and memory bandwidth vary widely. Rather than predicting exact runtime, it offers a bound-and-bottleneck model that tells programmers and architects whether a kernel is limited by compute throughput or memory bandwidth.

### Core idea

The model plots attainable performance against operational intensity. Operational intensity is operations per byte of DRAM traffic after filtering by the cache hierarchy. Performance is bounded by the lower of peak compute and peak memory bandwidth times operational intensity.

### Relevance to this lecture

Partitioning changes how many bytes move between memory levels for the same number of arithmetic operations. Therefore, partitioning can move a kernel rightward on the Roofline plot by increasing operational intensity. It can also reveal when adding more spatial PEs will not help because bandwidth is already the limiting roof.

### Key claims used in this chapter

- Operational intensity is defined in terms of operations per byte of DRAM traffic, measured after cache filtering. Source: Roofline paper, Roofline Model section, CACM 2009, pp. 66-67.
- The Roofline bound is $\min(\text{peak compute},\text{peak bandwidth}\times\text{operational intensity})$. Source: Roofline paper, Roofline Model section, formula on p. 67.
- The ridge point indicates the minimum operational intensity required to reach peak compute. Source: Roofline paper, discussion of Figure 1, p. 67.

### What students should remember

- Roofline does not choose the partition for you; it explains why a partition matters.
- A mapping that reuses data locally can increase operational intensity.
- Spatial parallelism without enough bandwidth may just push harder against the same slanted roof.

### Limitations and assumptions

Roofline is a bound model, not an exact simulator. It abstracts away many details such as control overhead, synchronization, bank conflicts, and irregular sparsity. For DNN accelerators, it is best used as intuition before more detailed mapping and energy tools.

### Suggested insertion points

Use this bridge when explaining why temporal partitioning reduces bandwidth pressure and when evaluating whether a spatial partition is likely to be compute-bound or bandwidth-bound.

---

## Paper Bridge: TeAAL

### Bibliographic identity

- **Title:** "TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators"
- **Authors:** Nayak et al.
- **Year / venue:** MICRO 2023
- **Used in lecture(s):** L01 pyramid context; L06 mapping formalism; later sparse accelerator lectures.

### Problem addressed

Sparse tensor accelerators are hard to compare because each design combines an algorithm, tensor formats, mappings, architecture resources, and bindings. TeAAL provides a declarative way to describe these concerns and generate accelerator models.

### Core idea

TeAAL separates tensor computation from mapping. Extended einsums specify what computation is performed, while mapping specifications describe rank ordering, partitioning, and scheduling choices. This distinction matches L06: partitioning changes the mapping without changing the mathematical einsum.

### Relevance to this lecture

L06 is essentially a hand walkthrough of one part of TeAAL's mapping layer. When we split $i$ into $(i_1,i_0)$ and decide whether $i_1$ is temporal or spatial, we are making the kind of mapping choice TeAAL aims to express explicitly.

### Key claims used in this chapter

- TeAAL uses einsums to specify computation and mapping specifications to describe how ranks are ordered, partitioned, and scheduled. Source: TeAAL Section 2.2 and Section 2.3.
- TeAAL includes separate specifications for mapping, tensor format, architecture, and binding. Source: TeAAL Sections 3-4.
- The framework targets sparse tensor accelerators, where mapping and format choices interact strongly. Source: TeAAL abstract and Section 2.

### What students should remember

- Einsum says what is computed; mapping says how it is executed.
- Partitioning is a mapping operation, not a change to the mathematical result.
- Explicit rank splits are useful because tools can analyze locality, parallelism, and communication.

### Limitations and assumptions

TeAAL is a modeling framework, not a universal optimizer that magically finds the best mapping. The chapter uses TeAAL to clarify abstractions, not to claim any specific speedup for L06 examples.

### Suggested insertion points

Reference TeAAL after the partition identity and again when explaining distributed tensors as mapping contracts.

---

## Standalone Study Guide

### How to study this lecture

1. Practice converting one index $i$ into $(i_1,i_0)$ until the coordinate mapping feels automatic.
2. For each split, ask whether the split rank is free or reduced.
3. For each outer split rank, label it temporal or spatial.
4. Estimate the active working set of the inner tile.
5. Identify whether any communication or reduction is needed.

### Self-check questions

1. Why does partitioning $i$ into $(i_1,i_0)$ add a tensor rank?
2. In $Z_m=\sum_k A_{k,m}B_k$, why is splitting $m$ different from splitting $k$?
3. What data must fit in the local buffer for the partitioned matrix-vector tile?
4. Why does delayed reduction introduce $ZT_{k_1,m_1,m_0,n}$?
5. In attention, why is head parallelism usually easier than splitting a contraction dimension?
6. How can partitioning increase operational intensity?

### Exercises

1. **Conceptual:** Explain the difference between a tile loop and a spatial loop using one sentence each.
2. **Small calculation:** Let $I=24$ and $I_0=6$. What are $I_1$ and the partitioned coordinate of $i=17$?
3. **Loop-nest rewrite:** Partition $Z_m=\sum_k A_{k,m}B_k$ with $M_0=2$ and $K_0=4$, then write the loop nest.
4. **Design tradeoff:** You have four PEs. Would you split $m$ or $k$ for matrix-vector multiply? Explain the communication tradeoff.
5. **Paper bridge:** In Roofline terms, explain why a tile that reduces DRAM traffic can improve attainable performance even if the MAC count is unchanged.
6. **Open-ended architecture reasoning:** For attention with $H=16$ heads and $8$ PE groups, propose a head-parallel split and list which tensors are sharded or replicated.

---

## Key Terms

### Partitioning

Splitting an index range into multiple rank levels, such as $i \rightarrow (i_1,i_0)$. It matters in hardware because the new rank levels can become tile loops or PE-assignment loops.

### Tile

A subset of a tensor or iteration space selected by fixed outer partition indices. A good tile is large enough for reuse but small enough to fit in the intended buffer.

### Reuse distance

The number of intervening accesses between uses of the same value. Temporal partitioning tries to reduce reuse distance so values remain in fast storage.

### Temporal partitioning

A partition whose outer tile loop runs sequentially. It is primarily a locality and buffer-management mechanism.

### Spatial partitioning

A partition whose outer tile loop is distributed across PEs. It is primarily a parallelism mechanism.

### Free rank

An index that appears in the output. Splitting a free rank often creates independent output partitions.

### Reduction rank

An index that appears only on the right-hand side of an einsum and is summed away. Splitting it creates partial sums that need reduction.

### Working set

The data that must be live during a tile's inner loops, including input tiles and partial sums.

### Delayed reduction

A transformation that keeps a partitioned reduction rank in a temporary output, then reduces it later. It exposes the communication stage.

### Operational intensity

Operations per byte moved from a chosen memory level, often DRAM in classic Roofline. Higher operational intensity usually means better ability to use compute throughput.

### Ridge point

The Roofline point where the bandwidth roof meets the compute roof. It indicates the operational intensity needed to become compute-bound.

### Distributed tensor

A tensor whose indices include a rank that identifies PE or device ownership, such as $g$ in $AD_{g,k_0,m_1,m_0}$.

---

## Takeaways

- Partitioning changes tensor rank structure, not the mathematical result.
- Temporal partitioning is about locality; spatial partitioning is about parallel work.
- Free-rank splits are usually easier to parallelize than reduction-rank splits.
- Delayed reduction makes communication explicit and gives the mapper a place to schedule collectives.
- Attention partitioning is the same rank-splitting idea applied to transformer dimensions.
- Roofline explains why partitioning can improve performance by reducing bytes moved per operation.

---

## Connections

This lecture connects backward to L05 because dataflow needs tile shapes and spatial loops to become a complete mapping. It connects sideways to L03/L04 because einsum notation makes rank splitting precise. It connects forward to L07-L10 because sparse tensors complicate partitioning with irregular densities and load balance. It connects directly to L13 because calculating data motion requires knowing the exact partitioned loop nest.

---

## Appendix - Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L06-1 | Title and metadata | Lecture identity |
| L06-2 | Main narrative | Partitioning topic setup |
| L06-3 | What problem this lecture solves | Objectives are directly slide-stated |
| L06-4 | Temporal partitioning | Unpartitioned matrix-vector example |
| L06-5 | Partitioning adds structure | Vector rank split |
| L06-6 | Partitioning adds structure | Matrix rank split |
| L06-7 | Temporal partitioning | Partitioned matrix-vector loop nest |
| L06-8 | Spatial partitioning | `spatial_for` example |
| L06-9-L06-18 | Distributed matrix multiply | Expanded with reduction/communication explanation |
| L06-19-L06-22 | Attention partitioning | Expanded with transformer-rank interpretation |
| L06-23-L06-76 | Source notes | Mostly blank or animation-frame pages in extracted text; treated as no additional conceptual content |

---

## Source Notes

- The lecture ordering and core examples follow `Lecture/L06-Mapping-Partitioning.pdf`.
- The objectives of partitioning are directly from L06 slide 3.
- The vector and matrix partition identities are directly from L06 slides 5-7.
- The distributed matrix multiply notation and delayed reduction are based on L06 slides 10-18. ThunderKittens is cited by the slide deck, but its paper PDF was not found locally for this worker pass, so this chapter treats ThunderKittens-specific claims as slide-stated only.
- The attention partitioning strategies are based on L06 slides 20-22, with standard transformer background from earlier L04 material.
- The Roofline bridge uses `papers/Roofline Model.pdf`, especially the Roofline Model section and Figure 1 discussion.
- The TeAAL bridge uses `papers/TeAAL.pdf`, especially Sections 2.2, 2.3, 3, and 4.
- Worked examples are original teaching examples unless explicitly marked otherwise.

## Uncertainty Notes

- The live lecture may have emphasized the later animation frames differently; the extracted slide text after L06-22 contains mostly blank animation pages.
- Exact ThunderKittens implementation details should be reviewed against the original paper if it is later added locally.
- This chapter does not delete or audit existing slide-derived assets under `assets/L06`; it simply avoids adding new copied figures.
