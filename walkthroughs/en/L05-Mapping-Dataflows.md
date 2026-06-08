# L05 - Mapping: Dataflows

> **Course:** 6.5930/1 - Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer and Vivienne Sze (MIT EECS)
> **Lecture date:** February 17, 2026. **Slides:** 111. **Source:** [`Lecture/L05-Mapping.pdf`](../../Lecture/L05-Mapping.pdf)
>
> This chapter reconstructs the missing lecture narration from the public slides. It is not a slide summary. Slide numbers are used as source anchors so a reader can cross-check the original ordering.

---

## TL;DR

Mapping is the set of choices that turns an abstract tensor computation into an executable schedule on a particular accelerator. L03 and L04 told us what the computation is: an Einsum. L05 asks a different question: in what order should the loops run, where should data live, and which operand should stay close to the MAC units?

The central idea is **stationarity**. A dense convolution repeatedly touches three data types: weights, input activations, and partial sums. A dataflow chooses which one is kept stationary in low-cost local storage while the others move. **Output Stationary (OS)** keeps partial sums local. **Weight Stationary (WS)** keeps weights local. **Input Stationary (IS)** keeps input activations local. All three compute the same mathematical result, but they create different memory traffic, interconnect pressure, buffer needs, and utilization patterns.

The lecture's quantitative motivation is stark: in the Eyeriss 65 nm energy table used by the slides, a DRAM access is about 200x the energy of a MAC, while a register-file access is about 1x. Therefore, a good dataflow is not mainly about reducing arithmetic. It is about replacing expensive movement with cheap reuse.

---

## What Problem This Lecture Solves

An Einsum is a mathematical contract. For a 1-D convolution,

```text
O[q] = sum_s I[q+s] * F[s]
```

the equation says which products must be accumulated into each output. It does not say whether the accelerator should compute all taps for one output before moving to the next output, or sweep one filter weight across all output positions, or iterate over input positions first. Those choices do not change the numerical result, but they change which data is read from DRAM, which data is kept in the register file, and which data must be communicated among PEs.

L05 solves this scheduling problem at the level of **dataflow**, the loop-order part of mapping. The question is:

> Given a fixed spatial accelerator and a fixed DNN layer, how should the loop nest be ordered so frequently reused data stays near the MAC units?

This question matters because DNN accelerators are usually limited less by the cost of multiplication than by the cost of feeding the multipliers. Slides L05-10 to L05-19 build this argument by counting memory reads and writes around a MAC. The worst case for AlexNet is stated as 724 million MACs and 2,896 million DRAM accesses; the best-case reduction shown in the slides is 61 million DRAM accesses. Those numbers are slide-stated for AlexNet and should be read as a motivation for reuse, not as a universal guarantee.

---

## Why This Lecture Matters

The naive mental model is "more PEs means faster and better." L05 corrects that model. A large PE array only helps when it is fed efficiently. If every MAC fetches a weight, an input, and a partial sum from DRAM, the array spends energy moving data rather than doing useful arithmetic.

The hardware architect's version of the problem is:

- **Energy:** which accesses happen at DRAM, global buffer, NoC, or RF?
- **Bandwidth:** can the memory system deliver operands fast enough to keep PEs busy?
- **Latency:** does the schedule create long reductions or serialized movement?
- **Area:** how much RF, global buffer, and interconnect support does a dataflow require?
- **Utilization:** do the layer dimensions match the parallel ranks exposed by the array?
- **Programmability:** can the accelerator change mappings across layers, or is it locked into one dataflow?

This is why mapping sits between algorithm and architecture in the TeAAL pyramid. The algorithm says what must be computed. The architecture provides storage, compute, and communication resources. Mapping is the plan that tries to make those resources look well matched to the workload.

---

## Prerequisites and Mental Model

You should bring three ideas from earlier lectures.

First, from L02, a convolution has named ranks. For a dense 2-D convolution:

```text
O[m][p][q] = sum_c sum_r sum_s I[c][p+r][q+s] * F[m][c][r][s]
```

Here `m` is output channel, `c` is input channel, `p,q` are output spatial positions, and `r,s` are filter spatial positions. A dataflow is often just a different permutation of these loop ranks.

Second, from L03 and L04, the Einsum fixes the arithmetic but not the traversal order. The same equation can be evaluated by many legal loop nests because addition and multiplication over the reduction ranks can be scheduled in different orders, subject to correctness of accumulation.

Third, from L01, memory hierarchy matters. L05 gives concrete normalized costs from the Eyeriss energy model used in the slides: ALU 1x, RF 1x, neighboring PE over NoC 2x, global buffer 6x, DRAM 200x. These values are attributed in the slides and in this chapter to Chen, Emer, and Sze, ISCA 2016, Table IV.

The mental model for this lecture is a MAC unit with three incoming operands:

```text
weight  ----\
input   ----- MAC ----> updated partial sum
psum    ----/
```

Every dense convolution MAC consumes a weight, an input activation, and an old partial sum, then produces an updated partial sum. Dataflow asks: which of these values should remain near the MAC long enough to be reused?

---

## Learning Objectives

After studying this chapter, you should be able to:

- Define the five aspects of mapping: partitioning, dataflow, data placement, compute placement, and partition sizing.
- Explain why local reuse and local accumulation reduce DRAM traffic.
- Read a loop nest and identify whether it is OS, WS, or IS.
- Explain how the outer loops encode stationarity.
- Count a small example showing that identical MACs can produce different memory traffic.
- Describe why a spatial dataflow implies a physical interconnect pattern.
- Explain why the energy comparison in L05 does not produce one universal winner.
- Read LoopTree as a notation for loop order plus storage placement.
- Distinguish slide-stated facts, paper-derived claims, background explanation, and teaching interpretation.

---

## 1. Mapping Has Five Aspects

**Source anchor:** slides L05-2 to L05-7.

Mapping is not one decision. Slides L05-3 to L05-5 divide it into five aspects.

| Aspect | What it chooses | Loop-nest effect | Hardware meaning |
|---|---|---|---|
| Partitioning | How tensors are split into tiles | Adds partitioned ranks and tile loops | Determines what can fit in buffers or be spread across PEs |
| Dataflow | The order of loops | Reorders loop ranks | Determines which operand changes slowest and can stay local |
| Data placement | Which memory level holds each tensor tile | Adds storage annotations | Controls whether accesses hit RF, global buffer, or DRAM |
| Compute placement | Which loops are temporal vs. parallel | Uses ordinary loops or `parallel_for` | Controls PE utilization and spatial sharing |
| Partition sizing | Exact tile sizes and loop bounds | Sets numeric loop extents | Balances capacity, bandwidth, and parallelism |

This lecture focuses on **dataflow**. That focus is deliberately narrow. A real mapping needs all five aspects, but loop order is the first lever because it determines the natural reuse pattern. L06 then studies partitioning in more depth.

### Intuition

Imagine a tensor tile sitting in a small register file. If the next many MACs use that same tile, the tile was worth loading. If the very next MAC needs a different tile, the register file did not help much. Dataflow is the choice that tries to make the next many MACs reuse the same data.

### Precise meaning

In this lecture, **dataflow** means the loop order of the computation, including which ranks are parallelized. It is not merely the geometric direction in which values move on a diagram. It is a compute, storage, and communication policy expressed through a loop nest.

### Common misconception

**Misconception:** Mapping is the same thing as hardware architecture.

**Correction:** Architecture provides resources: PEs, buffers, NoC links, and memory ports. Mapping decides how a particular computation uses those resources. The same hardware can sometimes run multiple mappings, and the same mapping can sometimes be implemented on different hardware with different costs.

---

## 2. Why Memory Access Is the Bottleneck

**Source anchor:** slides L05-8 to L05-27.

For the 1-D convolution

```text
O[q] = sum_s I[q+s] * F[s]
```

there are `Q * S` MACs. In the worst case, each MAC requires:

- read one filter weight,
- read one input activation,
- read one old partial sum,
- write one updated partial sum.

That is four memory operations per MAC. Slide L05-11 applies this worst-case counting to AlexNet: 724 million MACs imply 2,896 million DRAM accesses if all four operations go to DRAM.

The point is not that every accelerator actually does this. The point is that this is the baseline a memory hierarchy is trying to avoid.

### Two opportunities

Slides L05-13 to L05-19 identify two opportunities.

**Data reuse:** fetch a value once, then use it for multiple MACs. CNNs have several structural sources of reuse:

| Reuse type | Where it appears | What gets reused | Why it happens |
|---|---|---|---|
| Convolutional reuse | CONV | input activations and filter weights | sliding windows overlap |
| Fmap reuse | CONV and FC | input activations | one activation is multiplied by many filters |
| Filter reuse | CONV and FC, batch > 1 | filter weights | one weight is used for multiple input examples |

**Local accumulation:** keep the partial sum in local storage until it becomes a final output. Without local accumulation, every intermediate update can become a DRAM read and write. With local accumulation, the intermediate updates can happen in RF or local buffer, and only the final output must be written out.

### Computational intensity

Slide L05-12 asks when extra local memory levels help. The answer is:

```text
computational intensity > 1
```

For this lecture, read that as:

```text
MACs served by a fetched value / unique fetched values > 1
```

If a fetched word feeds only one MAC, storing it locally gives little benefit. If a fetched word feeds 10, 100, or 500 MACs, local storage can dramatically reduce DRAM traffic. Slide L05-17 states that filter/fmap DRAM reads can be reduced by up to 500x for AlexNet CONV layers under favorable reuse.

### Hardware implication

The memory hierarchy is not just a cache added after the fact. Its size, bandwidth, and placement determine which reuse patterns are exploitable. L05's spatial architecture model has DRAM feeding a 100-500 kB global buffer, feeding a PE array with 200-1000 PEs, where each PE has a 0.5-1.0 kB RF. Those capacities are slide-stated examples, not universal accelerator requirements.

---

## 3. Stationarity: The Reading Rule for Dataflow

**Source anchor:** slides L05-28 to L05-35 and L05-49 to L05-55.

The most useful reading rule in the lecture is:

> The tensor whose identifying ranks are placed in the outer loops changes slowest. That tensor is the most stationary.

For 1-D convolution, compare two legal loop nests.

```text
# Output Stationary
for q in [0, Q):
  for s in [0, S):
    o[q] += i[q+s] * f[s]
```

For a fixed outer `q`, the same output partial sum `o[q]` is updated for all `s`. The output stays local.

```text
# Weight Stationary
for s in [0, S):
  for q in [0, Q):
    o[q] += i[q+s] * f[s]
```

For a fixed outer `s`, the same weight `f[s]` is used for all `q`. The weight stays local.

The loop nest is doing the same nine, million, or billion MACs. It is not doing the same memory traffic.

### Important nuance

Stationary does not mean "never moves." It means a value stays resident over a useful interval. A WS accelerator still eventually loads new weights. An OS accelerator still eventually writes final outputs. The difference is how much reuse happens before that movement.

---

## 4. Worked Example: Same MACs, Different Traffic

**Teaching interpretation:** this example is original. It uses the energy ratios from Eyeriss Table IV as cited by the slides, but the tiny tensor is constructed for pedagogy.

Use a valid 1-D convolution with:

```text
W = 5 inputs
S = 3 filter taps
Q = W - S + 1 = 3 outputs
```

The computation has `Q * S = 9` MACs. The outputs are:

```text
O[0] = I[0]F[0] + I[1]F[1] + I[2]F[2]
O[1] = I[1]F[0] + I[2]F[1] + I[3]F[2]
O[2] = I[2]F[0] + I[3]F[1] + I[4]F[2]
```

Now isolate only the partial-sum operand.

| Policy | Psum reads | Psum writes | Approximate psum energy |
|---|---:|---:|---:|
| No local reuse | 9 DRAM | 9 DRAM | `(9 + 9) * 200 = 3600` |
| Output stationary | 9 RF | 6 RF intermediate + 3 DRAM final | `(9 + 6) * 1 + 3 * 200 = 615` |

The arithmetic is identical. The partial-sum traffic is not. OS avoids repeatedly sending intermediate partial sums to DRAM. This is the concrete mechanism behind the lecture's statement that memory access is the bottleneck.

### What this example does not claim

It does not count weight or activation traffic. A full model must count all three operands at all memory levels. L13 later formalizes that calculation with ISL. This example is deliberately small so the stationarity mechanism is visible.

---

## 5. Output Stationary Dataflow

**Source anchor:** slides L05-28 to L05-48.

**Definition:** In Output Stationary (OS), output partial sums are kept local while weights and input activations stream through.

For 1-D convolution:

```text
for q in [0, Q):
  for s in [0, S):
    o[q] += i[q+s] * f[s]
```

For each `q`, the accelerator performs all `S` contributions to that output before moving to the next output. If the partial sum fits in the PE's RF, intermediate psum reads and writes are local.

For a dense 2-D convolution, the slide loop nest is:

```text
for p in [0, P):
  for q in [0, Q):
    for r in [0, R):
      for s in [0, S):
        parallel-for c in [0, C):
          parallel-for m in [0, M):
            o[m][p][q] += i[c][p+r][q+s] * f[m][c][r][s]
```

The output spatial ranks `p,q` are outer temporal loops. The channel ranks `c,m` are parallelized in the slide example. The hardware meaning is that many PEs contribute to output channels and input channels while the relevant output partial sums are accumulated locally.

### Intuition

OS is the "finish this output before evicting it" strategy. It is attractive because partial sums are inherently write-heavy: every MAC updates one. If those updates spill to DRAM, energy becomes terrible.

### Hardware implications

Slides L05-29 to L05-31 cite OS examples such as ShiDianNao and KU Leuven designs. The common pattern is that activations and weights are delivered across the array, while partial sums accumulate in or near PEs. This favors hardware that can broadcast or multicast operands and provide low-cost local accumulation.

### Common misconception

**Misconception:** OS means outputs never move.

**Correction:** Final outputs still leave the PE or local buffer. OS reduces movement of intermediate partial sums, not final output storage.

---

## 6. Weight Stationary Dataflow

**Source anchor:** slides L05-48 to L05-84.

**Definition:** In Weight Stationary (WS), weights are kept local while input activations and partial sums move.

For 1-D convolution:

```text
for s in [0, S):
  for q in [0, Q):
    o[q] += i[q+s] * f[s]
```

For a fixed `s`, the same weight `f[s]` is used across all output positions `q`. If the weight is stored in a PE RF, one weight fetch can serve many MACs.

Slides L05-54 and L05-55 also show a parallel WS design:

```text
parallel_for s in [0, S):
  for q in [0, Q):
    o[q] += i[q+s] * f[s]
```

Here different PEs hold different weights. Activations are multicast to those PEs, and the resulting partial sums must be accumulated.

### NVDLA as a WS example

Slides L05-57 to L05-80 use a simplified NVDLA example. The slide-stated PE array is organized as `M * C` MACs, where `M` is output channels and `C` is input channels. The simplified loop nest is:

```text
for r in [0, R):
  for s in [0, S):
    for p in [0, P):
      for q in [0, Q):
        parallel-for m in [0, M):
          parallel-for c in [0, C):
            o[m][p][q] += i[c][p+r][q+s] * f[m][c][r][s]
```

The top loops are `r,s`, the filter spatial ranks, so the dataflow is weight stationary. For a fixed filter position, many input/output positions are processed while the relevant weights stay resident.

### Hardware implications

WS reduces weight read energy when each weight is used many times. It also creates different physical requirements from OS. Activations often need to be broadcast or multicast to PEs holding weights, and partial sums may require spatial accumulation or movement through an output path.

Slide L05-57 also exposes a utilization issue: if the layer's `M` and `C` dimensions do not match the array shape, some MACs can be idle. This is a mapping-architecture interaction, not a mathematical property of convolution.

### Common misconception

**Misconception:** WS is always best because weights are model parameters and should be reused.

**Correction:** WS is strong when weights are reused enough and fit well in local storage. It can be weaker when activation or partial-sum traffic dominates, when the layer shape underutilizes the array, or when reduction traffic becomes expensive.

---

## 7. Input Stationary Dataflow

**Source anchor:** slides L05-85 to L05-90.

**Definition:** In Input Stationary (IS), input activations are kept local while weights and partial sums move.

The 1-D convolution equation uses the compound input index `q+s`, so there is no explicit input loop in the OS form:

```text
for q in [0, Q):
  for s in [0, S):
    w = q + s
    o[q] += i[w] * f[s]
```

To make input stationarity explicit, change variables:

```text
w = q + s
q = w - s
```

Then iterate over raw input position `w`:

```text
for w in [0, W):
  for s in [0, S):
    q = w - s
    if 0 <= q < Q:
      o[q] += i[w] * f[s]
```

The guard matters. Without it, the loop would produce invalid output indices near the convolution boundaries.

### Why IS appears with sparse CNNs

Slide L05-86 says IS is used for sparse CNNs, citing SCNN (Parashar et al., ISCA 2017), and notes that it is not analyzed for dense CNNs in this lecture. The reason is structural: if inputs are large and many weights are zero, holding nonzero inputs and combining them with relevant weights can reduce reads from larger memory. Later sparse-architecture lectures return to this idea in more detail.

### Hardware implications

IS can create scattered updates to outputs because a fixed input `i[w]` contributes to several `o[w-s]` positions. That makes the accumulator path more complex than the simple "one output stays here" OS picture. In sparse settings, this complexity may be worth paying because zero skipping changes the traffic balance.

### Common misconception

**Misconception:** IS is just OS with loop names changed.

**Correction:** IS requires changing the traversal around the raw input coordinate and handling the `q = w - s` projection. That projection changes the output update pattern and the hardware path needed for accumulation.

---

## 8. Energy Comparison and Design Trade-offs

**Source anchor:** slides L05-91 to L05-94.

After introducing OS, WS, and IS, the slides revisit OS with three tiling variants:

| Variant | Output channels | Output activations | Slide note |
|---|---|---|---|
| OSA | single `M` | multiple `P * Q` | targeting CONV layers |
| OSB | multiple `M` | multiple `P * Q` | - |
| OSC | multiple `M` | single `P * Q` | targeting FC layers |

Slide L05-93 compares WS, OSA, OSB, OSC, and NLR under equal total area, 256 PEs, AlexNet CONV layers, and batch size 16. The main teaching result is not a single number. It is that WS and OS variants are much better than the no-local-reuse baseline, but none of them universally dominates all others.

### How to interpret the comparison

Layer shape changes the best dataflow. Large spatial maps, many channels, small filters, large filters, and batch size all change which operand has the most valuable reuse. A fixed-dataflow chip is therefore likely to be suboptimal for at least some layers.

### Hardware implication

This comparison motivates flexible or reconfigurable mappings. Flexibility is not free: it costs control complexity, storage flexibility, and often a more general NoC. But the alternative is committing silicon to one reuse pattern and hoping the workload matches it.

---

## 9. LoopTree: Dataflow Plus Data Placement

**Source anchor:** slides L05-95 to L05-111.

Pseudocode loop nests are good at expressing order. They are weaker at expressing where data is stored. LoopTree is introduced to represent both.

### Workload example: matrix multiplication

The slides switch to matrix multiplication:

```text
Z[m][n] = sum_ni A[m][ni] * B[ni][n]
```

The equation names tensor ranks, the binary operation, and the reduction rank `ni`. It still does not assume an operation order.

### Dataflow in LoopTree

In LoopTree, loop nodes encode order. An OS-like matrix multiplication might place `m` and `n` above `ni`, meaning one output `Z[m][n]` is accumulated across `ni`.

### Partitioning and rank swizzling

Slides L05-102 to L05-105 show that partitioning a rank such as `NI` creates sub-ranks such as `NI1` and `NI0`. The Einsum is rewritten with those sub-ranks, a process the slides call swizzling ranks. Dataflow is then specified after partitioning because the loop tree must order the partitioned ranks, not just the original rank.

### Storage plan

Slides L05-106 to L05-111 add storage nodes:

- DRAM is backing storage for all tensors.
- A global buffer can fetch all weights.
- Each iteration of `for m` can fetch a chunk of `A`.
- Each iteration of `for ni1` can fetch a chunk of the other operand.

The important point is that a complete mapping needs both loop nodes and storage nodes. Loop order says when values are used; storage placement says from which memory level they are supplied.

### Common misconception

**Misconception:** LoopTree is just a prettier loop nest.

**Correction:** LoopTree is a mapping specification. It can represent loop order, partitioned ranks, and storage placement in a single structure. That is why it becomes useful for TeAAL-style analysis and later data-motion counting.

---

## 10. Key Equations and How to Read Them

### 1-D convolution

```text
O[q] = sum_s I[q+s] * F[s]
```

`q` selects an output position. `s` selects a filter tap. `q+s` selects the input position used by that tap. Hardware meaning: the compound input coordinate is why input stationarity needs a projection.

### Dense 2-D convolution

```text
O[m][p][q] = sum_c sum_r sum_s I[c][p+r][q+s] * F[m][c][r][s]
```

`c,r,s` are reduction ranks. `m,p,q` name the output. Hardware meaning: keeping `m,p,q` fixed while iterating reductions favors local output accumulation; keeping `r,s` fixed favors weight reuse.

### Dataflow reading rule

```text
outer loops change slowest -> the associated tensor is most stationary
```

This rule is a teaching shorthand. A full mapping must also consider partitioning and data placement. An outer loop alone does not guarantee stationarity if the tile does not fit or the storage plan evicts it.

### Energy model reminder

```text
energy roughly tracks access count * energy per access level
```

This is why the same number of MACs can have different energy. It is also why quantitative claims require a source: access counts and per-level costs depend on workload, technology, and architecture.

---

## 11. Hardware Implications

**Energy:** Dataflow changes which accesses are RF, NoC, global buffer, or DRAM accesses. Since the cited hierarchy ranges from 1x to 200x, this can dominate arithmetic energy.

**Bandwidth:** A dataflow can reduce total traffic but demand high instantaneous bandwidth from a particular level. WS, for example, can require activation multicast; OS can require weight and activation delivery to psum-holding PEs.

**Latency:** Parallel loops reduce cycles, but reductions and psum movement can create latency if the accumulation path is poorly matched to the dataflow.

**Area:** Stationarity requires storage. OS needs local psum capacity. WS needs local weight capacity. IS needs local input capacity and often more flexible output update paths.

**Utilization:** Mapping choices interact with layer dimensions. NVDLA's simplified `M * C` array example shows that an array can leave MACs idle if channel dimensions do not match the exposed parallelism.

**Programmability:** A fixed dataflow can be simpler and efficient for matching layers. A flexible dataflow can adapt across layers but needs more mapping machinery.

**Correctness:** Loop reordering is legal only if accumulation dependencies are preserved. Partial sums can move through different storage levels, but the final reduction must include the same products.

---

## 12. Common Misconceptions

### Misconception: Dataflow means the direction data moves on a picture.

In this course, dataflow is primarily the loop-order policy that determines stationarity, reuse, and accumulation. A diagram of arrows is a consequence of the dataflow, not the definition.

### Misconception: The best dataflow is the one with the fewest MACs.

Dense OS, WS, and IS compute the same MACs for the same layer. Their difference is memory traffic, bandwidth, and utilization.

### Misconception: Keeping one operand stationary solves all movement.

It solves one part of movement. OS can still move many weights and inputs. WS can still move many activations and partial sums. IS can still create scattered output updates.

### Misconception: More local memory always helps.

Local memory helps when it captures reuse. If the dataflow does not reuse a tile before evicting it, the extra storage may add area without much benefit.

### Misconception: A dataflow comparison number transfers directly to any modern model.

The slide comparison uses AlexNet CONV layers, batch size 16, 256 PEs, and an energy model from a particular technology context. The conclusion "reuse matters and no fixed dataflow always wins" transfers more safely than the exact normalized bars.

---

## 13. Takeaways

- Mapping is the hardware schedule around an Einsum: loop order, tiling, storage placement, parallelism, and tile sizes.
- Dataflow is the loop-order part of mapping; it determines which operand is most stationary.
- OS keeps partial sums local, WS keeps weights local, and IS keeps inputs local.
- The same MAC count can have very different energy because memory accesses occur at different hierarchy levels.
- No single dataflow universally wins; layer shape, buffer capacity, interconnect, and PE utilization all matter.
- LoopTree extends loop nests with partitioned ranks and storage nodes so mappings can be analyzed by tools.

---

## 14. Connections to Previous and Later Lectures

**Builds on L01:** L01 introduced the memory-energy argument and the separation-of-concerns pyramid. L05 makes the mapping layer concrete through loop order and stationarity.

**Builds on L02:** L02 introduced convolution ranks. L05 turns those ranks into loop nests and asks which ranks should be outer, inner, or parallel.

**Builds on L03-L04:** Einsum expresses what to compute without committing to order. L05 shows why that separation is essential: many legal orders have different hardware costs.

**Leads to L06:** L06 studies partitioning. L05 says "which operand should stay local"; L06 asks "how large should each tile be, and how do partitioned ranks expose temporal and spatial reuse?"

**Leads to L07-L10:** Sparse architectures change the access counts and can make input-stationary or hybrid schedules more attractive. The IS projection in this lecture reappears when sparse fibers are traversed and projected.

**Leads to L13:** L05 uses intuitive access counting. L13 formalizes data-motion counting with sets, maps, timestamps, and shrink/delta calculations.

---

## 15. Paper Bridge: Eyeriss (Chen, Emer, and Sze, ISCA 2016)

### Bibliographic identity

- **Title:** *Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks*
- **Authors:** Yu-Hsin Chen, Joel Emer, Vivienne Sze
- **Year / venue:** 2016, ISCA
- **Used in lecture:** L05 dataflow taxonomy, energy hierarchy, and energy comparison; later lectures return to Eyeriss and sparse successors.

### Problem addressed

The paper addresses the energy cost of data movement in CNN accelerators. Highly parallel MAC arrays can provide throughput, but if weights, activations, and partial sums move through expensive memory levels too often, energy efficiency remains poor.

### Core idea

The paper systematizes existing dataflows by which data movement they minimize, including weight-stationary, output-stationary, and no-local-reuse styles. It then proposes **Row Stationary (RS)**, which tries to exploit reuse of filters, input feature maps, and partial sums together rather than optimizing only one operand.

### Relevance to this lecture

Slide L05-28 attributes the dataflow taxonomy to Chen, ISCA 2016. Slide L05-93 attributes the energy comparison to Chen et al., ISCA 2016. The normalized memory hierarchy used in slides L05-23 to L05-24 is aligned with the Eyeriss energy table, especially Table IV.

### Key claims used in this chapter

- **Energy hierarchy:** DRAM 200x, global buffer 6x, inter-PE movement 2x, RF 1x, relative to the MAC reference. Source anchor: Eyeriss Table IV and slides L05-23 to L05-24.
- **Taxonomy:** prior CNN accelerator dataflows can be characterized by which operand they keep stationary or which movement they reduce. Source anchor: Eyeriss Section V and slide L05-28.
- **Fair comparison method:** dataflows should be compared under constrained hardware resources such as equal area and equal parallelism. Source anchor: Eyeriss evaluation methodology and slide L05-93.
- **Row Stationary result:** the paper reports RS as more energy efficient than the compared existing dataflows under its AlexNet/equal-area methodology. This is paper-derived, not directly developed in the L05 slides.

### What students should remember

- OS and WS are not just names; they are reuse policies with hardware consequences.
- The energy numbers in this lecture come from a specific paper and technology context.
- A fair dataflow comparison fixes hardware assumptions before comparing mappings.
- "No universal winner" motivates more flexible dataflows and mapping tools.

### Limitations and assumptions

The Eyeriss results are tied to the networks, technology assumptions, area constraints, and architecture model in the paper. The exact improvement numbers should not be generalized without re-evaluating the workload and hardware. This chapter uses the paper as a bridge for L05 concepts, not as a full paper summary.

### Suggested insertion points

Read this bridge after Sections 8 and 9. It explains where the taxonomy and energy comparison come from and why the lecture naturally leads to more flexible dataflow design.

---

## 16. Standalone Study Guide

### What to master

- Explain mapping as a set of loop, storage, and parallelism decisions.
- Given a loop nest, identify the stationary operand.
- Explain why DRAM traffic can dominate energy even when MAC count is unchanged.
- Use a tiny convolution to count partial-sum traffic under OS and no local reuse.
- Explain the hardware path implied by OS, WS, and IS.
- Explain what LoopTree adds beyond an Einsum.

### Self-check questions

1. In the 1-D convolution `O[q] = sum_s I[q+s]F[s]`, why does `for q` outside `for s` make the output stationary?
2. Why does `for s` outside `for q` make the weight stationary?
3. What guard is needed in the input-stationary loop nest, and why?
4. Why can two mappings with the same MAC count have different energy?
5. Which of the five mapping aspects does dataflow control? Which aspects does it not control?
6. Why does NVDLA's simplified `M * C` PE organization create possible utilization loss?
7. What information does a LoopTree storage node express that an Einsum does not?
8. Why is "dataflow X is best" an incomplete claim unless the layer shape and hardware assumptions are stated?

### Exercises

1. **Conceptual:** For OS, WS, and IS, name the stationary operand and the operand most likely to create movement pressure.
2. **Small calculation:** Repeat the worked example for `W = 6`, `S = 3`, `Q = 4`. Count only partial-sum traffic for no local reuse and OS.
3. **Loop reading:** Given the loop order `r, s, p, q, parallel m, parallel c`, identify the dataflow and explain your reasoning.
4. **Design trade-off:** Suppose an accelerator has a very small RF but a larger global buffer. Which assumptions behind OS or WS become fragile?
5. **Paper bridge:** Read the Eyeriss abstract and Table IV. Which claims in this chapter depend on the paper rather than only on the slides?
6. **Open-ended architecture reasoning:** Design a NoC primitive for OS and one for WS. Explain why they are not identical.

---

## 17. Key Terms

### Mapping

The set of scheduling and placement decisions that map a tensor computation onto hardware. It includes partitioning, dataflow, data placement, compute placement, and partition sizing. Hardware relevance: mapping determines which memory levels and PEs are used for each part of the computation.

### Dataflow

The loop-order policy that determines which tensor values change slowly and can be reused locally. It is not just arrow direction in a figure. Hardware relevance: dataflow shapes memory traffic, NoC traffic, buffer requirements, and PE utilization.

### Stationarity

The property of a value staying in low-cost storage across multiple useful MACs. Hardware relevance: stationarity converts repeated DRAM or global-buffer accesses into RF or NoC-local accesses. Common confusion: stationary does not mean permanent.

### Output Stationary (OS)

A dataflow where output partial sums remain local while reduction terms are accumulated. Hardware relevance: reduces intermediate psum movement and favors local accumulation structures.

### Weight Stationary (WS)

A dataflow where weights remain local while activations and partial sums move. Hardware relevance: reduces weight read energy when each weight is reused many times, but can require activation multicast and psum reduction.

### Input Stationary (IS)

A dataflow where input activations remain local. Hardware relevance: useful in some sparse settings, but can create scattered output updates through the projection `q = w - s`.

### Partial Sum (Psum)

An intermediate accumulated value that is not yet a final output. Hardware relevance: psums are updated every MAC, so poor psum placement can dominate traffic.

### Local Accumulation

Keeping psum updates in RF or local buffer until the output is complete. Hardware relevance: avoids repeated DRAM reads and writes of intermediate psums.

### Data Reuse

Using a fetched value for multiple MACs. Hardware relevance: reuse is what makes small local memories worth their area and energy.

### Convolutional Reuse

Reuse caused by overlapping sliding windows in convolution. Hardware relevance: one input activation can contribute to several output windows.

### Fmap Reuse

Reuse of input activations across multiple filters or output channels. Hardware relevance: multicast or buffering can let one activation feed multiple MACs.

### Filter Reuse

Reuse of weights across multiple images in a batch or multiple spatial positions. Hardware relevance: weight storage and scheduling can amortize expensive weight fetches.

### Data Placement

The decision of which memory level holds each tensor tile at each point in the loop nest. Hardware relevance: placement determines whether a logical access costs RF, NoC, global-buffer, or DRAM energy.

### LoopTree

A tree notation for mapping. Loop nodes express dataflow; storage nodes express data placement; partitioned ranks express tiling. Hardware relevance: LoopTree can be analyzed by tools to estimate data movement.

### NLR (No Local Reuse)

A baseline with no useful local reuse. Hardware relevance: it is a warning case, not a good accelerator design.

---

## 18. Appendix - Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L05-1 | Title and metadata | Course framing |
| L05-2 | What problem this lecture solves | Mapping in separation of concerns |
| L05-3 to L05-5 | Section 1 | Five aspects of mapping |
| L05-6 to L05-7 | Learning objectives and source notes | Goals and background reading |
| L05-8 to L05-13 | Section 2 | 1-D convolution, MAC memory traffic, computational intensity |
| L05-14 to L05-19 | Section 2 | Reuse types and AlexNet DRAM reduction |
| L05-20 to L05-27 | Sections 2 and 11 | Spatial architecture and low-cost local access |
| L05-28 to L05-35 | Sections 3 and 5 | Taxonomy and 1-D OS |
| L05-36 to L05-48 | Section 5 | CONV-layer OS and OS examples |
| L05-49 to L05-55 | Sections 3 and 6 | 1-D WS and parallel WS |
| L05-56 to L05-84 | Section 6 | nn-X, NVDLA, and WS examples |
| L05-85 to L05-90 | Section 7 | IS, coordinate projection, sparse CNN note |
| L05-91 to L05-94 | Section 8 | OS variants and energy comparison |
| L05-95 to L05-111 | Section 9 | LoopTree, partitioning, swizzling, storage plan |
| Background | Sections 4, 10 to 17 | Teaching examples, misconceptions, key terms, exercises |

---

## 19. Source Notes

- **Slides:** The lecture ordering, mapping-aspects list, reuse taxonomy, OS/WS/IS loop nests, NVDLA simplified example, OS variants, energy comparison setup, and LoopTree introduction are based on `Lecture/L05-Mapping.pdf`.
- **Energy ratios:** The 1x/2x/6x/200x hierarchy is shown on slides L05-23 to L05-24 and is attributed here to Chen, Emer, and Sze, ISCA 2016, Table IV.
- **AlexNet access counts:** 724M MACs and 2,896M worst-case DRAM accesses are stated on slide L05-11. The 61M best-case DRAM access figure is stated on slide L05-19.
- **Up-to-500x reuse claim:** Slide L05-17 states that DRAM reads of filter/fmap can be reduced by up to 500x for AlexNet CONV layers.
- **Dataflow taxonomy:** Slide L05-28 cites Chen, ISCA 2016.
- **Hardware examples:** ShiDianNao, KU Leuven, nn-X/NeuFlow, NVDLA, TPU, ISAAC, PRIME, and others are named in slides L05-30 to L05-84. This chapter uses them only as brief examples, not as full paper summaries.
- **Input Stationary and sparse CNNs:** Slide L05-86 cites SCNN, Parashar, ISCA 2017.
- **Paper bridge:** The Eyeriss discussion uses the paper only to support the lecture concepts: energy hierarchy, taxonomy, fair comparison, and row-stationary motivation.
- **Teaching interpretation:** The worked examples, misconception sections, hardware implication synthesis, and several cross-lecture connections are original explanations created for self-study.

## 20. Uncertainty Notes

- The live lecture may have emphasized the animations more heavily than this chapter. This text replaces animations with loop-nest reasoning so it can stand alone without video.
- The AlexNet DRAM reduction numbers are slide-stated best-case figures. Actual reductions depend on layer dimensions, tile sizes, memory capacity, and implementation.
- The Eyeriss energy table is technology-specific. The qualitative hierarchy remains useful, but exact ratios should not be treated as universal constants.
- Row Stationary is included in the paper bridge to explain the motivation after the slide's "Is it possible to do better?" question. L05 itself does not fully teach RS.
- Existing `assets/L05/*.png` files appear to be extracted slide images. This rewrite does not embed them, but the files remain in the repository and should be reviewed separately for copyright risk before public release.
