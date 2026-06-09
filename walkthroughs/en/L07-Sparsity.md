# L07 - Co-Design of DNN Models and Hardware: Sparsity

> **Course:** 6.5930/1 - Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze (MIT EECS)
> **Lecture date:** February 25, 2026 · **Slides:** 53 · **Source:** [`Lecture/L07 - Sparsity.pdf`](../../Lecture/L07%20-%20Sparsity.pdf)
>
> This walkthrough reconstructs the missing lecture narration from the slides and local papers. It is a self-study chapter, not a slide transcript.

---

## TL;DR

Sparsity means many tensor values are repeated, usually repeated zeros. In DNN hardware, zeros create two opportunities: do not store/move the zero, and do not execute a multiply-add whose result cannot change the output. L07 explains where zeros come from and why they are not automatically useful. Activation sparsity can arise naturally from ReLU, correlations in input data, or sparse graph structure. Weight sparsity is usually created by pruning, a model-hardware co-design process that chooses which weights or groups to remove, how to recover accuracy, and what hardware metric to optimize.

The central warning is that sparsity is an opportunity, not a guarantee. Skipping zeros requires metadata, compression formats, zero detection, scheduling, and load balancing. A dense accelerator may still perform the ineffectual work. A sparse accelerator may skip work but pay overhead. The best design depends on sparsity level, granularity, dataflow, memory hierarchy, and the actual deployment platform.

---

## What Problem This Lecture Solves

Earlier lectures showed that DNN accelerator efficiency is often dominated by memory movement and PE utilization, not just MAC count. L07 asks: if many DNN values are zero, can the system avoid moving and computing with them?

The naive answer is "yes, because anything times zero is zero." The hardware answer is more careful. To save energy or time, the system must know which values are zero early enough to avoid fetching, routing, or multiplying them. It must also keep PEs busy after removing work. If the nonzeros are irregular, one PE may receive many nonzeros while another receives few. The sparse representation may require indices, run lengths, masks, or other metadata. That metadata costs storage, bandwidth, and control logic.

This lecture therefore frames sparsity as co-design. The model decides where zeros come from; the hardware decides whether those zeros become real savings.

Source note: L07 slides 2-6 directly state the goals, sparsity definition, and effectual/ineffectual operation framework.

---

## Why This Lecture Matters

Sparsity is one of the main levers for reducing DNN inference cost, but it sits across several layers of the accelerator stack.

- **Algorithm/model:** pruning changes the trained network and may change accuracy.
- **Format:** compressed representations decide how nonzeros and metadata are stored.
- **Mapping:** the dataflow decides where sparse values are reused and where zero-skipping happens.
- **Architecture:** PEs need gating, skipping, intersection, buffering, or load-balancing support.
- **Evaluation:** the right metric might be energy, latency, storage, throughput, accuracy, or a full tradeoff curve.

The lecture prepares the ground for L08-L10. L07 says where sparsity comes from and what design choices it introduces. Later lectures show sparse accelerator mechanisms.

---

## Prerequisites and Mental Model

You should know:

- ReLU: $\operatorname{ReLU}(x)=\max(0,x)$.
- Dense MAC: a multiply-accumulate such as $z \leftarrow z + a\cdot w$.
- Memory hierarchy: moving data from farther memory usually costs more energy.
- Dataflow and partitioning: mapping determines where values live and when they are reused.

Mental model: imagine a dense accelerator as a factory line that processes every box, even empty boxes. Sparsity asks whether empty boxes can be detected before they consume conveyor bandwidth, storage, and worker time. The answer depends on the label system, routing system, and how evenly the remaining boxes are distributed.

---

## Learning Objectives

After this lecture, you should be able to:

- Define sparsity, density, effectual operation, and ineffectual operation.
- Distinguish activation sparsity from weight sparsity.
- Explain why ReLU creates natural activation zeros.
- Explain why graph adjacency matrices introduce structural sparsity.
- Describe the pruning loop: scoring, grouping, ranking, fine-tuning, and scheduling.
- Compare magnitude-based, feature-based, energy-aware, and platform-aware pruning objectives.
- Explain why unstructured sparsity and structured sparsity create different hardware tradeoffs.
- Explain why MAC count and weight count can be poor proxies for latency or energy.
- Identify what L08-L10 must add to turn sparse workloads into hardware speedups.

---

## Main Textbook-Style Narrative

### 1. Sparsity Creates Effectual and Ineffectual Work

In this lecture, sparsity usually means zeros. A tensor with 75% zeros has density 25%, meaning one quarter of its entries are nonzero.

The hardware opportunity comes from two identities:

$$
a \times 0 = 0, \qquad a + 0 = a.
$$

If a MAC uses a zero activation or zero weight, the multiply either produces zero or the add does not change the accumulator. The slides call the useful operations **effectual operations** and the useless zero-related operations **ineffectual operations**:

$$
\text{total operations}=\text{effectual operations}+\text{ineffectual operations}.
$$

But hardware usually performs:

$$
\text{actual operations}=\text{effectual operations}+\text{unexploited ineffectual operations}.
$$

The gap between ineffectual operations and unexploited ineffectual operations is the sparse accelerator design space. A perfect sparse machine would skip all ineffectual work. A realistic sparse machine skips some work and pays overhead for doing so.

Hardware implication: zero-skipping is valuable only if the saved arithmetic and movement exceed the costs of detecting zeros, carrying metadata, steering data, and balancing work.

Source note: the effectual/ineffectual operation framework is directly from L07 slides 5-8.

### 2. Compression Saves Movement, Not Just Storage

Compression stores only useful values plus enough metadata to reconstruct their positions. For sparse tensors, that could mean coordinate lists, compressed sparse rows, run-length encoding, masks, or structured patterns.

A compression format must be uniquely decodable. For DNN inference, it is usually lossless, because changing the tensor values can affect accuracy unless the approximation is explicitly part of model design.

The hardware reason compression matters is not merely capacity. If a compressed activation tile fits in a local buffer but the dense tile does not, the accelerator may avoid DRAM reads. The storage win becomes a data-movement win.

Common misconception: compression alone guarantees speedup. Compression can reduce bytes moved, but if the compute engine must decompress into a dense stream and still execute dense MACs, the operation count may not fall. Speedup requires the compute path to exploit the sparse representation.

Source note: compression requirements and benefits are slide-stated in L07 slide 11.

### 3. Activation Sparsity: Natural Zeros From Data and Nonlinearities

Activation sparsity refers to zeros in intermediate feature maps. L07 identifies three origins: ReLU, input correlations, and sparse input structure.

**ReLU:** if a pre-activation is negative, ReLU outputs zero. For the tiny matrix

$$
\begin{bmatrix}
9 & -1 & -3\\
1 & -5 & 5\\
-2 & 6 & -1
\end{bmatrix},
$$

ReLU produces

$$
\begin{bmatrix}
9 & 0 & 0\\
1 & 0 & 5\\
0 & 6 & 0
\end{bmatrix}.
$$

Five of nine values are zero. In a convolution layer, each zero activation can eliminate many weight multiplications if the hardware can skip them.

The slides report that AlexNet convolutional output feature maps after ReLU have about 75% zeros across the shown convolution layers. This is a slide-stated quantitative claim from L07 slide 10.

**Spatial correlation:** neighboring activations in an image feature map are often similar. Instead of processing full values, a system may process differences or deltas. If neighboring values are close, those deltas can be sparse. This is application- and representation-dependent.

**Temporal correlation:** consecutive video frames are similar. A system may reuse prior-frame computation or process motion/delta information, but it needs extra storage and a way to find redundancy. This is useful mainly when the same operation is applied repeatedly across time.

**Graph structure:** graph neural networks use adjacency matrices. Most real graphs are not complete, so adjacency matrices are usually sparse. A layer can be written as:

$$
X^{(\ell+1)}=\sigma(\hat{A}X^{(\ell)}W^{(\ell)}),
$$

where $\hat{A}$ is a normalized sparse adjacency matrix, $X^{(\ell)}$ is the dense node-feature matrix, and $W^{(\ell)}$ is the dense weight matrix. The order of multiplication matters: $(\hat{A}X)W$ and $\hat{A}(XW)$ have the same mathematical result under associativity, but the intermediate density and memory traffic can differ.

Hardware implication: activation sparsity is dynamic. The zero pattern can change with each input. Hardware needs runtime detection or a format that carries the positions of nonzeros.

Source note: ReLU, Cnvlutin, SnaPEA, PredictiveNet/Song, Diffy, temporal-correlation examples, and GNN setup are based on L07 slides 10-23. Only the slide deck was used for the cited architecture result numbers in this worker pass.

### 4. Weight Sparsity: Zeros Created by Model Design

Weight sparsity refers to zeros in learned parameters. Unlike activation sparsity, weight sparsity can be fixed after training. That makes it easier to store compressed weights offline and plan mappings around them.

The lecture first notes that weights may contain redundancy even before pruning. If a filter is $[A,B,A]$ and the input is $[1,2,3]$, dense processing computes:

$$
A\cdot1+B\cdot2+A\cdot3.
$$

Because $A$ appears twice, we can rewrite:

$$
A(1+3)+B\cdot2.
$$

This reduces three multiplications and three weight reads to two multiplications and two weight reads, at the cost of an extra addition and possibly a wider value entering the multiply. This example is directly based on L07 slide 26.

This is not pruning yet. It exploits repeated weights. Pruning goes further by setting selected weights or groups to zero.

### 5. The Pruning Pipeline

The pruning pipeline has five conceptual pieces.

**Scoring:** assign a score to a weight or group. Magnitude-based pruning uses $|w|$. Feature-based pruning estimates impact on output features. More advanced methods may use gradients, saliency, energy, or measured latency.

**Grouping:** decide the granularity. You may remove individual weights, rows, channels, filters, blocks, or patterns.

**Ranking:** compare scores and choose what to remove.

**Fine-tuning:** update the surviving weights to recover accuracy.

**Scheduling:** decide how aggressively to prune at each iteration.

A compact version of the loop is:

```text
train dense model
repeat:
    score weights or groups
    remove the lowest-ranked weights or groups
    fine-tune the remaining weights
until target accuracy/energy/latency/storage budget is reached
```

The classic early method, Optimal Brain Damage, uses second-derivative saliency to estimate impact on training error before deletion. The slides cite LeCun, NeurIPS 1989 for this history.

Source note: the pipeline follows L07 slides 27-30 and 46-48. Blalock et al., "What is the State of Neural Network Pruning?" Sections 2.1-2.4 provides the local-paper support for pruning masks, high-level prune/fine-tune algorithm, structure/scoring/scheduling/fine-tuning categories, and evaluation metrics.

### 6. Energy-Aware and Platform-Aware Pruning

Magnitude-based pruning asks, "Which weights look least important to accuracy?" Energy-aware pruning asks a different question: "Which removals reduce energy most for this hardware and dataflow?"

This distinction matters because one weight is not equal to another in hardware cost. A weight fetched from DRAM is much more expensive than a weight already in a register file. A weight in a high-reuse layer may have a different energy impact than a weight in a layer with little reuse.

L07 slide 32 gives a memory hierarchy energy table from Yang, CVPR 2017: ALU/RF as $1\times$, NoC as $2\times$, global buffer as $6\times$, DRAM as $200\times$. Treat these ratios as slide-stated values for the course example, not universal constants.

The slides report that Energy-Aware Pruning reduces AlexNet energy by $3.7\times$ and improves over magnitude-based pruning by $1.7\times$ in the cited work. Source: L07 slide 35, citing Yang, CVPR 2017.

Platform-aware pruning pushes the idea further. NetAdapt measures latency or energy on the actual target platform instead of relying only on MAC count. The lecture source uses this example to show that optimizing MACs can fail to optimize real latency. Source: L07 slides 37-41, citing Yang, ECCV 2018.

Hardware implication: the pruning objective must match the deployment objective. If the goal is phone latency, measure phone latency. If the goal is accelerator energy, model or measure accelerator energy with the actual mapping and memory hierarchy.

### 7. Structured vs. Unstructured Sparsity

Grouping is the bridge between model accuracy and hardware regularity.

**Unstructured sparsity:** remove individual weights. This offers maximum flexibility and often preserves accuracy well at a given sparsity level, but the zero pattern is irregular. Hardware needs indices, intersection, or sparse scheduling.

**Structured sparsity:** remove rows, channels, filters, blocks, or fixed patterns. This is easier for dense SIMD, systolic, or vector hardware because the remaining computation has regular shapes. The cost is that removing an entire structure is a coarser model change and may hurt accuracy earlier.

A useful way to read the granularity spectrum:

| Granularity | What disappears | Hardware effect | Model risk |
|---|---|---|---|
| Weight | individual scalar | highest irregularity, best flexibility | lower |
| Row/block/pattern | small regular group | some vector regularity | moderate |
| Channel | full input channel | removes repeated dense work | higher |
| Filter | full output channel/filter | shrinks dense layer shape | higher |

The slides cite Scalpel as pruning to match SIMD organization, pattern-based pruning as a middle ground, and mixture-of-experts as dynamic coarse-grained sparsity. These are slide-stated examples from L07 slides 43-45.

Common misconception: a model with 90% unstructured sparsity is automatically faster than a model with 50% structured sparsity. The first may have better compression but worse hardware utilization; the second may map to dense kernels and run faster.

### 8. Pruning Interacts With Layer Type and Starting Architecture

Pruning does not affect all layers equally. In AlexNet, the slides report larger weight reduction in fully connected layers than convolutional layers, and a smaller MAC reduction than weight reduction. Source: L07 slide 49, citing Han, NeurIPS 2015.

Modern efficient models are also harder to prune. If a model such as MobileNet or EfficientNet was already designed to remove redundancy, pruning may reduce accuracy faster than pruning an older overparameterized model. The slides explicitly state that accuracy drops more quickly for modern efficient DNN models and that an unpruned efficient model can outperform a pruned inefficient model. Source: L07 slides 50-51, citing Hoefler JMLR 2021 and Blalock MLSys 2020.

Hardware implication: pruning is not a substitute for architecture design. It is one tool in a broader model-hardware co-design loop.

### 9. Why Specialized Hardware Is Needed

L07 closes by noting that practical DNN sparsity is often on the order of 30-70%, while conventional sparse software libraries are designed for much higher sparsity levels. Source: L07 slide 52.

This explains why later lectures are needed. At 30-70% sparsity, zeros are common enough to matter but not so dominant that generic sparse libraries automatically win. Hardware must exploit moderate sparsity with low overhead.

Specialized sparse hardware may need:

- compressed storage formats,
- zero detection before fetch or before MAC,
- gating to avoid useless switching,
- metadata decode and address generation,
- sparse intersection logic,
- buffering for variable-rate nonzero streams,
- load balancing across PEs,
- mapping strategies that account for density.

---

## Worked Examples

### Example 1: Effectual Fraction

Suppose a dot product has $8$ weights and $8$ activations. The activation vector has zeros at positions 1, 4, and 7. The weight vector has zeros at positions 2 and 7.

A MAC is ineffectual if either operand is zero. The zero-affected positions are $\{1,2,4,7\}$, so $4$ of $8$ MACs are ineffectual. The effectual fraction is $4/8=50\%$.

Hardware meaning: a dense PE performs all $8$ MACs. A sparse PE might perform only $4$, but only if it can identify and schedule the nonzero pairs efficiently.

### Example 2: Metadata Can Overwhelm Tiny Values

Suppose a sparse vector of length $16$ has four nonzeros. If each value is 8 bits and each index is 4 bits, the compressed representation stores $4(8+4)=48$ bits. The dense representation stores $16\cdot8=128$ bits. Compression helps.

If the vector has twelve nonzeros, compressed storage is $12(8+4)=144$ bits, worse than dense storage. This is why sparse format choice depends on density and metadata cost.

### Example 3: Pruning Objective Changes the Answer

Suppose two candidate weights can be pruned with equal accuracy loss. Weight $w_1$ is fetched once from a local buffer. Weight $w_2$ is fetched many times from DRAM because the mapping provides little reuse. Magnitude pruning may treat them similarly if $|w_1| \approx |w_2|$. Energy-aware pruning should prefer removing $w_2$ because it saves more movement.

---

## Key Equations and How to Read Them

### Sparsity and Density

$$
\text{sparsity}=\frac{\#\text{zeros}}{\#\text{total values}}, \qquad
\text{density}=1-\text{sparsity}.
$$

Sparsity tells you the opportunity size. Density tells you how much useful work remains. Hardware cost depends on both plus metadata and load balance.

### Effectual Operation Accounting

$$
\text{actual operations}
=\text{effectual operations}+\text{unexploited ineffectual operations}.
$$

The goal of sparse hardware is to reduce the second term without making each remaining operation too expensive.

### GNN Layer

$$
X^{(\ell+1)}=\sigma(\hat{A}X^{(\ell)}W^{(\ell)}).
$$

$\hat{A}$ is sparse graph structure. $X^{(\ell)}$ and $W^{(\ell)}$ are often dense. Mapping must decide whether to multiply by $\hat{A}$ early or late, because intermediate density affects memory traffic.

### Pruned Model Mask

$$
f(x;M\odot W)
$$

$M$ is a binary mask and $W$ is the weight tensor. If $M_i=0$, weight $W_i$ is pruned. This notation is based on Blalock et al., Section 2.1.

---

## Hardware Implications

**Energy:** skipping a zero can save compute energy and data movement energy, but metadata and control consume energy.

**Bandwidth:** compressed tensors can reduce bandwidth, but irregular nonzero streams may create bursty or hard-to-coalesce accesses.

**Latency:** skipping work can reduce latency only if the scheduler avoids bubbles and load imbalance.

**Area:** sparse support needs extra hardware: decoders, comparators, masks, queues, arbiters, or intersection units.

**Utilization:** random sparsity can leave PEs idle unless work is dynamically balanced.

**Correctness:** pruning can change model outputs; activation skipping must preserve exact dense semantics unless approximate inference is intended.

**Programmability:** sparse mappings are harder to specify because format, mapping, and architecture interact.

---

## Common Misconceptions

### Misconception: Data movement is saved automatically when values are zero.

The system saves movement only if zeros are not fetched or if compressed storage avoids moving them. If zeros are fetched in dense format and discarded at the PE, DRAM bandwidth was already spent.

### Misconception: Higher sparsity always means faster execution.

Higher sparsity can mean less work, but it may also mean more irregularity, metadata overhead, and load imbalance. Granularity matters.

### Misconception: Weight count is a good energy metric.

Different weights can cause different memory traffic depending on layer shape, reuse, and dataflow. Energy-aware pruning exists because weight count alone is not enough.

### Misconception: Pruning an inefficient model is equivalent to designing an efficient model.

Pruning can help, but starting from a better architecture can dominate. L07 explicitly points to cases where an unpruned efficient model outperforms a pruned inefficient one.

---

## Connections to Previous and Later Lectures

**L01-L03:** energy and memory hierarchy motivate why skipping movement can matter more than skipping MACs.

**L05-L06:** pruning energy depends on mapping; the same zero may save different memory levels under different dataflows and partitions.

**L08-L10:** these lectures provide the accelerator mechanisms for exploiting sparse weights and activations.

**L12:** precision and sparsity are complementary compression axes: one reduces bits per value, the other reduces number of represented values.

**L13:** calculating data motion for sparse workloads requires counting nonzeros, metadata, and movement across each memory level.

---

## Paper Bridge: Computing's Energy Problem

### Bibliographic identity

- **Title:** "Computing's Energy Problem (and what we can do about it)"
- **Author:** Mark Horowitz
- **Year / venue:** ISSCC 2014
- **Used in lecture(s):** Supports L01/L03 memory-energy themes and L07's motivation for avoiding data movement.

### Problem addressed

The paper explains why energy, not just transistor count or peak operations, became a central computing constraint. It emphasizes that memory movement can dominate energy.

### Core idea

Energy-efficient systems require locality and specialization. DRAM accesses are much more expensive than internal cache accesses or functional operations, so algorithms and hardware should maximize reuse and avoid unnecessary movement.

### Relevance to this lecture

Sparsity is attractive partly because zeros need not move. Horowitz provides the energy context for why avoiding a DRAM fetch of a zero can matter more than avoiding a single arithmetic operation.

### Key claims used in this chapter

- DRAM access energy is listed as roughly 1-2 nJ, while an internal cache access or functional operation is around 10 pJ. Source: Horowitz ISSCC 2014, Section 5 "Don't Forget the Memory Energy."
- Energy-efficient computation requires strong locality and many operations per memory fetch. Source: Horowitz ISSCC 2014, Sections 5-6.

### What students should remember

- Sparse acceleration is as much about memory movement as arithmetic.
- A zero fetched from DRAM has already cost energy.
- Locality and format choices determine whether sparsity becomes a real hardware benefit.

### Limitations and assumptions

The numerical energy values are technology- and system-dependent. Use them as scale intuition, not universal constants.

### Suggested insertion points

Reference this paper when explaining compression, energy-aware pruning, and why skipping a zero late in the PE is less valuable than never moving it.

---

## Paper Bridge: What is the State of Neural Network Pruning?

### Bibliographic identity

- **Title:** "What is the State of Neural Network Pruning?"
- **Authors:** Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, John Guttag
- **Year / venue:** MLSys 2020
- **Used in lecture(s):** L07 pruning methodology and evaluation discipline.

### Problem addressed

The paper asks whether the pruning literature has reliable, comparable evidence about which methods work best. It finds that inconsistent datasets, architectures, metrics, and baselines make comparison difficult.

### Core idea

The paper frames pruning as applying a binary mask to a model, then compares pruning methods by structure, scoring, scheduling, fine-tuning, and evaluation metrics. It argues for standardized benchmarking and introduces ShrinkBench.

### Relevance to this lecture

L07 teaches the pruning design space. Blalock et al. explain why that design space must be evaluated carefully: parameter count, FLOPs, theoretical speedup, latency, and accuracy are not interchangeable.

### Key claims used in this chapter

- A pruned model can be represented as $f(x;M\odot W')$, where $M$ is a binary mask. Source: Blalock et al., Section 2.1.
- Many pruning strategies follow train, score/prune, fine-tune, iterate. Source: Algorithm 1 and Section 2.2.
- Methods differ mainly in structure, scoring, scheduling, and fine-tuning. Source: Section 2.3.
- Evaluation goals differ; parameter count and FLOPs are loose proxies for latency, throughput, memory usage, and power. Source: Section 2.4.
- The paper identifies inconsistent metrics and benchmarking as a major problem in pruning research. Source: abstract and Section 5.2.

### What students should remember

- Pruning is not one method; it is a family of choices.
- Always ask what metric a pruning result optimizes.
- A sparse model should be evaluated across an accuracy-efficiency curve, not by one cherry-picked point.
- Hardware-facing pruning should report direct metrics when possible.

### Limitations and assumptions

The paper is a pruning survey and benchmarking study, not a sparse accelerator architecture paper. It supports pruning methodology and evaluation discipline, not a particular hardware mechanism.

### Suggested insertion points

Use this bridge in the pruning pipeline section, the direct-metrics discussion, and the warning about efficient architectures versus pruned inefficient ones.

---

## Standalone Study Guide

### How to study this lecture

1. Start with effectual vs. ineffectual operations.
2. Separate activation sparsity from weight sparsity.
3. For each sparsity source, ask whether the zero pattern is known before runtime.
4. For each pruning strategy, ask what metric it optimizes.
5. For each sparse representation, estimate metadata and load-balance cost.

### Self-check questions

1. What is the difference between sparsity and density?
2. Why does ReLU create activation sparsity?
3. Why does skipping a zero at the PE save less than never fetching that zero?
4. Why can splitting pruning by weight magnitude miss energy opportunities?
5. What is the difference between unstructured and structured sparsity?
6. Why are MAC count and latency not interchangeable metrics?
7. Why do L08-L10 need specialized hardware mechanisms?

### Exercises

1. **Conceptual:** Explain effectual and ineffectual operations using a two-element dot product.
2. **Small calculation:** A vector has 64 entries, 20 nonzeros, 8-bit values, and 6-bit indices. Compare dense and coordinate-list storage.
3. **Design tradeoff:** Pick one layer and decide whether weight pruning or activation skipping is easier to exploit. State your assumptions.
4. **Pruning pipeline:** Design a pruning loop for a small MLP. Specify scoring, grouping, ranking, fine-tuning, and stopping criterion.
5. **Paper-reading bridge:** From Blalock et al., explain why reporting only parameter reduction can be misleading.
6. **Architecture reasoning:** Design a PE-level zero-skipping mechanism and list one benefit and one overhead.

---

## Key Terms

### Sparsity

The fraction of tensor values that are zero or otherwise repeated enough to exploit. In this lecture it usually means zeros.

### Density

The fraction of values that are nonzero. A 70% sparse tensor has 30% density.

### Effectual operation

A computation that changes the output. In a MAC, it usually means both operands matter.

### Ineffectual operation

A computation involving a zero that does not change the result. Sparse hardware tries to skip these.

### Activation sparsity

Zeros in intermediate activations. Often dynamic and input-dependent, especially with ReLU.

### Weight sparsity

Zeros in model parameters. Usually created by pruning and often known before inference.

### Compression format

A representation that stores nonzeros plus metadata. It matters because metadata can erase savings at moderate density.

### Pruning

Removing weights or groups of weights, often by applying a binary mask and fine-tuning the remaining model.

### Scoring

Assigning importance values to weights or groups before pruning.

### Grouping

Choosing the granularity of pruning: individual weights, blocks, rows, channels, filters, or patterns.

### Ranking

Ordering scored weights or groups to decide what to remove.

### Fine-tuning

Retraining the remaining weights after pruning to recover accuracy.

### Scheduling

Choosing how many weights or groups to prune at each iteration.

### Unstructured sparsity

Arbitrary individual zeros. Flexible for the model, hard for hardware.

### Structured sparsity

Zeros arranged in regular groups. Easier for hardware, often more restrictive for accuracy.

### Energy-aware pruning

Pruning that scores removals partly by their energy impact, not only by weight magnitude or accuracy.

### Platform-aware pruning

Pruning or adapting a network using direct measurements such as latency or energy on the target platform.

### Metadata

Extra information needed to locate nonzeros, such as indices, masks, or run lengths.

---

## Takeaways

- Sparsity creates the possibility of saving storage, data movement, and computation.
- Effectual/ineffectual accounting separates mathematical opportunity from what hardware actually skips.
- Activation sparsity is often dynamic; weight sparsity can often be fixed after training.
- Pruning is a pipeline of scoring, grouping, ranking, fine-tuning, and scheduling.
- The right pruning metric depends on the deployment objective.
- Unstructured sparsity favors model flexibility; structured sparsity favors hardware regularity.
- Moderate DNN sparsity requires specialized hardware support to become reliable speedup or energy savings.

---

## Connections

L07 connects to L01-L03 through memory energy, to L05-L06 through mapping-dependent reuse, to L08-L10 through sparse accelerator mechanisms, to L12 through precision/compression tradeoffs, and to L13 through explicit data-movement accounting. The unifying question is: when a value is zero, which layer of the stack knows it soon enough to avoid cost?

---

## Appendix - Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L07-1 | Title and metadata | Lecture identity |
| L07-2-L07-8 | Sparsity creates effectual and ineffectual work | Expanded with hardware-overhead explanation |
| L07-9-L07-17 | Activation sparsity | ReLU, compression, activation skipping, correlation |
| L07-18-L07-23 | Graph sparsity | GNN notation and operation-order discussion |
| L07-24-L07-26 | Weight redundancy | Gauss/UCNN-style repeated-weight intuition |
| L07-27-L07-31 | Pruning pipeline | Scoring and classic pruning |
| L07-32-L07-41 | Energy/platform-aware pruning | Expanded with direct-metric reasoning |
| L07-42-L07-45 | Structured vs. unstructured sparsity | Expanded with granularity tradeoff |
| L07-46-L07-48 | Ranking, fine-tuning, scheduling | Integrated into pruning pipeline |
| L07-49-L07-51 | Interplay with layer type and model architecture | Expanded with Blalock connection |
| L07-52-L07-53 | Summary and readings | Used for takeaways and paper bridge |

---

## Source Notes

- The lecture ordering follows `Lecture/L07 - Sparsity.pdf`.
- The effectual/ineffectual operation definitions, sparsity sources, ReLU/AlexNet activation sparsity, activation-sparsity architecture examples, GNN setup, pruning pipeline, energy-aware pruning examples, NetAdapt examples, structured-pruning examples, and summary sparsity range are based on L07 slides 2-53.
- Quantitative claims such as ~75% zeros in AlexNet feature maps, Cnvlutin speedup/area overhead, Minerva speed/power results, Energy-Aware Pruning energy improvements, NetAdapt latency improvements, Han pruning reductions, and 30-70% practical sparsity are slide-stated claims from L07 and should be checked against their original papers before reuse outside this companion.
- The memory-energy motivation uses `papers/L07_ComputingsEnergyProblem_Horowitz_ISSCC2014.pdf`, especially Section 5.
- The pruning methodology and evaluation discipline use `papers/L16_StateOfPruning_Blalock_MLSys2020.pdf`, especially Sections 2.1-2.4 and 5.2.
- Worked examples are original teaching examples.

## Uncertainty Notes

- Several architecture examples cited by the slides, including Cnvlutin, Minerva, SnaPEA, Diffy, UCNN, EAP, NetAdapt, Scalpel, PCONV, and PatDNN, were not independently checked against local PDFs in this worker pass.
- The chapter does not remove existing slide-derived assets under `assets/L07`; it avoids adding new copied figures.
- Exact sparse-hardware mechanisms are intentionally deferred to L08-L10.
