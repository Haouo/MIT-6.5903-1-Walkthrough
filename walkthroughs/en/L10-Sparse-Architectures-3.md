# L10 — Sparse Architectures 3

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze
> **Lecture date:** March 9, 2026 · **Slides:** 66 · **Source:** [`Lecture/L10-Sparse_Architectures-3.pdf`](../../Lecture/L10-Sparse_Architectures-3.pdf)
>
> This chapter reconstructs the missing teaching layer from the slides. It does not reproduce slide or paper figures; diagrams are described textually for copyright safety.
>
> **Traceability note:** the PDF internally labels these slides as `L06-*`. This repository treats the deck as Lecture 10, so source anchors below use both names: Lecture 10, slide label `L06-n`.

---

## TL;DR

Sparse acceleration becomes genuinely hard when **both** weights and activations are compressed. If only one operand is sparse, a hardware designer can traverse the sparse operand and directly index the dense operand. If both operands are sparse, the accelerator must also solve a coordinate problem: which nonzero weight and nonzero activation pairs actually meet at the same output coordinate?

Lecture 10 moves from the easy cases to the joint case. It begins with **gating** versus **skipping**, revisits sparse-weight-only and sparse-activation-only convolution, then studies two approaches to joint sparsity. **SCNN** uses an input-stationary Cartesian product of nonzero activations and nonzero weights, then scatters products to output accumulators. **ISOSceles** uses a two-stage **IS-OS dataflow**: an input-stationary pass creates a small intermediate tensor, then an output-stationary pass consumes it after rank swizzling. The central lesson is that sparsity is not only a format decision. It couples representation, loop order, coordinate generation, routing, buffer sizing, and load balance.

---

## What Problem This Lecture Solves

Previous sparse lectures established that zeros can reduce arithmetic and memory traffic. The remaining question is architectural:

> If weights and activations are both sparse, how can a real accelerator skip useless multiplies without drowning in metadata, random access, scatter traffic, and load imbalance?

The naive answer is "compress both tensors and loop over their nonzeros." That is incomplete. A convolution product is legal only when the weight coordinate and activation coordinate imply a valid output coordinate. In dense code, the nested loops enforce legality for free. In sparse code, legality must be reconstructed from metadata.

This lecture therefore solves a mapping problem, not merely a storage problem. It asks which dataflow makes the coordinate problem cheap enough to implement.

---

## Why This Lecture Matters

Sparsity looks mathematically attractive because a zero product contributes nothing: $0 \times x = 0$. Hardware does not get that benefit automatically. A zero can save:

- **Energy only**, if the machine gates the multiplier but still occupies a cycle.
- **Energy and time**, if the machine removes the operation from the schedule.
- **Nothing**, if metadata and routing overhead exceed the saved work.

For a hardware architect, the useful question is not "is the tensor sparse?" but "where does the sparsity appear, is the coordinate stream regular enough to exploit, and what hardware must be added to keep the nonzero work flowing?"

---

## Prerequisites and Mental Model

You should remember three ideas from L07-L09:

- A **fiber** is a one-dimensional slice of a tensor, often represented as coordinate-payload pairs such as `(coord, value)`.
- A **concordant traversal** visits compressed coordinates in their stored order; a **discordant traversal** asks for data in an order that the format does not naturally provide.
- A **dataflow** chooses which tensor stays near the PE and which tensor streams through memory and interconnect.

The mental model for this lecture is a small 1-D convolution:

$$
o[q] = \sum_s i[q+s] \cdot f[s].
$$

If `f` is sparse and `i` is dense, a nonzero weight coordinate `s` directly tells the hardware which input location `q+s` to read. If `i` is sparse and `f` is dense, a nonzero input coordinate `w` directly tells the hardware which filter location `w-q` to read. If both are sparse, neither direct lookup is free: the machine must discover coordinate matches or produce products and scatter them to their output coordinates.

---

## Learning Objectives

After this lecture, you should be able to:

- Distinguish **gating** from **skipping**, and explain why only skipping reduces latency.
- Rewrite a 1-D convolution loop for sparse weights, sparse inputs, and joint sparsity.
- Explain why sparse-weight-only and sparse-input-only accelerators are easier than joint-sparse accelerators.
- Describe how **Cambricon-X** and **Cnvlutin** fit the single-sparsity cases shown in the slides.
- Explain SCNN's **input-stationary Cartesian-product** dataflow and why it needs a scatter accumulator.
- Explain why SCNN's useful work scales with the product of activation density and weight density, while its hardware overhead does not disappear.
- Explain ISOSceles's **IS-OS** split, why an intermediate tensor appears, and why **swizzling** changes a discordant traversal into a concordant one.
- Evaluate sparse-accelerator claims by asking what is counted: arithmetic, memory traffic, metadata, routing, utilization, or end-to-end speedup.

---

## Main Textbook-Style Narrative

### 1. Gating Is Not Skipping

Lecture 10 starts by reminding the reader that a zero operand offers two different opportunities. **Gating** detects a zero and disables parts of the datapath, such as a multiplier or memory read port. The cycle still occurs, so latency does not improve. **Skipping** removes the operation from the dynamic schedule; the machine executes fewer useful cycles.

Lecture 10 slide `L06-7` gives Eyeriss as the gating example and reports a 45% PE power reduction when input activations are zero. This is a slide-derived quantitative claim. The teaching interpretation is that gating is an excellent low-risk first step: it preserves the dense schedule and avoids complicated metadata-driven load balancing. Its limitation is equally important: if half the activations are zero, the accelerator does not automatically finish in half the time.

### 2. Sparse Weights Only: Traverse Weights, Index Inputs

For sparse weights, the compressed filter stores only nonzero `(s, f_val)` pairs. A simple output-stationary loop is:

```text
for q in [0, Q):
  for (s, f_val) in f:
    w = q + s
    o[q] += i[w] * f_val
```

The sparse tensor is traversed concordantly. The dense input tensor is still directly indexed. The important hardware implication is that the coordinate generator must compute `w = q + s`, but it does not need to search for a matching input coordinate. This is why sparse-weight-only designs can be relatively clean.

The weight-stationary variant swaps the first two loops:

```text
for (s, f_val) in f:
  for q in [0, Q):
    w = q + s
    o[q] += i[w] * f_val
```

Now the weight can remain close to the PE while multiple outputs reuse it. The tradeoff is output partial-sum traffic. If outputs are many and the local accumulator capacity is small, the design may save weight reads while increasing partial-sum movement.

Lecture 10 slides `L06-22` through `L06-24` introduce parallel sparse-weight traversal by splitting a fiber into equal chunks. The subtlety is that a compressed fiber can be split by **position in the compressed stream** rather than by coordinate value. Position splitting gives each PE a similar number of nonzero entries, but the coordinate ranges may be irregular. Coordinate splitting gives clean spatial ownership, but load balance may become poor if nonzeros cluster.

### 3. Sparse Inputs Only: Traverse Activations, Index Weights

For sparse input activations and dense weights, the roles reverse:

```text
for q in [0, Q):
  for (w, i_val) in i if q <= w < q + S:
    s = w - q
    o[q] += i_val * f[s]
```

The input fiber is sparse, so only nonzero activations produce MACs. The dense filter is directly indexed by `s = w - q`. Lecture 10 slides `L06-28` through `L06-35` present this as a sparse sliding window. The window condition matters: not every nonzero activation contributes to every output.

Cnvlutin, shown in Lecture 10 slides `L06-36` through `L06-38`, is the case-study architecture for activation skipping. The slide-derived lesson is that per-channel encoders remove zero activations, and the hardware uses activation coordinates to select the proper weight. The architectural constraint is that the dense side must remain cheap to index. If the weights were also compressed, direct `getPayload(s)` would no longer be a simple array lookup.

### 4. Joint Sparsity: The Coordinate Problem Appears

When both tensors are sparse, the loop cannot simply traverse one compressed stream and directly index the other. Suppose:

- Nonzero input coordinates are $i = \{0, 3, 4\}$.
- Nonzero filter coordinates are $f = \{0, 2\}$.
- Output coordinate is computed by $q = w - s$.

The Cartesian products are:

| Input `w` | Weight `s` | Output `q = w-s` | Legal if output range includes `q`? |
|---:|---:|---:|---|
| 0 | 0 | 0 | yes |
| 0 | 2 | -2 | usually no |
| 3 | 0 | 3 | yes |
| 3 | 2 | 1 | yes |
| 4 | 0 | 4 | maybe yes |
| 4 | 2 | 2 | yes |

This example shows the new work: the accelerator must compute output coordinates, reject illegal products, and route legal products to accumulators. Dense loop indices used to do that silently.

### 5. SCNN: Cartesian Products Plus Scatter

SCNN, attributed on Lecture 10 slides `L06-45` through `L06-51` to Parashar et al., ISCA 2017, chooses an **input-stationary Cartesian-product** strategy. In each PE, a vector of nonzero activations and a vector of nonzero weights are multiplied all-to-all. If there are $I$ activation lanes and $F$ weight lanes, the PE creates $I \times F$ candidate products per step.

The conceptual loop is:

```text
for (w, i_val) in sparse_inputs:
  for (s, f_val) in sparse_weights if product_is_legal(w, s):
    q = w - s
    scatter_add(o[q], i_val * f_val)
```

The multiplication step is regular. The accumulation step is irregular. Each product may target a different output coordinate, so SCNN needs a scatter network and banked accumulators. This is a classic sparse-hardware pattern: removing zero MACs exposes irregular communication.

From the SCNN paper, the architecture stores both weights and activations in compressed-sparse form, computes a Cartesian product inside a PE, computes output coordinates from sparse indices, and routes products through a scatter accumulator array (SCNN Sections III-IV, especially Figure 5 and Figure 6 in the paper). The paper reports 2.7x performance improvement and 2.3x energy reduction over a dense accelerator across evaluated networks (SCNN Abstract and Section VI). These numbers are paper-derived claims.

### 6. ISOSceles: Split the Work Into IS Then OS

Lecture 10 slides `L06-53` through `L06-66` present another way to handle joint sparsity. Instead of generating many irregular products and scattering them immediately, ISOSceles uses an **IS-OS** decomposition.

The output-stationary joint-sparse idea can be written as an intersection:

```text
for q in [0, Q):
  for (coord, (f_val, i_val)) in f.project(+q) & i:
    o[q] += i_val * f_val
```

The projection shifts filter coordinates so they align with input coordinates. The intersection emits only jointly nonzero pairs. This is elegant mathematically, but it can become discordant if the storage order of the participating fibers does not match the loop order.

ISOSceles splits the computation:

- **IS pass:** traverse input-oriented sparse streams and create an intermediate tensor `T`.
- **OS pass:** traverse `T` in output order and accumulate final outputs.

The key transformation is **swizzling**, a rank reordering of `T` so that the second pass can consume the intermediate tensor concordantly. In teaching terms, ISOSceles pays for a small intermediate buffer to avoid a more chaotic scatter pattern.

Lecture 10 slide `L06-66` reports up to 7.5x speedup and about 1.7x average speedup for ISOSceles. This is a slide-derived quantitative claim attributed in the slides to Yang et al., HPCA 2023. Because the ISOSceles paper PDF is not in the provided local paper list for this worker, this chapter treats the detailed ISOSceles explanation as slide-derived plus teaching interpretation rather than independently paper-verified.

---

## Worked Examples

### Example 1: Gating Versus Skipping

Assume a PE receives eight activation-weight pairs. Four activations are zero.

- With **gating**, the PE still consumes eight cycle slots. Four multiplier operations are disabled, so PE dynamic energy drops, but latency remains eight slots.
- With **skipping**, the PE schedules only four nonzero products. Latency can drop toward four slots, but only if the compressed stream, coordinate generator, and accumulator can keep up.

Hardware meaning: gating is easy because the dense schedule remains intact. Skipping is more powerful because it changes the schedule, but it needs metadata and load balancing.

### Example 2: Sparse Sliding Window

Let $Q = 4$, $S = 3$, sparse input coordinates be $\{0, 2, 4, 5\}$, and dense weights be $f[0], f[1], f[2]$. For output $q = 2$, the window is $2 \le w < 5$, so the active sparse inputs are $w = 2$ and $w = 4$.

The weight coordinates are:

- For $w = 2$, $s = w-q = 0$.
- For $w = 4$, $s = w-q = 2$.

Thus output $o[2]$ receives only two products: $i[2]f[0]$ and $i[4]f[2]$. A dense loop would also test $i[3]f[1]$, but sparse traversal avoids it because `i[3]` is zero or absent.

### Example 3: Joint Density Is a Product, Not a Sum

If activation density is $d_i = 0.3$ and weight density is $d_f = 0.2$, then the expected fraction of nonzero products under an independent-density model is $d_i d_f = 0.06$.

Teaching interpretation: this is why joint sparsity can look so attractive. However, the accelerator still pays fixed and semi-fixed costs: metadata reads, coordinate arithmetic, scatter/intersection logic, underutilized lanes, and synchronization. A sparse accelerator is excellent only when the skipped work dominates those overheads.

---

## Key Equations and How to Read Them

The dense 1-D convolution used throughout the lecture is:

$$
o[q] = \sum_s i[q+s] f[s].
$$

Read it as: for one output coordinate $q$, slide the filter coordinate $s$ over the input coordinate $w=q+s$.

The output coordinate for input-stationary traversal is:

$$
q = w - s.
$$

Read it as: once a nonzero input at coordinate $w$ and a nonzero filter entry at coordinate $s$ meet, the output coordinate is not an independent loop index; it must be computed and used for accumulation.

The simple independent-density model is:

$$
d_{\text{joint}} = d_i d_f.
$$

Read it as: if activations and weights are independently nonzero with densities $d_i$ and $d_f$, only the product of the densities is expected to produce nonzero multiplication work. The equation does not include metadata, routing, or load-balance overhead.

---

## Hardware Implications

- **Metadata bandwidth:** Sparse formats replace some payload reads with coordinate reads. If coordinates are wide or irregular, metadata can become a first-order cost.
- **Coordinate generation:** Sparse convolution requires arithmetic such as $w=q+s$ or $q=w-s$. This logic is modest compared with MAC arrays, but it must run at the PE rate.
- **Accumulator design:** SCNN-style scatter needs banked accumulators and conflict management. ISOSceles-style decomposition needs an intermediate buffer and rank-swizzled access.
- **Load balance:** Equal nonzero counts do not always imply equal execution time. Products may be illegal, scatter conflicts may differ, and layers have different densities.
- **Memory hierarchy:** Sparse weights, sparse activations, and sparse outputs stress different memories. The best dataflow depends on which buffer is small, which tensor is reused, and which tensor can be streamed.
- **Programmability:** Dataflow-specific sparse hardware is efficient but less general. A framework such as TeAAL, introduced earlier in the course, is useful because it separates format, mapping, architecture, and binding decisions.

---

## Common Misconceptions

### Misconception: Sparsity automatically gives speedup.

Sparsity gives speedup only if the machine skips work rather than merely gating it, and only if metadata and routing overhead do not dominate.

### Misconception: Joint sparsity is just sparse weights plus sparse activations.

The joint case introduces coordinate matching. With both operands compressed, the accelerator must either intersect coordinate streams or produce products and scatter them to computed output locations.

### Misconception: Cartesian-product multiplication is wasteful because it creates many products.

It can be wasteful if many products are illegal or collide in the accumulator, but it is also regular and parallel. SCNN uses this regularity to exploit two compressed streams at once.

### Misconception: A peak sparse speedup number describes the whole network.

Layer densities vary. Early layers often have different activation density from later layers, and some layers may be too dense or too small to amortize sparse overhead.

---

## Connections to Previous and Later Lectures

- **L05 Mapping/Dataflows:** L10 is a direct application of mapping. Output-stationary, weight-stationary, input-stationary, and IS-OS choices determine which sparse operations are cheap.
- **L07-L09 Sparse Formats:** The ideas of fibers, concordant traversal, discordant traversal, projection, and intersection become concrete hardware mechanisms here.
- **L11 Advanced Technologies:** Compute-in-memory changes the cost of moving weights and partial sums, but it does not remove the metadata and utilization questions raised by sparse dataflows.
- **L12 Reduced Precision:** Quantization can create additional zeros and can also reduce metadata/payload width. Precision and sparsity interact in both accuracy and hardware cost.
- **Later accelerator modeling:** The sparse cases in this lecture are exactly the kind of design points that require source discipline in cost models: arithmetic count, memory traffic, metadata traffic, and utilization must be separated.

---

## Paper Bridge: SCNN

### Bibliographic Identity

- **Title:** SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks
- **Authors:** Angshuman Parashar et al.
- **Year / venue:** ISCA 2017
- **Local PDF:** [`papers/L17_SCNN_Parashar_ISCA2017.pdf`](../../papers/L17_SCNN_Parashar_ISCA2017.pdf)
- **Used in lecture:** Lecture 10 slides `L06-45` through `L06-51`

### Problem Addressed

SCNN addresses the gap between sparse models and dense accelerator schedules. Earlier designs could save energy by gating zeros or exploit only one sparse operand. SCNN asks how an accelerator can keep both weights and activations compressed, skip zero-valued products, and still accumulate correct convolution outputs.

### Core Idea

The paper's core abstraction is the **PlanarTiled-InputStationary-CartesianProduct-sparse** dataflow. Each PE fetches a vector of nonzero activations and a vector of nonzero weights, forms their Cartesian product, computes output coordinates from sparse indices, and routes products through a scatter accumulator.

### Relevance to This Lecture

SCNN is the concrete architecture behind Lecture 10's transition from single-sparse skipping to joint-sparse skipping. It clarifies why joint sparsity needs more than compressed storage: the PE must compute coordinates and route partial sums.

### Key Claims Used in This Chapter

- SCNN stores both sparse weights and sparse activations in compressed form and exploits both forms of sparsity; this is stated in the paper abstract and developed in Sections III-IV.
- The PE forms Cartesian products of compressed activation and weight vectors, then computes output coordinates and scatters products; see SCNN Section III-B and Section IV, especially the descriptions around Figures 5 and 6.
- The evaluated design uses 64 PEs with 16 multipliers per PE, for 1024 multipliers total; see SCNN Table IV and Section IV.
- The paper reports 2.7x performance improvement and 2.3x energy reduction over a dense accelerator; see the abstract and Section VI.
- The paper notes that accumulator and activation memories are a large part of PE area; see the area discussion around Table III.

### What Students Should Remember

1. SCNN's multiplication is regular; its accumulation is irregular.
2. Keeping both operands compressed saves payload movement but introduces coordinate and scatter overhead.
3. Sparse speedup depends on density and on how well the PE array avoids bank conflicts and load imbalance.
4. SCNN is a design point, not a universal sparse recipe. It chooses scatter hardware as the price of joint skipping.

### Limitations and Assumptions

SCNN's published results depend on evaluated CNNs, pruning/activation sparsity patterns, and a specific area/performance model. The paper's speedup should not be generalized to all networks or all sparsity structures without checking density, layer shape, and accumulator behavior.

### Suggested Insertion Points

Reference SCNN when explaining joint-sparse Cartesian products, scatter accumulators, and the difference between arithmetic reduction and end-to-end accelerator speedup.

---

## Paper Bridge: State of Pruning

### Bibliographic Identity

- **Title:** What is the State of Neural Network Pruning?
- **Authors:** Davis Blalock et al.
- **Year / venue:** MLSys 2020
- **Local PDF:** [`papers/L16_StateOfPruning_Blalock_MLSys2020.pdf`](../../papers/L16_StateOfPruning_Blalock_MLSys2020.pdf)
- **Used in lecture:** Background bridge for where sparse weights come from

### Problem Addressed

The paper surveys the pruning literature and asks whether reported pruning results are comparable and what consistent conclusions can be drawn. This matters for Lecture 10 because sparse accelerators often assume pruned weights as input.

### Core Idea

Pruning is not a single technique. The paper distinguishes unstructured pruning, structured pruning, scoring rules, local versus global pruning, fine-tuning schedules, compression ratios, and theoretical speedups.

### Relevance to This Lecture

Lecture 10 treats sparse weights as available. The pruning survey reminds the reader that weight sparsity is produced by an algorithmic pipeline and that reported compression or theoretical speedup may not translate into hardware speedup.

### Key Claims Used in This Chapter

- The paper defines pruning as producing a masked or reduced model $f(x; M \odot W')$; see Section 2.
- It distinguishes unstructured pruning from structured pruning and explains why unstructured pruning may not map cleanly to speedups on modern libraries and hardware; see Section 2.
- It argues that pruning papers often use inconsistent metrics and baselines; see Sections 3-5.
- It recommends reporting compression, theoretical speedup, controls, and tradeoff curves; see Section 6.

### What Students Should Remember

1. Sparse weights are not free; they come from an accuracy-efficiency tradeoff.
2. Unstructured sparsity is attractive for compression but harder for hardware.
3. Theoretical FLOP reduction is not the same as accelerator speedup.
4. A sparse architecture paper should state the sparsity pattern, density, and evaluation baseline clearly.

### Limitations and Assumptions

This survey is about pruning methodology, not accelerator microarchitecture. It supports the source of sparse weights but does not validate any specific sparse accelerator.

### Suggested Insertion Points

Use this paper when discussing why sparse weight density varies across layers and why hardware speedup claims need careful evaluation.

---

## Standalone Study Guide

### What to Master Before Moving On

- Explain the difference between a zero detected late and a zero skipped before scheduling.
- Derive sparse-weight-only and sparse-input-only loops from $o[q] = \sum_s i[q+s]f[s]$.
- Explain why joint sparsity requires coordinate matching.
- Compare SCNN's scatter-based design with ISOSceles's intermediate-tensor design.
- Read sparse speedup results as density plus overhead, not density alone.

### Self-Check Questions

1. Why does Eyeriss gating save PE power but not latency?
2. In sparse-weight-only traversal, why is the dense input easy to index?
3. In sparse-input-only traversal, what does the sparse sliding window restrict?
4. Why does SCNN need a scatter accumulator?
5. What does rank swizzling accomplish in the IS-OS dataflow?
6. Why can pruning reduce theoretical MACs but fail to produce proportional hardware speedup?

### Exercises

1. For input coordinates $\{1, 4, 5\}$ and filter coordinates $\{0, 2, 3\}$, list all legal products for outputs $q \in [0, 4)$.
2. Rewrite the dense 1-D convolution loop into sparse-weight-only, sparse-input-only, and joint-sparse input-stationary loops.
3. Assume activation density $0.4$ and weight density $0.25$. Compute the independent-density estimate of nonzero product density. Then list two hardware costs ignored by this estimate.
4. Design a small accumulator banking scheme for four simultaneous SCNN products. What happens if two products target the same bank?
5. Paper-reading bridge: read SCNN Section III-B and explain why the Cartesian product is paired with coordinate computation.

---

## Key Terms

| Term | Meaning |
|---|---|
| **Gating** | Disabling hardware activity when an operand is zero. It saves dynamic energy but keeps the dense schedule. |
| **Skipping** | Removing zero-valued operations from the dynamic schedule. It can save time and energy but needs compressed traversal. |
| **Fiber** | A one-dimensional tensor slice, often represented by coordinate-payload pairs. |
| **Concordant traversal** | Reading a compressed fiber in its natural coordinate order. |
| **Discordant traversal** | Requesting data in an order that does not match the stored sparse order. |
| **Fiber projection** | Shifting coordinates so two fibers can be aligned for intersection. |
| **Fiber intersection** | Emitting only coordinates present in both sparse fibers. |
| **Output-stationary** | A dataflow that keeps output accumulators local while inputs and weights stream through. |
| **Weight-stationary** | A dataflow that keeps weights local to maximize weight reuse. |
| **Input-stationary** | A dataflow that keeps input activations local while weights produce contributions to multiple outputs. |
| **Cartesian product** | All-to-all multiplication between a group of nonzero activations and a group of nonzero weights. |
| **Scatter accumulator** | Accumulation hardware that routes products to output coordinates computed at runtime. |
| **IS-OS dataflow** | A two-pass sparse dataflow: input-stationary production of an intermediate tensor, then output-stationary reduction. |
| **Swizzling** | Reordering tensor ranks so a later traversal becomes concordant. |
| **Joint density** | The expected nonzero product density, often approximated as activation density times weight density under independence. |

---

## Takeaways

- Sparse acceleration is a scheduling and communication problem, not just a compression problem.
- Gating is useful but does not reduce latency; skipping changes the executed schedule.
- Single-sparse cases are easier because the other operand can remain dense and directly indexed.
- Joint sparsity introduces coordinate matching, legality checks, and irregular accumulation.
- SCNN pays for scatter hardware to exploit both compressed operands directly.
- ISOSceles pays for an intermediate tensor and swizzled traversal to make the second pass more structured.
- Quantitative sparse claims must state their source and scope: layer density, architecture baseline, metadata cost, and measured versus theoretical speedup.

---

## Connections

This lecture closes the sparse-architecture arc that began in L07. It also sets up L11 by sharpening the question of where data should move. Advanced memory technologies can reduce some movement costs, but they do not make sparse coordinate management disappear. The same separation of concerns from TeAAL remains useful: format determines what is stored, mapping determines the traversal, architecture determines the available hardware, and binding determines which hardware resource performs each operation.

---

## Appendix — Slide-to-Section Map

| Slide label | Chapter section | Notes |
|---|---|---|
| `L06-1` | Title and framing | PDF label differs from repository lecture number. |
| `L06-2`-`L06-7` | Gating vs. skipping | Includes Eyeriss 45% PE power slide-derived claim. |
| `L06-8`-`L06-25` | Sparse weights only | Output-stationary, weight-stationary, fiber splitting, Cambricon-X. |
| `L06-26`-`L06-38` | Sparse inputs only | Sparse sliding window and Cnvlutin. |
| `L06-39`-`L06-51` | SCNN and joint sparsity | Expanded with SCNN paper bridge. |
| `L06-52`-`L06-66` | IS-OS and ISOSceles | Slide-derived explanation of projection, intersection, swizzling, and reported speedup. |
| Background | State of pruning bridge | Added to explain where sparse weights come from and why speedup metrics need care. |

---

## Source Notes

- The lecture ordering and loop-nest examples follow Lecture 10 slides `L06-1` through `L06-66`.
- Eyeriss gating 45% PE power reduction is stated on Lecture 10 slide `L06-7`.
- Cambricon-X and Cnvlutin are used as slide-stated architecture examples from Lecture 10 slides `L06-23` and `L06-36` through `L06-38`; their original PDFs were not part of this worker's provided local paper list.
- SCNN details and numerical results are derived from `papers/L17_SCNN_Parashar_ISCA2017.pdf`, especially the abstract, Sections III-IV, Section VI, and Tables III-IV.
- Pruning context is derived from `papers/L16_StateOfPruning_Blalock_MLSys2020.pdf`, especially Sections 2-6.
- ISOSceles details and speedup numbers are slide-derived from Lecture 10 slides `L06-53` through `L06-66`; the original ISOSceles PDF was not available in the specified local paper inputs.
- Worked examples are original teaching examples constructed for this chapter.

## Uncertainty Notes

- The live lecture may have emphasized different implementation details for Cambricon-X, Cnvlutin, or ISOSceles than can be recovered from the slides alone.
- The independent-density equation $d_i d_f$ is a teaching model, not a guarantee. Real activation and weight sparsity can be correlated by layer, channel, and data distribution.
- Existing repository assets under `assets/L10/` may be copyright-sensitive slide captures. This chapter no longer embeds them, but this worker did not delete assets outside the owned walkthrough files.
