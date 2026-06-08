# L03 - Memory, Metrics, Einsums, and Transformers

> **Course:** 6.5930/1 - Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer and Vivienne Sze (MIT EECS)
> **Lecture date:** February 9, 2026. **Slides:** 127. **Source:** [`Lecture/L03-Memory+Metrics+Einsums+Transformers.pdf`](../../Lecture/L03-Memory+Metrics+Einsums+Transformers.pdf)
>
> This chapter reconstructs the missing lecture narration from the public slides. It is not a slide summary. Slide page references are used as source anchors; equations and examples are rewritten for self-study.

---

## TL;DR

L03 connects four ideas that will drive the rest of the course.

First, DNN hardware is often limited by **data movement**, not arithmetic. The lecture uses Horowitz's energy table: a 32-bit DRAM read is 640 pJ, while a 32-bit floating-point multiply is 3.7 pJ. That slide-stated ratio is about 170x. The architect's job is therefore not only to add MAC units, but to arrange memory hierarchy and computation order so weights, activations, and partial sums move as little as possible.

Second, accelerator evaluation needs a **set of metrics**, not a single number such as GOPS/W. Accuracy, throughput, latency, energy, power, cost, flexibility, and scalability answer different questions. A design can look excellent under one metric and fail the application under another.

Third, the course needs a notation that says **what** a DNN computation is without saying **how** it is scheduled. That notation is Einsum. An Einsum such as $Z_{m,n} = A_{m,k} B_{n,k}$ defines a tensor contraction and its iteration space, but does not prescribe loop order, tiling, data placement, or parallelization.

Fourth, Transformer self-attention is not a separate kind of magic workload. It is a cascade of tensor contractions: input projection, $QK^T$, softmax, $AV$, and output projection. The key hardware warning is that ordinary self-attention materializes an $M \times M$ attention matrix for sequence length $M$.

---

## What Problem This Lecture Solves

Earlier lectures introduced deep learning workloads and accelerator motivation. L03 supplies the common language needed to compare and map those workloads.

The problem is that "a DNN layer" is too vague for hardware design. A hardware architect needs to know:

- What tensors exist?
- Which indices are preserved in the output?
- Which indices are reduced?
- How many bytes move between memory levels?
- Which metric is being optimized?
- Whether a reported result is a real system result or a proxy such as weight count or peak TOPS?

L03 solves this by building three bridges:

| Bridge | From | To | Why it matters |
|---|---|---|---|
| Memory hierarchy | Physical memory technologies | DNN data movement cost | Explains why local reuse is valuable |
| Evaluation metrics | Model/application goals | Hardware comparison | Prevents misleading one-number claims |
| Einsum notation | Neural-network layers | Loop nests and mapping | Lets later lectures discuss dataflow precisely |

The final part of the lecture applies the same Einsum language to Transformer attention, because modern DNN accelerators must handle more than CNNs.

---

## Why This Lecture Matters

The naive mental model is: "A neural network is mostly matrix multiplication, so the best accelerator is the one with the most multiply units." L03 corrects that model.

The multiplication units matter, but they are only useful when the system can feed them. If every MAC fetches operands from DRAM, the system spends far more energy moving data than multiplying it. If a benchmark reports peak TOPS but does not report PE utilization, batch size, off-chip bandwidth, or accuracy, the result may not describe a deployable system.

The same correction appears in the notation. A formula such as $O = AB$ is not enough. Hardware design needs to know whether $A$ and $B$ are weights, activations, or intermediate tensors; whether the reduced rank can be tiled; whether an intermediate is materialized; and whether a loop order exposes reuse.

L03 is therefore the "language and measurement" lecture. L05 and L06 will ask how to map an Einsum onto hardware. L07 and later lectures will ask how sparsity, precision, and specialized architectures change the same cost model.

---

## Prerequisites and Mental Model

You should bring three ideas.

First, a DNN layer is a tensor computation. A tensor is an array with named dimensions, such as channel, height, width, batch, sequence position, or embedding dimension.

Second, a DNN accelerator has a memory hierarchy. A useful simplified picture is:

```text
DRAM -> global buffer / SRAM -> PE-local RF or registers -> MAC units
```

The farther data travels away from the MAC unit, the more energy and latency it tends to cost. This is a source-based hardware principle from the memory slides, not merely a software locality slogan.

Third, many DNN computations are reductions. A dot product, matrix multiplication, convolution, and attention score all compute products and then sum over at least one rank. Einsum gives the course a way to name those ranks cleanly.

The mental model for this lecture is:

```text
mathematical layer -> Einsum -> possible loop orders -> memory traffic -> metrics
```

The Einsum fixes the mathematical contract. Mapping, introduced later, chooses the loop order and data placement. Metrics tell us whether the resulting system is useful.

---

## Learning Objectives

After studying this chapter, you should be able to:

- Explain why large memories are slower and more energy-intensive than small nearby memories.
- Use the Horowitz table in the slides to compare arithmetic energy with SRAM and DRAM access energy.
- Define accuracy, throughput, latency, energy, power, hardware cost, flexibility, and scalability as accelerator metrics.
- Explain why weights and operation counts are proxy metrics rather than direct measurements of energy or latency.
- Read an Einsum and identify output ranks, input ranks, and reduction ranks.
- Explain the Operational Definition for Einsums as traversal of an iteration space.
- Convert a fully connected layer into a matrix-vector or matrix-matrix multiplication by flattening ranks.
- Explain how Toeplitz/im2col conversion turns convolution into a matrix multiplication and why that conversion repeats data.
- Trace the self-attention cascade through $I$, $Q$, $K$, $V$, $QK$, softmax, $A$, $AV$, and $Z$.
- Explain the hardware implications of an $M \times M$ attention matrix.
- Distinguish slide-stated facts, paper/source-derived claims, standard background, and teaching interpretation.

---

## 1. Memory Is a First-Class Design Constraint

**Source anchor:** PDF pages 2-19. The energy table is attributed in the slides to Horowitz, ISSCC 2014.

### Intuition

A memory is not just a passive container. It is a circuit. Larger memories usually have longer wires, larger capacitance, more peripheral circuitry, and more energy per access. The lecture states the physical rule as $E = C V^2$: for a fixed voltage, more capacitance means more energy.

This is why memory hierarchy exists. We put small, fast, low-energy storage close to compute, and large, dense storage farther away.

### Precise meaning

The lecture compares four storage technologies.

| Storage | Typical role in this lecture | Strength | Cost |
|---|---|---|---|
| Latches / flip-flops | Very small local state, pipeline registers | Very low latency, near logic | Low density, many transistors per bit |
| SRAM | Register files, buffers, on-chip memories | On-chip, reusable, faster than DRAM | More area than DRAM; peripheral circuits matter |
| DRAM | Main memory, often off-chip | Large capacity, low cost per bit | High access energy and latency |
| Flash | Persistent storage | Very dense, non-volatile | Writes are expensive and slow for compute use |

The important DNN point is not that every accelerator uses exactly these levels. The point is that capacity, latency, bandwidth, density, and energy pull against one another. A local buffer helps only if the computation reuses data before evicting it.

### Quantitative anchor

The slide table gives these energy values:

| Operation | Energy |
|---|---:|
| 8-bit add | 0.03 pJ |
| 32-bit add | 0.1 pJ |
| 32-bit floating-point add | 0.9 pJ |
| 8-bit multiply | 0.2 pJ |
| 32-bit multiply | 3.1 pJ |
| 32-bit floating-point multiply | 3.7 pJ |
| 32-bit SRAM read from an 8 kB SRAM | 5 pJ |
| 32-bit DRAM read | 640 pJ |

Two ratios are worth reading carefully:

- A 32-bit DRAM read costs $640 / 5 = 128$ times a 32-bit SRAM read in this table.
- A 32-bit DRAM read costs $640 / 3.7 \approx 173$ times a 32-bit floating-point multiply in this table.

These are slide-stated values from a particular technology context. They are not universal constants, but they are strong enough to justify the architectural direction: reduce large-memory traffic.

### Worked example: why one reused word matters

Suppose a weight is used by 16 MACs. If the system reads the weight from DRAM for every MAC, the weight traffic costs $16 \times 640 = 10{,}240$ pJ for that one 32-bit value in the Horowitz table.

If the system reads the weight once from DRAM and keeps it in local storage, the DRAM part costs $640$ pJ. Even if each local reuse costs an SRAM-like $5$ pJ, the 16 local reads cost $80$ pJ, for a total of $720$ pJ.

This toy example is not a full accelerator energy model. It deliberately ignores address generation, interconnect, control, and writes. Its purpose is to show the scale of the reuse opportunity.

### Hardware implication

The memory hierarchy changes the mapping problem. A loop order that reuses a weight 16 times while it is in a PE-local register file can be much cheaper than a loop order that uses the same weight 16 times after repeatedly evicting and refetching it. The arithmetic is identical; the traffic is not.

### Common misconception

**Misconception:** Data movement is expensive only because DRAM is slow.

**Correction:** Latency matters, but energy is also central. Large memories and off-chip links have high capacitance. The lecture explicitly connects the energy cost to capacitance and voltage, then uses the Horowitz table to show that memory movement can dominate arithmetic.

---

## 2. Efficient Models Are Not Automatically Efficient Hardware

**Source anchor:** PDF pages 20-39.

The lecture includes an efficient-CNN interlude before the metric section. This is not a detour. It sets up a warning: model-design papers often report number of weights and number of operations as "complexity," but those are indirect metrics.

### What the model side tries to reduce

The slides list several ways CNN designers reduce apparent complexity:

- Replace one large spatial filter with stacked smaller filters, such as replacing a $5 \times 5$ filter with two $3 \times 3$ filters.
- Use $1 \times 1$ bottleneck convolutions to reduce channel count before an expensive layer.
- Use grouped or depthwise convolutions so each filter sees only a subset of channels.
- Reuse feature maps across layers, as in DenseNet-style connectivity.
- Search architectures automatically with NAS.

The MobileNet slide gives a compact formula. A standard convolution has work proportional to $H W C R S M$. A depthwise-separable version has depthwise work $H W C R S$ plus pointwise work $H W C M$, so its work is proportional to $H W C (R S + M)$. The standard-to-depthwise-separable MAC ratio is therefore:

$$
\frac{H W C R S M}{H W C (R S + M)} = \frac{R S M}{R S + M}.
$$

This is a real algorithmic reduction, source-anchored to the MobileNets slide. But the lecture immediately warns that fewer operations do not automatically imply lower latency or energy.

### Why the hardware answer is more subtle

An operation count ignores at least five hardware facts:

- Whether the reduced operation exposes enough parallelism to fill the PE array.
- Whether the smaller tensors still have good reuse.
- Whether grouped/depthwise layers create awkward memory access patterns.
- Whether metadata, layout conversion, or kernel launch overhead appears.
- Whether the hardware was designed for dense GEMM-like work or for many small irregular kernels.

This is a teaching interpretation based on the slide warning on pages 36-37 and the metrics section on pages 40-56. The lecture's point is not that efficient CNN techniques are bad. The point is that algorithmic complexity and hardware cost must be connected through a real mapping and measurement method.

### Common misconception

**Misconception:** If a model has 50x fewer parameters, it should use 50x less energy.

**Correction:** Parameter count estimates storage for weights, but energy also depends on activations, outputs, partial sums, reuse, memory level, data layout, utilization, and batch size. The slides later use AlexNet versus SqueezeNet as a warning that proxy metrics can mislead.

---

## 3. Evaluation Metrics: What Must Be Reported

**Source anchor:** PDF pages 40-56.

### Intuition

One metric answers one question. It cannot answer every question.

For example, throughput asks "how many inputs per second?" Latency asks "how long does one input wait?" Energy asks "how much work per joule?" Accuracy asks "is the result useful?" A system that wins one of these may lose another.

The lecture makes this concrete with the ring-oscillator example: high TOPS/W can be manufactured by a circuit that toggles quickly but performs no useful DNN inference. Peak arithmetic divided by peak power is not enough.

### The metric set

| Metric | What it asks | Required context |
|---|---|---|
| Accuracy | Does the model solve the task? | Dataset, task difficulty, training/evaluation procedure |
| Throughput | How many operations or inferences per second? | Actual model, PE count, utilization, batch size |
| Latency | How long from input to output? | Batch size and end-to-end path |
| Energy and power | How much energy per inference and power while running? | Model, memory traffic, off-chip bandwidth, measurement/simulation method |
| Hardware cost | How expensive is the implementation? | Area, process node, on-chip storage, PE count, external interfaces |
| Flexibility | How many workloads run efficiently? | Range of models, shapes, precisions, sparsity, and layer types |
| Scalability | What happens when resources increase? | Scaling variable and bottleneck, such as PEs or memory bandwidth |

### Throughput, latency, and batch size

Throughput and latency are related but not identical. A batch of 64 may improve throughput because the same weights are reused across more inputs, but it can increase the waiting time for a single input. This is why the slides explicitly state that low latency has the additional constraint of small batch size.

### Energy and off-chip memory

The metrics slides emphasize off-chip memory access. If a paper reports only chip power, it may hide the energy of DRAM traffic outside the accelerator core. A design with many multipliers and insufficient local storage can look cheap on-chip while pushing cost into the memory system.

### Worked example: operational intensity

Operational intensity is commonly read as $\text{ops}/\text{byte}$. The lecture references the Roofline paper in the metrics references; this chapter uses the idea as standard background.

Suppose a tiny kernel performs 1024 MACs. If we count one MAC as one operation for this toy example, and the kernel reads 2048 bytes from DRAM, then its operational intensity is:

$$
\text{OI} = \frac{1024\ \text{ops}}{2048\ \text{bytes}} = 0.5\ \text{ops/byte}.
$$

If a different loop order reuses local data and reads only 512 bytes from DRAM for the same 1024 MACs, then:

$$
\text{OI} = \frac{1024}{512} = 2\ \text{ops/byte}.
$$

The arithmetic did not change. The memory traffic changed. A higher operational intensity usually means each fetched byte supports more computation, which is exactly what local reuse tries to accomplish.

### Source bridge: MLPerf, Accelergy, and AccelForge

The lecture points to three kinds of evaluation infrastructure.

| Source/tool | Problem addressed | Relevance to L03 |
|---|---|---|
| MLPerf | Standardized benchmarking across models and platforms | Reduces cherry-picking by using common workloads and divisions |
| Accelergy, Wu et al., ICCAD 2019 | Architecture-level energy estimation | Connects components and actions to estimated energy |
| AccelForge | DNN mapping and performance simulation | Produces action counts that can feed an energy estimator |

The chapter uses these as source bridges, not as full paper summaries. The key takeaway is that fair accelerator evaluation needs workload shape, architecture description, mapping, action counts, and energy modeling, not just peak TOPS.

### Common misconception

**Misconception:** GOPS/W is the same as energy efficiency for a useful application.

**Correction:** GOPS/W can be meaningful only when the operations are useful, the workload is specified, utilization is measured, accuracy is preserved, and memory-system energy is included. Otherwise it is easy to optimize a ratio rather than the application.

---

## 4. Einsum: The Contract for Tensor Computation

**Source anchor:** PDF pages 57-72.

### Intuition

Einsum separates two questions:

- What values must be multiplied, added, or otherwise combined?
- In what order should a machine perform those operations?

The first question is the Einsum. The second question is mapping.

For example, $Z_{m,n} = A_{m,k} B_{n,k}$ says that each output element $Z_{m,n}$ sums products over $k$. It does not say whether the loop over $m$, $n$, or $k$ should run first, nor whether $m$ and $n$ should be parallelized.

### Operational Definition for Einsums

The slides define an Einsum operationally:

1. Traverse all legal values of the rank variables. This set is the **iteration space**.
2. At each point, compute the right-hand side at those rank-variable values.
3. Assign the result into the left-hand-side tensor.
4. If the target location already has a value, reduce into it, usually by addition.

For $Z_{m,n} = A_{m,k} B_{n,k}$, the legal points are triples $(m,n,k)$. The output location is identified by $(m,n)$. Because $k$ appears only on the right-hand side, multiple iteration points contribute to the same $Z_{m,n}$, so $k$ is a reduction rank.

### Precise vocabulary

| Term | Meaning |
|---|---|
| Rank variable | The index used in the expression, such as $m$, $n$, or $k$ |
| Rank name | The dimension label, such as $M$, $N$, or $K$ |
| Rank shape | The extent of the dimension, such as $M=64$ |
| Uncontracted rank | A rank appearing on both sides; it remains in the output |
| Contracted rank | A rank appearing on the right but not the left; it is reduced |
| Iteration space | The Cartesian product of all rank-variable ranges |

### Common patterns

| Einsum | Reading | Reduced rank |
|---|---|---|
| $Z_{m,n} = A_{m,k} B_{k,n}$ | Matrix-matrix multiply | $k$ |
| $Z_m = A_{k,m} B_k$ | Matrix-vector multiply | $k$ |
| $Z_{m,n} = A_m B_n$ | Cartesian product | none |
| $Z_m = A_m B_m$ | Element-wise multiply | none |
| $Z_m = A_m + B_m$ | Element-wise addition | none |

The names of the variables are not special. $Z_{p,q} = A_{p,r} B_{q,r}$ is still a matrix-matrix-style contraction because $r$ is reduced and $p,q$ remain.

### Partitioning and flattening

The slides introduce a rank split: if $i = i_1 I_0 + i_0$, then one original rank $i$ can be represented by two ranks $(i_1,i_0)$. This is partitioning.

Flattening is the inverse. A pair $(i_1,i_0)$ can be treated as one flattened coordinate $i$. This matters because later lectures describe tiling as rank partitioning, not as an ad hoc code trick.

### Worked example: reading an Einsum

Consider:

$$
Y_{b,m} = X_{b,k} W_{m,k}.
$$

The rank variables are $b$, $m$, and $k$. The output ranks are $b$ and $m$. The rank $k$ appears only on the right-hand side, so it is reduced. If $B=2$, $M=3$, and $K=4$, then the iteration space has $2 \times 3 \times 4 = 24$ points, and the output has $2 \times 3 = 6$ elements. Each output element accumulates 4 products.

### Hardware implication

Einsum gives the hardware architect a workload-independent way to talk about data reuse. If $k$ is reduced, partial sums for the left-hand-side tensor must be accumulated. If $m$ or $n$ is uncontracted, those ranks may expose parallel output elements. If a rank is partitioned, the tile size can be chosen to fit a buffer.

### Common misconception

**Misconception:** Einsum is just a compact way to write matrix multiplication.

**Correction:** Matrix multiplication is one Einsum pattern. The point of the notation is broader: it describes tensor contractions, element-wise operations, Cartesian products, convolution index relationships, and attention projections without committing to a loop order.

---

## 5. Fully Connected and Convolution Layers as Matrix Multiplication

**Source anchor:** PDF pages 73-109.

### Fully connected layer

A fully connected layer can be seen as a convolution whose filter covers the whole input spatial extent. The slide equation is:

$$
O_m = I_{c,h,w} F_{m,c,h,w}.
$$

The repeated ranks $c,h,w$ are reduced. To turn this into a matrix-vector multiply, flatten $(c,h,w)$ into one rank $chw$:

$$
O_m = I_{chw} F_{m,chw}.
$$

With batch size $N$, the input gets a batch rank $n$:

$$
O_{n,m} = I_{n,chw} F_{m,chw}.
$$

This is matrix-matrix multiplication in Einsum form. The output has ranks $(n,m)$, and $chw$ is the reduction rank.

### Why flattening is not just notation

Flattening changes how we view the memory layout. If the tensor is stored so that consecutive $chw$ elements are contiguous, the matrix-vector view can be efficient. If the layout is mismatched, the hardware may need strided accesses or layout conversion. The mathematical contraction is the same, but the memory behavior can differ.

### Convolution and Toeplitz/im2col

For a 1-D convolution:

$$
O_q = I_{q+s} F_s.
$$

The input index is not simply $q$ or $s$; it is $q+s$. That coupling is what makes convolution different from ordinary matrix multiplication.

The slides break the conversion into two steps:

$$
T_{q,s} = I_{q+s}
$$

and then:

$$
O_q = T_{q,s} F_s.
$$

The first step creates a Toeplitz/im2col matrix of shifted input windows. The second step is a matrix multiplication.

For 2-D convolution with batch, the slide-stated matrix dimensions are:

$$
\text{Filters }[M \times C R S] \times \text{Input-Toeplitz }[C R S \times P Q N] = \text{Output }[M \times P Q N],
$$

where $P = H - R + 1$ and $Q = W - S + 1$ for the no-padding, unit-stride case shown in the slides.

### Worked example: tiny 1-D Toeplitz conversion

Let the input be $I=[1,2,3,4,5]$ and the filter be $F=[a,b,c]$. A valid 1-D convolution has three output positions:

$$
O_0 = 1a + 2b + 3c,
$$

$$
O_1 = 2a + 3b + 4c,
$$

$$
O_2 = 3a + 4b + 5c.
$$

The Toeplitz matrix is:

```text
T = [1 2 3
     2 3 4
     3 4 5]
```

Then $O = T F$. The important hardware observation is that the input values are repeated in $T$. The value 3 appears in three windows. Materializing $T$ may make the operation look like GEMM, but it can increase memory traffic unless the implementation avoids physically writing all repeated entries.

### Hardware implication

The conversion explains why GEMM engines can run FC and CONV layers. It also explains a tradeoff: im2col can produce regular matrix multiplication, but the regularity may come from duplicating data. Later mapping lectures ask whether the accelerator can exploit convolutional reuse without fully materializing the Toeplitz matrix.

### Common misconception

**Misconception:** Since convolution can be converted to matrix multiplication, convolution hardware only needs a generic GEMM block.

**Correction:** GEMM is a powerful abstraction, but the conversion may duplicate input data and change memory traffic. A convolution-aware mapping can exploit overlap directly, while a naive im2col implementation may spend energy moving repeated data.

---

## 6. Transformer Self-Attention as Einsums

**Source anchor:** PDF pages 110-127.

### Mental model

Self-attention takes a sequence of $M$ tokens. Each token starts as an embedding vector of dimension $D$. The block computes three projections:

- Query $Q$: what this position is asking for.
- Key $K$: what each position offers for matching.
- Value $V$: what information each position contributes if attended to.

The attention score between a query position and a key position comes from a dot product. The softmax turns scores into weights. The value vectors are mixed using those weights. Finally, an output projection returns the result to the model's embedding space.

### Rank dictionary

| Rank | Meaning |
|---|---|
| $M$ | Sequence length for keys, values, and self-attention positions |
| $P$ | Alias of sequence length used for query/output position in the slides |
| $R$ | Query sequence length in non-self-attention |
| $C$ | Vocabulary size |
| $D$ | Input/global embedding dimension, often $d_{\text{model}}$ |
| $E$ | Query/key projection dimension, often $d_k$ |
| $F$ | Value projection dimension, often $d_v$ |
| $G$ | Output embedding dimension |
| $B$ | Batch size |
| $H$ | Number of attention heads |

The slides use $M$ and $P$ as aliases for sequence length so the two axes of the attention matrix can be named separately.

### Single-head attention cascade

For the first layer only, raw one-hot input $IR$ is embedded:

$$
I_{m,d} = IR_{m,c} W^I_{c,d}.
$$

Then the input is projected to key, query, and value spaces:

$$
K_{m,e} = I_{m,d} W^K_{d,e},
$$

$$
Q_{m,e} = I_{m,d} W^Q_{d,e},
$$

$$
V_{m,f} = I_{m,d} W^V_{d,f}.
$$

The pre-softmax score matrix is:

$$
QK_{m,p} = Q_{p,e} K_{m,e}.
$$

Here $e$ is reduced. For each query position $p$, the score compares that query with every key position $m$. The slides note that some constant scaling steps are not illustrated, so this chapter does not treat scaling as a source-stated equation.

The softmax components in the slide convention are:

$$
SN_{m,p} = \exp(QK_{m,p}),
$$

$$
SD_p = \sum_m SN_{m,p},
$$

$$
A_{m,p} = SN_{m,p} / SD_p.
$$

Then values are mixed:

$$
AV_{p,f} = A_{m,p} V_{m,f}.
$$

Finally the output projection is:

$$
Z_{p,g} = AV_{p,f} W^Z_{f,g}.
$$

### Worked example: attention shapes

Suppose $M=4$, $D=8$, $E=2$, and $F=3$ for a single head and no batch.

| Tensor | Shape | Why |
|---|---|---|
| $I$ | $4 \times 8$ | Four tokens, eight-dimensional embeddings |
| $W^Q$ and $W^K$ | $8 \times 2$ | Project from $D$ to $E$ |
| $Q$ and $K$ | $4 \times 2$ | One 2-D query/key vector per token |
| $QK$ | $4 \times 4$ | Every query position compares with every key position |
| $W^V$ | $8 \times 3$ | Project from $D$ to $F$ |
| $V$ | $4 \times 3$ | One value vector per token |
| $AV$ | $4 \times 3$ | One mixed value vector per query position |

The quadratic tensor is $QK$ or $A$: it has $M^2 = 16$ entries here. If $M$ doubles, this intermediate grows by about 4x, before considering batch or heads.

### Batched and multi-headed attention

Batching adds a rank $b$:

$$
QK_{b,m,p} = Q_{b,p,e} K_{b,m,e}.
$$

Multi-headed attention adds a head rank $h$:

$$
QK_{b,h,m,p} = Q_{b,h,p,e} K_{b,h,m,e}.
$$

Each head produces $AV_{b,h,p,f}$. The head and value dimensions are then concatenated or flattened before the output projection. In the slide convention:

$$
C_{b,p,hF+f} = AV_{b,h,p,f},
$$

followed by an output projection such as:

$$
Z_{b,p,d} = C_{b,p,g} W^Z_{g,d}.
$$

### Hardware implication

Attention has two different cost profiles in one block. The projection layers are GEMM-like and scale roughly with $M D E$, $M D F$, and $M F G$. The attention score and value-mixing steps create and consume tensors shaped by $M^2$. This means sequence length can dominate memory capacity and bandwidth even if embedding dimensions are moderate.

The hardware question is therefore not just "can the accelerator multiply matrices?" It is also "does it materialize the $M \times M$ attention matrix, where is it stored, and can softmax be fused with the surrounding operations?"

### Common misconception

**Misconception:** Transformer attention is fundamentally different from the tensor computations used for CNNs.

**Correction:** The data dependencies differ, but the lecture expresses attention as the same kind of Einsum cascade used for FC and CONV. The new challenge is the shape and lifetime of intermediates, especially the $M \times M$ attention matrix and softmax normalization.

---

## 7. Key Equations and How to Read Them

### Energy ratio

$$
\frac{E_{\text{DRAM read, 32b}}}{E_{\text{FP multiply, 32b}}} = \frac{640}{3.7} \approx 173.
$$

Read this as a source-stated motivation for minimizing DRAM access. It is not a universal technology constant.

### Operational intensity

$$
\text{Operational intensity} = \frac{\text{operations}}{\text{bytes moved from the measured memory level}}.
$$

The denominator must be specified. DRAM operational intensity and SRAM operational intensity are different measurements.

### Matrix multiplication Einsum

$$
Z_{m,n} = A_{m,k} B_{k,n}.
$$

Ranks $m$ and $n$ remain in the output. Rank $k$ is reduced. A mapping may run the loops in many legal orders.

### Fully connected layer

$$
O_m = I_{c,h,w} F_{m,c,h,w}
$$

becomes:

$$
O_m = I_{chw} F_{m,chw}.
$$

Flattening turns the input dimensions into one reduction rank.

### Convolution

$$
O_{n,m,p,q} = I_{n,c,U p + r,U q + s} F_{m,c,r,s}.
$$

The ranks $c,r,s$ are reduced. The output ranks $n,m,p,q$ remain. The input spatial coordinates are derived from output location and filter offset.

### Attention score

$$
QK_{m,p} = Q_{p,e} K_{m,e}.
$$

The rank $e$ is reduced. The result compares every query position $p$ with every key position $m$.

---

## 8. Hardware Implications

- **Energy:** DRAM traffic can dominate arithmetic energy; local reuse is a first-order design goal.
- **Bandwidth:** A PE array cannot reach peak throughput unless the memory system supplies operands fast enough.
- **Latency:** Batching may improve throughput while hurting latency; low-latency applications require small-batch evaluation.
- **Area:** More local SRAM/RF can reduce traffic, but it consumes area and may increase access energy if arrays become too large.
- **Utilization:** Model shapes such as depthwise convolution may reduce MACs but also reduce dense parallel work available to a fixed array.
- **Interconnect:** Tensor contractions create producer-consumer relationships; where intermediates are stored determines network traffic.
- **Correctness:** Einsum reductions require correct accumulation. Changing loop order is legal only if the reduction semantics are preserved.
- **Programmability:** Flexible hardware must handle CNNs, FC layers, attention, varying precision, sparsity, and layer shapes.

---

## 9. Common Misconceptions

### Misconception: Data movement is a secondary detail after compute.

Compute and movement are inseparable in accelerator design. The lecture's energy table shows that a DRAM read can cost orders of magnitude more than a multiply.

### Misconception: Fewer MACs always means lower latency.

Fewer MACs help only if the hardware bottleneck is arithmetic and the remaining work maps well to the hardware. If the bottleneck is memory bandwidth, layout conversion, synchronization, or poor utilization, latency may not fall proportionally.

### Misconception: Accuracy is an algorithm metric, not a hardware metric.

Hardware choices can constrain precision, sparsity support, model size, and memory capacity. If those choices change the model or numerical behavior, accuracy must be part of the evaluation.

### Misconception: Einsum tells the accelerator how to run the loops.

Einsum tells what must be computed. Mapping chooses loop order, tiling, storage placement, and parallelism.

### Misconception: im2col is always efficient because it turns convolution into GEMM.

im2col exposes regular GEMM structure, but it may duplicate input data. Efficient implementations often avoid materializing the full expanded matrix.

### Misconception: Attention only needs matrix multiplication acceleration.

Matrix multiplication is central, but the $M \times M$ score/attention matrix, softmax, memory capacity, and fusion opportunities are equally important.

---

## 10. Takeaways

- The lecture's Horowitz table gives a 32-bit DRAM read as 640 pJ and a 32-bit floating-point multiply as 3.7 pJ, motivating data-movement-aware design.
- A memory hierarchy exists because no memory is simultaneously large, fast, dense, cheap, and low-energy.
- Efficient model techniques reduce proxy costs such as weights and OPs, but hardware efficiency depends on mapping, locality, utilization, and memory traffic.
- Fair accelerator evaluation needs accuracy, throughput, latency, energy, power, cost, flexibility, and scalability, with enough context to reproduce the meaning of each metric.
- Einsum is the course's mathematical contract for tensor computations. It fixes output, input, and reduction ranks without fixing loop order.
- FC layers become matrix-vector or matrix-matrix multiplication by flattening ranks and adding batch.
- Convolution becomes matrix multiplication through Toeplitz/im2col conversion, but that conversion can repeat data.
- Transformer self-attention is an Einsum cascade with an $M \times M$ attention intermediate.

---

## 11. Connections to Previous and Later Lectures

- **L01:** The accelerator motivation introduced in L01 becomes quantitative here through memory energy and evaluation metrics.
- **L02:** CNN layer shapes and efficient model designs reappear in L03 as examples where proxy metrics can mislead.
- **L04:** The Einsum formalism becomes the foundation for describing more DNN operations and tensor transformations.
- **L05-L06:** Mapping takes the Einsum as input and chooses loop order, partitioning, data placement, and parallelism.
- **Sparsity lectures:** Sparsity changes the tensors inside the same Einsum framework, but adds metadata, irregularity, and load-balancing issues.
- **Precision lectures:** Bit width changes arithmetic energy and storage traffic, but must be evaluated with accuracy and system metrics.
- **Attention accelerators:** The attention equations here explain why later systems try to fuse or avoid materializing $QK$ and $A$.

---

## 12. Source Bridge

### Paper Bridge: Computing's Energy Problem

**Bibliographic identity:** Mark Horowitz, *Computing's Energy Problem (and what we can do about it)*, ISSCC 2014. Local PDF: `papers/L07_ComputingsEnergyProblem_Horowitz_ISSCC2014.pdf`.

**Problem addressed:** Technology scaling no longer gives architects enough automatic energy improvement. The paper asks how computing systems can keep improving when power and energy are first-order constraints.

**Core idea:** Energy efficiency requires both low-energy operations and **extreme locality**. The paper emphasizes that memory-system energy can dwarf efficient computation and that specialization can be useful when it reduces movement and overhead.

**Relevance to L03:** This is the paper-level support behind the lecture's memory-vs-compute argument. It explains why L03 treats memory hierarchy, action counts, and operational intensity as architectural concepts rather than bookkeeping details.

**Key claims used in this chapter:**

- DRAM accesses are orders of magnitude more expensive than internal cache accesses or simple functional operations. Source anchor: Section 5, "Don't Forget the Memory Energy."
- Figure 1.1.9 gives rough energy costs for operations and memory accesses; the lecture's energy table is the slide-level presentation of this idea.
- High energy efficiency requires data locality so one expensive memory access can support many operations. Source anchor: Sections 5-6.

**What students should remember:** The memory hierarchy is not a secondary detail. It is the reason DNN accelerators care so much about reuse, tiling, and dataflow.

**Limitations and assumptions:** The numeric energy values are technology-specific. Use them as order-of-magnitude motivation, not universal constants.

### Paper Bridge: AlexNet

**Bibliographic identity:** Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton, *ImageNet Classification with Deep Convolutional Neural Networks*, NeurIPS 2012. Local PDF: `papers/L03_AlexNet_Krizhevsky_NeurIPS2012.pdf`.

**Problem addressed:** Train a large supervised CNN on ImageNet-scale data, where both model capacity and computation exceed what earlier small-scale CNN examples required.

**Core idea:** Combine large convolutional/fully connected layers, ReLU activations, GPU implementation, data augmentation, and dropout to make a deep CNN trainable at ImageNet scale.

**Relevance to L03:** AlexNet is a useful bridge from "DNNs are tensor programs" to "DNNs are hardware workloads." It shows why convolution, fully connected layers, memory capacity, GPU parallelism, and regularization all matter in the same design story.

**Key claims used in this chapter:**

- The model has eight learned layers and about 60 million parameters. Source anchor: Abstract and Section 3.5.
- The paper explicitly discusses GPU memory limits and training across two GTX 580 GPUs. Source anchor: Introduction and Section 3.2.
- ReLU nonlinearity is used to speed training compared with saturating nonlinearities. Source anchor: Section 3.1.
- Dropout is applied to reduce overfitting in fully connected layers. Source anchor: Section 4.

**What students should remember:** AlexNet is not just "a CNN." It is an early example where dataset scale, model size, GPU memory, and implementation choices all shaped the architecture.

**Limitations and assumptions:** AlexNet's exact architecture is historically important, but later models changed the accuracy/efficiency tradeoff. L03 uses it as workload history and hardware motivation, not as a recommended modern design.

### Paper Bridge: Deep Residual Learning

**Bibliographic identity:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016. Local PDF: `papers/L03_ResNet_He_CVPR2016.pdf`.

**Problem addressed:** Simply making plain networks deeper can make optimization worse; the paper calls this the **degradation problem**, which is not just overfitting.

**Core idea:** Learn a residual function $F(x) = H(x) - x$ and add the input back through a shortcut, so the block computes $F(x) + x$. Identity shortcuts let very deep networks learn perturbations around the identity mapping.

**Relevance to L03:** ResNet supports the chapter's warning that model complexity cannot be reduced to layer count or MAC count. The graph structure introduces elementwise additions and skip paths that affect memory traffic, buffering, and fusion.

**Key claims used in this chapter:**

- The degradation problem is introduced in the paper's Introduction. Source anchor: Section 1.
- The residual formulation $F(x)+x$ and identity shortcut are defined in the residual-learning section. Source anchor: Section 3.
- The paper reports 18/34-layer comparisons and deeper bottleneck ResNets. Source anchor: Tables 1-4.

**What students should remember:** Skip connections are mathematically simple but architecturally visible: tensors must be preserved or reloaded for the later addition.

**Limitations and assumptions:** L03 does not use ResNet's accuracy numbers as current benchmark claims. It uses the paper to explain why modern layer graphs contain non-chain data dependencies.

### Paper Bridge: MobileNets

**Bibliographic identity:** Andrew Howard et al., *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*, 2017. Local PDF: `papers/L03_MobileNet_Howard_2017.pdf`.

**Problem addressed:** Mobile and embedded systems need CNNs with lower computation and model size while preserving useful accuracy.

**Core idea:** Replace standard convolution with **depthwise separable convolution**: a depthwise spatial filter per input channel followed by a $1 \times 1$ pointwise convolution that mixes channels. The paper also introduces width and resolution multipliers.

**Relevance to L03:** MobileNet is the paper behind the chapter's efficient-CNN warning: reducing MACs is useful, but hardware still has to evaluate where the work moved. In MobileNet, much of the computation shifts to dense $1 \times 1$ convolution, which has different reuse and bandwidth behavior.

**Key claims used in this chapter:**

- Standard convolution cost is $D_K^2 M N D_F^2$ in the paper's notation. Source anchor: Section 3.1.
- Depthwise separable convolution splits filtering and channel mixing, reducing computation and model size. Source anchor: Section 3.1.
- For $3 \times 3$ filters, the paper states an 8-9x computation reduction with modest accuracy loss. Source anchor: Section 3.1.
- Table 2 notes that MobileNet spends most computation in $1 \times 1$ convolution. Source anchor: Table 2.

**What students should remember:** A lower-MAC model can move the bottleneck. Depthwise layers may be cheap arithmetically, while pointwise layers can dominate compute and memory traffic.

**Limitations and assumptions:** The paper's accuracy/latency tradeoffs are tied to its training setup and hardware context. This chapter uses the operator decomposition, not the exact benchmark ranking, as the durable concept.

### Paper Bridge: EfficientNet

**Bibliographic identity:** Mingxing Tan and Quoc Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019. Local PDF: `papers/L03_EfficientNet_Tan_ICML2019.pdf`.

**Problem addressed:** Scaling a CNN by only depth, only width, or only input resolution gives diminishing returns and requires manual tuning.

**Core idea:** Compound scaling jointly scales depth, width, and resolution using a coefficient $\phi$, with $d=\alpha^\phi$, $w=\beta^\phi$, and $r=\gamma^\phi$ under a resource constraint.

**Relevance to L03:** EfficientNet sharpens the chapter's message that model-level efficiency is multi-dimensional. Depth, width, and resolution change arithmetic, activation sizes, memory footprint, and achievable hardware utilization differently.

**Key claims used in this chapter:**

- Single-dimension scaling improves accuracy but saturates for larger models. Source anchor: Section 3.2 and Figure 3.
- Compound scaling balances depth, width, and resolution. Source anchor: Section 3.3 and Equation 3.
- The paper compares scaled EfficientNets against other ConvNets in accuracy, parameters, FLOPs, and latency. Source anchor: Tables 2 and 4.

**What students should remember:** "Efficient" is not one scalar. A model can trade depth, channel width, resolution, parameter count, activation size, and latency in different ways.

**Limitations and assumptions:** EfficientNet's reported performance depends on architecture search, training recipe, and evaluation hardware. L03 uses it to explain scaling dimensions and proxy metrics, not to make a current leaderboard claim.

### Source Bridge: Evaluation Tools and Benchmarks

**Bibliographic identity:** The slides cite MLPerf, Accelergy, Timeloop, and AccelForge as examples of benchmark and modeling infrastructure.

**Relevance to L03:** These sources support the claim that realistic accelerator evaluation must specify workload, mapping, precision, batch size, memory hierarchy, and action counts.

**Key claims used in this chapter:** the list of required metric specifications is taken from the L03 PDF page 48; attention equations and rank names are taken from PDF pages 115-127.

**Limitations:** This bridge is slide-anchored. The corresponding original papers are not all present as local PDFs in this repository.

---

## 13. Standalone Study Guide

### What to master

- Be able to explain the memory hierarchy without saying only "data movement is expensive."
- Memorize the qualitative ordering: local register-like storage is small and cheap to access; DRAM is large and expensive to access.
- Practice reading an Einsum by marking output ranks and reduction ranks.
- Practice converting FC and convolution into matrix multiplication while noting what data gets repeated.
- Trace self-attention shapes from $I$ through $QK$ and $AV$.

### Self-check questions

1. Why does a small local buffer reduce energy only when there is reuse?
2. Why is batch size required context for a latency claim?
3. In $Z_{m,n} = A_{m,k}B_{k,n}$, which rank is reduced and which ranks remain?
4. Why can im2col increase memory traffic even though it enables GEMM?
5. In the slide convention for attention, what does $QK_{m,p}$ store?
6. Why might depthwise convolution have fewer MACs but still be awkward for a dense accelerator?
7. Which metric would expose a design that reports low chip power but uses large off-chip bandwidth?

### Exercises

1. **Conceptual:** Explain why a ring oscillator can produce a misleading TOPS/W-style number.
2. **Small calculation:** Using the Horowitz table, compute the energy ratio between a 32-bit SRAM read and an 8-bit multiply.
3. **Einsum reading:** For $Y_{b,p,f} = A_{b,m,p} V_{b,m,f}$, identify all output and reduction ranks.
4. **Convolution:** Build the Toeplitz matrix for input $[2,4,6,8]$ and a length-2 filter.
5. **Attention shapes:** If $B=2$, $H=8$, $M=128$, $E=64$, what is the shape of $QK$?
6. **Design tradeoff:** Compare an im2col-based convolution implementation with a direct convolution implementation using energy, bandwidth, and programmability.
7. **Paper/source bridge:** Read the Accelergy slide and explain why action counts are needed before energy can be estimated.

---

## 14. Key Terms

### Memory hierarchy

A layered storage system that places small, fast, low-energy memories near compute and large memories farther away. In accelerator design, it exists to exploit reuse and reduce expensive traffic.

### Data movement

The transfer of weights, activations, partial sums, or intermediates between memory levels or PEs. It affects energy, bandwidth, latency, and utilization.

### Operational intensity

The ratio $\text{operations}/\text{bytes moved}$ at a specified memory boundary. It measures how much computation is supported by each byte fetched.

### Throughput

Work completed per unit time, such as inferences/s or operations/s. It must be reported for a real workload and with utilization context.

### Latency

Elapsed time from input to output. Batch size is essential context because large batches can improve throughput while increasing per-input waiting time.

### PE utilization

The fraction of processing elements doing useful work. Peak TOPS assumes high utilization; real workloads may not achieve it.

### Einsum

A tensor expression that names output, input, and reduction ranks without specifying traversal order. It is the mathematical contract used by later mapping lectures.

### Rank variable

An index variable in an Einsum, such as $m$, $n$, or $k$.

### Contracted rank

A rank that appears on the right-hand side but not on the left-hand side. It is reduced, usually by summation.

### Uncontracted rank

A rank that appears in the output and is preserved.

### Partitioning

Splitting one rank into multiple ranks, such as $i = i_1 I_0 + i_0$. Later lectures use this idea for tiling.

### Flattening

Collapsing multiple ranks into one rank, such as $(c,h,w)$ into $chw$.

### Toeplitz / im2col

A transformation that represents convolution as matrix multiplication by collecting shifted input windows into a matrix. It can expose GEMM structure but may duplicate data.

### Query, Key, Value

The three projections in attention. Queries ask, keys match, and values provide the information that is mixed by attention weights.

### Attention matrix

The softmax-normalized score tensor $A_{m,p}$ in the slide convention. It scales with sequence length squared.

---

## 15. Appendix - Slide-to-Section Map

| PDF pages | Slide label in PDF | Chapter section | Notes |
|---|---|---|---|
| 1 | L02-1 | Title | PDF labels this lecture deck as L02 internally, but repository file is L03 |
| 2-19 | L02-2 to L02-19 | Memory is a first-class design constraint | Expanded with reuse example and hardware implications |
| 20-39 | L02-20 to L02-39 | Efficient models are not automatically efficient hardware | Treated as a metrics bridge, not a full CNN-model survey |
| 40-56 | L02-40 to L02-56 | Evaluation metrics | Expanded with operational-intensity example and tool bridge |
| 57-72 | Extended Einsums 2/37 to 18/37 | Einsum contract | Expanded with vocabulary and worked reading example |
| 73-90 | Kernel Computation 1 to 18 | Fully connected as matrix multiplication | Explains flattening and batch |
| 91-109 | Kernel Computation 19 to 37 | Convolution as matrix multiplication | Explains Toeplitz/im2col and data repetition |
| 110-127 | Attention Einsums 20/37 to 37/37 | Transformer self-attention as Einsums | Expanded with shape example and hardware implications |

---

## 16. Source Notes

- Direct slide-stated claims include the Horowitz energy table values, the memory hierarchy tradeoffs, the key metric list, the metric specification checklist, the MobileNets MAC ratio, the FC/CONV conversion dimensions, and the attention rank/equation cascade.
- Paper/source-derived claims are limited to what the slides cite: Horowitz ISSCC 2014, MobileNets 2017, MLPerf, Accelergy, and related evaluation tools.
- Standard background explanations include operational intensity, matrix multiplication reading, batch-vs-latency interpretation, and the explanation of query/key/value roles.
- Teaching interpretations include the small energy-reuse example, the operational-intensity toy calculation, and the attention-shape example.
- No slide figures or paper figures are embedded in this rewritten chapter. Existing `assets/L03/*.png` files remain in the repository but are not used here.

---

## 17. Uncertainty Notes

- The public PDF labels many early pages as L02 even though the repository treats this as L03. This chapter cites PDF pages to avoid ambiguity.
- The live lecture may have emphasized some efficient-CNN examples differently. This chapter uses pages 20-39 mainly to support the proxy-metric warning.
- The Horowitz energy values are technology-specific; use them as order-of-magnitude guidance, not as universal constants.
- The attention softmax axis follows the slide convention, where $p$ is the query/output position and normalization sums over $m$.
