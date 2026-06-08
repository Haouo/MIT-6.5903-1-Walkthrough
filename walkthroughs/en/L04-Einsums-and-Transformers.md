# L04 - Einsums and Transformers

> **Course:** 6.5930/1 - Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer and Vivienne Sze (MIT EECS)
> **Lecture date:** February 13, 2026. **Slides:** 198. **Source:** [`Lecture/L04-Einsums+Transformers.pdf`](../../Lecture/L04-Einsums+Transformers.pdf)
>
> This chapter reconstructs the missing lecture narration from the public slides. It is not a slide summary. L04 is animation-heavy, especially in the FC/CONV lowering section, so slide ranges are used as source anchors while the explanation is organized as a self-contained textbook chapter.

---

## TL;DR

L04 deepens the course's formal compute language: **Einsum**. An Einsum describes tensor computation by naming operands, ranks, and arithmetic relationships, but it deliberately does not choose a loop order, dataflow, tiling, or memory placement. In hardware terms, an Einsum is the contract for *what* must be computed; later mapping lectures decide *how* to execute it.

The lecture has three connected messages. First, rank variables tell us which tensor dimensions are preserved and which are reduced. A rank that appears on the right-hand side but not on the left-hand side is a **contracted rank**; it becomes a summation. Second, common DNN kernels can be rewritten into matrix multiplication by **flattening** or **partitioning** ranks. Fully connected layers flatten naturally. Convolution can be lowered to matrix multiplication by exposing its sliding-window, Toeplitz-like structure. Third, Transformer self-attention is also a cascade of Einsums: projections form $Q$, $K$, and $V$; $QK^T$ creates attention scores; $\mathrm{softmax}$ normalizes them; $AV$ combines value vectors; and a final projection produces the output.

The hardware lesson is not "everything is just GEMM." The lesson is sharper: once computation is expressed as ranks and contractions, an architect can reason about reuse, movement, parallelism, intermediate storage, and reductions systematically. L05, L06, L09, and L13 all depend on this rank-level view.

---

## What Problem This Lecture Solves

L02 introduced DNN layers as tensor operations, and L03 introduced the memory/metric reason that computation alone is not enough. L04 solves the representation problem that sits between those ideas:

> How can we write FC, CONV, and self-attention in one notation that is precise enough for hardware mapping but abstract enough not to prematurely choose an implementation?

Ordinary layer names hide too much. "Convolution" tells us a filter slides over an input, but not which loop ranks exist or which rank is reduced. "Attention" tells us tokens interact dynamically, but not which matrix products create the $M \times M$ score tensor. A hardware architect needs the rank structure, because ranks become loops, loops become schedules, and schedules determine movement through the memory hierarchy.

The solution is to use Einsums as the compute-level interface. The same notation can express $Z_{m,n} = \sum_k A_{m,k}B_{k,n}$, $O_q = \sum_s I_{q+s}F_s$, and $AV_{p,f} = \sum_m A_{m,p}V_{m,f}$. These look different as neural-network layers, but they are the same kind of object for mapping: named output ranks, named reduction ranks, and tensor accesses.

---

## Why This Lecture Matters

For a student without the lecture video, the most important transition is this: L04 is not trying to teach linear algebra for its own sake. It is teaching the notation that later lets the course ask hardware questions rigorously.

The same mathematical Einsum can be executed in many orders. For $Z_{m,n} = \sum_k A_{m,k}B_{k,n}$, a CPU loop nest might iterate $m$, then $n$, then $k$. A systolic array might stream $A$ and $B$ in different directions. A tensor compiler might tile all three ranks. All are valid if they visit the same iteration space and accumulate the same contracted rank. The numerical result is fixed; data movement is not.

This distinction matters for:

- **Energy:** rank order controls whether operands are reused from RF, SRAM, or DRAM.
- **Bandwidth:** tensor lowering can create regular GEMM traffic, but may duplicate data.
- **Latency:** reductions and softmax create dependencies that cannot always be parallelized away.
- **Area:** supporting large intermediates such as the attention matrix $A \in \mathbb{R}^{M \times M}$ requires storage or fusion.
- **Utilization:** flattened and partitioned ranks expose different parallel loops to a PE array.
- **Correctness:** contracted ranks must be accumulated exactly once per legal coordinate, even if the loop order changes.

---

## Prerequisites and Mental Model

You should bring three ideas from earlier lectures.

First, from L02, tensors have named dimensions. A 2-D convolution uses output channels, input channels, output spatial ranks, and filter spatial ranks. These are not just shapes; they are the loops a machine must eventually execute.

Second, from L03, memory movement is expensive and evaluation must include data movement, not only MAC count. L04 does not yet choose mappings, but it exposes the ranks that mappings will reorder and tile.

Third, from L03's first introduction to Einsum, the expression specifies a mathematical computation, not an execution order.

The mental model for this chapter is:

```text
Einsum expression
    -> ranks and tensor accesses
    -> legal iteration space
    -> many possible loop nests
    -> many possible data-movement costs
```

When you see an Einsum, ask four questions:

1. Which ranks appear on the output?
2. Which ranks appear only on the right-hand side and are therefore reduced?
3. Do any tensor accesses combine ranks, such as $q+s$ or $U \times p+r$?
4. Which ranks could be flattened, partitioned, tiled, or parallelized later?

---

## Learning Objectives

After studying this chapter, you should be able to:

- State the Operational Definition for Einsums (ODE) and use it to evaluate a small expression.
- Distinguish rank variables, rank names, and rank shapes.
- Identify contracted and uncontracted ranks in matrix multiplication, convolution, and attention.
- Explain why renaming rank variables does not change the computation.
- Explain partitioning and flattening as inverse rank transformations.
- Convert a fully connected layer to matrix-vector or matrix-matrix multiplication by flattening ranks.
- Explain how convolution lowering creates a Toeplitz/im2col tensor and why that tensor may duplicate input values.
- Trace the self-attention cascade as Einsums: embedding, $Q/K/V$ projections, $QK$, $\mathrm{softmax}$, $AV$, and output projection.
- Explain why standard attention scales quadratically with sequence length $M$.
- Connect the L04 rank notation to L05 dataflows, L06 partitioning, sparse traversal, and L13 data-movement analysis.

---

## 1. Einsum Is a Compute Contract

**Source anchor:** slides L04-4 to L04-9.

The lecture begins with an example: $Z_{m,n} = A_{m,k} \times B_{n,k}$. The slides define the **Operational Definition for Einsums (ODE)**:

> Traverse all legal rank-variable values in the iteration space. At each point, compute the right-hand side at those rank values. Assign the result to the left-hand side coordinate, unless that coordinate already has a value, in which case reduce into it.

For multiply-based contractions, "reduce" usually means add. Therefore, $Z_{m,n} = A_{m,k} \times B_{n,k}$ should be read as $Z_{m,n} = \sum_k A_{m,k}B_{n,k}$.

### Intuition

An Einsum is like a precise recipe with the cooking order removed. It lists every ingredient and every destination, but it does not say whether to prepare one output at a time, one reduction rank at a time, or one tile at a time. This is why it is useful for hardware: the same computation can later be mapped to a CPU loop nest, GPU kernel, systolic array, or custom accelerator.

### Precise meaning

A **rank variable** is an index such as $m$, $n$, or $k$. A **rank name** is the semantic dimension name such as $M$ or $K$. A **rank shape** is the size of that dimension. In $A^{K,M}_{k,m}$, the rank variables are $k,m$, the rank names are $K,M$, and the rank shapes are the extents of those dimensions.

A rank is **uncontracted** if it appears on the left-hand side. It identifies output coordinates. A rank is **contracted** if it appears on the right-hand side but not on the left-hand side. Its values are reduced away.

### Worked example: reading a small Einsum

Let $Z_{m,n} = \sum_k A_{m,k}B_{n,k}$ with $M=2$, $N=2$, and $K=3$. To compute $Z_{1,0}$, hold the output ranks fixed at $m=1,n=0$ and traverse the contracted rank $k$:

$Z_{1,0} = A_{1,0}B_{0,0} + A_{1,1}B_{0,1} + A_{1,2}B_{0,2}$.

The output shape is $M \times N$ because $m,n$ are uncontracted. The amount of work per output is $K$ multiply-accumulates because $k$ is contracted.

### Equivalent forms

Slides L04-8 to L04-9 show that $Z_{m,n} = A_{m,k}B_{k,n}$, $Z_{n,m} = A_{k,m}B_{n,k}$, and $Z_{p,q} = A_{p,r}B_{q,r}$ are all matrix multiplication patterns. The letters are not the meaning. The pattern of shared, output, and reduction ranks is the meaning.

### Hardware implication

The ODE creates a correctness boundary. A mapper may reorder loops, tile ranks, or parallelize work, but it must preserve the same set of rank tuples and the same reductions. This is the formal reason L05 can compare output-stationary, weight-stationary, and input-stationary loop nests without changing the layer's mathematical result.

### Common misconception

**Misconception:** An Einsum is just a compact way to write a matrix equation.

**Correction:** In this course, an Einsum is the compute specification consumed by mapping and analysis tools. It exposes loop ranks and reductions explicitly, which is exactly what a hardware schedule must manipulate.

---

## 2. Rank Transformations: Flattening and Partitioning

**Source anchor:** slides L04-10 to L04-14.

L04 next studies transformations that change how ranks are named without changing the underlying computation.

### Flattening

If two coordinates always move together, they can be treated as one compound coordinate. For a 2-D elementwise multiply, $Z_{i,j} = A_{i,j}B_{i,j}$ can be flattened into $Z_{ij} = A_{ij}B_{ij}$ by defining $ij = i \times J + j$.

Flattening removes ranks. It is the algebraic reason a tensor of shape $C \times H \times W$ can become a vector of length $CHW$.

### Partitioning

Partitioning does the opposite. A single rank $i$ can be split into $i_1$ and $i_0$ by $i = i_1 \times I_0 + i_0$. Then $Z_i = A_iB_i$ becomes $Z_{i_1,i_0} = A_{i_1,i_0}B_{i_1,i_0}$.

Partitioning adds ranks. It is the algebraic basis of tiling: $i_1$ names the tile, and $i_0$ names the position inside the tile.

### Worked example: split and recover an index

Suppose a vector has length 8 and we choose $I_0=4$. Then $i = i_1 \times 4 + i_0$.

| Original $i$ | $i_1$ | $i_0$ |
|---:|---:|---:|
| 0 | 0 | 0 |
| 3 | 0 | 3 |
| 4 | 1 | 0 |
| 7 | 1 | 3 |

No data changed. Only the coordinate system changed. In hardware, however, this coordinate change is powerful: $i_1$ can become an outer memory tile loop, while $i_0$ can fit inside a local buffer.

### Hardware implication

Flattening often makes a computation look like GEMM, which is convenient for hardware that already has efficient matrix engines. Partitioning often makes a computation fit a memory hierarchy or PE array. Neither transformation is physically free. Flattening changes address generation and may obscure locality; partitioning adds loop structure and may require boundary handling.

### Common misconception

**Misconception:** If two computations flatten to the same matrix multiplication, they must have the same hardware cost.

**Correction:** The flattened algebra may match, but the data layout, reuse pattern, and cost of forming the flattened tensor can differ. Convolution lowering is the most important example.

---

## 3. Convolution as an Einsum

**Source anchor:** slides L04-15 to L04-18 and L04-42 to L04-59.

The simplest convolution in the slides is one-dimensional: $O_q = \sum_s I_{q+s}F_s$. The output rank $q$ is preserved. The filter rank $s$ is contracted. The input access $q+s$ is the key. It says that output position $q$ sees a shifted window of the input.

### Worked example: 1-D convolution

Let $I=[2,1,3,0]$ and $F=[4,5]$ with valid convolution. The output length is $Q=W-S+1=3$.

$O_0 = I_0F_0 + I_1F_1 = 2 \times 4 + 1 \times 5 = 13$.

$O_1 = I_1F_0 + I_2F_1 = 1 \times 4 + 3 \times 5 = 19$.

$O_2 = I_2F_0 + I_3F_1 = 3 \times 4 + 0 \times 5 = 12$.

The filter values are reused across output positions, and input values can also be reused by overlapping windows. L05 will turn this observation into dataflow choices.

### 2-D convolution

For a batch $N$, output channel $M$, input channel $C$, output spatial ranks $P,Q$, filter ranks $R,S$, and stride $U$, the dense 2-D convolution is:

$$O_{n,m,p,q} = B_m + \sum_{c,r,s} I_{n,c,U \times p+r,U \times q+s}F_{m,c,r,s}.$$

The contracted ranks are $c,r,s$. The output ranks are $n,m,p,q$. The combined input indices $U \times p+r$ and $U \times q+s$ are what make convolution a sliding-window operation rather than a plain dense layer.

### Hardware implication

The expression reveals reuse directly. A weight $F_{m,c,r,s}$ can be used for many output positions $p,q$ and many batch elements $n$. An input activation $I_{n,c,h,w}$ can contribute to several nearby outputs because multiple pairs $(p,r)$ and $(q,s)$ may map to the same $h,w$. A partial sum $O_{n,m,p,q}$ receives many updates over $c,r,s$. These are exactly the three operand classes L05 will try to keep stationary.

### Common misconception

**Misconception:** Convolution is fundamentally different from matrix multiplication, so matrix-multiply hardware is irrelevant.

**Correction:** Convolution has a special indexing pattern, but it can be lowered into matrix multiplication. The important question is whether the lowering is materialized in memory or represented implicitly by address generation.

---

## 4. FC and CONV Lowering to Matrix Multiplication

**Source anchor:** slides L04-20 to L04-172. This section synthesizes a long animation sequence; the slide text is sparse because many pages are visual build states.

### Fully connected layer

For an input activation tensor $I_{c,h,w}$ and $M$ output neurons, a fully connected layer computes:

$$O_m = \sum_{c,h,w} I_{c,h,w}F_{m,c,h,w}.$$

Here $m$ is uncontracted, while $c,h,w$ are contracted. Flatten the input coordinates into $chw$, and the expression becomes $O_m = \sum_{chw} I_{chw}F_{m,chw}$, a matrix-vector multiply.

With a batch rank $n$, the same layer becomes:

$$O_{n,m} = \sum_{chw} I_{n,chw}F_{m,chw}.$$

This is matrix-matrix multiplication. The batch rank turns many independent matrix-vector products into one larger matrix computation.

### Loop nest view

The FC Einsum corresponds to a loop nest such as:

```text
for m in [0, M):
  O[m] = 0
  for c in [0, C):
    for h in [0, H):
      for w in [0, W):
        O[m] += I[c,h,w] * F[m,c,h,w]
```

This code block is not the definition of the Einsum. It is one legal traversal order. Other loop orders compute the same output but create different reuse.

### Convolution lowering

For convolution, flattening alone is not enough because the input uses $p+r$ and $q+s$. The standard lowering idea is to create a patch tensor:

$$T_{n,p,q,c,r,s} = I_{n,c,U \times p+r,U \times q+s}.$$

Then the convolution becomes:

$$O_{n,m,p,q} = \sum_{c,r,s} T_{n,p,q,c,r,s}F_{m,c,r,s}.$$

Flatten $(c,r,s)$ into $crs$ and flatten $(n,p,q)$ into $npq$:

$$O_{m,npq} = \sum_{crs} F_{m,crs}T_{npq,crs}.$$

This is matrix multiplication: a filter matrix of shape $M \times CRS$ multiplies a patch matrix of shape $NPQ \times CRS$ after transposition/convention choices.

### Worked example: 2-D im2col

Take a single-channel $3 \times 3$ input and a $2 \times 2$ filter with stride 1. The output has $P=2,Q=2$, so there are four output positions. The patch matrix has four rows, one for each output position, and four columns, one for each filter coordinate:

```text
input coordinates in each row of T[pq,rs]

pq=00: (0,0) (0,1) (1,0) (1,1)
pq=01: (0,1) (0,2) (1,1) (1,2)
pq=10: (1,0) (1,1) (2,0) (2,1)
pq=11: (1,1) (1,2) (2,1) (2,2)
```

Notice the duplication. Input coordinate $(1,1)$ appears in all four rows. Materialized im2col makes GEMM easy but can increase memory traffic and storage. A convolution accelerator may instead generate the same addresses on the fly and avoid storing $T$ explicitly.

### Hardware implication

Lowering explains why GEMM engines are broadly useful, but it also reveals a tradeoff. A lowered matrix can regularize compute and improve PE utilization, but materializing the patch matrix can amplify memory traffic. A custom convolution dataflow can exploit the same mathematics without physically duplicating input values. This is the bridge into L05's question: which traversal order and storage placement minimizes movement for the same Einsum?

---

## 5. From Convolution to Attention

**Source anchor:** slides L04-174 to L04-178.

The lecture then changes workload family. Convolution uses a fixed local receptive field: an output position sees the region determined by the filter shape $R \times S$. Attention uses a dynamic global receptive field: a token can assign high weight to any other token, regardless of distance.

Slides L04-175 to L04-177 state the motivation in three parts:

- Convolution models spatial-neighbor dependencies naturally, but the filter window is fixed.
- Attention is used to model long-range dependencies, with the slide quoting Vaswani et al. 2017 that it allows modeling global dependencies without regard to distance.
- Inputs are broken into tokens: words for text, image patches for vision, and spectrogram patches for audio.

### Teaching interpretation

Attention is not "more connected convolution." It changes the dependency rule. In convolution, the dependency is determined by geometry: output $(p,q)$ reads a fixed window. In attention, the dependency is determined by data: a query vector compares against all key vectors, and $\mathrm{softmax}$ turns those scores into weights.

### Hardware implication

This dynamic global dependency creates an $M \times M$ attention tensor for sequence length $M$. That intermediate can dominate storage and traffic for long sequences. The hardware problem is therefore not only multiplying matrices quickly; it is also deciding whether to materialize, tile, stream, fuse, sparsify, or approximate the score and attention tensors.

---

## 6. Self-Attention as Einsums

**Source anchor:** slides L04-179 to L04-198.

The self-attention part of L04 writes each stage as a tensor expression. The slides omit some constant scaling factors, so this chapter follows the slide-level computation and marks the omission in the source notes.

### Rank names

| Rank | Meaning |
|---|---|
| $M$ | Sequence length for query, key, and value in self-attention |
| $P$ | Alias for sequence length on the query/output side of $QK$ |
| $C$ | Dictionary or vocabulary size |
| $D$ | Input/global embedding dimension, often $d_{\text{model}}$ |
| $E$ | Query/key local embedding dimension, often $d_k$ |
| $F$ | Value local embedding dimension, often $d_v$ |
| $G$ | Output embedding dimension |
| $B$ | Batch size |
| $H$ | Number of attention heads |

### Single-head computation

For the first layer, one-hot or raw token input $IR_{m,c}$ is embedded by $I_{m,d} = \sum_c IR_{m,c}WI_{c,d}$. Later layers already receive dense $I$.

Queries and keys are projections into $E$-dimensional space:

$$Q_{m,e} = \sum_d I_{m,d}WQ_{d,e}, \qquad K_{m,e} = \sum_d I_{m,d}WK_{d,e}.$$

The pre-softmax score tensor compares every query position $p$ with every key position $m$:

$$QK_{m,p} = \sum_e Q_{p,e}K_{m,e}.$$

The slide convention normalizes over $m$ for each fixed $p$:

$$SN_{m,p} = \exp(QK_{m,p}), \qquad SD_p = \sum_m SN_{m,p}, \qquad A_{m,p} = SN_{m,p}/SD_p.$$

Values are projected by $V_{m,f} = \sum_d I_{m,d}WV_{d,f}$. The attention output is a weighted sum of values:

$$AV_{p,f} = \sum_m A_{m,p}V_{m,f}.$$

Finally, $Z_{p,g} = \sum_f AV_{p,f}WZ_{f,g}$ projects back to the output embedding space.

### Worked example: two-token attention scores

Let $M=2$ and $E=2$. Suppose $Q_0=[1,0]$, $Q_1=[0,1]$, $K_0=[1,1]$, and $K_1=[2,0]$. For query position $p=0$, the scores are:

$QK_{0,0}=Q_0 \cdot K_0=1$ and $QK_{1,0}=Q_0 \cdot K_1=2$.

The normalized attention weights for this query are:

$A_{0,0}=e^1/(e^1+e^2)$ and $A_{1,0}=e^2/(e^1+e^2)$.

Then $AV_{0,f}=A_{0,0}V_{0,f}+A_{1,0}V_{1,f}$. This is the core meaning of attention: the output at position 0 becomes a data-dependent weighted mixture of value vectors from all token positions.

### Computation properties

Slides L04-190 to L04-191 state the important scheduling properties:

- The $Q$, $K$, and $V$ projections are independent and can be computed in parallel.
- Within attention, $QK$ must be computed before $\mathrm{softmax}$, and $\mathrm{softmax}$ must be computed before $AV$.
- $WQ$, $WK$, $WV$, and $WZ$ are static at inference time; $Q$, $K$, $V$, $QK$, $A$, $AV$, and $Z$ are dynamic.
- MAC count scales quadratically with token count because $QK$ costs proportional to $M^2E$ and $AV$ costs proportional to $M^2F$.

### Batched attention

Adding batch is mechanical. Prepend a batch rank $b$ to dynamic tensors: $QK_{b,m,p} = \sum_e Q_{b,p,e}K_{b,m,e}$, $A_{b,m,p}=SN_{b,m,p}/SD_{b,p}$, and $AV_{b,p,f}=\sum_m A_{b,m,p}V_{b,m,f}$. Weight matrices are shared across batch elements.

### Multi-head attention

Multi-head attention adds a head rank $h$ so each head can use separate projections:

$$Q_{b,h,m,e} = \sum_d I_{b,m,d}WQ_{d,h,e}, \quad K_{b,h,m,e} = \sum_d I_{b,m,d}WK_{d,h,e}, \quad V_{b,h,m,f} = \sum_d I_{b,m,d}WV_{d,h,f}.$$

Each head computes its own $QK$, $\mathrm{softmax}$, and $AV$. The per-head outputs are concatenated along the head/value dimensions, for example $C_{b,p,h \times F+f}=AV_{b,h,p,f}$, then projected by $Z_{b,p,d}=\sum_g C_{b,p,g}WZ_{g,d}$, with $g$ representing the flattened head-value dimension.

### Hardware implication

Attention has both favorable and difficult hardware properties. The projection matrices are static and reusable, which suits weight reuse. The $QK$ and $AV$ products are dense matrix multiplications, which suits array compute. But the score and attention tensors are dynamic, sequence-length dependent, and often large. Softmax introduces exponentiation, reduction, division, and ordering constraints. Attention accelerators and fused attention kernels exist because materializing $QK$ and $A$ naively can waste bandwidth.

### Common misconception

**Misconception:** Multi-head attention is just bigger single-head attention.

**Correction:** It adds an explicit head rank $H$. That rank exposes parallelism and independent projections, but it also changes storage layout, concatenation, and the final projection. A mapper must decide whether heads are processed in parallel, sequentially, or tiled.

---

## 7. Paper and Source Bridge

**Source anchor:** slides L04-175 to L04-178 and L04-193 to L04-194.

### Paper Bridge: Attention Is All You Need

**Bibliographic identity:** Ashish Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. Local PDF: `papers/Transformer (Attention).pdf`.

**Problem addressed:** Sequence transduction models traditionally relied on recurrence or convolution to propagate information across positions. The paper asks whether a model can use attention mechanisms alone to connect positions and build sequence representations.

**Core idea:** The Transformer replaces recurrence with stacked self-attention and position-wise feed-forward layers. Its scaled dot-product attention computes $\mathrm{softmax}(QK^T/\sqrt{d_k})V$, and multi-head attention runs several projected attention heads in parallel before combining them.

**Relevance to L04:** L04 uses Transformer attention as the second major workload, after convolution, for practicing Einsum thinking. The paper supplies the semantic meaning of $Q$, $K$, $V$, scaled dot-product attention, multi-head attention, and the reason token-token interaction creates a different hardware problem from convolution.

**Key claims used in this chapter:**

- Attention maps a query and key-value pairs to an output. Source anchor: Section 3.2.
- Scaled dot-product attention is defined as $\mathrm{softmax}(QK^T/\sqrt{d_k})V$. Source anchor: Section 3.2.1, Equation 1.
- Multi-head attention projects queries, keys, and values into multiple subspaces and performs attention in parallel. Source anchor: Section 3.2.2.
- Self-attention connects all positions with constant sequential path length, while its per-layer complexity is $O(n^2d)$ for sequence length $n$ and representation dimension $d$. Source anchor: Section 4 and Table 1.
- The paper uses positional encodings because the model lacks recurrence or convolution to encode position by default. Source anchor: Section 3.5.

**What students should remember:** Attention is not just a new neural-network layer. It changes the tensor shape of the workload: the intermediate attention matrix scales with token-token pairs, which is why hardware discussions later worry about materialization, tiling, fusion, and long-sequence memory pressure.

**Limitations and assumptions:** The paper is used here to support the mathematical structure of attention, not to reproduce translation benchmark claims. L04 follows the slide convention for rank names, which may transpose indices relative to the paper's matrix notation.

**Suggested insertion points:** Read this bridge before Section 6 if $Q$, $K$, $V$, or multi-head attention feel like unexplained names.

### Paper Bridge: Squeeze-and-Excitation Networks

**Bibliographic identity:** Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu, *Squeeze-and-Excitation Networks*, CVPR 2018. Local PDF: `papers/L03_SENet_Hu_CVPR2018.pdf`.

**Problem addressed:** Standard convolution mixes spatial and channel information locally, but it does not explicitly model global channel interdependencies.

**Core idea:** An SE block first **squeezes** each channel into a global descriptor, then **excites** channels with learned, input-dependent weights. The block recalibrates feature maps by multiplying channels by these learned gates.

**Relevance to L04:** L04 contrasts convolution's fixed local receptive field with attention-like mechanisms that allocate emphasis dynamically. SENet is not Transformer self-attention, but it gives a concrete CNN example of data-dependent weighting: the network changes channel importance based on the current input.

**Key claims used in this chapter:**

- The paper frames the SE block as explicitly modeling channel interdependencies and adaptively recalibrating channel-wise feature responses. Source anchor: Abstract.
- The squeeze stage uses global information to form a channel descriptor. Source anchor: Section 3.1.
- The excitation stage maps the descriptor to channel weights and applies channel-wise scaling. Source anchor: Section 3.2.
- The paper reports that SE blocks can be added to existing architectures with slight additional computational cost. Source anchor: Abstract and Section 4.

**What students should remember:** Attention is not only a Transformer equation. The broader architectural idea is data-dependent emphasis. SENet emphasizes channels; Transformer attention emphasizes token-token interactions.

**Limitations and assumptions:** SENet does not support the L04 equations for $QK^T$, softmax over token positions, or multi-head attention. It is included only to clarify the local-vs-dynamic-emphasis concept raised by the slides.

### Other slide-stated examples

Slides L04-176 and L04-178 cite GPT-3 (Brown et al., NeurIPS 2020), AST (Gong et al., Interspeech 2021), ViT (Dosovitskiy et al., ICLR 2021), Jalammar's illustrated Transformer, and Dive into Deep Learning as examples or figure sources. This chapter does not reproduce their figures and does not rely on paper-specific quantitative claims from them.

---

## Hardware Implications

The main hardware implications of L04 are:

- **Einsum separates compute from schedule.** This allows a mapper to search loop orders without changing the mathematical layer.
- **Contracted ranks are reduction work.** They affect accumulator lifetime, reduction trees, and partial-sum storage.
- **Uncontracted ranks are output space.** They often expose parallel work, but the best parallel rank depends on hardware shape and data reuse.
- **Flattening regularizes compute.** It can expose GEMM structure, but it may change address generation and hide original locality.
- **Partitioning is the algebra of tiling.** It creates tile ranks that later map to memory levels or PE array dimensions.
- **CONV lowering trades regularity for possible duplication.** Materialized im2col can expand traffic; implicit lowering can preserve storage but requires more specialized address generation.
- **Attention introduces large dynamic intermediates.** The $M \times M$ score and attention tensors create storage and bandwidth pressure, especially for long sequences.
- **Softmax is a scheduling boundary.** It contains reductions and normalization, so it constrains fusion and parallel execution.

---

## Common Misconceptions

### Misconception: The left-hand side tells the whole cost.

The output shape is only part of the story. The contracted ranks determine how many products contribute to each output, and tensor access functions determine reuse. $O_{n,m,p,q}$ says nothing by itself about the cost of summing over $c,r,s$ or moving $I$ and $F$.

### Misconception: Lowering convolution to GEMM means im2col must be materialized.

The algebraic lowering defines a patch matrix $T$. Hardware may materialize $T$, generate its elements on demand, or use a direct convolution dataflow that never names $T$ in storage.

### Misconception: Softmax is a small detail after matrix multiplication.

Softmax changes the scheduling problem. To normalize over $m$ for each $p$, the machine needs the relevant score values, their exponentials or a numerically stable equivalent, a denominator, and then normalized weights before $AV$ can finish.

### Misconception: Attention's quadratic cost is caused by the embedding projection.

The projections scale like $MDE$, $MDF$, or $MFG$. The quadratic terms come from token-token interactions: $QK$ scales like $M^2E$ and $AV$ scales like $M^2F$.

---

## Connections

- **L02:** Supplies the DNN layer vocabulary: FC, CONV, channels, batches, filters, and tokens.
- **L03:** Introduces memory hierarchy and the first Einsum/attention examples. L04 makes the rank manipulation more explicit.
- **L05:** Uses the same Einsums and asks which loop order creates output-stationary, weight-stationary, or input-stationary reuse.
- **L06:** Turns partitioning into temporal and spatial tiling, including distributed matrix multiplication and attention.
- **L07-L10:** Sparse tensors are still tensors in an Einsum. The difference is that formats and traversal must skip or represent missing coordinates.
- **L12:** Precision choices apply to operands and accumulators inside these Einsums; reductions often require wider accumulators.
- **L13:** Converts ranks, tensor accesses, and schedules into formal spaces and maps for exact data-movement calculation.

---

## Standalone Study Guide

### What to master before moving on

- Given an Einsum, mark output ranks and contracted ranks.
- Explain the ODE in your own words without mentioning any particular loop order.
- Flatten and unflatten a small coordinate such as $(i,j)$.
- Convert $O_m = \sum_{c,h,w} I_{c,h,w}F_{m,c,h,w}$ into matrix-vector form.
- Explain why $O_q = \sum_s I_{q+s}F_s$ has sliding-window reuse.
- Describe the difference between materialized im2col and implicit convolution lowering.
- Trace the attention chain from $I$ to $Z$ and identify which tensors are static or dynamic.

### Self-check questions

1. In $Z_{m,n} = \sum_k A_{m,k}B_{n,k}$, which ranks determine output shape and which determine work per output?
2. Why does changing rank-variable names not change an Einsum?
3. If $i=i_1 \times I_0+i_0$, what are $i_1$ and $i_0$ for $i=11,I_0=4$?
4. In $O_q = \sum_s I_{q+s}F_s$, which values are reused across neighboring $q$?
5. What data duplication can appear when a $3 \times 3$ input is lowered for a $2 \times 2$ filter?
6. Why does $QK_{m,p} = \sum_e Q_{p,e}K_{m,e}$ create an $M \times M$ tensor?
7. Which attention operations can be parallelized before the core attention step, and which must be sequential?
8. Why is softmax not merely an elementwise operation in this context?

### Exercises

1. Write a loop nest for $Z_{m,n} = \sum_k A_{m,k}B_{k,n}$. Then write a second loop nest with a different order and explain why both are valid.
2. Flatten $(c,h,w)$ into one rank $chw$ for $C=2,H=3,W=4$. Compute the flat index for $(c,h,w)=(1,2,3)$ using $chw=c \times H \times W+h \times W+w$.
3. For $I=[1,2,0,3,4]$ and $F=[2,-1,1]$, compute valid 1-D convolution outputs.
4. Build the im2col coordinate table for a single-channel $4 \times 4$ input and a $2 \times 2$ filter with stride 1. How many rows and columns does the patch matrix have?
5. For $M=4,E=8,F=8$, count the MACs in $QK$ and $AV$. Then repeat for $M=8$. What changed?
6. Design question: If an accelerator cannot store the full $M \times M$ attention matrix, what two implementation strategies could avoid materializing it?

---

## Key Terms

| Term | Meaning |
|---|---|
| **Einsum** | A tensor expression that specifies operands, ranks, and reductions without choosing an execution order. |
| **Operational Definition for Einsums (ODE)** | The rule that evaluates every legal rank tuple, computes the right-hand side, and assigns or reduces into the left-hand side. |
| **Rank variable** | The index symbol used in an expression, such as $m$ or $k$. |
| **Rank name** | The semantic name of a dimension, such as $M$ for output channels or sequence length depending on context. |
| **Rank shape** | The extent of a rank. |
| **Contracted rank** | A rank that appears on the right-hand side but not on the left-hand side, so it is reduced. |
| **Uncontracted rank** | A rank that appears on the output and identifies output coordinates. |
| **Iteration space** | The set of all legal rank-variable tuples traversed by the ODE. |
| **Flattening** | Combining multiple ranks into one compound rank, such as $(c,h,w) \rightarrow chw$. |
| **Partitioning** | Splitting one rank into multiple ranks, such as $i \rightarrow (i_1,i_0)$. |
| **Toeplitz/im2col lowering** | Rewriting convolution as matrix multiplication by forming an input patch tensor or equivalent address pattern. |
| **Static tensor** | A tensor such as a trained weight matrix that does not change across input examples during inference. |
| **Dynamic tensor** | An activation or intermediate tensor recomputed for each input. |
| **Self-attention** | Attention where $Q$, $K$, and $V$ are all derived from the same input sequence. |
| **Attention tensor** | The softmax-normalized $M \times M$ tensor $A$ that weights value vectors for each query position. |
| **Multi-head attention** | Attention with an added head rank $H$, allowing separate projections and attention patterns per head. |
| **Quadratic scaling** | The $M^2$ growth in standard attention's token-token products as sequence length $M$ increases. |

---

## Takeaways

- Einsum is the course's precise language for compute: it specifies ranks, tensor accesses, and reductions while leaving schedule open.
- Contracted ranks are the key to reading work and reductions; uncontracted ranks are the key to reading output shape.
- Flattening and partitioning are not cosmetic. They are the algebraic tools behind GEMM lowering, tiling, and spatial mapping.
- FC layers flatten directly into matrix-vector or matrix-matrix multiplication.
- CONV lowers to matrix multiplication by exposing a patch tensor, but materializing that tensor can duplicate input data.
- Attention is a cascade of matrix-like Einsums plus $\mathrm{softmax}$; its token-token products create $M^2$ work and intermediates.
- Static weights and dynamic intermediates should be separated in hardware reasoning because they have different reuse opportunities.
- L04 prepares the formal rank language needed for mapping, partitioning, sparse traversal, precision choices, and exact data-movement analysis.

---

## Appendix - Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L04-1 to L04-3 | Header and source context | Title and acknowledgements |
| L04-4 to L04-9 | Sections 1, Key Terms | ODE, tensor references, matrix/einsum patterns |
| L04-10 to L04-14 | Section 2 | Rank tuples, partitioning, flattening, matrix-multiply variants |
| L04-15 to L04-18 | Section 3 | 1-D convolution and rank-shape examples |
| L04-20 to L04-41 | Section 4 | FC animation sequence synthesized into FC lowering explanation |
| L04-42 to L04-159 | Sections 3 and 4 | CONV visual build sequence synthesized; no slide images embedded |
| L04-160 to L04-172 | Section 4 | Toeplitz/im2col and CONV-to-matmul summary |
| L04-174 to L04-178 | Sections 5 and 7 | Convolution vs. attention, Transformers, tokens, source examples |
| L04-179 to L04-189 | Section 6 | Attention mechanism as Einsums, full cascade |
| L04-190 to L04-192 | Sections 6 and Hardware Implications | Computation properties, static/dynamic tensors, batched attention |
| L04-193 to L04-194 | Section 6 | Multi-head attention |
| L04-195 to L04-198 | Section 6 and Key Terms | Rank names and tensor glossary |

## Source Notes

- The lecture ordering, formulas, and attention rank names follow `Lecture/L04-Einsums+Transformers.pdf`.
- The ODE wording is paraphrased from slide L04-4.
- The convolution and CONV-to-matmul explanation is reconstructed from slides L04-15 to L04-18 and L04-42 to L04-172; many of those slides are animation frames with little extractable text.
- The attention formulas follow slides L04-183 to L04-194. The slides note that some constant scaling steps are not illustrated; this chapter follows the slide scope and does not add the omitted scaling factor.
- The examples using small vectors, small im2col coordinate tables, and two-token attention are original teaching examples.
- The paper bridges for Vaswani et al. 2017 and Hu et al. 2018 use the local PDFs `papers/Transformer (Attention).pdf` and `papers/L03_SENet_Hu_CVPR2018.pdf`; Brown et al. 2020, Gong et al. 2021, Dosovitskiy et al. 2021, Jalammar, and D2L remain slide-stated examples rather than independently reviewed paper sources.

## Uncertainty Notes

- The live lecture may have emphasized particular animation frames differently; this chapter reconstructs the likely narration from the slide sequence.
- Some CONV lowering details are teaching interpretation because the corresponding slide pages are mostly visual builds.
- The attention notation follows the slide convention where $A_{m,p}$ normalizes over key/source position $m$ for each query/output position $p$. Other textbooks often transpose this convention.
