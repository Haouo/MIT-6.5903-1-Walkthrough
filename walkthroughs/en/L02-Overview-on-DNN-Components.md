# L02 — Overview on DNN Components

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer and Vivienne Sze (MIT EECS)
> **Lecture date:** February 4, 2026 · **Slides:** 102 · **Source:** [`Lecture/L02-Overview_on_DNN_components.pdf`](../../Lecture/L02-Overview_on_DNN_components.pdf)
>
> This chapter reconstructs the teaching layer behind the slides. It uses the slides as the ordering backbone, but the prose is written as a self-contained study companion for a reader who cannot watch the lecture video.

---

## TL;DR

Lecture 02 teaches two languages that the rest of the course will use constantly.

The first language is the **workload-description language**: tensors, ranks, Einsums, iteration spaces, compute intensity, and Roofline reasoning. These let a hardware architect say exactly what a DNN layer computes, how much arithmetic it requires, how much data it ideally needs to move, and whether a particular implementation is limited by memory bandwidth or compute parallelism. The key teaching point is that an Einsum specifies **what** computation is performed, while a loop order or mapping specifies **how** the iteration space is traversed.

The second language is the **CNN-component language**: CONV, activation, NORM, POOL, and FC layers; feature maps; filters; stride; padding; channels; batch size; and the standard loop variables $N, C, H, W, R, S, M, P, Q, U$. The main workload is convolution: each output activation is a sum over a local receptive field and input channels, and the full layer performs $N \times M \times P \times Q \times C \times R \times S$ multiply-accumulates.

The reason these two languages appear in the same lecture is architectural: once a CONV layer is written as an Einsum, the accelerator designer can reason about data reuse, memory traffic, stationarity, mapping, tiling, and parallelism. Later lectures will keep returning to this exact bridge.

---

## What Problem This Lecture Solves

Lecture 01 motivated why DNN acceleration matters: compute demand is high, data movement is expensive, and general-purpose processors do not always expose the right structure for efficient DNN execution. Lecture 02 asks the next question: **what exactly is the workload that the accelerator is supposed to run?**

That question is more subtle than it first appears. Saying "run a CNN" is not precise enough for architecture. A hardware designer needs to know:

- which tensors exist and what their shapes are,
- which indices are output indices and which are reduction indices,
- how many multiply-accumulates the layer performs,
- which values can be reused across many operations,
- how stride and padding change output shape,
- whether a loop order repeatedly spills partial sums,
- and whether more MAC units would help or whether memory traffic is the real limit.

The lecture therefore starts with a modeling discipline: describe the workload as tensor algebra, evaluate it with compute and traffic metrics, and only then talk about hardware choices. It then applies that discipline to CNN components, especially convolution and fully connected layers.

**Source note:** This ordering follows Lecture 02 slides 2-44 for the accelerator design methodology and slides 45-102 for CNN components.

---

## Why This Lecture Matters

Most beginner explanations of CNNs emphasize the machine-learning view: filters detect edges, deeper layers detect higher-level concepts, and FC layers produce class scores. That view is useful, but it is incomplete for this course.

For a hardware architect, a DNN layer is also a structured data-movement problem. A convolutional layer does not merely contain many multiplications. It contains repeated use of the same weights, repeated use of nearby activations, and repeated updates to partial sums. These reuse patterns decide whether an accelerator spends its energy doing arithmetic or moving data between DRAM, SRAM, interconnect, register files, and processing elements.

The lecture's deeper lesson is:

> A DNN component is not just a neural-network concept. It is an iteration space over tensors, and the way that iteration space is traversed determines memory traffic, reuse, utilization, and bottlenecks.

That lesson sets up L03-L04, where tensor algebra and memory metrics are expanded, and L05-L06, where mapping/dataflow and partitioning decide how CONV loop nests run on real hardware.

---

## Prerequisites and Mental Model

You should be comfortable with:

- vectors and matrices,
- matrix-vector and matrix-matrix multiplication,
- basic CNN terminology such as input image, feature map, filter, and layer,
- and the idea that memory accesses can be much more expensive than arithmetic.

The mental model for this chapter is:

1. A tensor is a multi-dimensional box of values.
2. An Einsum describes a set of points in an iteration space.
3. Each point usually performs a multiply and contributes to an output.
4. If an index appears on the right-hand side but not the left-hand side, the computation reduces over that index.
5. Hardware efficiency depends on how much useful arithmetic is obtained per value moved through the memory hierarchy.

A useful analogy is a nested loop nest. An Einsum says which loops must exist, but not the order of those loops. The mapping chooses the order, tiling, and parallelism. Later, dataflow names such as output-stationary and weight-stationary will describe particular ways of making some values stay close to the processing elements.

---

## Learning Objectives

After studying this chapter, you should be able to:

1. Explain the five-step TeAAL accelerator design methodology and distinguish architecture, workload, mapping, format, and binding.
2. Define tensor rank, rank shape, tensor size, iteration space, free index, and reduction index.
3. Read an Einsum such as $Z_m = A_{k,m} \times B_k$ and derive its iteration-space size, number of multiplications, and reduction behavior.
4. Compute best-case and achieved compute intensity for the matrix-vector example used in the lecture.
5. Use Roofline reasoning to explain when more compute parallelism helps and when memory bandwidth is the bottleneck.
6. Describe the role of CONV, activation, NORM, POOL, and FC layers in a CNN.
7. Derive the output dimensions of a 2-D convolution with stride and optional padding.
8. Explain the meaning of the CNN decoder-ring variables $N, C, H, W, R, S, M, P, Q, U$.
9. Write the CONV layer as an Einsum and identify the output indices and reduction indices.
10. Explain why an FC layer is a special case of CONV and why batching turns FC into matrix-matrix multiplication.
11. Connect these DNN components to hardware concerns: data reuse, partial sums, memory traffic, bandwidth, latency, utilization, and mapping.

---

## Main Narrative: From Workload to Hardware

### TeAAL's Five-Step Methodology

Lecture 02 begins with a design method rather than with a neural-network layer. This is deliberate. A DNN accelerator is not designed by first choosing a PE array and hoping the workload fits. The course wants a repeatable path from workload to hardware.

Lecture 02 uses TeAAL as the organizing framework. In the slides, the five steps are:

1. **Describe the architecture.** Choose hardware components, such as processing elements (PEs), ALUs, register files, global buffers, and DRAM, and organize them into an accelerator specification.
2. **Develop the workload.** Describe the computation as a cascade of Einsums, then specify mapping, format, and binding.
3. **Evaluate the workload.** Count operations, memory traffic, and compute intensity for the workload on the architecture.
4. **Compare implementations.** Normalize hardware parameters and compare alternative designs.
5. **Optimize the design.** Modify architecture, mapping, format, or binding and evaluate again.

This is not a one-pass checklist. It is a loop. A poor memory-traffic result may force a different mapping; a mapping that exposes more reuse may require more local storage; a sparse format may reduce data movement but add metadata and control complexity.

**Source note:** The five-step framing is directly stated in Lecture 02 slides 4-8 and 21-44, which cite TeAAL and HiFiber and Nayak, MICRO 2023.

### Separation of Concerns

The most important part of the methodology is the separation between **what is computed** and **how it is computed**.

The slides describe four workload concerns, from most concise to most detailed:

| Concern | Question it answers | Example |
|---|---|---|
| Cascade of Einsums | What computations make up the workload? | A CONV followed by activation and pooling |
| Mapping | In what order is the iteration space traversed? | Loop order, tiling, parallelism |
| Format | How is data represented? | Dense, sparse, compressed |
| Binding | Which hardware resource performs each part? | Which PE, buffer, or memory level holds a tensor tile |

This separation matters because many design choices are independent only if the model keeps them independent. For example, the CONV Einsum can remain the same while the mapping changes from output-stationary to weight-stationary. Likewise, a sparse format can be introduced without changing the mathematical layer definition.

**Teaching interpretation:** The "pyramid" in slides 8 and 28 is a warning against mixing concerns too early. If you fuse the computation definition with a specific loop order, you make it harder to explore alternative accelerators.

---

## Tensors, Ranks, and Einsums

### Tensor Terminology

A **tensor** is a multi-dimensional array of values. In this course, a dimension of a tensor is called a **rank**. This differs from some math contexts where "rank" has a different meaning; here, rank simply means one named dimension.

Examples:

| Object | Number of ranks | Shape example | Size |
|---|---:|---|---:|
| Scalar | 0 | `[]` | $1$ |
| Vector | 1 | $[K]$ | $K$ |
| Matrix | 2 | $[M, K]$ | $M \times K$ |
| 3-D activation tensor | 3 | $[C, H, W]$ | $C \times H \times W$ |
| Batched activation tensor | 4 | $[N, C, H, W]$ | $N \times C \times H \times W$ |

The **rank shape** is the number of elements along one rank. The **size** of the tensor is the product of all rank shapes.

**Source note:** Lecture 02 slides 9-11 introduce tensors, ranks, rank shapes, and tensor size.

### What an Einsum Means

An **Einsum** is a compact way to describe tensor algebra. It tells us which operands are multiplied, which output element is updated, and which indices are reduced.

For matrix multiplication, the slides write:

$$Z_{m,n} = A_{k,m} \times B_{k,n}.$$

The index $k$ appears on the right-hand side but not on the left-hand side, so $k$ is a **reduction index**. More explicitly:

$$Z_{m,n} = \sum_k A_{k,m} B_{k,n}.$$

For matrix-vector multiplication:

$$Z_m = A_{k,m} \times B_k.$$

Here, $m$ is a **free index** because it appears in the output. The index $k$ is a **reduction index** because it appears only on the right-hand side. The computation means: for every output coordinate $m$, sum over all $k$.

The lecture gives an operational definition of an Einsum:

1. Build the iteration space from all legal values of the unique indices.
2. At each point, read the operand values selected by those indices.
3. Multiply the selected values.
4. Accumulate into the output if a reduction index is present.

For $Z_m = A_{k,m} \times B_k$, the iteration space is $K \times M$. Each point $(k,m)$ performs one multiplication, and points with the same $m$ reduce into the same $Z_m$.

### Small Worked Example: Matrix-Vector Einsum

Suppose $K=3$ and $M=2$. Let the output be $Z_0, Z_1$. The iteration space has $3 \times 2 = 6$ points:

```text
(k,m): (0,0), (1,0), (2,0), (0,1), (1,1), (2,1)
```

The output equations are:

$$Z_0 = A_{0,0}B_0 + A_{1,0}B_1 + A_{2,0}B_2,$$

$$Z_1 = A_{0,1}B_0 + A_{1,1}B_1 + A_{2,1}B_2.$$

There are $K \times M = 6$ multiplications. There are $(K-1) \times M = 4$ additions if we count additions needed to combine $K$ products per output. The exact addition count can vary depending on whether initialization and accumulation are counted separately, which is why the lecture focuses on multiplication count and memory traffic for compute intensity.

**Hardware implication:** The same value $B_0$ is used for both $m=0$ and $m=1$. If a PE loads $B_0$ once and keeps it in a register while it multiplies by $A_{0,0}$ and $A_{0,1}$, it reduces memory traffic. This is the first appearance of the idea later called **stationarity**.

---

## Compute Intensity and Roofline Reasoning

### Best-Case Compute Intensity

Lecture 02 defines compute intensity as:

$$\mathrm{CI} = \frac{\text{multiplications}}{\text{values accessed}}.$$

The slides intentionally use **multiplications per value** instead of the common FLOPs/byte definition. This avoids two ambiguities: whether a MAC is one operation or two, and what bitwidth the values use.

For $Z_m = A_{k,m} \times B_k$:

- Multiplications: $K \times M$.
- Best-case traffic: load every $A_{k,m}$ once, load every $B_k$ once, and store every $Z_m$ once.
- Best-case values accessed: $K \times M + K + M$.

So:

$$\mathrm{CI}_{\text{best}} = \frac{K \times M}{K \times M + K + M}.$$

For the slide example $K=250$ and $M=100$:

$$\mathrm{CI}_{\text{best}} = \frac{250 \times 100}{250 \times 100 + 250 + 100} \approx 0.99\ \text{multiplications/value}.$$

This is an upper bound for this simple memory model. It assumes the implementation can exploit all reuse needed to avoid extra traffic.

**Source note:** Lecture 02 slides 23-26 define compute intensity and derive the best-case traffic for the matrix-vector example. Slide 41 gives the $K=250, M=100$ numerical example.

### Achieved Compute Intensity Depends on Mapping

The best-case CI is not automatically achieved. The actual loop order and storage behavior determine achieved traffic.

The lecture uses the loop order:

```text
for k in range(K):
    keep B[k] in a register
    for m in range(M):
        load A[k,m]
        load current Z[m] partial sum when needed
        update Z[m]
        store Z[m]
```

With this processing order and simple storage model, the achieved traffic is:

- $K \times M$ loads of $A_{k,m}$,
- $K$ loads of $B_k$,
- $(K-1) \times M$ loads of $Z_m$ partial sums,
- $K \times M$ stores of $Z_m$.

Therefore:

$$\text{traffic}_{\text{achieved}} = 3KM - M + K,$$

and:

$$\mathrm{CI}_{\text{achieved}} = \frac{K \times M}{3KM - M + K}.$$

For $K=250$ and $M=100$:

$$\mathrm{CI}_{\text{achieved}} = \frac{250 \times 100}{3 \times 250 \times 100 - 100 + 250} \approx 0.33\ \text{multiplications/value}.$$

The gap between $0.99$ and $0.33$ is not a mathematical property of matrix-vector multiplication. It is a property of this implementation. The loop order keeps $B_k$ stationary but repeatedly reloads and stores partial sums $Z_m$ across the $k$ loop.

**Common misconception:** "The Einsum determines the memory traffic." It does not. The Einsum determines the mathematical iteration space. The mapping and hardware storage choices determine achieved traffic.

### Roofline Model

The **Roofline Model** connects compute intensity to throughput. In its simplest form:

$$\text{achievable throughput} \le \min(\text{peak compute throughput},\ \mathrm{CI} \times \text{memory bandwidth}).$$

The diagonal line is the memory-bandwidth limit. The horizontal line is the compute limit. A low-CI workload sits on the diagonal and is **memory-bound**. A high-CI workload can reach the horizontal roof and become **compute-bound**.

The architectural lesson is direct:

- If the workload is memory-bound, adding more MAC lanes may not improve throughput.
- If the workload is compute-bound, improving memory bandwidth may not be the main bottleneck.
- If the measured implementation is far below the roof, the gap may come from stalls, instruction overhead, poor mapping, insufficient buffering, or utilization problems.

**Source note:** Lecture 02 slides 42-43 introduce the Roofline Model and cite Williams, Waterman, and Patterson, CACM 2009. The local PDF `papers/Roofline Model.pdf` defines operational intensity as operations per byte of DRAM traffic and presents the attainable-performance bound used here. The slides generalize this intuition to memory-hierarchy levels beyond DRAM.

---

## DNN Workloads and CNN Components

### Why CNNs Appear After Einsums

The lecture's transition from Einsums to CNNs is not a topic jump. It is the moment where the abstract workload language gets applied to a real DNN family.

CNNs are useful for computer vision, speech spectrograms, gameplay, and medical imaging. A modern deep CNN can contain roughly 5 to 1000 layers. Its layers transform low-level input features into higher-level features and eventually into class scores.

The common components are:

| Component | What it does | Hardware perspective |
|---|---|---|
| CONV | Applies learned filters over local regions of feature maps | Dominant MAC count and rich reuse patterns |
| Activation | Applies a pointwise nonlinearity such as ReLU | Usually simple elementwise logic; often fused with CONV/FC output |
| NORM | Normalizes activations to stabilize training or inference behavior | More control/data movement than arithmetic in many cases |
| POOL | Downsamples spatial dimensions | Reduces later work; max/average reductions over local windows |
| FC | Connects all input neurons to all output neurons | Dense matrix-vector or matrix-matrix multiply after flattening |

The slides state that convolutions account for more than 90% of overall computation in typical CNNs, dominating runtime and energy consumption. This is why the lecture spends most of its DNN-component time on CONV.

**Source note:** CNN applications and components are directly shown in Lecture 02 slides 45-52. The more-than-90% computation claim is on slide 52.

### CONV Layer: Intuition and Precise Meaning

A convolutional layer slides a learned filter over an input feature map. At each output position, the layer multiplies the filter weights by the input values under the filter and sums the products. That sum is one output activation.

For a single-channel 2-D convolution:

- input feature map shape: $H \times W$,
- filter shape: $R \times S$,
- output feature map shape: $P \times Q$.

For one output location $(p,q)$, the layer computes:

$$O_{p,q} = \sum_{r=0}^{R-1}\sum_{s=0}^{S-1} I_{Up+r,\ Uq+s}F_{r,s},$$

where $U$ is stride. The term $Up+r$ selects the input row under the filter, and $Uq+s$ selects the input column.

The filter's support, $R \times S$, is also called the **receptive field** for one output activation in that layer. If $R=S=3$, each output activation uses 9 input values and 9 weights, before considering channels.

### Worked Example: 5-by-5 Input and 3-by-3 Filter

Lecture 02 uses a $5 \times 5$ input and a $3 \times 3$ filter with stride $U=1$. With no padding, the filter can start at rows $0,1,2$ and columns $0,1,2$, so the output is $3 \times 3$.

The output-size formula used in the slides for no padding is:

$$P = \frac{H - R + U}{U}, \qquad Q = \frac{W - S + U}{U},$$

when the division is exact. More generally, many frameworks use floor-style shape rules for valid convolution:

$$P = \left\lfloor \frac{H - R}{U} \right\rfloor + 1, \qquad Q = \left\lfloor \frac{W - S}{U} \right\rfloor + 1.$$

For $H=W=5$, $R=S=3$, and $U=1$:

$$P=Q=\left\lfloor \frac{5-3}{1} \right\rfloor + 1 = 3.$$

Each output activation needs $R \times S = 9$ multiplications. The whole layer has $P \times Q = 9$ output activations, so it performs $9 \times 9 = 81$ multiplications for this single-channel example.

**Hardware implication:** Those 81 multiplications do not require 81 independent filter loads if the hardware can reuse the same $3 \times 3$ filter across all 9 output positions. Likewise, neighboring output positions reuse many input pixels because their sliding windows overlap.

### Stride

**Stride** is the distance the filter moves between adjacent output positions. Stride $U=1$ evaluates every valid window. Stride $U=2$ skips every other starting position. Stride $U=3$ skips even more.

Using the same $5 \times 5$ input and $3 \times 3$ filter:

- $U=1$ gives $P=Q=3$, so 9 output activations.
- $U=2$ gives $P=Q=2$, so 4 output activations.
- $U=3$ gives $P=Q=1$, so 1 output activation.

Stride is therefore a downsampling mechanism. It reduces output size and later computation, but it also changes which input positions are sampled. Architecturally, larger stride reduces the number of output partial sums but may reduce overlap reuse between neighboring windows.

**Source note:** Lecture 02 slides 64-71 show the output sizes for stride 1, stride 2, and stride 3 and state that stride greater than 1 is equivalent to downsampling the stride-1 output feature map.

### Zero Padding

Without padding, convolution shrinks the spatial dimensions. A $5 \times 5$ input with a $3 \times 3$ filter and stride 1 produces a $3 \times 3$ output. Repeating that shrinkage across many layers would quickly collapse feature-map size.

**Zero padding** adds zeros around the input boundary so the filter can be centered near the edge. For a $3 \times 3$ filter, padding one row/column on each side turns the effective input into $7 \times 7$, and a stride-1 valid convolution over that padded input produces a $5 \times 5$ output.

For symmetric padding $A_h$ rows vertically and $A_w$ columns horizontally, a common formula is:

$$P = \left\lfloor \frac{H + 2A_h - R}{U} \right\rfloor + 1,\qquad
Q = \left\lfloor \frac{W + 2A_w - S}{U} \right\rfloor + 1.$$

For odd filter sizes and stride $U=1$, choosing $A_h=(R-1)/2$ and $A_w=(S-1)/2$ keeps spatial size unchanged.

**Important caveat:** Different frameworks define padding modes slightly differently. Lecture 02 explicitly notes PyTorch examples and warns that padding is not always explicitly defined but can often be inferred from feature-map sizes.

### Depth and Receptive Field Growth

As a CNN gets deeper, one output activation in a later layer depends on a larger region of the original input. A single $3 \times 3$ convolution sees a $3 \times 3$ patch. A second $3 \times 3$ convolution over the first layer's outputs combines neighboring first-layer outputs, each of which already depended on a $3 \times 3$ patch. The effective receptive field grows with depth.

This explains the usual CNN story in hardware terms:

- early layers operate on large spatial maps and often have fewer channels,
- later layers operate on smaller spatial maps but often have more channels,
- deeper features become more semantic because each activation aggregates information from a wider input region.

**Source note:** Lecture 02 slides 47-48 and 75 connect CNN depth with low-level and high-level features and show receptive-field growth.

---

## Multichannel CONV and the CNN Decoder Ring

### From One Channel to Many Channels

The simple 2-D example hides the channel dimension. Real CNN layers usually have many input channels and many output channels.

For a batched CONV layer:

| Tensor | Shape | Meaning |
|---|---|---|
| Input activations $I$ | $N \times C \times H \times W$ | $N$ input feature maps, each with $C$ channels |
| Filter weights $F$ | $M \times C \times R \times S$ | $M$ filters, each spanning all $C$ input channels |
| Bias $B$ | $M$ | one bias per output channel |
| Output activations $O$ | $N \times M \times P \times Q$ | $N$ output feature maps, each with $M$ channels |

One filter produces one output channel. If a layer has $M$ output channels, it has $M$ filters. The next layer's input-channel count is usually the previous layer's output-channel count.

### Decoder-Ring Variables

Lecture 02 defines a standard notation set that the rest of the course reuses:

| Symbol | Meaning |
|---|---|
| $N$ | batch size, number of input/output feature maps |
| $C$ | number of input channels |
| $H$ | input feature-map height |
| $W$ | input feature-map width |
| $R$ | filter height |
| $S$ | filter width |
| $M$ | number of output channels / number of filters |
| $P$ | output feature-map height |
| $Q$ | output feature-map width |
| $U$ | convolution stride |

These variables are not just notation. They are loop bounds. When you see $N \times M \times P \times Q \times C \times R \times S$, you should see seven loops and a large design space for reordering, tiling, and parallelizing those loops.

**Source note:** Lecture 02 slides 76-80 introduce many input channels, many output channels, batch size, and the CNN decoder ring.

### CONV Einsum

The full dense CONV layer can be written:

$$O_{n,m,p,q} = B_m + I_{n,c,U p+r,U q+s} \times F_{m,c,r,s}.$$

The reduction over $c,r,s$ is implicit. Expanded:

$$O_{n,m,p,q} = B_m +
\sum_{c=0}^{C-1}\sum_{r=0}^{R-1}\sum_{s=0}^{S-1}
I_{n,c,U p+r,U q+s}F_{m,c,r,s}.$$

The free output indices are $n,m,p,q$. The reduction indices are $c,r,s$.

The total number of multiplications is:

$$N \times M \times P \times Q \times C \times R \times S.$$

This equation is one of the most important in the course. It tells you how work scales with batch size, channel count, output spatial size, and filter size. It also shows where reuse can come from:

- a weight $F_{m,c,r,s}$ can be reused across $N \times P \times Q$ output positions,
- an input activation can be reused by multiple filters $M$ and by overlapping spatial windows,
- an output partial sum $O_{n,m,p,q}$ is updated $C \times R \times S$ times before it is complete.

### Naive Loop Nest

The slides show a naive seven-loop implementation. A readable pseudocode version is:

```text
for n in [0, N):
  for m in [0, M):
    for q in [0, Q):
      for p in [0, P):
        O[n,m,p,q] = B[m]
        for c in [0, C):
          for r in [0, R):
            for s in [0, S):
              O[n,m,p,q] += I[n,c,U*p+r,U*q+s] * F[m,c,r,s]
        O[n,m,p,q] = Activation(O[n,m,p,q])
```

The loop nest enforces an order. The Einsum does not. That difference is the bridge to mapping and dataflow.

**Hardware implication:** In this loop order, the output partial sum is naturally kept local while the inner $c,r,s$ loops run. That resembles output-stationary behavior. A different loop order might keep weights stationary or input activations stationary instead.

---

## Fully Connected Layers

### FC as a Special Case of CONV

A fully connected layer connects every input neuron to every output neuron. From the CONV point of view, an FC layer is a convolution whose filter covers the entire input feature map:

$$R=H,\qquad S=W.$$

There is no spatial sliding. Each output channel is produced by one filter that spans all $C \times H \times W$ input values.

For a single input example:

$$O_m = I_{c,h,w} \times F_{m,c,h,w},$$

or explicitly:

$$O_m = \sum_{c=0}^{C-1}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1} I_{c,h,w}F_{m,c,h,w}.$$

### Flattening to Matrix-Vector Multiplication

The ranks $C,H,W$ can be flattened into one rank $CHW$:

$$chw = H W c + W h + w.$$

Then:

$$O_m = I_{chw} \times F_{m,chw}.$$

This is matrix-vector multiplication. The weight tensor becomes a matrix with shape $M \times CHW$, the input becomes a vector of length $CHW$, and the output is a vector of length $M$.

For example, if $C=2$, $H=2$, $W=2$, and $M=3$, then $CHW=8$. The FC layer multiplies a $3 \times 8$ weight matrix by an $8$-element input vector to produce a $3$-element output vector. It performs $M \times CHW = 24$ multiplications.

**Hardware implication:** At batch size 1, each weight is typically used once for one input example. This can make FC layers memory-bandwidth intensive even when their loop structure is simpler than CONV.

### Batching Turns FC into Matrix-Matrix Multiplication

If the batch size is $N$, the input is $N \times CHW$ and the output is $N \times M$:

$$O_{n,m} = I_{n,chw} \times F_{m,chw}.$$

This is equivalent to matrix-matrix multiplication, with reduction over $chw$. In typical matrix multiplication notation:

$$C_{m,n} = A_{m,k} \times B_{k,n},$$

where $k$ corresponds to $chw$.

Batching improves weight reuse: the same weight matrix $F$ is used across $N$ input examples. This is one reason large-batch FC layers can achieve high arithmetic utilization on dense linear algebra hardware.

**Source note:** Lecture 02 slides 85-102 derive FC as a CONV variant, flatten it to matrix-vector multiplication, and show that batch size $N$ turns the computation into matrix-matrix multiplication.

---

## Worked Examples

### Example 1: CONV Shape and Work Count

Suppose a layer has:

- $N=1$,
- $C=3$,
- $H=W=32$,
- $R=S=3$,
- $M=16$,
- stride $U=1$,
- padding $A_h=A_w=1$.

The output size is:

$$P=Q=\left\lfloor \frac{32 + 2 - 3}{1} \right\rfloor + 1 = 32.$$

The output tensor has:

$$N \times M \times P \times Q = 1 \times 16 \times 32 \times 32 = 16{,}384$$

output activations.

Each output activation reduces over:

$$C \times R \times S = 3 \times 3 \times 3 = 27$$

products.

The layer therefore performs:

$$1 \times 16 \times 32 \times 32 \times 3 \times 3 \times 3 = 442{,}368$$

multiplications.

Hardware meaning: the filter tensor has only $M \times C \times R \times S = 432$ weights, but those weights support 442,368 multiplications. A good accelerator tries to avoid fetching those same 432 weights from expensive memory over and over.

### Example 2: Output Partial Sum Lifetime

For the same layer, one output value $O_{0,5,10,12}$ is not produced by one multiplication. It is an accumulation over $C \times R \times S = 27$ products:

$$O_{0,5,10,12} = B_5 + \sum_{c=0}^{2}\sum_{r=0}^{2}\sum_{s=0}^{2} I_{0,c,10+r,12+s}F_{5,c,r,s}.$$

Until all 27 products have been accumulated, the output is a **partial sum**. If the partial sum stays in a register or local buffer, the hardware avoids repeatedly loading and storing it. If it spills to DRAM after each product, memory traffic explodes.

This is why later lectures care so much about output-stationary dataflow.

### Example 3: FC Batch Reuse

Suppose an FC layer has $CHW=1024$ input values and $M=100$ outputs. For one input example, it performs:

$$1024 \times 100 = 102{,}400$$

multiplications and uses $1024 \times 100 = 102{,}400$ weights.

If the batch size is $N=16$, the same weight matrix is reused across 16 examples. The multiplication count becomes:

$$16 \times 1024 \times 100 = 1{,}638{,}400.$$

The weight traffic can ideally be amortized across the batch if the weight matrix or tiles of it remain close to the compute units. This is why batch size changes the hardware behavior of FC layers, even though the mathematical layer is the same.

---

## Key Equations and How to Read Them

### Matrix-Vector Einsum

$$Z_m = A_{k,m} \times B_k.$$

Read this as: for each $m$, sum over $k$. The output index is $m$; the reduction index is $k$.

### Best-Case Compute Intensity

$$\mathrm{CI}_{\text{best}} = \frac{K \times M}{K \times M + K + M}.$$

Read this as: the numerator is the work; the denominator is the minimum number of values accessed under the simple model used in the lecture.

### Achieved Compute Intensity for the Lecture's Loop Order

$$\mathrm{CI}_{\text{achieved}} = \frac{K \times M}{3KM - M + K}.$$

Read this as: the same mathematical work has lower CI because the implementation repeatedly moves partial sums.

### No-Padding CONV Output Shape

$$P = \left\lfloor \frac{H-R}{U} \right\rfloor + 1,\qquad
Q = \left\lfloor \frac{W-S}{U} \right\rfloor + 1.$$

Read this as: count how many legal top-left filter positions fit in the input.

### Padded CONV Output Shape

$$P = \left\lfloor \frac{H + 2A_h - R}{U} \right\rfloor + 1,\qquad
Q = \left\lfloor \frac{W + 2A_w - S}{U} \right\rfloor + 1.$$

Read this as: padding increases the effective input size before the same legal-window count is applied.

### CONV Einsum

$$O_{n,m,p,q} = B_m +
\sum_{c=0}^{C-1}\sum_{r=0}^{R-1}\sum_{s=0}^{S-1}
I_{n,c,U p+r,U q+s}F_{m,c,r,s}.$$

Read this as: each output activation is one bias plus a reduction over input channels and filter positions.

### CONV Work Count

$$\text{multiplications} = N \times M \times P \times Q \times C \times R \times S.$$

Read this as: number of output activations times number of products per output activation.

### FC Flattening

$$O_m = I_{c,h,w} \times F_{m,c,h,w}
\quad \Longrightarrow \quad
O_m = I_{chw} \times F_{m,chw}.$$

Read this as: flatten the input-channel and spatial ranks into one reduction rank.

---

## Hardware Implications

### Data Reuse Is the Central Resource

The lecture's CI examples and CONV equations point to the same hardware issue: efficiency improves when values are reused near compute.

CONV has three major reuse opportunities:

- **Weight reuse:** one weight $F_{m,c,r,s}$ contributes to many output positions and batch elements.
- **Input reuse:** one input activation can be used by multiple overlapping windows and multiple filters.
- **Output reuse:** one partial sum is updated many times before it becomes a final output value.

The accelerator's mapping decides which reuse is easiest to exploit.

### Partial Sums Are Data, Too

Beginners often count only input activations and weights as memory traffic. That misses partial sums. A partial sum may be read and written many times if the mapping cannot keep it local. The achieved-CI example in the first half of the lecture is deliberately chosen to show this.

### More Parallelism Is Not Always More Throughput

The Roofline Model explains why. If the implementation sits on the memory-bandwidth slope, more MAC lanes can sit idle waiting for data. The right fix may be tiling, a different loop order, more local buffering, better reuse, compression, or a different dataflow.

### Shape Parameters Are Hardware Parameters

Changing $R,S,C,M,P,Q,N,U$ changes more than neural-network accuracy. It changes loop bounds, buffer capacity needs, interconnect traffic, and reuse. For example:

- increasing $M$ increases output channels and filter count,
- increasing $C$ increases reduction work per output,
- increasing $R$ and $S$ increases receptive field and products per output,
- increasing $U$ reduces output spatial size,
- increasing $N$ can improve weight reuse but increases activation and output storage.

### FC and CONV Stress Memory Differently

CONV often has rich spatial reuse due to sliding windows. FC after flattening has a simpler dense matrix structure, but at batch size 1 its weights may have little reuse. With larger batch size, FC becomes matrix-matrix multiplication and can reuse weights across examples.

---

## Common Misconceptions

### Misconception: An Einsum Is a Loop Nest

An Einsum defines the mathematical computation and iteration space. A loop nest chooses a traversal order. Many loop nests can implement the same Einsum, and they can have very different memory traffic.

### Misconception: Compute Intensity Is a Fixed Property of a Layer

Best-case CI is a theoretical upper bound under a traffic model. Achieved CI depends on mapping, buffering, and hardware behavior. The same layer can have different achieved CI on different accelerators.

### Misconception: Convolution Means the Filter Is Mathematically Flipped

Many deep-learning libraries implement cross-correlation while calling it convolution. For this hardware lecture, the important issue is not the signal-processing convention but the sliding-window multiply-accumulate pattern and its data reuse.

### Misconception: Stride Only Reduces Compute

Stride reduces output size and therefore compute, but it also changes sampling and overlap reuse. A larger stride can reduce the number of partial sums while also reducing how much neighboring windows share input data.

### Misconception: FC Is Completely Different From CONV

FC is a special case of CONV with $R=H$ and $S=W$. The difference is not a new kind of arithmetic; it is the shape and reuse pattern.

### Misconception: Activation, NORM, and POOL Are Architecturally Irrelevant

They usually do not dominate MAC count like CONV, but they can still affect fusion, buffering, memory traffic, precision, and control. A good accelerator often handles them near the CONV/FC datapath to avoid extra memory round trips.

---

## Paper and Source Bridge

### Local PDF Note

The repository currently includes local PDFs for Batch Normalization, TeAAL, and the original Roofline CACM 2009 paper. These bridges are therefore paper-verified as well as slide-anchored.

### Paper Bridge: Batch Normalization

**Bibliographic identity:** Sergey Ioffe and Christian Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, ICML 2015. Local PDF: `papers/L02_BatchNorm_Ioffe_ICML2015.pdf`.

**Problem addressed:** Deep networks are hard to train partly because the distribution of a layer's inputs changes as earlier layers update. The paper names this effect **internal covariate shift** and proposes to stabilize layer inputs during training.

**Core idea:** For each mini-batch, normalize activations using mini-batch mean and variance, then apply learned scale and shift parameters. In the paper notation, a normalized activation is transformed by learned $\gamma$ and $\beta$ so the layer can still represent useful scales, including the identity-like case when needed.

**Relevance to this lecture:** L02 introduces DNN components beyond CONV and FC. BatchNorm is a good example of a layer whose MAC count is not the main story: it changes training behavior, inference-time computation, fusion opportunities, and memory traffic around adjacent layers.

**Key claims used here:**

- The paper defines internal covariate shift as a change in the distribution of network activations caused by changing network parameters during training. Source anchor: Section 2.
- The mini-batch transform computes mini-batch mean and variance, normalizes the activation, then applies learned scale and shift. Source anchor: Algorithm 1.
- During inference, BatchNorm should not depend on the current mini-batch; the paper uses population statistics and a fixed linear transform. Source anchor: Section 3 and Algorithm 2.
- For convolutional layers, the paper applies normalization over both mini-batch examples and spatial locations for a feature map. Source anchor: Section 3.2.

**What students should remember:** BatchNorm is not just another arithmetic layer. It changes the training/inference distinction and often becomes a fusion target in hardware or compiler implementations because repeatedly materializing BN outputs can waste memory bandwidth.

**Limitations and assumptions:** BatchNorm's training benefit is paper-derived and optimizer/model dependent. For accelerator design in this chapter, the important point is not the exact training-speed claim, but that non-CONV components can affect scheduling, fusion, storage, and inference datapaths.

**Suggested insertion points:** Read this bridge after the chapter's discussion of activation, normalization, and pooling layers. It explains why "non-CONV layers" still matter architecturally.

### Source Bridge: TeAAL and HiFiber

**Bibliographic identity:** *TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators*, MICRO 2023. Local PDF: `papers/TeAAL.pdf`.

**Problem addressed:** Accelerator design needs concise workload descriptions and precise implementation descriptions. Without separation of concerns, it is hard to compare mappings, formats, and hardware bindings systematically.

**Core idea:** Use Einsums to describe tensor algebra workloads, then separately describe mapping, format, and binding.

**Relevance to this lecture:** Lecture 02 uses TeAAL to introduce the course methodology and to justify why Einsum is the input format for workload modeling tools.

**Key claims used here:**

- TeAAL expresses computations as extended Einsums and leaves iteration order to mapping. Source anchor: paper Section 2.2.
- Mapping includes loop order, rank partitioning, and work scheduling. Source anchor: paper Section 2.3.
- TeAAL specifications include computation, mapping, format, architecture, and binding information that can be lowered into performance models. Source anchor: paper Sections 3-4.
- The lecture's design flow of architecture description, workload development, evaluation, comparison, and optimization is the course-facing version of this specification discipline. Source anchor: Lecture 02 slides 4-8, 21, and 44.

**What students should remember:** TeAAL is not introduced as a software detail. It is introduced to discipline architectural thinking: specify the computation first, then explore mappings and hardware.

### Paper Bridge: Roofline Model

**Bibliographic identity:** Samuel Williams, Andrew Waterman, and David Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, Communications of the ACM, 2009. Local PDF: `papers/Roofline Model.pdf`.

**Problem addressed:** Multicore systems were becoming diverse, and programmers, compiler writers, and architects needed a simple model that did not merely predict performance but explained which bottleneck mattered. The paper frames Roofline as a bound-and-bottleneck model: it is intentionally not perfect, but it should expose whether a kernel is constrained by memory bandwidth or compute throughput.

**Core idea:** Plot attainable throughput as a function of operational intensity. In the paper, operational intensity means operations per byte of DRAM traffic after cache filtering. The upper bound is $P_\text{attainable}=\min(P_\text{peak}, B_\text{mem}\times I_\text{op})$, where $P_\text{peak}$ is peak compute throughput, $B_\text{mem}$ is sustainable memory bandwidth, and $I_\text{op}$ is operational intensity. The intersection of the sloped bandwidth line and the flat compute roof is the **ridge point**: the minimum operational intensity needed to reach peak compute.

**Relevance to this lecture:** L02 uses compute intensity to make memory traffic concrete. The paper's operational-intensity language explains the same hardware tradeoff: a low-intensity implementation cannot use more MAC lanes unless it reduces data movement, increases bandwidth, or changes the mapping so more operations are supported by each byte moved.

**Key claims used here:**

- The model ties floating-point performance, operational intensity, and memory performance into a 2D graph. Source anchor: paper p. 66.
- Operational intensity is measured as operations per byte of DRAM traffic, with traffic counted after cache hierarchy filtering. Source anchor: paper p. 66.
- Attainable performance is bounded by the smaller of peak compute throughput and memory bandwidth times operational intensity. Source anchor: paper p. 67.
- The ridge point gives the minimum operational intensity required to achieve peak performance; a ridge far to the right means only high-intensity kernels can reach the compute roof. Source anchor: paper p. 67.
- The paper adds **ceilings** below the roofline to represent missing optimizations such as ILP/SIMD, floating-point balance, unit-stride access, memory affinity, and software prefetching. Source anchor: paper pp. 68-69.
- Cache optimizations can move a kernel rightward by increasing operational intensity, because caches reduce traffic to main memory. Source anchor: paper p. 69.

**What students should remember:** Roofline is not merely a graph. It is a design diagnostic: it tells you whether to spend effort on reuse and bandwidth, on compute parallelism, or on lower-level implementation ceilings before expecting more throughput.

**Limitations and assumptions:** The original paper is written for multicore floating-point kernels and DRAM traffic. L02 adapts the same idea to DNN accelerators and sometimes to other memory-hierarchy levels. That adaptation is a teaching interpretation: the diagnostic remains useful, but the exact bandwidth roof, traffic definition, and ceilings must be remeasured for each accelerator and memory level.

---

## Connections

### Connection to L01

L01 motivates DNN accelerators by discussing AI demand, energy cost, and the need for specialized architectures. L02 gives the vocabulary needed to make that motivation precise: tensors, Einsums, compute intensity, memory traffic, and CNN layer shapes.

### Connection to L03 and L04

L03 and L04 extend the tensor-algebra language. The matrix-vector and CONV Einsums here are the first examples. Later lectures apply similar reasoning to more complex operations, including transformer and attention computations.

### Connection to L05 and L06

L05 and L06 are about mapping, dataflow, and partitioning. The seven-loop CONV nest in this lecture is the object being mapped. Output-stationary, weight-stationary, and input-stationary dataflows are different answers to the question: which tensor should stay close to the PEs?

### Connection to L07-L10

The dense work count $NMPQCRS$ becomes the baseline for sparsity. Sparse accelerators try to skip zeros in weights or activations, but they must pay for metadata, irregular traversal, and load balancing.

### Connection to L11-L13

The FC-as-matrix-multiply view becomes important for advanced technologies and precision. Matrix-vector and matrix-matrix kernels are common demonstrations for reduced precision and compute-in-memory ideas.

---

## Standalone Study Guide

### How to Study This Lecture Without Video

1. First, make sure you can explain $Z_m = A_{k,m} \times B_k$ in words. Identify $m$ as free and $k$ as reduced.
2. Re-derive the best-case and achieved CI formulas. Do not memorize the final numbers; understand which data movement terms appear and why.
3. Draw a $5 \times 5$ input and a $3 \times 3$ filter. Count legal filter positions for stride 1 and stride 2.
4. Memorize the CNN decoder-ring variables only after you understand which tensor each variable belongs to.
5. Write the CONV Einsum and point to the output indices $n,m,p,q$ and reduction indices $c,r,s$.
6. Explain FC as CONV with $R=H,S=W$, then flatten $C,H,W$ into $CHW$.
7. For each equation, ask: what values are reused, where could partial sums live, and what would happen if they spilled to DRAM?

### Self-Check Questions

1. In $Z_m = A_{k,m} \times B_k$, why is $k$ a reduction index and $m$ a free index?
2. Why is the iteration-space size for the matrix-vector example $K \times M$?
3. What data movement terms appear in the best-case CI denominator, and why?
4. Why does the achieved-CI example include $(K-1) \times M$ loads of $Z_m$?
5. What does the Roofline Model say if an implementation is on the memory-bandwidth slope?
6. For a $7 \times 7$ input, $3 \times 3$ filter, stride $1$, and no padding, what are $P$ and $Q$?
7. For a $5 \times 5$ input, $3 \times 3$ filter, stride $1$, and padding $1$, why does the output remain $5 \times 5$?
8. In the CONV Einsum, which indices define the output tensor shape?
9. Why does each CONV output activation require $C \times R \times S$ products?
10. Why does batching make FC closer to matrix-matrix multiplication?

### Exercises

1. **Conceptual:** Explain why "more parallelism" is not a complete accelerator optimization strategy. Use Roofline vocabulary.
2. **Calculation:** Let $K=64$ and $M=32$ for $Z_m = A_{k,m} \times B_k$. Compute $\mathrm{CI}_{\text{best}}$ and $\mathrm{CI}_{\text{achieved}}$ using the lecture's formulas.
3. **Shape reasoning:** A CONV layer has $N=4$, $C=16$, $H=W=28$, $R=S=3$, $M=32$, stride $U=1$, and padding $1$. Compute $P$, $Q$, output tensor size, and multiplication count.
4. **Data reuse:** For the layer in exercise 3, list one reuse opportunity for weights, inputs, and output partial sums.
5. **Design tradeoff:** Suppose a mapping keeps weights stationary but spills output partial sums frequently. Which traffic term might grow, and how would that affect CI?
6. **FC bridge:** For $C=8$, $H=W=4$, $M=10$, and batch size $N=1$, write the FC matrix-vector shape. Then repeat for $N=16$ as matrix-matrix multiplication.
7. **Source bridge:** Read Lecture 02 slides 42-43. Explain in your own words why a workload can be far from the roof even if its CI is high enough to be compute-bound.

---

## Key Terms

| Term | Definition |
|---|---|
| Tensor | A multi-dimensional array of values. In this course, tensors describe activations, weights, outputs, and other DNN data. |
| Rank | One named dimension of a tensor, such as $C$, $H$, or $W$. |
| Rank shape | The number of elements along a rank. |
| Tensor size | The product of all rank shapes. |
| Einsum | A compact notation for tensor algebra; it specifies the computation without fixing loop order. |
| Free index | An index that appears in the output; it names output coordinates. |
| Reduction index | An index that appears on the right-hand side but not the output; values are summed over it. |
| Iteration space | The Cartesian product of all legal index values in an Einsum. Its size is the amount of loop work. |
| Mapping | The traversal policy for the iteration space, including loop order, tiling, and parallelism. |
| Format | The data representation, such as dense or sparse encoding. |
| Binding | The assignment of mapped computation and data to hardware resources. |
| Compute intensity (CI) | Multiplications per value accessed. It indicates how much arithmetic is obtained per data movement. |
| Best-case CI | Theoretical upper-bound CI under a minimum-traffic assumption. |
| Achieved CI | CI produced by a specific implementation, mapping, and storage behavior. |
| Roofline Model | A throughput model that compares compute intensity, memory bandwidth, and peak compute throughput. |
| CNN | Convolutional neural network, a deep network built largely from convolutional layers plus auxiliary layers. |
| CONV layer | A layer that applies learned filters over local input regions and sums products to produce output feature maps. |
| Feature map (fmap) | An activation tensor, often viewed as channels of spatial maps. |
| Filter / kernel | The learned weight tensor used by a CONV layer. |
| Receptive field | The input region that contributes to one output activation. |
| Stride ($U$) | The step size between neighboring filter positions. |
| Zero padding | Zeros added around input boundaries to control output size. |
| Channel ($C$ or $M$) | A feature dimension; $C$ is input channels and $M$ is output channels in this course's CONV notation. |
| Batch size ($N$) | The number of examples processed together. |
| Partial sum (psum) | An intermediate output accumulation before all reduction products have been added. |
| Activation | A pointwise nonlinear operation, often applied after CONV or FC. |
| NORM | A normalization layer used to control activation statistics. |
| POOL | A downsampling layer over local spatial regions. |
| FC layer | Fully connected layer; equivalent to CONV with $R=H$ and $S=W$, and to matrix-vector/matrix-matrix multiplication after flattening. |

---

## Takeaways

1. Lecture 02 is not just a CNN overview. It establishes the tensor-algebra language used throughout the course.
2. An Einsum specifies computation; a mapping specifies traversal. Confusing these two hides the real accelerator design space.
3. Compute intensity links workload structure to memory traffic and throughput.
4. The gap between best-case and achieved CI is a concrete way to see unexploited reuse.
5. Roofline reasoning tells the architect whether to optimize bandwidth/reuse or compute parallelism first.
6. CONV dominates typical CNN computation and has the key loop structure $N,M,P,Q,C,R,S$.
7. Stride and padding are not cosmetic neural-network details; they change output shape, work, reuse, and buffer needs.
8. FC is a special CONV case and becomes matrix-matrix multiplication when batched.
9. Partial sums are first-class hardware data; failing to keep them local can dominate traffic.
10. This lecture prepares the ground for dataflow, partitioning, sparsity, precision, and advanced accelerator technologies.

---

## Appendix

### Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L02-1-L02-2 | Title and outline | Used to frame the two-part lecture structure |
| L02-3-L02-8 | Main Narrative: From Workload to Hardware | TeAAL methodology and separation of concerns |
| L02-9-L02-20 | Tensors, Ranks, and Einsums | Expanded with free/reduction index explanations |
| L02-21-L02-26 | Compute Intensity and Roofline Reasoning | Best-case CI derivation |
| L02-27-L02-40 | Compute Intensity and Roofline Reasoning | Mapping-dependent achieved traffic and stationarity |
| L02-41 | Compute Intensity and Roofline Reasoning | Numerical CI example with $K=250$, $M=100$ |
| L02-42-L02-43 | Compute Intensity and Roofline Reasoning; Paper and Source Bridge | Roofline Model |
| L02-44 | Main Narrative | Methodology recap |
| L02-45-L02-52 | DNN Workloads and CNN Components | CNN applications, layer types, CONV dominance |
| L02-53-L02-64 | CONV Layer | Single-channel convolution and stride-1 worked example |
| L02-65-L02-71 | CONV Layer | Stride examples and downsampling interpretation |
| L02-72-L02-74 | CONV Layer | Zero padding and framework caveats |
| L02-75 | CONV Layer | Depth and receptive-field growth |
| L02-76-L02-80 | Multichannel CONV and Decoder Ring | $N,C,H,W,R,S,M,P,Q,U$ |
| L02-81-L02-83 | CONV Einsum and Naive Loop Nest | Expanded with output/reduction index interpretation |
| L02-84-L02-91 | Fully Connected Layers | FC as full-input CONV and flattening |
| L02-92-L02-102 | Fully Connected Layers | Matrix-vector and matrix-matrix views |

## Source Notes

- The lecture ordering and terminology follow Lecture 02 slides.
- The TeAAL methodology and separation-of-concerns discussion is based on Lecture 02 slides 3-8, 21, 27-28, and 44 plus the local `papers/TeAAL.pdf`, especially Sections 2.2, 2.3, and 3-4.
- The tensor, rank, Einsum, and ODE explanations are based on Lecture 02 slides 9-20.
- The compute-intensity formulas and $K=250, M=100$ numerical values are based on Lecture 02 slides 23-26 and 38-41.
- The Roofline discussion is based on Lecture 02 slides 42-43 and the local `papers/Roofline Model.pdf`, especially pp. 66-69.
- The CNN component taxonomy and CONV dominance claim are based on Lecture 02 slides 45-52.
- The CONV shape, stride, padding, decoder-ring, and Einsum material is based on Lecture 02 slides 53-83.
- The FC-to-matrix-vector and FC-to-matrix-matrix derivations are based on Lecture 02 slides 85-102.
- Worked examples not directly shown in the slides are original teaching examples derived from the slide equations.

## Uncertainty Notes

- This chapter reconstructs the likely spoken explanation from slides and source anchors; the live lecture may have emphasized some examples differently.
- The chapter uses standard floor-form convolution output formulas as background explanation. Lecture 02 presents a simplified exact-division formula on slides 64, 69, 70, and 74.
- The chapter does not embed slide images. Existing local assets may still contain slide-derived images; a repository-level copyright audit should decide whether to keep, remove, or replace them.
