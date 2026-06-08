# L02 — Overview on DNN Components

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze (MIT EECS)
> **Lecture date:** February 4, 2026 · **Slides:** 102 · **Source:** [`Lecture/L02-Overview_on_DNN_components.pdf`](../../Lecture/L02-Overview_on_DNN_components.pdf)
>
> *This is a conceptual walkthrough that reconstructs the lecture's narrative from the slides. It is organized by idea, not slide-by-slide. Each section cites the slide range it draws from so you can follow along with the original deck.*

---

## TL;DR

This lecture builds the language every hardware architect needs to reason about DNN workloads. The first half formalizes the **accelerator design methodology** through the TeAAL framework — defining Einsums as the workload specification format, deriving compute intensity as the key efficiency metric, and connecting both to the Roofline Model. The second half systematically dissects **Convolutional Neural Network (CNN) components**: the CONV layer (with its sliding-window computation, stride, padding, and multi-channel extensions), the Fully-Connected (FC) layer (which is CONV with filter size equaling feature-map size, collapsing to matrix–vector/matrix–matrix multiply), and the auxiliary NORM and POOL layers. The throughline is **data reuse**: every architectural dimension — spatial, channel, batch — creates an opportunity to amortize weight or activation memory traffic, and the challenge for the hardware architect is to exploit that reuse efficiently.

---

## Learning Objectives

After this lecture you should be able to:

- Describe the **five-step TeAAL accelerator design methodology** and explain what each step decides.
- Write a **CONV layer computation as an Einsum** and read off the iteration-space size and the seven canonical loop variables (N, C, H, W, R, S, M, P, Q, U).
- Define **compute intensity (CI)** as multiplications per value, compute the best-case CI for a given Einsum, and use the **Roofline Model** to interpret what limits throughput.
- Explain how **stride** and **zero padding** control the spatial dimensions of a CONV output.
- Show that an **FC layer** is a special case of a CONV layer (R = H, S = W), and that with batch size N > 1 it reduces to **matrix–matrix multiplication**.
- Identify NORM and POOL as auxiliary layers and state that CONV accounts for **>90%** of overall computation in a typical CNN.

---

## Chapter 1 — Accelerator Design Methodology

> *Slides: L02-3 … L02-44*

### From workload to hardware: the five-step cycle

The lecture opens by establishing that hardware architects need a **principled, repeatable methodology** to move from a DNN workload specification all the way to an optimized accelerator. The TeAAL framework (introduced in Nayak, MICRO 2023) formalizes this as five steps that can be iterated:

![TeAAL five-step accelerator design methodology](../../assets/L02/L02-p04-accel-design-methodology.png)

1. **Describe the architecture** — Select from a library of hardware components (PEs with ALUs and register files, global SRAM buffers, DRAM) and organize them into an accelerator specification.
2. **Develop the workload** — Write a cascade of Einsums that describes the computation, together with mapping, format, and binding specifications.
3. **Evaluate the workload** — Model the workload on the hardware: count computes, compute memory traffic, and derive compute intensity.
4. **Compare implementations** — Normalize across hardware parameters and re-evaluate to compare design alternatives fairly.
5. **Optimize the design** — Incrementally modify one or more specifications (architecture, mapping, format, binding) and re-evaluate.

This cycle is not sequential; optimizing often sends you back to step 1 or 2. Steps 3–4 are what **Lab 1** is about; steps 2 and 3 (mapping / traverse order) are what **Labs 2–3** focus on.

### The separation of concerns inside step 2

Within "develop the workload," TeAAL draws a strict hierarchy — from coarsest and most concise to finest-grained:

![Separation of concerns: Cascade → Mapping → Format → Binding](../../assets/L02/L02-p08-separation-of-concerns.png)

- **Cascade of Einsums** — *what* computation is performed.
- **Mapping** — *how* the iteration space is traversed (loop order, tiling, parallelism).
- **Format** — *how data is encoded* (dense vs. sparse, compression scheme).
- **Binding** — *which* hardware resources execute which parts of the mapping.

This mirrors the TeAAL Pyramid of Concerns from L01, and it is the reason the course can discuss a change in loop order without touching the hardware description.

### Tensors, ranks, and Einsums

A **tensor** is a multi-dimensional array. In this course, a tensor's dimensionality is called its number of **ranks**, and the size of each rank is its **shape**. The size of the tensor is the product of all rank shapes.

Matrix multiplication is the archetype. The Einsum notation captures it in one line:

$$Z_{m,n} = A_{k,m} \times B_{k,n}$$

Every index that appears on the right but *not* on the left is summed over (the "reduction" index). For matrix-vector multiplication the same idea holds:

$$Z_m = A_{k,m} \times B_k$$

The **Operational Definition of an Einsum (ODE)** makes this precise:

1. Form the **iteration space** — the Cartesian product of all legal index values (e.g., K × M for the vector case).
2. For each point in the iteration space, read the operand values at the specified indices, multiply them, and accumulate into the output at the left-hand-side indices.
3. Indices that appear only on the right are **reduction indices** and are summed over implicitly.

The size of the iteration space (K × M) is exactly the amount of work (number of multiplications) the Einsum requires.

### Compute intensity and the Roofline Model

**Compute intensity (CI)** is defined as *multiplications per value* (rather than the ambiguous FLOPs/byte):

$$\text{CI} = \frac{\text{number of multiplications}}{\text{number of values accessed}}$$

For the matrix-vector Einsum Z_m = A_{k,m} × B_k:

- **Best-case CI** (minimum traffic, maximum reuse): The numerator is K × M multiplications; the denominator is K×M (loads of A) + K (loads of B) + M (stores of Z) = K×M + K + M values. For K=250, M=100 this gives ≈ **0.99 multiplications/value**.

- **Achieved CI** (with a specific loop nest): If the outer loop is over k and the inner loop is over m — the natural loop order — achieved traffic is K×M (loads of A) + K (loads of B) + (K−1)×M (loads of Z for partial sums) + K×M (stores of Z) ≈ 3K×M values. For K=250, M=100 this drops to ≈ **0.33 multiplications/value**.

The achieved CI is always ≤ the best-case CI. The gap comes from not exploiting reuse fully — here, the Z[m] partial sum is repeatedly loaded from DRAM across the k-loop because there is only one register (no tiling).

The **Roofline Model** turns CI into a throughput prediction:

![Roofline Model — memory-bound vs. compute-bound regimes](../../assets/L02/L02-p42-roofline-model.png)

- The horizontal roof is the **compute ceiling** (e.g., 8 MACs/cycle for L=8 lanes), limited by the number of parallel multiply units.
- The diagonal ramp is the **memory bandwidth ceiling** — throughput = CI × bandwidth.
- A workload's CI places it on the x-axis; it falls in the **memory-bound region** (on the ramp) or the **compute-bound region** (under the roof).

Key implication: when a workload is memory-bound, adding more compute lanes does *not* increase throughput — only reducing memory traffic (increasing CI) helps.

> **Why it matters:** Compute intensity is the single most important number linking a workload specification to a hardware design. It tells the architect whether the bottleneck is memory bandwidth or arithmetic throughput — and therefore which lever to pull.

---

## Chapter 2 — CNN Structure and Layer Types

> *Slides: L02-45 … L02-52*

### CNNs: deep stacks of heterogeneous layers

A modern CNN is a sequence of **5 to 1000 layers** organized to transform a raw input (an image, a speech spectrogram, game state, or medical scan) into a prediction:

![CNN layer stack: CONV → NORM → POOL → FC](../../assets/L02/L02-p47-cnn-overview.png)

The canonical layer types are:

| Layer type | Role |
|---|---|
| **CONV** (Convolution) | Extracts spatial features using learned filters; hierarchically from low-level edges to high-level semantics. |
| **Activation (nonlinear)** | Pointwise nonlinearity applied after each CONV or FC (e.g., ReLU). Enables networks to learn non-linear decision boundaries. |
| **NORM** (Normalization) | Stabilizes training; occurs between CONV and POOL layers. |
| **POOL** (Pooling) | Spatially downsamples the feature map, reducing computation and providing translation invariance. |
| **FC** (Fully-Connected) | Produces the final class-score vector from high-level features; typically 1–3 layers at the network's end. |

The critical quantitative fact: **CONV layers account for >90% of overall computation** in a typical CNN, dominating both runtime and energy. This single statistic motivates why the rest of the lecture — and much of the course — focuses on understanding and optimizing the CONV computation.

Depth creates hierarchy: each successive CONV layer combines local patches from the previous layer's output, so by layer 3 or 4, each output activation "sees" a region covering much of the original input. Low layers respond to edges and textures; deep layers respond to parts, objects, and scenes.

> **Why it matters:** Understanding which layer type costs what — and that CONV dominates — is the first step toward a principled hardware budget. Architectural choices for NORM, POOL, and activation are largely free-riders on the CONV and FC engines.

---

## Chapter 3 — The CONV Layer in Depth

> *Slides: L02-53 … L02-83*

### Anatomy of one CONV layer

A single CONV layer slides a **filter** (a learned weight tensor) across an **input feature map** to produce an **output feature map**:

![CONV layer anatomy — input fmap, filter (weights), output fmap](../../assets/L02/L02-p53-conv-layer-anatomy.png)

For the 2D case:

- The **input feature map (fmap)** is an H × W plane of activations.
- The **filter** is an R × S grid of weights (R and S are typically 1×1, 3×3, 5×5, or 7×7).
- The filter slides across the input in a **sliding window**: at each position, the R×S activations under the filter are multiplied element-wise by the R×S weights and summed to produce one **output activation** (a partial sum accumulation).
- The resulting **output feature map** is P × Q in spatial dimensions.

### Worked 2D convolution example

The slides walk through a concrete 5×5 input, 3×3 filter example with stride 1:

![2D convolution example — 5×5 input, 3×3 filter, stride 1](../../assets/L02/L02-p57-2d-conv-example-stride1.png)

Each output activation requires 3×3 = 9 element-wise multiplications and 8 additions. Sliding the filter across all valid positions with stride 1 produces a **3×3 output** (since (5−3+1)/1 = 3). The output feature map size formula is:

$$P \times Q = \left\lfloor\frac{H - R + U}{U}\right\rfloor \times \left\lfloor\frac{W - S + U}{U}\right\rfloor$$

where **U** is the stride.

### Stride and zero padding

**Stride** controls how many pixels the filter moves between output positions:

![Impact of stride — stride 1 (3×3 output), stride 2 (2×2), stride 3 (1×1)](../../assets/L02/L02-p71-stride-impact.png)

- Stride 1 → 3×3 output (9 values).
- Stride 2 → 2×2 output (4 values), which is equivalent to downsampling the stride-1 result.
- Stride 3 → 1×1 output (1 value).

Stride > 1 is a spatial downsampling mechanism: it reduces the number of output activations (and hence subsequent computation) without requiring a separate pooling layer.

**Zero padding** surrounds the input with zeros to control the output size. Without padding, every CONV layer shrinks the spatial dimensions — for a deep network this quickly collapses the feature maps to zero size. Setting padding to (R−1)/2 for each spatial dimension (with stride U=1) keeps the output the same spatial size as the input, which simplifies network design.

### Multichannel: C input channels, M output channels

The 2D description above assumed a single-channel input. In practice:

- The input fmap has **C channels** (for the first layer, C=3 for RGB; for subsequent layers, C equals the number of output channels of the preceding layer).
- The filter is now an **R × S × C** tensor — one R×S plane per input channel.
- Each filter is applied to all C input channels simultaneously, producing one scalar output activation by summing across all R×S×C products.
- There are **M such filters**, one per desired output channel.
- The output fmap therefore has **M channels**, each a P×Q spatial map.

With batch size **N** (processing N images simultaneously), the four tensor shapes are:

| Tensor | Shape |
|---|---|
| Input fmap | N × C × H × W |
| Filter weights | M × C × R × S |
| Output fmap | N × M × P × Q |
| Bias | M (one per output channel) |

### The CNN "decoder ring"

The lecture provides a definitive table of all CONV layer loop variables — the **CNN decoder ring**:

![CNN decoder ring — all 10 canonical variable definitions](../../assets/L02/L02-p79-cnn-decoder-ring.png)

| Variable | Meaning |
|---|---|
| **N** | Batch size (number of input/output fmaps) |
| **C** | Number of input channels |
| **H** | Height of input fmap |
| **W** | Width of input fmap |
| **R** | Height of filter (kernel) |
| **S** | Width of filter (kernel) |
| **M** | Number of output channels (number of filters) |
| **P** | Height of output fmap |
| **Q** | Width of output fmap |
| **U** | Stride |

These ten symbols will appear repeatedly throughout the course. Every CONV layer is fully characterized by specifying values for all of them.

### The CONV Einsum and the 7-nested-loop implementation

The full CONV computation is compactly expressed as an Einsum:

![CONV layer Einsum notation](../../assets/L02/L02-p82-conv-einsum.png)

$$O_{n,m,p,q} = B_m + I_{n,c,U\cdot p + r, U\cdot q + s} \times F_{m,c,r,s}$$

The naïve implementation is a 7-nested loop nest: outer loops over n (batch), m (output channels), q and p (output spatial position); inner loops over c (input channel), r and s (filter position). The inner body performs one multiply-accumulate and adds the bias once per output activation. The total work is N × M × P × Q × C × R × S multiply-accumulate operations.

The Einsum is more general than the loop nest: it specifies *what* is computed without fixing any loop order — that choice is the **mapping**, addressed in L05–L06.

> **Why it matters:** The CONV Einsum is the workload specification that flows through the entire TeAAL pipeline. Its seven reduction indices (n, m, p, q, c, r, s) correspond to seven loop levels, and the choice of which to tile, reorder, or parallelize is the central hardware design question of the course.

---

## Chapter 4 — The Fully-Connected Layer

> *Slides: L02-84 … L02-102*

### FC is CONV with full-size filters

A Fully-Connected layer connects every input neuron to every output neuron. From the CONV layer's point of view, this is just a CONV layer where the filter size equals the input feature map size: **R = H, S = W**. There is no sliding window — each filter covers the entire spatial extent of the input.

The FC Einsum (for a single image) is:

$$O_m = I_{c,h,w} \times F_{m,c,h,w}$$

By **flattening** the three ranks C, H, W into a single rank CHW (of size C×H×W), this becomes a **matrix–vector multiply**:

$$O_m = I_{chw} \times F_{m,chw}$$

![FC layer as matrix-vector multiply](../../assets/L02/L02-p92-fc-matrix-vector.png)

The weight matrix F has shape M × CHW; the input vector I has shape CHW × 1; the output vector O has shape M × 1.

### Batch size N > 1 promotes to matrix–matrix multiply

When N images are processed simultaneously (batch mode), the input becomes an N × CHW matrix and the output becomes an N × M matrix:

$$O_{n,m} = I_{n,chw} \times F_{m,chw}$$

This is a standard **matrix–matrix multiply** — the same operation that dense linear algebra libraries (cuBLAS, MKL) are highly optimized for. This equivalence is why FC layers on GPUs often achieve near-peak arithmetic throughput at inference time (for large N), while CONV layers typically cannot match it because of the more complex address pattern imposed by the filter's receptive field.

The Einsum for the flattened FC with batch:

$$O_{n,m} = I_{n,chw} \times F_{m,chw}$$

is directly analogous to generic matrix multiplication $C_{m,n} = A_{m,k} \times B_{k,n}$ with the index mapping k ↔ chw. The reduction is over the chw rank; the free ranks are n and m.

> **Why it matters:** Recognizing FC as a special CONV case unifies the hardware design: the same PE array and memory hierarchy that runs CONV can run FC. At the same time, FC's low compute intensity (each weight is used for only one output) makes it memory-bandwidth-bound at batch size 1 — a hardware challenge distinct from CONV's bandwidth challenges.

---

## Key Terms

| Term | Gloss |
|---|---|
| **Tensor** | A multi-dimensional array; characterized by its number of ranks and the shape of each rank. |
| **Rank** | One dimension of a tensor (this course's preferred term over "dimension"). |
| **Einsum** | Einstein summation notation: a compact expression specifying what computation is performed, without fixing loop order. Indices appearing only on the right are summed over. |
| **Iteration space** | The Cartesian product of all legal index values in an Einsum; its size equals the number of multiplications. |
| **Compute intensity (CI)** | Multiplications per value accessed; measures the potential for data reuse. |
| **Best-case CI** | CI achieved when every value is accessed the minimum number of times (maximum reuse). |
| **Achieved CI** | CI realized by a specific loop order / mapping; always ≤ best-case CI. |
| **Roofline Model** | A visual model showing achievable throughput as a function of CI, memory bandwidth, and compute parallelism. |
| **CONV layer** | Convolutional layer: slides R×S×C filters across an H×W×C input to produce P×Q×M output. |
| **FC layer** | Fully-Connected layer: a special CONV with R=H, S=W; equivalent to matrix–vector (or matrix–matrix) multiply. |
| **Input fmap** | Input feature map — the activation tensor input to a layer. |
| **Output fmap** | Output feature map — the activation tensor produced by a layer. |
| **Filter / kernel / weight** | The learned weight tensor for a CONV or FC layer. |
| **Stride (U)** | Step size of the sliding window; stride > 1 downsamples the output. |
| **Zero padding** | Zeros added around the input boundary to control output spatial size. |
| **N, C, H, W, R, S, M, P, Q, U** | The ten canonical CONV loop variables (CNN decoder ring). |
| **Partial sum (psum)** | An intermediate accumulation result in the CONV inner loop; must be stored until the full sum over c, r, s is complete. |
| **Batch size (N)** | Number of images/samples processed simultaneously; N > 1 promotes FC to matrix–matrix multiply. |
| **Receptive field** | The region of the input fmap that contributes to one output activation; grows with depth and filter size. |
| **Cascade of Einsums** | A sequence of Einsums describing the full forward pass of a network (TeAAL workload specification). |
| **Stationarity** | Keeping a data operand stationary in a register across multiple MACs to exploit reuse (e.g., keeping B[k] in a register across the m-loop). |

---

## Takeaways

- The **TeAAL five-step methodology** (describe architecture → develop workload → evaluate → compare → optimize) is the structured approach to accelerator design used throughout this course.
- An **Einsum** precisely specifies what a DNN layer computes without imposing any loop order. The loop order is the **mapping** — a separate, hardware-critical decision.
- **Compute intensity (multiplications/value)** is the workload's fundamental characterization. The gap between best-case CI and achieved CI directly reflects unexploited data reuse.
- The **Roofline Model** reveals whether a design is **memory-bound** (CI too low → add reuse, not lanes) or **compute-bound** (CI high enough → add parallelism).
- **CONV layers dominate CNN computation (>90%)**: seven nested loops over N, M, P, Q, C, R, S; total work = N × M × P × Q × C × R × S MACs.
- **Stride** controls spatial downsampling; **zero padding** controls output size preservation. Both appear in the output-size formula P = (H − R + U) / U.
- An **FC layer is CONV with R = H, S = W** (filter covers the full input). With batch size N, it reduces to **matrix–matrix multiply** — a well-studied high-throughput kernel.

---

## Connections to Later Lectures

- **Einsums as the universal workload language** → **L03–L04** (memory traffic analysis, Einsum for other DNN operations including transformers and attention). The Einsum framework introduced here is the standard input to all modeling tools used in the labs.
- **Compute intensity and the Roofline Model** → **L03** (memory and metrics): the CI analysis begun here is extended to the full memory hierarchy and to realistic hardware bandwidth constraints.
- **CONV mapping — which loop to tile and in what order** → **L05–L06** (dataflows and partitioning). The seven-nested-loop CONV is exactly the loop nest whose order and tiling are the central design variables of those lectures.
- **Data reuse and stationarity** → **L05** (dataflows): weight-stationary, output-stationary, and input-stationary dataflows are strategies for keeping one of the three CONV tensors (weights F, output psum O, or input activations I) stationary in the register file to maximize reuse.
- **CONV operation counts and sparse CONV** → **L07–L10** (sparsity): once you know the dense operation count (N × M × P × Q × C × R × S), sparsity in weights or activations reduces the effective iteration space — but only if the hardware can find and skip the zeros.
- **FC as matrix multiply** → **L11–L13** (reduced precision, compute-in-memory): the matrix–matrix multiply structure of batched FC is the canonical kernel for in-memory computing and reduced-precision arithmetic demonstrations.

---

## Appendix — Slide-to-Section Map

| Slides | Section |
|---|---|
| L02-1 | Title |
| L02-2 | Outline |
| L02-3 … L02-8 | Ch.1 — TeAAL design methodology and separation of concerns |
| L02-9 … L02-20 | Ch.1 — Tensors, ranks, and Einsum definition (ODE) |
| L02-21 … L02-43 | Ch.1 — Workload evaluation: compute intensity, Roofline Model |
| L02-44 | Ch.1 — Full methodology recap |
| L02-45 … L02-52 | Ch.2 — CNN structure and layer types |
| L02-53 … L02-75 | Ch.3 — CONV layer anatomy, 2D worked example, stride, zero padding |
| L02-76 … L02-83 | Ch.3 — Multi-channel CONV (C inputs, M outputs), decoder ring, Einsum, loop nest |
| L02-84 … L02-91 | Ch.4 — FC layer definition and equivalence to CONV |
| L02-92 … L02-102 | Ch.4 — FC as matrix-vector and matrix-matrix multiply, batch Einsum |
