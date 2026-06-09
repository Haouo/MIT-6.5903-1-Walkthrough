# L12 — Precision

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze (MIT EECS)
> **Lecture date:** March 16, 2025 · **Slides:** 76 · **Source:** [`Lecture/L12 - Precision_r1.pdf`](../../Lecture/L12%20-%20Precision_r1.pdf)
>
> *This is a conceptual walkthrough that reconstructs the lecture's narrative from the slides. It is organized by idea, not slide-by-slide. Each section cites the slide range it draws from so you can follow along with the original deck.*

---

## TL;DR

Every bit you save in a neural network operand saves **energy twice over** — once for the arithmetic and once for the memory traffic. This lecture is about **reduced precision**: the systematic replacement of 32-bit floating-point numbers with narrower formats (FP16, bfloat16, INT8, INT4, and even 1-bit binary) and the hardware consequences of doing so. The lecture builds a precise mental model of *why* reducing bit-width cuts area and energy (O(n²) for multipliers), *how* to quantize values with minimal accuracy loss (uniform vs. non-uniform, post-training vs. quantization-aware), and *what* the industry has shipped — from Google's 8-bit integer TPU to Nvidia's mixed-precision GPU tensor cores to the open MX (Microscaling) standard. The recurring tension is the **accuracy–efficiency tradeoff**: each bit removed shrinks cost but risks degrading accuracy, and careful co-design of the model, the number format, and the hardware determines how far you can push.

---

## What Problem This Lecture Solves

Earlier lectures treated a DNN layer mostly as a loop nest: choose a dataflow, place tensors in the memory hierarchy, and try to reduce data movement. Lecture 12 asks a different question: **what if the values themselves do not need 32 bits?**

The naive approach is to keep every weight, activation, gradient, and partial sum in FP32 because FP32 is convenient, expressive, and familiar from software training. The hardware problem is that FP32 is wasteful for many inference tensors and sometimes for training tensors too. A 32-bit value consumes four times the storage of an 8-bit value, creates wider memory transfers, and requires a much larger multiplier. The lecture's goal is to teach when those extra bits are useful numerical information and when they are just cost.

The central hardware design question is:

> For each tensor and each phase of the workload, what is the narrowest representation that preserves the needed model behavior while reducing arithmetic, storage, and data-movement cost?

This is why reduced precision is a co-design topic. A hardware designer cannot choose INT4, FP8, or bfloat16 by looking only at the multiplier. The choice interacts with training recipe, activation statistics, accumulator growth, layer sensitivity, scale-factor granularity, and the memory hierarchy.

## Why This Lecture Matters

Precision is one of the few optimizations that changes both sides of the accelerator cost equation:

- It reduces **compute cost** because narrower multipliers and adders need less switching capacitance and less area.
- It reduces **memory cost** because fewer bits are read, written, buffered, and routed.
- It increases **effective on-chip capacity** because a fixed SRAM can hold more values.
- It can improve **throughput density** if a MAC can be spatially or temporally reused for multiple low-precision operations.
- It can hurt **accuracy or convergence** if quantization noise, clipping, overflow, underflow, or accumulator truncation is mishandled.

That last point is the boundary between reduced precision and earlier mapping/dataflow lectures. A poor dataflow may waste energy but should compute the same mathematical result. A poor precision choice may compute a different model.

## Prerequisites and Mental Model

Before reading the main narrative, keep four ideas in view:

1. **A number format is a budget.** A bit can buy sign, range, or resolution. You rarely get all three.
2. **Quantization is a lossy encoding unless the original values already lie on the quantization grid.** The engineering question is whether the loss matters for the model.
3. **A MAC has internal values, not just inputs and outputs.** Multiplying two 8-bit operands can produce a 16-bit product, and summing many products needs extra accumulator bits.
4. **Scale granularity is part of the format.** "INT8" is incomplete unless you say whether the scale is per tensor, per layer, per channel, per block, or per value.

The most useful mental model is a ruler. Uniform quantization lays equally spaced tick marks across a range. Floating-point moves the ruler by storing a per-value exponent. Dynamic fixed-point and MX formats share one ruler position across a group of values. Learned and log-domain quantization bend the ruler so that dense regions of the distribution receive more tick marks.

---

## Learning Objectives

After this lecture you should be able to:

- Quantify the **energy and area savings** from reducing operand bit-width, and explain why multiplier cost scales as O(n²) while adder cost scales as O(n).
- Define **quantization** and distinguish uniform from non-uniform schemes (log, learned/codebook, floating-point).
- Explain the roles of **mantissa bits (M)** and **exponent bits (E)** in floating-point formats, and compare FP32, FP16, bfloat16, INT8, and dynamic fixed-point on the range–precision tradeoff.
- Describe the **precision taxonomy** (uniform integer, fixed-point, log-domain, learned lookup-table, floating-point, dynamic fixed-point, per-layer/per-channel scaling, binary/ternary nets).
- Explain **mixed precision** strategies for both training (FP16 compute + FP32 master weights) and inference (INT8/INT4).
- Describe **precision-scalable MAC** architectures (spatial, temporal / bit-serial) and their area–throughput–energy tradeoffs.
- Summarize the **hardware ecosystem** (TPU, Nvidia Pascal/Ampere, NVDLA, MX standard, BitNet).

---

## Chapter 1 — Why Reduce Precision? The Energy and Area Case

> *Slides: L12-2 … L12-7*

### The co-design framing

Lecture 12 is the second of two co-design lectures (after sparsity). The key distinction the slides make at the outset: co-design approaches can **affect accuracy**, so every optimization must be evaluated against an accuracy–efficiency tradeoff — unlike purely architectural changes that are accuracy-neutral.

Reduced precision belongs to the category of **"reduce size of operands"**: shrinking the bit-width of weights, activations, and partial sums so that they consume less storage and less compute.

### The numbers: energy and area vs. bit-width

The most important table in this chapter (Horowitz, ISSCC 2014) shows the **energy cost (pJ) and silicon area (µm²)** for common arithmetic and memory operations:

Reading the table:

| Operation | Energy (pJ) | Area (µm²) |
|---|---|---|
| 8b Add | 0.03 | 36 |
| 16b Add | 0.05 | 67 |
| 32b Add | 0.1 | 137 |
| 16b FP Add | 0.4 | 1,360 |
| 32b FP Add | 0.9 | 4,184 |
| 8b Mult | 0.2 | 282 |
| 32b Mult | 3.1 | 3,495 |
| 16b FP Mult | 1.1 | 1,640 |
| 32b FP Mult | 3.7 | 7,700 |
| 32b SRAM Read (8 KB) | 5 | — |
| 32b DRAM Read | 640 | — |

Three observations stand out:
1. **Integer multiply energy and area scale roughly as O(n²) with bit-width.** Going from 8-bit to 32-bit integer multiply costs ~15× more energy and ~12× more area.
2. **Floating-point is expensive.** A 32-bit FP multiplier costs ~3.7 pJ and 7,700 µm² — about **18× more energy and 27× more area** than an 8-bit integer multiply.
3. **Memory dominates.** A 32-bit DRAM read is 640 pJ — more than **170× a 32-bit integer multiply**. Reducing bit-width shrinks memory traffic, not just arithmetic cost.

### Propagation through the MAC

The MAC (multiply-accumulate) has three operands with different precision requirements: input activations (n_i bits), weights (n_f bits), and the partial sum accumulator. Because the accumulator sums up to RSC (receptive field spatial cardinality) partial products, it needs **⌈log₂(RSC)⌉ additional bits** to avoid overflow without loss of precision.

For real networks: AlexNet (RSC up to 9,216) needs 14 extra bits; VGG-16 (RSC up to 25,088) needs 15 extra bits. This is why accumulators are often kept at 32 bits even when inputs are 8 bits — the accumulator is the internal "headroom" for the operation.

Small example: suppose an activation and a weight are both signed 8-bit values. A single product may need about 16 bits. If one output accumulates \(K = 9\) such products, a conservative lossless accumulator needs about \(16 + \lceil \log_2 9 \rceil = 20\) bits. If \(K = 1024\), it needs about \(16 + 10 = 26\) bits. This is the reason "INT8 inference" often means **8-bit inputs and weights, but 32-bit accumulation**. The stored operands are narrow; the internal running sum is wider to preserve correctness.

> **Why it matters:** Reducing operand bit-width from 32 to 8 cuts integer multiplier energy by ~15× and area by ~12×, and it also reduces memory traffic — both reads and writes. These are the two biggest costs in a DNN accelerator. Every technique in this lecture is a different strategy for achieving those savings while preserving accuracy.

---

## Chapter 2 — Quantization: The Core Idea

> *Slides: L12-10 … L12-15*

### What quantization is

**Quantization** maps a real-valued distribution of numbers to a finite set of discrete **quantization levels** {q₀, q₁, …, q_{L-1}}, separated by **decision boundaries** {d₁, d₂, …, d_{L-1}}. The goal is to minimize the **quantization error** (the difference between original and reconstructed values) subject to the constraint of using only L levels.

The number of bits required is log₂(L). "Reduced precision" means reducing L (and hence the bit count), trading finer representation for smaller storage and compute. The optimal placement of levels and boundaries depends on the **probability density function** of the data — an insight that motivates non-uniform quantization.

### Range selection: symmetric vs. asymmetric, and clipping

Two modes of mapping the real-valued range to quantization levels:

- **Symmetric mode**: the quantization range is centered at zero. Simpler hardware (no zero-point offset).
- **Asymmetric mode**: the range is shifted to best cover the actual data distribution, using a zero-point offset. Better utilizes the full bit-width for skewed distributions (e.g., post-ReLU activations, which are non-negative).

**Clipping (saturation)** deliberately excludes outliers:
- **DoReFa**: clips activations to [0, 1].
- **ReLU6**: clips activations to [0, 6].
- **PACT**: clips activations to [0, α], where α is a **learned parameter** (trained jointly with the network weights).

Clipping sacrifices a small amount of dynamic range to buy higher resolution for the bulk of the distribution.

Worked example: take symmetric 4-bit signed quantization over \([-1, 1]\). If we reserve integer codes \(-7, \ldots, 7\), the scale is \(s = 1/7\). A real value \(x = 0.37\) becomes \(q = \mathrm{round}(x/s) = \mathrm{round}(2.59) = 3\), and reconstructs to \(\hat{x} = q s \approx 0.429\). The quantization error is \(\hat{x} - x \approx 0.059\). If the tensor contains a rare outlier \(x = 4.0\), expanding the range to cover it would make the scale \(4/7\), so values near zero would be much coarser. Clipping the outlier may improve the average error for the dense part of the distribution.

> **Why it matters:** The accuracy of quantization depends not just on the number of bits but on *how* those bits are allocated across the value range. Poor range selection wastes levels on rarely-occurring extreme values.

---

## Chapter 3 — A Taxonomy of Number Formats

> *Slides: L12-16 … L12-36*

### Numerical format anatomy

Every numerical representation has three components: a **sign (S)**, a **mantissa/significand (M)** encoding the number of unique values within a scale, and (for floating-point) an **exponent (E)** encoding the scale. Total bits = n_S + n_E + n_M.

| Format | S | E | M | Range |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | ~10⁻³⁸ to 10³⁸ |
| FP16 | 1 | 5 | 10 | ~6×10⁻⁵ to 6×10⁴ |
| bfloat16 | 1 | 8 | 7 | ~10⁻³⁸ to 3×10³⁸ |
| Int32 | 1 | 0 | 31 | 0 to 2×10⁹ |
| Int8 | 1 | 0 | 7 | 0 to 127 |

The key design choice is how to **allocate bits between E and M**:
- More **E bits** → wider dynamic range (important for gradients in training, which span many orders of magnitude).
- More **M bits** → finer resolution within a range (important for inference accuracy with quantized weights).

### bfloat16: the training-optimized format

**bfloat16** keeps 8 exponent bits (equal to FP32) but reduces the mantissa to 7 bits. This gives **the same dynamic range as FP32** at half the storage cost, which is why it is preferred for training (gradients need wide range). FP16, by contrast, has only 5 exponent bits — it runs out of range for gradients in some models.

### Fixed-point: when range is bounded

When the value range is known a priori, you can eliminate the exponent entirely and use a **fixed-point format**: a sign bit plus a mantissa, with the binary point at a pre-determined (fixed) position. An 8-bit fixed-point number represents values from −128 to +127 as integers, or sub-integer values depending on where the binary point is placed.

Fixed-point hardware is simpler (no exponent alignment logic) and cheaper. The tradeoff is inflexibility: the same scale factor applies to all values.

### Dynamic fixed-point: the middle ground

**Dynamic fixed-point** (a.k.a. block floating-point) shares a single scale factor **f** across a *group* of values (e.g., all weights in a layer or a channel). Within the group, each value is stored as a fixed-point mantissa. The scale factor is stored once and amortized over the group.

This is strictly between fixed-point (one global scale) and floating-point (one scale per value). The scale can differ across layers, data types (weights vs. activations), and channels — but not per-value. The result is high accuracy with modest storage overhead.

Small example: a channel's weights might be stored as 8-bit integers with shared scale \(2^{-5}\). The stored code \(q = -13\) reconstructs to \(-13 \cdot 2^{-5} = -0.40625\). Another channel may use scale \(2^{-7}\), giving finer resolution but narrower range. The arithmetic unit can still operate on integers; the scale is accounted for when aligning products, accumulating, or dequantizing outputs.

### The mantissa–exponent tradeoff

The lecture source directly comparing fp16 and bfloat16 makes the tradeoff concrete:

- **fp16** (S=1, E=5, M=10): range ~5.9×10⁻⁸ to ~6.5×10⁴ — fine precision, limited range.
- **bfloat16** (S=1, E=8, M=7): range ~1×10⁻³⁸ to ~3×10³⁸ — wide range, coarser precision.

The slides extend this to **AdaptivFloat** (Tambe, DAC 2020), which makes the exponent bias configurable per layer, and **CFloat** (Tesla Dojo 2021), which makes the partition between M and E bits fully configurable.

### Microscaling (MX) formats

A consortium (AMD, ARM, Intel, Meta, Microsoft, Nvidia, Qualcomm) standardized **MX (Microscaling) data formats** in 2023. MX formats share a single scaling factor across a *block* of narrow-precision elements (e.g., MXFP8, MXFP6, MXFP4, MXINT8). This is essentially block floating-point applied at fine granularity, enabling:
- Inference with MXINT8/MXFP8 on FP32-pretrained models with minimal accuracy loss.
- Training with MXFP6 weights, activations, and gradients with minimal loss (without changing the training recipe).
- Training with MXFP4 weights and MXFP6 activations/gradients with only minor accuracy loss.

> **Why it matters:** The format determines how many bits you need and what dynamic range you preserve. The proliferation of formats (FP32 → FP16 → bfloat16 → INT8 → MX formats → FP4) reflects the ongoing search for the sweet spot on the accuracy–efficiency Pareto front.

---

## Chapter 4 — Non-Uniform Quantization

> *Slides: L12-23 … L12-30*

### Motivation: not all distributions are uniform

Uniform quantization spaces levels equally. But DNN weight distributions are often **non-uniform** — concentrated near zero with heavy tails. Non-uniform quantization allocates more levels where the distribution is dense and fewer where it is sparse, reducing average quantization error for the same number of bits.

### Log-domain quantization

**Logarithmic quantization** maps the mantissa bits to a logarithmic scale. The key hardware benefit: a **multiplication in the log domain becomes an addition** (and an addition becomes a bitshift and comparison). In the lecture source, when both weights and activations are represented in the log domain, multiply-accumulate becomes shift-and-add — a fundamentally cheaper hardware operation.

The LogNet (Lee, ICASSP 2017; Miyashita, arXiv 2016) demonstrated:
- 5-bit weights (CONV), 4-bit weights (FC), 4-bit activations on AlexNet.
- Accuracy loss: only **3.2%** top-1 on AlexNet.
- Hardware: shift-and-add instead of multiplier.

### Learned (codebook) quantization

**Weight sharing** (Han, ICLR 2016) applies k-means clustering to find U representative values per layer. Each weight stores an index (log₂U bits) into a codebook of U full-precision weights. Example results on AlexNet with **no accuracy loss**:
- CONV layers: 256 unique weights per layer (8 bits for the index).
- FC layers: 16 unique weights per layer (4 bits for the index).

Hardware implication: the weight memory reads a narrow index; a small dequantization table maps the index to the full-precision weight value before the MAC. This reduces weight **storage** but does not reduce the precision of the MAC itself — the multiplier still operates at full precision.

The precision taxonomy that emerges:

- **Uniform**: direct integer, fixed-point.
- **Non-uniform constrained**: log-domain (a function of the binary value).
- **Non-uniform unconstrained**: learned codebook (arbitrary mapping, implemented as a lookup table).
- **Scaled binary**: floating-point, dynamic fixed-point, MX formats.
- **Shared hardware**: precision-scalable MACs (varying mantissa width).

> **Why it matters:** Non-uniform quantization can deliver better accuracy than uniform quantization at the same bit-width, at the cost of additional hardware (shift logic for log; a lookup table for codebook). The tradeoff is between hardware simplicity and quantization fidelity.

---

## Chapter 5 — Accuracy Impact and Mixed Precision

> *Slides: L12-21 … L12-22, L12-35, L12-48, L12-70 … L12-74*

### Accuracy impact of dynamic fixed-point

The lecture source reports measured top-1 accuracy of CaffeNet on ImageNet as the bit-width is reduced from 16 bits to as few as 2 bits, with and without fine-tuning:

Key findings:
- **8-bit dynamic fixed-point without fine-tuning**: 0.4% accuracy loss vs. 32-bit float.
- **8-bit dynamic fixed-point with fine-tuning**: 0.6% accuracy loss.
- Fine-tuning (retraining with quantized arithmetic) is critical as bit-widths drop below 8.

### The comprehensive accuracy summary

The lecture's summary table (slide 69) is the single most useful quantitative reference:

| Category | Method | Weights (bits) | Activations (bits) | Accuracy loss (%) |
|---|---|---|---|---|
| Uniform | Dynamic Fixed Point (w/o fine-tuning) | 8 | 10 | 0.4 |
| Uniform | Dynamic Fixed Point (w/ fine-tuning) | 8 | 8 | 0.6 |
| Ternary | TWN | 2* | 32 | 3.7 |
| Ternary | TTQ | 2* | 32 | 0.6 |
| Binary | Binary Connect (BC) | 1 | 32 | 19.2 |
| Binary | Binary Weight Net (BWN) | 1* | 32 | 0.8 |
| Binary | Binarized Neural Net (BNN) | 1 | 1 | 29.8 |
| Binary | XNOR-Net | 1* | 1 | 11 |
| Non-uniform | LogNet | 5(conv), 4(fc) | 4 | 3.2 |
| Non-uniform | Weight Sharing | 8(conv), 4(fc) | 16 | 0 |

*\* first and last layers are kept at 32-bit float*

The pattern is clear: **full binarization of both weights and activations** (BNN) causes catastrophic accuracy loss on a non-trivial task like ImageNet; **binarizing only weights** (BWN, with scaling) keeps loss under 1%; **8-bit uniform quantization with fine-tuning** is essentially lossless.

### Mixed precision for training

Training with reduced precision is harder than inference because **gradients** have large dynamic ranges that vary across layers. The standard industry solution (Narang, ICLR 2018) uses **FP16 for forward and backward computations** but keeps **FP32 master weights** for the weight update. This avoids gradient underflow while halving memory for the compute-intensive passes.

For inference, 8-bit integer is already industry-standard. Research has pushed to 4-bit:
- **HFP8** (Sun, NeurIPS 2019): forward pass uses E=4, M=3; backward uses E=5, M=2.
- **FP4 training** (Sun, NeurIPS 2020): gradients use a radix-4 logarithmic format with per-layer trainable scale factors and two-phase rounding.

### Varying precision across layers

Not every layer needs the same precision. The lecture source reports that varying precision across layers (e.g., using 4-bit for some layers, 8-bit for others) can achieve a better accuracy–efficiency tradeoff than applying a uniform low bit-width everywhere. Early and late layers (first and last) are consistently kept at higher precision across virtually every method in the summary table.

> **Why it matters:** 8-bit integer for inference is mature and industry-standard. The frontier is **4-bit and below**, where quantization-aware training, per-channel scaling, and careful format design are necessary to preserve accuracy. The summary table is the empirical scorecard for that frontier.

---

## Chapter 6 — Hardware for Reduced Precision

> *Slides: L12-37 … L12-58*

### Industry hardware: from TPU to MX tensor cores

The lecture surveys how industry has shipped reduced-precision hardware:

- **Google TPU (Jouppi, ISCA 2017)**: 8-bit integer MACs with 32-bit accumulators. The design choice was deliberate: cut precision rather than throw bandwidth at the problem.
- **Nvidia Pascal (2016)**: first GPU to add 16-bit FP (half-precision) tensor instructions (>21 TFLOPS) and 8-bit integer inference instructions (47 TOPS).
- **Nvidia Ampere and later**: mixed-precision tensor cores supporting FP64, FP32, TF32, FP16, bfloat16, INT8, INT4.
- **NVDLA**: supports Binary/INT4/INT8/INT16/INT32/FP16/FP32/FP64.
- **Microsoft BrainWave**: custom 8-bit and 9-bit floating-point for RNN/LSTM inference on FPGAs.
- **Intel Nervana (FlexPoint)**: custom format for training.
- **TPU v2 & v3**: bfloat16 for training.
- **Nvidia NVFP4** (2024): 4-bit floating-point for inference, delivering higher compute throughput with carefully managed quantization overhead.
- **DeepSeek-V3**: validated FP8 mixed-precision training at 671B parameter scale.

### Precision-scalable MACs

The key hardware challenge: can a single multiplier unit serve **different bit-widths** without requiring a separate physical multiplier for each precision? The answer is yes, via **precision-scalable MACs**.

The lecture classifies them into three families:

**Spatial precision-scalable MACs**: reconfigure the wiring of the partial-product tree to serve multiple precisions. At 8×8 bits, all adders are used; at 4×4 bits, you run multiple 4×4 multiplications in parallel using different portions of the same adder tree. The gain is **throughput at lower precision** (more parallel operations per cycle), without a separate multiplier.

**Temporal precision-scalable MACs (bit-serial)**: process one bit-plane of the operand per clock cycle, accumulating partial results. Reducing bit-width reduces the number of cycles — a direct speed-up proportional to the bit-width reduction. Stripes (Judd, MICRO 2016) demonstrated a **1.92× speedup** vs. 16-bit fixed on AlexNet using bit-serial processing.

**Power via voltage scaling**: with shorter critical paths (from fewer bits), the MAC can run at a **lower supply voltage**, cutting dynamic power. Moons (VLSI 2016) showed a **2.5× power reduction** on AlexNet Layer 2 vs. 16-bit fixed.

The Camus (JETCAS 2019) benchmark of 19 precision-scalable MAC designs found that adding logic to increase utilization at lower precision can *reduce* net benefits when the precision distribution is skewed (e.g., 95% of values at 2- or 4-bit, 5% at 8-bit). The conventional data-gated design achieved 1.3× energy reduction; adding spatial decomposition pushed to 1.6× — but the additional complexity had diminishing returns.

### Binary and ternary nets — and LLMs

The extreme case of reduced precision:

- **Binary Connect** (Courbariaux, NeurIPS 2015): weights ∈ {−1, +1}, activations at 32-bit float. MAC becomes addition/subtraction; no multiplier. Accuracy loss: 19% on AlexNet.
- **Binary Weight Net (BWN)** (Rastegari, ECCV 2016): weights ∈ {−α, +α} with a per-filter scale factor α (determined by the ℓ₁-norm of filter weights). Accuracy loss: only **0.8%** on AlexNet.
- **Binarized Neural Net (BNN)** (Courbariaux, arXiv 2016): both weights and activations ∈ {−1, +1}. MAC becomes XNOR-popcount — a tiny gate. Accuracy loss: 29.8%.
- **XNOR-Net** (Rastegari, ECCV 2016): weights ∈ {−α, +α} and activations ∈ {−βᵢ, +βᵢ} with per-position scale factors. Accuracy loss: 11%.
- **Ternary Weight Net (TWN)**: weights ∈ {−w, 0, +w} adding a zero value, increasing sparsity. Accuracy loss: 3.7%.
- **Trained Ternary Quantization (TTQ)**: weights ∈ {−w₁, 0, +w₂} with asymmetric learned thresholds. Accuracy loss: 0.6%.

**BitNet** (Microsoft 2024) extends binarization to **large language models**: 1-bit weights (BitNet) and ~1.58-bit weights (Ternary BitNet, which is log₂(3) bits), both with 8-bit activations, trained from scratch. This shows that extreme precision reduction is viable even at LLM scale — a significant finding given the memory and energy cost of serving LLMs.

> **Why it matters:** The hardware design space mirrors the numerical format design space: spatial reconfigurability gives throughput, temporal bit-serial gives proportional speedup, and the ability to gate unused logic gates gives energy reduction. The challenge is that adding flexibility always adds some overhead, so the net benefit depends on the actual precision distribution of the workload.

---

## Worked Examples

### Example 1: Accumulator Headroom

Suppose a convolution output sums \(R \times S \times C = 3 \times 3 \times 64 = 576\) products. If each product is kept in \(16\) bits, a lossless accumulator needs an additional \(\lceil \log_2 576 \rceil = 10\) bits, for about \(26\) bits total. A 16-bit accumulator would not be safe for the worst case even though the input operands are only 8 bits.

Hardware implication: the PE datapath may use an 8-bit multiplier but a wider accumulator register file. This increases PE storage and routing width, so accumulator precision is a first-class hardware parameter, not a footnote.

### Example 2: Per-Tensor vs. Per-Channel Scale

Imagine two output channels. Channel A has weights mostly in \([-0.25, 0.25]\); channel B has weights in \([-2, 2]\). A single per-tensor INT8 scale must cover \([-2, 2]\), so the quantization step is roughly \(2/127\). Channel A then uses only a small part of the code range. Per-channel scaling lets channel A use a much smaller step, roughly \(0.25/127\), improving resolution without changing the stored bit-width.

Hardware implication: per-channel scale improves accuracy but adds scale storage and usually extra multiply/shift logic at channel boundaries. It is a classic model-hardware tradeoff: better numerical fit in exchange for more metadata and control.

### Example 3: Spatial vs. Temporal Precision Scaling

A spatial precision-scalable MAC may split an 8-bit by 8-bit multiplier into several lower-bit multipliers and complete multiple 4-bit operations in one cycle. A bit-serial MAC instead consumes one or a few bit planes per cycle; reducing precision from 8 bits to 4 bits reduces the number of cycles. Both exploit low precision, but they stress different parts of the design: spatial scaling needs more output/input bandwidth when it produces more results per cycle, while temporal scaling trades latency and control for smaller arithmetic.

---

## Key Equations and How to Read Them

- Number of quantization levels: \(L = 2^b\), where \(b\) is the number of stored bits. This equation measures representation capacity, not accuracy. Two 8-bit formats can have the same \(L\) but very different ranges and error distributions.
- Uniform affine quantization: \(q = \mathrm{clip}(\mathrm{round}(x/s) + z)\), with reconstruction \(\hat{x} = s(q - z)\). Here \(s\) is the scale and \(z\) is the zero-point. Symmetric quantization often uses \(z = 0\); asymmetric quantization uses \(z\) to represent shifted distributions more efficiently.
- Accumulator growth: \(b_{\mathrm{acc}} \ge b_{\mathrm{product}} + \lceil \log_2 K \rceil\), where \(K\) is the number of products summed. This is a worst-case correctness bound; real designs may use saturation, rounding, or statistical assumptions, but those choices must be explicit.
- Floating-point bit budget: \(b = 1 + E + M\). The sign bit stores polarity, \(E\) controls dynamic range, and \(M\) controls local resolution. bfloat16 and FP16 both use 16 bits, but spend the exponent/mantissa budget differently.

---

## Hardware Implications

- **Energy:** Narrower arithmetic reduces switching activity, and narrower values reduce SRAM/DRAM bitline activity. The Horowitz numbers in L12-3 are slide-stated anchors for this intuition.
- **Area:** A narrower multiplier can be much smaller; this can be spent on more MACs, more local storage, or lower cost.
- **Bandwidth:** INT8 halves FP16 traffic and quarters FP32 traffic before compression. This changes whether a design is compute-bound or memory-bound.
- **Buffer capacity:** A 128 KiB SRAM holds four times as many INT8 values as FP32 values. Precision therefore changes mapping feasibility, not just arithmetic cost.
- **Accumulator design:** Low-precision inputs do not imply low-precision partial sums. Accumulator width affects PE register files and interconnect width.
- **Programmability:** Variable precision requires metadata: scale factors, zero-points, per-layer precision choices, and sometimes calibration statistics.
- **Verification and correctness:** Saturation, rounding mode, NaN/Inf behavior, denormal handling, and overflow policy become observable hardware-software contracts.

---

## Common Misconceptions

### Misconception: "INT8" fully specifies the computation.

It does not. A complete INT8 specification must say the scale, zero-point, whether scaling is per tensor/channel/block, accumulator width, rounding mode, saturation policy, and where requantization occurs.

### Misconception: Lower precision always increases speed.

Only if the hardware can exploit it. A fixed 8-bit MAC running 4-bit operands may save some switching energy through gating, but it does not automatically produce twice as many operations per cycle. Throughput gains require spatial packing, temporal bit-serial shortening, or another precision-scalable mechanism.

### Misconception: Non-uniform quantization is always better.

Non-uniform levels can reduce quantization error, but the decoder, lookup table, shift logic, metadata, and irregularity may erase the savings. It is useful only when the accuracy improvement or storage reduction pays for the extra hardware.

### Misconception: If weights are low precision, activations and accumulators can be low precision too.

Weights, activations, gradients, and partial sums have different distributions and error sensitivity. Many successful systems use low-precision weights/activations with higher-precision accumulators or master weights.

---

## Paper Bridge

### Paper Bridge: In-Datacenter Performance Analysis of a Tensor Processing Unit

#### Bibliographic identity

- Title: *In-Datacenter Performance Analysis of a Tensor Processing Unit*
- Authors: Norman P. Jouppi et al.
- Year / venue: ISCA 2017
- Used in this lecture: L12's discussion of 8-bit integer inference hardware and industrial reduced-precision design.

#### Problem addressed

The TPU paper asks whether a domain-specific inference accelerator can deliver better latency, throughput, and energy efficiency than contemporary CPUs and GPUs for production datacenter neural-network workloads.

#### Core idea

The first-generation TPU uses a systolic matrix multiply unit with \(256 \times 256 = 65{,}536\) 8-bit MACs, 32-bit accumulators, and a large software-managed on-chip memory. Reduced precision is not an isolated arithmetic trick; it is what lets the chip pack many MACs and enough on-chip storage into a low-power inference accelerator.

#### Relevance to this lecture

The paper is the industrial example behind the lecture's claim that 8-bit integer inference is a mature hardware point. It also illustrates why accumulator precision remains wider than operand precision.

#### Key claims used in this chapter

- Abstract and Section 2 state that the TPU's matrix multiply unit contains 65,536 8-bit MACs, reaches 92 TOPS peak, and uses 28 MiB of software-managed on-chip memory.
- Section 1 says quantization transforms floating-point NN values into narrow integers, often 8-bit, for inference; it also states that 8-bit integer multiply can be much lower energy and area than 16-bit floating-point multiply, citing prior circuit data.
- Section 2 states that the matrix unit accumulates 16-bit products into 32-bit accumulators and that mixed 8/16-bit operation reduces throughput relative to pure 8-bit operation.
- Section 4 uses a TPU-specific roofline model with operational intensity defined in integer operations per byte of weights read, showing that memory bandwidth remains a limiter even with reduced precision.

#### What students should remember

- "8-bit inference" in a real accelerator usually means narrow stored operands plus wider internal accumulation.
- Reduced precision enabled the TPU to allocate die area to a large systolic array and on-chip buffers.
- Precision and memory hierarchy are coupled: smaller operands help, but workload operational intensity still determines whether the accelerator is bandwidth-bound.

#### Limitations and assumptions

The first-generation TPU was designed for inference workloads around 2015 and did not target training. It also omitted sparse architectural support for time-to-deploy reasons, so it should not be treated as a universal DNN-accelerator template.

#### Suggested insertion points

Use this paper when reading Chapter 6's TPU discussion, Chapter 1's energy/area motivation, and the accumulator-headroom example.

### Paper Bridge: Review and Benchmarking of Precision-Scalable MAC Architectures

#### Bibliographic identity

- Title: *Review and Benchmarking of Precision-Scalable Multiply-Accumulate Unit Architectures for Embedded Neural-Network Processing*
- Authors: Vincent Camus, Linyan Mei, Christian Enz, and Marian Verhelst
- Year / venue: IEEE JETCAS 2019
- Used in this lecture: L12's precision-scalable MAC taxonomy.

#### Problem addressed

Many accelerators claim support for variable precision, but their MAC designs are implemented in different processes, with different assumptions and benchmarks. The paper tries to make the design space comparable.

#### Core idea

The paper classifies precision-scalable MACs by whether they exploit low precision spatially or temporally, and by how partial products are accumulated. It introduces the Sum Apart (SA) and Sum Together (ST) distinction, then benchmarks representative designs in a common 28 nm CMOS flow.

#### Relevance to this lecture

Lecture 12 explains that lower precision is valuable only when the hardware can exploit it. This paper provides the microarchitectural bridge from "the model can use fewer bits" to "the MAC array can gain throughput, energy, or area efficiency."

#### Key claims used in this chapter

- Section II introduces the SA/ST taxonomy and ties precision scalability to dataflow and PE-array behavior.
- Section II-D and Table I organize precision-scalable MACs into spatial and temporal categories.
- Section V analyzes bandwidth, throughput, area, and energy, showing that precision-scalable designs can introduce bandwidth and area overheads.
- Section VII and Figure 30 show that the best MAC depends on the precision distribution of the workload, not simply on the lowest available bit-width.

#### What students should remember

- Spatial scaling can turn one full-precision MAC into multiple low-precision operations per cycle, but it may stress input/output bandwidth.
- Temporal or bit-serial scaling reduces cycles when fewer bit planes are used, but it changes latency and control.
- Flexibility has overhead. A precision-scalable MAC is worthwhile only if the workload spends enough time in the reduced-precision modes it accelerates.

#### Limitations and assumptions

The paper focuses on MAC-unit architectures and circuit-level benchmarking. A full accelerator still needs memory hierarchy, dataflow mapping, compiler support, and model-level precision selection.

#### Suggested insertion points

Use this paper when studying Chapter 6's spatial and temporal precision-scalable MAC discussion and Exercise 3.

### Paper Bridge: Eyeriss and Precision as a Data-Movement Co-Design Variable

#### Bibliographic identity

- Title: *Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks*
- Authors: Yu-Hsin Chen, Joel Emer, and Vivienne Sze
- Year / venue: ISCA 2016
- Related local source: `papers/L12_Eyeriss_Chen_ISCA2016.pdf`; the JSSC follow-up is also available locally.
- Used in this lecture: as context for how precision interacts with dataflow and memory hierarchy.

#### Problem addressed

Eyeriss asks how a spatial CNN accelerator should minimize data movement across DRAM, global buffer, inter-PE communication, and RF storage.

#### Core idea

The row-stationary dataflow tries to exploit reuse for weights, activations, and partial sums simultaneously, under equal hardware-resource constraints. Although Eyeriss is mainly a dataflow paper, it is relevant to precision because operand width changes every access count's byte cost and changes how much data fits in each storage level.

#### Relevance to this lecture

Reduced precision and dataflow are not separable. If values are narrower, the same buffer can hold more live data; if weights are much narrower than activations, a dataflow that previously prioritized weight reuse may no longer be optimal.

#### Key claims used in this chapter

- Abstract and Section I state that data movement can dominate CNN accelerator energy and that row-stationary reduces movement across the hierarchy.
- Section II describes the spatial-architecture memory hierarchy: DRAM, global buffer, inter-PE communication, and RF.
- Section VII reports that row-stationary is more energy efficient than compared dataflows under the paper's hardware constraints.
- The JSSC Eyeriss paper states the fabricated chip uses 16-bit fixed-point precision, which is a useful historical anchor for pre-INT8 CNN accelerators.

#### What students should remember

- Precision changes the cost per access; dataflow changes the number of accesses. Hardware efficiency depends on both.
- The right precision choice may change the best mapping choice because it changes the relative cost of weights, activations, and partial sums.
- A chapter about precision still needs the memory-hierarchy intuition from earlier lectures.

#### Limitations and assumptions

Eyeriss is not primarily a quantization paper. It should be used here as a bridge between precision and data movement, not as evidence for a specific low-bit quantization method.

#### Suggested insertion points

Use this paper when connecting L12 to L05/L06 mapping and when reasoning about how reduced precision affects buffer capacity and dataflow selection.

## Standalone Study Guide

### What to master before moving on

- Quantify why reducing bit-width saves arithmetic energy, arithmetic area, and memory traffic.
- Distinguish integer, fixed-point, floating-point, dynamic fixed-point, MX, log-domain, and codebook formats.
- Explain why accumulators often need more bits than activations and weights.
- Compare post-training quantization, quantization-aware training, and mixed precision.
- Describe spatial and temporal precision-scalable MACs.

### Self-check questions

1. Why does multiplier cost scale more steeply with bit-width than adder cost?
2. Why is bfloat16 better suited than FP16 for some training workloads?
3. Why can learned codebook quantization reduce storage without reducing MAC precision?

### Exercises

1. For a layer with RSC = 1024, compute the minimum extra accumulator bits needed to avoid lossless overflow.
2. Compare INT8, FP16, bfloat16, and MXFP8 for training versus inference. State the range/precision trade-off.
3. Choose a binary or ternary method from the summary table and explain what hardware it simplifies and what accuracy cost it pays.

### Common traps

- Saying "8-bit" without specifying integer, fixed-point, floating-point, scale granularity, and accumulator precision.
- Treating quantization as only a hardware issue. Low-bit accuracy usually depends on the training or fine-tuning recipe.
- Forgetting first and last layers, gradients, and accumulators often need special precision treatment.

---

## Key Terms

| Term | Gloss |
|---|---|
| **Quantization** | Mapping a real-valued distribution to a discrete set of L quantization levels, using ⌈log₂L⌉ bits. |
| **Quantization level** | One of L discrete values (qᵢ) that the quantized representation can take. |
| **Quantization error** | The difference between the original real value and its quantized reconstruction. |
| **Fixed-point** | Format with a sign bit, mantissa bits, and a pre-fixed binary point position; no per-value exponent. |
| **Floating-point (FP)** | Format with sign (S), exponent (E), and mantissa (M); the binary point position varies per value. |
| **FP32** | IEEE 754 single precision: S=1, E=8, M=23. The conventional baseline. |
| **FP16** | IEEE 754 half precision: S=1, E=5, M=10. Narrower range than FP32. |
| **bfloat16** | Brain float 16: S=1, E=8, M=7. Same range as FP32, coarser precision. Preferred for training. |
| **INT8 / INT4** | 8-bit / 4-bit integer formats; no exponent. Standard for inference. |
| **Dynamic fixed-point** | Block floating-point: a shared scale factor per group (layer/channel), fixed-point mantissa per value. |
| **Uniform quantization** | Quantization with equally-spaced levels (integer, fixed-point). |
| **Non-uniform quantization** | Quantization with unequally-spaced levels (log-domain, codebook/learned). |
| **Log-domain quantization** | Quantization on a logarithmic scale; converts multiplications to bitshifts. |
| **Weight sharing** | Learned codebook (k-means) quantization: many weights share a single representative value. |
| **Post-training quantization (PTQ)** | Quantizing a pre-trained model without retraining; fast but less accurate at low bit-widths. |
| **Quantization-aware training (QAT)** | Simulating quantization during training to minimize accuracy loss; required below ~4 bits. |
| **Mixed precision** | Using different bit-widths for different tensors or layers (e.g., FP16 compute + FP32 master weights). |
| **Symmetric / asymmetric quantization** | Whether the quantization range is centered at zero (symmetric) or shifted (asymmetric). |
| **Clipping / saturation** | Restricting the quantization range to exclude outliers (e.g., ReLU6, PACT). |
| **Per-tensor / per-channel scaling** | One scale factor for the whole tensor vs. one per output channel. Per-channel is more accurate. |
| **Binary / ternary nets** | Extreme reduction: weights ∈ {−1,+1} (binary) or {−w,0,+w} (ternary). |
| **XNOR-Net** | Binary-weight, binary-activation network where MAC becomes XNOR-popcount. |
| **MX (Microscaling)** | Industry-standard block-floating-point formats (MXFP8, MXFP6, MXFP4, MXINT8) for narrow precision. |
| **Bit-serial / temporal MAC** | MAC that processes one bit-plane per cycle; lower bit-width → fewer cycles → linear speedup. |
| **Precision-scalable MAC** | A single multiplier that reconfigures to serve multiple bit-widths (spatial or temporal decomposition). |
| **RSC** | Receptive field spatial cardinality; bounds the number of bits needed in the partial-sum accumulator. |

---

## Takeaways

- Reducing operand bit-width from 32 to 8 bits cuts **integer multiplier energy by ~15× and area by ~12×**, and also cuts memory traffic — both the dominant costs in a DNN accelerator.
- **8-bit integer for inference** is industry-standard and essentially lossless (< 1% accuracy degradation with fine-tuning). **4-bit training** is active research.
- The three-way format choice — **integer vs. fixed-point vs. floating-point** — trades off hardware simplicity against dynamic range. **bfloat16** (wide range) for training and **INT8** (simple hardware) for inference is the dominant industry pattern.
- **Non-uniform quantization** (log-domain, learned codebook) fits the data distribution better than uniform quantization at the same bit-width, at the cost of additional hardware (shift/lookup logic).
- **Quantization-aware training (QAT)** is necessary below ~4 bits; **post-training quantization (PTQ)** is sufficient above ~8 bits for most models.
- **Binary and ternary nets** achieve extreme compression (1–2 bits per weight) but incur large accuracy losses without careful scaling factors and QAT.
- **Precision-scalable MACs** (spatial and temporal/bit-serial) enable a single hardware design to serve multiple bit-widths, trading design complexity for flexibility.
- The **MX standard** and formats like NVFP4 signal that 4-bit and sub-8-bit is the emerging frontier, validated at LLM scale (DeepSeek-V3 with FP8, BitNet with 1-bit weights).

---

## Connections to Later Lectures

- **Data-attribute-specific optimizations (L01):** Reduced precision is the second of the two data-attribute optimizations in the TeAAL pyramid (the first is sparsity, L07–L10). Both cut data-movement cost by making representations smaller; precision also cuts arithmetic cost.
- **Sparsity (L07–L10):** Binary and ternary nets naturally introduce sparsity (zero weights in ternary nets). The two co-design axes interact: sparser + lower-precision networks compound the savings.
- **Advanced technologies (L11):** L11 covers compute-in-memory, which inherently operates with analog or near-digital precision — making quantization central to its accuracy analysis.
- **Dataflows and mapping (L05–L06):** Mixed precision interacts with dataflow choice; the UNPU accelerator (Lee, JSSC 2019) uses input-stationary dataflow specifically because its weights are reduced precision, changing the relative cost of weight vs. activation reuse.
- **Model co-design:** WRPN (Mishra, ICLR 2018) shows that increasing channel width can compensate for accuracy lost to reduced precision — a direct example of model-hardware co-design where the model shape and the number format are jointly optimized.

---

## Appendix — Slide-to-Section Map

| Slides | Section |
|---|---|
| L12-1 | Title |
| L12-2 … L12-9 | Ch.1 — Why Reduce Precision? Energy and Area Case; determining operand and accumulator bit-width |
| L12-10 … L12-15 | Ch.2 — Quantization: The Core Idea |
| L12-16 … L12-36 | Ch.3 — A Taxonomy of Number Formats |
| L12-23 … L12-30 | Ch.4 — Non-Uniform Quantization |
| L12-21 … L12-22, L12-35, L12-48, L12-70 … L12-74 | Ch.5 — Accuracy Impact and Mixed Precision |
| L12-37 … L12-58 | Ch.6 — Hardware for Reduced Precision |
| L12-59 … L12-68 | Ch.6 — Binary and Ternary Nets |
| L12-69 | Ch.5 — Summary Table |
| L12-75 … L12-76 | Summary and Recommended Reading |

## Source Notes

- The lecture ordering, precision taxonomy, accuracy summary table, MX discussion, binary/ternary examples, and industry-hardware survey follow `Lecture/L12 - Precision_r1.pdf`.
- The operation energy and area table is slide-stated in L12-3 and attributed there to Horowitz, ISSCC 2014.
- The TPU discussion is based on Jouppi et al., *In-Datacenter Performance Analysis of a Tensor Processing Unit*, ISCA 2017, especially the Abstract, Sections 1-4, Figures 1-5, and Tables 1-3.
- The precision-scalable MAC discussion is based on Camus et al., *Review and Benchmarking of Precision-Scalable Multiply-Accumulate Unit Architectures for Embedded Neural-Network Processing*, JETCAS 2019, especially Sections II, V, and VII plus Table I.
- The Eyeriss bridge uses Chen, Emer, and Sze, *Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks*, ISCA 2016, and the local JSSC follow-up for the fabricated-chip precision context.
- The small quantization, accumulator-headroom, and per-channel-scaling examples are original teaching examples constructed for this walkthrough.

## Uncertainty Notes

- The live lecture may have emphasized a subset of the many industry formats differently; this chapter groups them pedagogically around range, resolution, scaling granularity, and hardware support.
- Several accuracy numbers in the summary table are slide-stated literature results. This revision did not independently re-read every original binary/ternary/log-quantization paper behind that table.
- The chapter discusses modern formats such as MX and NVFP4 using the lecture slides as immediate source anchors. Treat product-specific deployment details as time-sensitive if using this chapter for hardware selection.
