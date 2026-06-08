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

![Energy and area cost vs. operation precision — Horowitz ISSCC 2014](../../assets/L12/L12-p03-energy-area-cost.png)

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

![Bit-width growth inside a MAC — partial sum needs extra bits](../../assets/L12/L12-p07-mac-bit-widths.png)

For real networks: AlexNet (RSC up to 9,216) needs 14 extra bits; VGG-16 (RSC up to 25,088) needs 15 extra bits. This is why accumulators are often kept at 32 bits even when inputs are 8 bits — the accumulator is the internal "headroom" for the operation.

> **Why it matters:** Reducing operand bit-width from 32 to 8 cuts integer multiplier energy by ~15× and area by ~12×, and it also reduces memory traffic — both reads and writes. These are the two biggest costs in a DNN accelerator. Every technique in this lecture is a different strategy for achieving those savings while preserving accuracy.

---

## Chapter 2 — Quantization: The Core Idea

> *Slides: L12-10 … L12-15*

### What quantization is

**Quantization** maps a real-valued distribution of numbers to a finite set of discrete **quantization levels** {q₀, q₁, …, q_{L-1}}, separated by **decision boundaries** {d₁, d₂, …, d_{L-1}}. The goal is to minimize the **quantization error** (the difference between original and reconstructed values) subject to the constraint of using only L levels.

![Quantization: mapping a distribution to L quantization levels with decision boundaries](../../assets/L12/L12-p10-quantization-levels.png)

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

> **Why it matters:** The accuracy of quantization depends not just on the number of bits but on *how* those bits are allocated across the value range. Poor range selection wastes levels on rarely-occurring extreme values.

---

## Chapter 3 — A Taxonomy of Number Formats

> *Slides: L12-16 … L12-36*

### Numerical format anatomy

Every numerical representation has three components: a **sign (S)**, a **mantissa/significand (M)** encoding the number of unique values within a scale, and (for floating-point) an **exponent (E)** encoding the scale. Total bits = n_S + n_E + n_M.

![Numerical format comparison: FP32, FP16, bfloat16, Int32, Int16, Int8](../../assets/L12/L12-p16-numerical-formats.png)

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

### The mantissa–exponent tradeoff

The slide directly comparing fp16 and bfloat16 makes the tradeoff concrete:

![Mantissa vs. exponent bit allocation: fp16 vs. bfloat16](../../assets/L12/L12-p31-mantissa-exponent-tradeoff.png)

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

**Logarithmic quantization** maps the mantissa bits to a logarithmic scale. The key hardware benefit: a **multiplication in the log domain becomes an addition** (and an addition becomes a bitshift and comparison). The slide shows that with both weights and activations in log domain, multiply-accumulate becomes shift-and-add — a fundamentally cheaper hardware operation.

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

![Full precision taxonomy: uniform, non-uniform, shared-exponent formats](../../assets/L12/L12-p51-precision-taxonomy.png)

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

The slides show measured top-1 accuracy of CaffeNet on ImageNet as the bit-width is reduced from 16 bits to as few as 2 bits, with and without fine-tuning:

![Impact on accuracy: dynamic fixed-point quantization of CaffeNet on ImageNet](../../assets/L12/L12-p35-dynamic-fixed-point-accuracy.png)

Key findings:
- **8-bit dynamic fixed-point without fine-tuning**: 0.4% accuracy loss vs. 32-bit float.
- **8-bit dynamic fixed-point with fine-tuning**: 0.6% accuracy loss.
- Fine-tuning (retraining with quantized arithmetic) is critical as bit-widths drop below 8.

### The comprehensive accuracy summary

The lecture's summary table (slide 69) is the single most useful quantitative reference:

![Summary table: accuracy loss for various reduced-precision methods on AlexNet](../../assets/L12/L12-p69-summary-table.png)

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

Not every layer needs the same precision. The slides show that varying precision across layers (e.g., using 4-bit for some layers, 8-bit for others) can achieve a better accuracy–efficiency tradeoff than applying a uniform low bit-width everywhere. Early and late layers (first and last) are consistently kept at higher precision across virtually every method in the summary table.

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
| L12-2 … L12-7 | Ch.1 — Why Reduce Precision? Energy and Area Case |
| L12-10 … L12-15 | Ch.2 — Quantization: The Core Idea |
| L12-16 … L12-36 | Ch.3 — A Taxonomy of Number Formats |
| L12-23 … L12-30 | Ch.4 — Non-Uniform Quantization |
| L12-21 … L12-22, L12-35, L12-48, L12-70 … L12-74 | Ch.5 — Accuracy Impact and Mixed Precision |
| L12-37 … L12-58 | Ch.6 — Hardware for Reduced Precision |
| L12-59 … L12-68 | Ch.6 — Binary and Ternary Nets |
| L12-69 | Ch.5 — Summary Table |
| L12-75 … L12-76 | Summary and Recommended Reading |
