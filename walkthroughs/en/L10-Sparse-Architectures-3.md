# L10 — Sparse Architectures 3

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze (MIT EECS)
> **Lecture date:** March 9, 2026 · **Slides:** 66 · **Source:** [`Lecture/L10-Sparse_Architectures-3.pdf`](../../Lecture/L10-Sparse_Architectures-3.pdf)
>
> *This is a conceptual walkthrough that reconstructs the lecture's narrative from the slides. It is organized by idea, not slide-by-slide. Each section cites the slide range it draws from so you can follow along with the original deck.*

---

## TL;DR

The third and final installment of the sparse-architectures series asks: **how do real accelerators exploit both sparse weights and sparse activations simultaneously?** The lecture begins with a crisp review of the two simpler cases — gating (energy savings only), and skipping sparse weights or sparse inputs separately — and then builds up to the hard joint case. Two landmark architectures anchor the analysis: **SCNN** (Sparse CNN, ISCA 2017), which uses a Cartesian-product multiplier array with a scatter network to handle input-stationary sparse-times-sparse multiplication; and **ISOSceles** (HPCA 2023), which splits the computation into a two-pass **IS→OS dataflow** pipeline that achieves up to **7.5× speedup** over a dense baseline by exploiting joint sparsity through fiber intersection and swizzled tensor traversal. The lecture ties together all the representation, intersection, projection, and skipping machinery from L08–L09 and shows how it is instantiated in concrete silicon.

---

## Learning Objectives

After this lecture you should be able to:

- Explain the difference between **gating** (saves energy, not time) and **skipping** (saves both energy and time) for zero-valued operands.
- Trace through the loop-nest transformations that produce **output-stationary, weight-stationary, and input-stationary** dataflows for sparse convolution, and identify which sparsity each exploits.
- Describe how **Cambricon-X** handles sparse-weight convolution via a compressed weight stream and indirect input-activation lookup.
- Explain how **Cnvlutin** achieves speedup by skipping zero input activations and why it uses a per-channel encoder + reindexed weight lookup.
- Describe the **SCNN** Cartesian-product architecture: why all-to-all multiplication works, what the scatter network does, and how **flattening** maps a 2-D convolution into a 1-D inner product.
- Explain how SCNN's **latency and energy** scale with joint activation and weight sparsity.
- Describe the **IS-OS two-pass dataflow** used by ISOSceles, including the role of tensor **swizzling** to convert a discordant traversal into a concordant one.
- Read the ISOSceles speedup chart and explain why the **IS-frontend / OS-backend pipeline** produces multiplicative gains from joint sparsity.

---

## Chapter 1 — Recap: Gating vs. Skipping and the 1-D Convolution Template

> *Slides: L06-1 … L06-9*

![L10 Title — Sparse Architectures Part 3](../../assets/L10/L10-p01-title.png)

### The two levers on zero-valued operands

Every sparse accelerator must decide what to do when an input activation or weight is zero. There are two options:

- **Gating:** execute the multiply-accumulate (MAC) cycle anyway, but shut off the power to the multiplier and the memory reads through clock- or data-gating. This saves **energy but not time** — the cycle slot is wasted.
- **Skipping:** actually remove the zero-valued operation from the schedule, so the hardware never issues that cycle. This saves **both energy and time**.

The distinction matters because skipping is harder to implement: it requires knowing *ahead of time* (or very quickly) which operands are zero, compressing the data stream, and being able to merge results that arrive at unpredictable positions.

### 1-D output-stationary convolution as the running example

The lecture uses a simple 1-D convolution `o[q] += i[w] * f[s]` with `w = q + s` as the pedagogical template. This output-stationary loop nest is the starting point from which all four cases — sparse weights only, sparse inputs only, both sparse — are derived by changing the loop order and which tensor is compressed.

**Eyeriss – Gating (slide L06-7):**
Eyeriss (Chen et al., ISSCC 2016) gates the 2-stage pipelined multiplier and both memory read ports whenever the input activation is zero, saving ~**45% of PE power** while keeping throughput unchanged.

![Eyeriss gating circuit — multiplier enable and zero buffer](../../assets/L10/L10-p07-eyeriss-gating.png)

> **Why it matters:** Gating is the simplest form of sparsity exploitation and requires no change to the loop structure or data format. Its limitation — saving energy but not time — motivates the more complex skipping designs studied in the rest of this lecture.

---

## Chapter 2 — Exploiting Sparse Weights Only

> *Slides: L06-8 … L06-25*

### Output-stationary with sparse weights

When weights are sparse and stored in compressed form (coordinate + payload list), the outer loop walks over non-zero filter entries `(s, f_val)` via **concordant traversal**, directly giving their coordinate `s`. For each non-zero weight, the inner output loop computes `w = q + s` and looks up the (dense, uncompressed) input at position `w`. This is the **output-stationary sparse-weight** dataflow:

```
for q in [0, Q):
  for (s, f_val) in f:          # concordant traversal of sparse filter
    w = q + s
    o[q] += i[w] * f_val
```

Because the filter fiber is traversed concordantly, the number of multiply-accumulates is exactly proportional to the number of non-zero weights — a direct, linear speedup.

### Weight-stationary with sparse weights

Swapping the loop order to `for (s, f_val) in f: for q in [0, Q):` gives the **weight-stationary** variant. The weight `f[s]` is loaded once and reused across all outputs — the hallmark of weight-stationary dataflow. Skipping is still over non-zero weights via concordant traversal.

### Parallelism: fiber splitting in position space

To compute multiple outputs in parallel, the filter fiber is split into **equal-sized chunks in position space** (`f.splitEqual(K)`), and each PE group works on one chunk. This exposes a `spatial-for` loop over the weight chunks. A key subtlety: splitting *by position* (not by coordinate) ensures each PE gets the same number of weight slots regardless of sparsity, enabling simple synchronization.

### Cambricon-X — an industrial implementation

**Cambricon-X** (Zhang et al., MICRO 2016) is a weight-stationary sparse accelerator. Compressed weights (metadata + values) are streamed in, and each PE looks up the corresponding input activation by using the weight's coordinate as an index into a shared input activation buffer.

![Cambricon-X activation access — weight metadata drives indirect input lookup](../../assets/L10/L10-p23-cambricon-x.png)

The coordinate of each non-zero weight directly addresses the input, making the lookup a simple indexed read. This is efficient when input activations are dense (or nearly so), but does not exploit input sparsity.

> **Why it matters:** The sparse-weight-only case admits a clean implementation: compress weights, traverse them concordantly, and look up dense inputs by coordinate. Cambricon-X shows this can be done at scale with modest hardware overhead. Its limitation is that it leaves the sparsity of activations unexploited.

---

## Chapter 3 — Exploiting Sparse Inputs Only

> *Slides: L06-26 … L06-38*

### Weight-stationary with sparse inputs

When inputs are sparse (coordinates + payloads) and weights are dense, the weight `f[s]` is held stationary and the input fiber `i` is traversed. The loop restricts input coordinates to those falling within the active window for the current weight:

```
for s in [0, S):
  for (w, i_val) in i if s <= w < Q + s:   # windowed traversal
    q = w - s
    o[q] += i_val * f[s]
```

The filter weight is fetched once per window position — it is stationary — and only the non-zero inputs generate MACs. This is the **weight-stationary sparse-input** skipping dataflow.

### Output-stationary with sparse inputs

The output-stationary variant iterates over outputs first, then restricts the input fiber to the window relevant to each output:

```
for q in [0, Q):
  for (w, i_val) in i if q <= w < q + S:   # sparse sliding window
    s = w - q
    o[q] += i_val * f[s]
```

The "sparse sliding window" visualization (slides L06-29 … L06-34) shows how the active input-coordinate set slides with `q`. The weight lookup `f[s]` requires computing `s = w - q` — a simple subtraction.

### Cnvlutin — skipping zero activations at scale

**Cnvlutin** (CNVLUTIN, ISCA 2016) exploits sparse activations across a full 2-D multi-channel convolution. Per-channel encoders compress the input activation maps, removing zero entries and recording their coordinates. The output-stationary computation then maps each non-zero activation to the appropriate weight via reindexing:

```
for q in [0, Q]:
  for m, f_c in f:
    for (c, (f_s, i_w)) in f_c & i_c:       # implicit intersection over channels
      for (w, i_val) in getWindow(i_w, q, S):
        s = w - q
        o[m, q] += i_val * f_s.getPayload(s)  # uncompressed weight lookup
```

By keeping weights uncompressed, the `getPayload(s)` lookup is cheap (direct index). Cnvlutin demonstrates that compressing zero activations and skipping their MACs translates directly into throughput speedup, with gains that scale with the activation sparsity of the network.

![Cnvlutin architecture — per-channel encoder feeds reindexed weights to PE array](../../assets/L10/L10-p36-cnvlutin.png)

> **Why it matters:** The sparse-input-only case is complementary to sparse weights: compress activations, skip their MACs via concordant traversal, and look up weights by coordinate. Cnvlutin shows this yields real speedup — but like Cambricon-X, it leaves the other dimension of sparsity on the table.

---

## Chapter 4 — Exploiting Sparse Weights and Sparse Inputs Simultaneously (SCNN)

> *Slides: L06-39 … L06-51*

### The challenge of joint sparsity

When both weights **and** activations are sparse, neither can serve as the "outer" loop driver while the other is looked up densely. Both are compressed, both must be traversed, and their products must be **scattered** to output positions that are not known until runtime.

### Input-stationary with both sparse — the Cartesian product idea

The **input-stationary** dataflow keeps each input activation stationary while multiplying it against all non-zero filter weights, producing partial results that scatter to different outputs:

```
for (w, i_val) in i:
  for (s, f_val) in f if w-Q <= s < w:    # restrict weights to current input
    q = w - s
    o[q] += i_val * f_val
```

When parallelized, both the input split and the filter split are traversed spatially. This yields **all-to-all (Cartesian product) multiplication**: each of the K spatially active input values multiplies each of the K' spatially active filter values, producing K×K' partial products simultaneously.

### SCNN architecture

**SCNN** (Parashar et al., ISCA 2017) implements exactly this:

- **Flattening:** The 2-D convolution is reformulated by substituting index variables to produce a 1-D inner product over flattened indices `(hw)` and `(mrs)`. This makes the all-to-all multiplication regular.
- **Sparse-compressed frontend:** Non-zero inputs (`i[C][W*H]`) and non-zero weights (`f[C][M*R*S]`) are stored in compressed format. The PE frontend consumes them in parallel.
- **Dense backend / scatter network:** Each of the K×K' products is accompanied by a computed output coordinate `(m, p, q)`. A scatter network routes each product to the correct accumulator in the output partial-sum buffer.

![SCNN architecture — Cartesian-product multiply + scatter network for sparse inputs and weights](../../assets/L10/L10-p45-scnn-architecture.png)

![SCNN PE microarchitecture — sparse-compressed frontend and dense scatter backend](../../assets/L10/L10-p49-scnn-pe-microarchitecture.png)

### Flattening: why it enables the Cartesian product

The substitution `h = p + r, w = q + s` converts the 2-D convolution `O[m,p,q] += I[c,h,w] * F[m,c,r,s]` into a form indexed over flat coordinates. Splitting both the flattened input and weight fibers equally and then traversing them spatially gives the all-to-all structure:

```
for (hw1, i_split) in i.splitEqual(4):
  for (mrs1, f_split) in f.splitEqual(4):
    spatial-for ((h,w), i_val) in i_split:
      spatial-for ((m,r,s), f_val) in f_split if "legal":
        p = h - r;  q = w - s
        o[m,p,q] += i_val * f_val
```

### SCNN latency and energy vs. joint density

The latency of SCNN scales with the **product of activation density × weight density** (the fraction of non-zero values in each tensor). At high joint sparsity (e.g., 90% zeros in both), SCNN achieves proportionally fewer MAC cycles. However, the scatter network introduces area and energy overhead that limits gains at moderate sparsity levels. The energy curve (slides L06-50 and L06-51) shows that SCNN's energy efficiency improves monotonically with joint sparsity, but the crossover point relative to a dense baseline depends on the scatter-network cost.

> **Why it matters:** SCNN demonstrates that joint weight-and-activation sparsity can be exploited with a relatively simple hardware mechanism (Cartesian product + scatter), but the irregular scatter step is a real cost. This motivates architectures like ISOSceles that seek a more structured approach to handling the joint sparse case.

---

## Chapter 5 — The IS-OS Two-Pass Dataflow and ISOSceles

> *Slides: L06-52 … L06-66*

### The weight-stationary reversal and its cost

Before introducing ISOSceles, the lecture shows a **weight-stationary variant** of the joint-sparse loop (slide L06-52) where the filter split is the outer loop and the input split is inner. This reversal means the filter is more stationary, but the *input* must now be read from a larger buffer more frequently — a disadvantage that shows the sensitivity of dataflow choice to memory hierarchy costs.

### The output-stationary joint-sparse case and fiber intersection

The **output-stationary joint-sparse** dataflow iterates over outputs `q` and performs **fiber intersection** to find the (input, weight) pairs that both contribute to output `q`:

```
for q in [0,Q):
  for (s, (f_val, i_val)) in f.project(+q) & i:
    o[q] += i_val * f_val
```

`f.project(+q)` shifts the weight fiber's coordinates by `+q`, aligning them with input coordinates. The intersection `&` then finds only the (coordinate, payload) pairs present in **both** the projected weight fiber and the input fiber. Only non-zero pairs in both tensors contribute a MAC — achieving true joint skipping.

The hardware component diagram (slide L06-56) shows the intersection unit sitting between the weight and input streams, feeding a single MAC unit and a partial-sum accumulator.

### IS-OS Dataflow: splitting the computation into two passes

The key insight of **ISOSceles** (Yang et al., HPCA 2023) is that a single-pass output-stationary loop that exploits both sparsities leads to a discordant traversal of an intermediate tensor `T`. Instead, the computation is split into two mathematically equivalent passes:

**Step 1 (IS pass — Input-Stationary):**

```
T[c,h-r,w-s] = I[c,h,w] × F[c,m,r,s]
```

Traverse `i` concordantly (input-stationary), multiply by intersected filter entries, and accumulate into a *temporary* tensor `T` indexed by `[h, r, w-s]`.

**Step 2 (OS pass — Output-Stationary):**

```
O[m,p,q] = T[c,p+r,q]
```

Traverse `T` and accumulate into output, but `T` must be accessed in a **different rank order** than it was written. This is a **discordant traversal** — naively expensive.

**Swizzling to fix the traversal order:**

The fix is to **swizzle the ranks** of `T`:

```python
t = t.swizzleRanks(["H", "R", "Q"] -> ["Q", "R", "H"])
```

After swizzling, the Step-2 traversal becomes concordant, and the overall pipeline is:

- IS frontend processes sparse inputs and sparse weights → writes partial results to `T` (small).
- OS backend reads `T` (concordantly after swizzle) → accumulates into outputs.

![ISOSceles IS-OS pipeline — IS frontend feeds OS backend via small intermediate buffer](../../assets/L10/L10-p64-isosceles-pipeline.png)

### Why the pipeline works

The IS frontend and OS backend operate as a **two-stage pipeline**:

1. **IS frontend:** Input wavefront traverses non-zero activations concordantly, intersects them with non-zero filter entries, and produces partial sums into a small `T` buffer.
2. **OS backend:** Output wavefront traverses `T` concordantly (after swizzle) and drains partial sums into the output map.

The "small" annotation on the `T` buffer in the diagram underlines a key implementation fact: because `T` is indexed by a shifted coordinate, its **active footprint at any moment is small**, fitting in on-chip memory and avoiding DRAM traffic.

### ISOSceles speedup

The measured speedup of ISOSceles versus a dense baseline ranges up to **7.5×**, with an average around **1.7×** across benchmarks.

![ISOSceles speedup — up to 7.5× over dense baseline, average 1.7×](../../assets/L10/L10-p66-isosceles-speedup.png)

The 7.5× peak arises at very high joint sparsity where both the IS and OS passes skip proportionally many operations. The gap between peak and average reflects the fact that not all layers are equally sparse, and the swizzle + pipeline overhead sets a floor on achievable speedup.

> **Why it matters:** ISOSceles closes the loop on the sparse-architecture series. It demonstrates that the full IS-OS split — combined with fiber intersection, projection, and tensor swizzling — can be implemented in real hardware and deliver substantial measured speedup. The two-pass structure trades the irregular scatter network of SCNN for a structured pipeline with a small intermediate buffer, a principled engineering tradeoff that generalizes to other workloads.

---

## Key Terms

| Term | Gloss |
|---|---|
| **Gating** | Shutting off multiplier/memory power when an operand is zero; saves energy but not time. |
| **Skipping** | Removing zero-operand cycles from the schedule entirely; saves both energy and time. |
| **Concordant traversal** | Iterating over a compressed fiber in coordinate order, so each non-zero element is visited exactly once in linear time. |
| **Discordant traversal** | Accessing a tensor in an order that does not match its stored rank order; requires random lookup or reordering. |
| **Fiber** | A 1-D slice of a multi-dimensional tensor, represented as a list of (coordinate, payload) pairs. |
| **Fiber splitting (splitEqual)** | Dividing a fiber into equal-sized chunks by position, used to create spatial parallelism. |
| **Fiber projection (project)** | Shifting a fiber's coordinates by an offset, used to align two fibers for intersection. |
| **Fiber intersection (&)** | Finding coordinate-payload pairs present in both of two fibers; produces only jointly non-zero entries. |
| **Output-stationary (OS)** | Dataflow where each output accumulator remains fixed while inputs and weights are streamed in. |
| **Weight-stationary (WS)** | Dataflow where each filter weight remains fixed while inputs and outputs are streamed. |
| **Input-stationary (IS)** | Dataflow where each input activation remains fixed while filter weights and outputs are cycled. |
| **Cartesian product multiplication** | All-to-all multiplication of a split of K non-zero inputs with K' non-zero weights, producing K×K' partial products. |
| **Scatter network** | Hardware network that routes each partial product from a Cartesian-product multiplier to the correct output accumulator address. |
| **Flattening** | Substituting 2-D spatial indices (h,w) with a single flat index (hw) to make a 2-D convolution look like a 1-D inner product. |
| **IS-OS dataflow** | Two-pass computation: IS pass produces an intermediate tensor T; OS pass reduces T to output. |
| **Tensor swizzling (swizzleRanks)** | Reordering the rank axes of a tensor to convert a discordant traversal into a concordant one. |
| **Eyeriss** | Weight-stationary accelerator (ISSCC 2016) with input-activation gating; saves ~45% PE power. |
| **Cambricon-X** | Weight-stationary sparse accelerator (MICRO 2016); exploits sparse weights via indirect activation lookup. |
| **Cnvlutin** | Output-stationary sparse accelerator (ISCA 2016); exploits sparse activations via per-channel encoding. |
| **SCNN** | Sparse CNN accelerator (ISCA 2017); exploits joint weight and activation sparsity via Cartesian product + scatter. |
| **ISOSceles** | IS-OS sparse accelerator (HPCA 2023); achieves up to 7.5× speedup via two-pass pipeline with tensor swizzling. |

---

## Takeaways

- **Gating saves energy; skipping saves time.** The two are architecturally distinct and require different hardware mechanisms. Most high-performance sparse accelerators aim for both.
- **Single-sparsity designs are tractable.** Sparse-weight-only (Cambricon-X) and sparse-input-only (Cnvlutin) architectures are each relatively straightforward, exploiting one compressed tensor with concordant traversal and a dense lookup for the other.
- **Joint sparsity requires either a scatter network or a two-pass pipeline.** SCNN uses a Cartesian-product multiplier with a scatter network; ISOSceles uses an IS→OS two-pass pipeline with fiber intersection and tensor swizzling.
- **SCNN's Cartesian product is conceptually clean** but pays an area and energy cost for the scatter network. Its gains scale with the **product** of weight and activation density.
- **ISOSceles's IS-OS split converts an irregular scatter into a structured pipeline** by storing partial results in a small intermediate tensor `T`, then swizzling `T`'s ranks to make the second pass concordant. This achieves up to **7.5×** measured speedup.
- **Fiber intersection is the mathematical primitive** that enables joint skipping in output-stationary and IS-OS dataflows. The hardware cost depends on the representation (bitmask and uncompressed are cheapest; coordinate lists require sorting or merge logic).
- **Dataflow choice and sparsity exploitation are coupled.** Changing from output-stationary to weight-stationary reverses which tensor is more stationary and changes which data must be read from the larger buffer — illustrating that mapping decisions (the TeAAL Mapping layer) directly interact with Format-layer sparsity choices.

---

## Connections to Later Lectures

- **This lecture closes the sparse-architecture series (L07–L10).** L07 introduced sparsity motivation and formats; L08 introduced fiber representations, concordant/discordant traversal, and gating; L09 covered skipping for single-sparse cases; L10 (this lecture) covers joint sparse and two case-study accelerators (SCNN and ISOSceles).
- **L11 — Advanced Technologies:** The next lecture shifts to novel device technologies (RRAM, optical, superconducting) that may enable different sparsity-exploitation strategies at the physical level.
- **L12 — Reduced Precision:** Quantization and low-bit arithmetic interact with sparsity — zero-valued entries in quantized networks may arise from different sources, and both techniques can be composed.
- **TeAAL Pyramid revisited:** The SCNN and ISOSceles case studies illustrate all four pyramid layers in action: **Format** (compressed fibers), **Mapping** (IS vs. OS vs. IS-OS dataflow), **Architecture** (Cartesian multiplier vs. IS-OS pipeline), and **Binding** (scatter network routing vs. swizzled tensor indexing).

---

## Appendix — Slide-to-Section Map

| Slides | Section |
|---|---|
| L06-1 | Title — Sparse Architectures Part 3, March 9, 2026 |
| L06-2 … L06-4 | Ch.1 — CONV layer review; 1-D output-stationary convolution loop nest |
| L06-5 … L06-6 | Ch.1 — Gating vs. skipping: energy vs. time savings |
| L06-7 | Ch.1 — Eyeriss gating: 45% PE power reduction |
| L06-8 … L06-9 | Ch.2 — Weight-stationary dataflow; dense vs. compressed representation |
| L06-10 … L06-14 | Ch.2 — Output-stationary sparse-weight dataflow and datapath diagram |
| L06-15 … L06-16 | Ch.2 — Weight-stationary sparse-weight dataflow and datapath diagram |
| L06-17 … L06-21 | Ch.2 — Fiber splitting in position space; extending to multiple dimensions |
| L06-22 … L06-24 | Ch.2 — Parallel weight-stationary sparse-weight loop nest |
| L06-23 | Ch.2 — Cambricon-X: weight metadata drives indirect activation lookup |
| L06-25 | Ch.3 — Transition: exploiting sparse inputs |
| L06-26 … L06-27 | Ch.3 — Weight-stationary sparse-input dataflow and datapath |
| L06-28 … L06-34 | Ch.3 — Output-stationary sparse-input dataflow; sparse sliding window |
| L06-35 | Ch.3 — Output-stationary sparse-input datapath diagram |
| L06-36 … L06-38 | Ch.3 — Cnvlutin: per-channel encoder, loop nest, speedup |
| L06-39 … L06-41 | Ch.4 — Input-stationary sparse weights & inputs; datapath diagram |
| L06-42 … L06-43 | Ch.4 — Fiber splitting for joint sparse; parallel IS loop nest |
| L06-44 | Ch.4 — Cartesian product multiplication visualization |
| L06-45 … L06-46 | Ch.4 — SCNN architecture overview and flattening |
| L06-47 … L06-48 | Ch.4 — SCNN tile loop nest (one channel; flattened) |
| L06-49 | Ch.4 — SCNN PE microarchitecture: sparse frontend + dense scatter backend |
| L06-50 … L06-51 | Ch.4 — SCNN latency and energy vs. joint density |
| L06-52 | Ch.5 — Weight-stationary joint-sparse: reversed loops and buffer cost |
| L06-53 … L06-56 | Ch.5 — Output-stationary joint-sparse: fiber projection + intersection |
| L06-57 … L06-62 | Ch.5 — IS-OS dataflow math: two-pass derivation and swizzleRanks |
| L06-63 … L06-65 | Ch.5 — ISOSceles IS-OS pipeline diagram (IS frontend → T → OS backend) |
| L06-66 | Ch.5 — ISOSceles speedup: up to 7.5×, average 1.7× |
