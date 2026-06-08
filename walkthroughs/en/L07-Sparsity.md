# L07 — Co-Design of DNN Models and Hardware: Sparsity

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze (MIT EECS)
> **Lecture date:** February 25, 2026 · **Slides:** 53 · **Source:** [`Lecture/L07 - Sparsity.pdf`](../../Lecture/L07%20-%20Sparsity.pdf)
>
> *This is a conceptual walkthrough that reconstructs the lecture's narrative from the slides. It is organized by idea, not slide-by-slide. Each section cites the slide range it draws from so you can follow along with the original deck.*

---

## TL;DR

Deep neural networks are full of **zeros** — and zeros are free if the hardware knows how to skip them. This lecture, which sits at the boundary of the **Format** layer of the TeAAL Pyramid (how data is represented) and the **Compute** layer (what gets computed), explains where those zeros come from and how to make more of them. **Activation sparsity** arises naturally from ReLU and from structured input data such as graphs; **weight sparsity** can be deliberately introduced through **pruning** — a four-step loop of scoring, grouping, ranking, and fine-tuning. The two big payoffs are **(1) reduced data movement and storage** (because zeros need not be stored or fetched) and **(2) reduced computation** (because anything multiplied by zero equals zero). The catch: fully exploiting sparsity in hardware is non-trivial and demands specialized support — which is exactly what L08–L10 address.

---

## Learning Objectives

After this lecture you should be able to:

- Distinguish **activation sparsity** (natural) from **weight sparsity** (engineered via pruning).
- Explain the two **hardware-level benefits** of sparsity: reducing data movement/storage and reducing computation.
- Define **effectual** vs. **ineffectual** operations and understand why the latter introduce a hardware cost tradeoff.
- Trace the **pruning pipeline** (scoring → grouping → ranking → fine-tuning → scheduling).
- Contrast **magnitude-based** and **energy-aware** pruning criteria, and explain why indirect metrics like MAC count can be misleading.
- Distinguish **fine-grained (unstructured)** from **coarse-grained (structured)** weight sparsity and articulate the hardware implications of each.
- Explain why sparsity on the order of **30–70%** already matters for DNN hardware even though software libraries target >99%.

---

## Chapter 1 — What Sparsity Buys (and What It Costs)

> *Slides: L07-2 … L07-8*

### The two goals

The lecture opens with a precise statement of scope: today's focus is **reducing the number of operations for storage and compute** by exploiting sparsity — broadly defined as repeated values, and in most DNN contexts, **repeated zeros**.

![Goals of today's lecture — exploit sparsity to reduce data movement and computation](../../assets/L07/L07-p02-goals.png)

Two distinct hardware benefits flow from a zero in a tensor:

1. **Reduce data movement and storage cost** — because `anything × 0 = 0` and `anything + 0 = anything`, a zero value need never be fetched from memory or sent across a network-on-chip. Skipping the load and the communication directly translates to energy savings, since (as L01 established) DRAM access costs ~200× an arithmetic operation.
2. **Reduce number of operations** — the multiply-accumulate can be entirely bypassed when one operand is zero, saving both compute time and the energy of the ALU itself.

### Effectual vs. ineffectual operations

The lecture introduces a precise accounting framework:

> **Total algorithmic operations** = effectual operations + ineffectual operations

An **effectual** operation is one that changes the result; an **ineffectual** one (involving a zero) does not. Hardware that fully exploits sparsity would execute only the effectual fraction. In reality:

> **Actual operations performed** = effectual + *unexploited* ineffectual operations

Avoiding *all* ineffectual operations is hard — it requires knowing at hardware time which inputs are zero, routing around them, and keeping the effective throughput high. This creates a central tension: more elaborate sparsity-exploitation machinery **reduces operations skipped but increases the cost per operation** (area, power, control overhead). Slides L07-7 and L07-8 make this tradeoff graphical: throughput and energy efficiency both depend on the interplay between hardware complexity and the actual sparsity level in the workload. The right design point depends on the deployment scenario.

> **Why it matters:** Sparsity is not free — it creates an opportunity that hardware must be built to capture. Understanding the effectual/ineffectual distinction is the first step toward designing or evaluating a sparse accelerator.

---

## Chapter 2 — Activation Sparsity: Where Natural Zeros Come From

> *Slides: L07-9 … L07-23*

### Sources of sparsity

The lecture identifies two broad origins of zeros in DNN tensors:

![Sources of sparsity — activation sparsity (ReLU, correlations, graph structure) and weight sparsity (reordering, pruning)](../../assets/L07/L07-p03-sources-of-sparsity.png)

**Activation sparsity** (zeros in the feature maps / intermediate tensors):
- **ReLU nonlinearity** — the dominant source: any negative pre-activation is clamped to zero.
- **Spatial and temporal correlation of inputs** — neighboring pixels in a feature map are correlated; frames in a video are similar.
- **Structural sparsity of the input representation** — graphs, for instance, have sparse adjacency matrices.

**Weight sparsity** (zeros in the learned parameters):
- **Redundant weights** — many weights are near-identical and can be consolidated before execution (without changing the model).
- **Network pruning** — deliberately removing weights whose removal has small impact on accuracy.

### ReLU and AlexNet activation sparsity

The landmark result: on **AlexNet**, the output feature maps of all five convolutional layers contain **~75% zeros** after ReLU. The bar chart on slide 10 shows that the fraction of non-zero activations never rises above ~25% across the five CONV layers. This means that in a naive dense implementation, three out of every four activations are zero — all of their associated multiplications are ineffectual.

![ReLU produces ~75% zeros in AlexNet feature maps](../../assets/L07/L07-p10-relu-activation-sparsity.png)

### Hardware responses to activation sparsity

The lecture surveys several architectural techniques for exploiting activation zeros:

- **Compression** — store only non-zero values with metadata (indices or run-length codes). This shrinks both storage footprint and memory bandwidth — meaning more useful data can be held at each level of the hierarchy.
- **Zero-skipping (Cnvlutin, ISCA 2016)** — built on top of DaDianNao, avoids fetching and multiplying zero activations, achieving 1.37× speedup (1.52× with additional activation pruning) at 4.49% area overhead.
- **Activation pruning (Minerva, ISCA 2016)** — goes further by removing small-but-nonzero activations; achieves 11% speedup on ImageNet, 2× power reduction on MNIST.
- **SnaPEA (ISCA 2018)** — predicts, *before* completing the convolution, whether the ReLU will output zero. If the partial sum already crosses a threshold indicating a negative result, the remaining computation is skipped. Additional hardware is required to decide when to safely terminate.
- **PredictiveNet / Song (ISCA 2018)** — simplifies the prediction: only compute on the high bits of each weight; if the high-bit result is already negative, skip the low-bit computation entirely.

### Spatial and temporal correlations

Beyond ReLU, sparsity can be extracted from the structure of the input itself:

- **Spatial correlation (Diffy, MICRO 2018)** — neighboring activations in a feature map differ only slightly. Processing *deltas* between neighbors rather than the full values introduces sparsity in the difference representation.
- **Temporal correlation** — consecutive video frames overlap significantly. Architectures such as EVA2, Euphrates, and FAST (all 2018) exploit inter-frame redundancy to skip redundant computation, at the cost of extra storage and motion-vector computation. This approach is application-specific (video only) and assumes the same operation is applied every frame.

### Graph Neural Networks: structural sparsity

GNNs operate on graphs (molecules, social networks, biological networks, financial graphs). The graph topology is encoded as an **adjacency matrix**, which is typically very sparse — most nodes are not connected to most other nodes. The GNN computation per layer is:

> X_(l+1) = σ(Â · X_(l) · W_(l))

where Â is the normalized adjacency matrix (sparse), X is the node feature matrix (dense), and W is the weight matrix (dense). The order in which the two matrix multiplications are performed materially affects how sparsity is exploited: computing `Â × (X × W)` versus `(Â × X) × W` can lead to very different effective densities of the intermediate result.

> **Why it matters:** Activation sparsity in CNNs is structural — ReLU guarantees it. Approximately 75% of activations in a standard network like AlexNet are zero. Any accelerator targeting CNN inference can plan around this fact. GNNs and video models add additional structured sparsity, but with different hardware challenges.

---

## Chapter 3 — Weight Sparsity: Pruning DNN Models

> *Slides: L07-24 … L07-51*

### Redundant weights and Gauss's trick

Before pruning, the lecture notes that weight redundancy can sometimes be exploited without accuracy loss. If two weights in a filter are equal (e.g., filter [A B A]), their corresponding inputs can be summed first, and only then multiplied by the weight — reducing 3 multiplies to 2. This is the idea behind **UCNN (ISCA 2018)**, which pre-processes weights to find and exploit such redundancies. The insight generalizes Gauss's multiplication algorithm (which trades multiplications for additions) to DNN convolutions.

### The pruning pipeline

Modern network pruning is a four-stage iterative loop:

![The pruning pipeline: scoring → grouping → ranking → fine-tuning → scheduling](../../assets/L07/L07-p29-pruning-pipeline.png)

**1. Scoring** — assign each weight (or group of weights) a score reflecting its importance to accuracy or efficiency:
- **Magnitude-based pruning** (most common): the score is simply |w|. Large-magnitude weights are deemed important; small-magnitude ones are candidates for removal. Typical results: 50% sparsity without retraining, 80% with retraining [Han, NeurIPS 2015].
- **Feature-based pruning**: score based on impact on the output feature map rather than weight magnitude [Yang, CVPR 2017].

**2. Grouping** — decide the *granularity* of removal: individual weights, rows, channels, or entire filters. This is the structured/unstructured dimension.

**3. Ranking** — sort weights (or groups) by their scores and select those below a threshold for removal.

**4. Fine-tuning and scheduling** — after each pruning step, retrain the surviving weights to recover accuracy. The schedule determines how aggressively to prune each iteration. A key refinement: **splicing (Guo, NeurIPS 2016)** allows previously pruned weights to be *restored* if the gradient later indicates they are important, roughly halving the number of non-zero weights needed to maintain accuracy compared to irreversible pruning.

The classic algorithm is **Optimal Brain Damage [LeCun, NeurIPS 1989]**: train → compute second derivatives for each weight → compute saliency → delete low-saliency weights → fine-tune → repeat.

### Energy-aware scoring

A critical insight: **weight count alone is not a good proxy for energy**. The energy cost of a weight depends on *where it lives in the memory hierarchy* and *which dataflow is in use*. On a 65 nm process, the hierarchy is (as established in L01):

| Data source | Relative energy |
|---|---|
| Register File (RF) | 1× |
| Neighbor PE (NoC) | 2× |
| Global Buffer | 6× |
| DRAM | 200× |

A weight in DRAM costs 200× more to access than one already in the register file. **Energy-aware pruning (EAP) [Yang, CVPR 2017]** directly incorporates this into the scoring metric: layers consuming most energy are pruned first.

![Scoring with energy awareness — the energy cost depends on the memory hierarchy](../../assets/L07/L07-p32-energy-aware-scoring.png)

The payoff is dramatic: EAP reduces AlexNet energy by **3.7×** vs. 2.1× for magnitude-based pruning, a **1.7× additional improvement**. The energy breakdown of GoogLeNet on slide L07-34 reveals why MAC count alone is misleading: output feature maps (43%), input feature maps (25%), and weights (22%) together vastly outweigh computation (10%) in total energy.

### Platform-aware adaptation: NetAdapt

Even energy is not always the right metric. **NetAdapt [Yang, ECCV 2018]** makes a deeper point: the relevant metric is **on-device latency or energy on the actual target platform**, measured empirically rather than estimated analytically. The reason is that the compiler, runtime, and hardware architecture all interact in ways that make the number of MACs a poor predictor of real latency.

NetAdapt's algorithm:
1. Start from a pretrained network.
2. At each iteration, propose multiple simplified networks (by reducing channels or filters per layer).
3. For each proposal, measure actual latency/energy on the target platform.
4. Among proposals that meet the resource budget, select the one with highest accuracy.
5. Fine-tune the selected network and repeat until the target budget is reached.

The result: NetAdapt increases the real inference speed of MobileNet by up to **1.7×** with similar accuracy on ImageNet, tested on a Google Pixel 1 CPU. Critically, a version of NetAdapt guided by MAC count (rather than latency) achieves a better MAC vs. accuracy tradeoff — but this does *not* translate to lower latency. The takeaway: **use direct metrics**.

### Structured vs. unstructured weight sparsity

The choice of pruning granularity — the **grouping** dimension of the pipeline — has profound hardware implications:

![Pruning granularity spectrum: from individual weights (fine-grained/unstructured) to entire filters (coarse-grained/structured)](../../assets/L07/L07-p42-structured-vs-unstructured.png)

Along a spectrum from fine to coarse:

| Granularity | What is removed | Hardware benefit | Accuracy cost |
|---|---|---|---|
| **Weight (unstructured)** | Individual scalar weights | Maximum compression | Minimum |
| **Row pruning** | Entire rows of a weight matrix | Some vectorization | Moderate |
| **Channel pruning** | Entire input channels | Eliminates whole MACs | Higher |
| **Filter pruning** | Entire output filters | Dense submatrix remains | Highest |

**Unstructured sparsity** offers the highest compression ratio and the least accuracy loss, but the resulting irregular zero pattern is hard to exploit in dense hardware. **Structured (coarse-grained) sparsity** produces regular zeros that map naturally onto SIMD units, systolic arrays, or other parallel data-path hardware — at the cost of accuracy for a given sparsity level.

**Scalpel [Yu, ISCA 2017]** bridges the gap by pruning to match the underlying hardware's data-parallel organization. For a 2-way SIMD unit, it ensures zeros appear in pairs, achieving 1.92× speedup over fully unstructured pruning.

**Pattern-based pruning [PCONV, AAAI 2020; PatDNN, ASPLOS 2020]** takes a middle path: prune based on a small set of structured zero patterns within each filter, achieving better hardware regularity than unstructured pruning with less accuracy loss than channel/filter pruning.

**Mixture-of-Experts (MoE)** models can also be understood as a form of coarse-grained sparsity: at inference time, only a small number of "expert" sub-networks are activated for any given input, leaving the rest idle.

### Pruning and accuracy: the interaction with model efficiency

Two important nuances close this chapter:

![Accuracy drops more quickly when pruning efficient models](../../assets/L07/L07-p50-pruning-accuracy-tradeoff.png)

1. **Efficient models are harder to prune.** Modern compact models (MobileNets, EfficientNets) already have most redundancy removed by design. Pruning them causes accuracy to drop faster than pruning older, overparameterized models like AlexNet. The "free compression" gains are smaller the more optimized the starting model.

2. **Starting from a better model matters more than pruning a worse one.** An unpruned efficient DNN model can outperform a pruned inefficient model at the same computational budget [Blalock, MLSys 2020]. The implication: **architecture search and pruning are complementary, not interchangeable strategies**.

![An unpruned efficient model can outperform a pruned inefficient model](../../assets/L07/L07-p51-pruning-dnn-model.png)

> **Why it matters:** Pruning is the primary lever for creating weight sparsity, but its design space is vast — scoring, grouping, scheduling, fine-tuning, and the choice of starting model all interact. Energy-aware and platform-aware approaches (EAP, NetAdapt) systematically outperform naive magnitude-based methods because they target the right objective from the start.

---

## Chapter 4 — The Hardware Gap and the Road Ahead

> *Slides: L07-52 … L07-53*

### What the lecture concludes

The summary slide crystallizes the key messages of L07:

![Summary — sparsity's benefits, typical sparsity levels, and the need for specialized hardware](../../assets/L07/L07-p52-summary.png)

- Sparsity can reduce **number of operations, data movement, and storage cost** simultaneously.
- **Fine-tuning** is essential: it allows the model to recover accuracy after weights are removed and to reach higher sparsity without accuracy collapse.
- Practical DNN sparsity is **30–70%** — meaningful, but far below the >99% threshold at which existing dense software libraries (cuSPARSE, MKL-sparse) begin to show gains. Off-the-shelf software does not help here.
- **Coarse-grained pruning** can also improve speed and storage cost without requiring custom sparse hardware.
- Directly targeting hardware metrics (energy, latency) produces better accuracy vs. complexity tradeoffs than indirect proxies (operation count, weight count).

### The missing piece: specialized hardware

The final message of L07 is a forward pointer: **exploiting 30–70% sparsity requires specialized hardware**. Dense accelerators waste cycles and energy on zero multiplications even when the software knows the zeros are there. Efficiently skipping those zeros requires:

- Detection of zero operands at the PE level (or before data is fetched).
- Compression formats that allow non-zero values to be stored and retrieved without the overhead of their zero neighbors.
- Load-balancing mechanisms that prevent idle PEs when one input stream is denser than another.

These are exactly the topics of L08–L10.

> **Why it matters:** Without hardware support, sparsity is theoretical. With the right hardware (the subject of the next three lectures), 30–70% weight or activation sparsity translates directly into near-proportional reductions in energy and execution time.

---

## Key Terms

| Term | Gloss |
|---|---|
| **Sparsity** | The fraction of values in a tensor that are zero (or below some threshold). High sparsity = many zeros. |
| **Density** | The complement of sparsity: fraction of *non-zero* values. A 75%-sparse tensor has 25% density. |
| **Activation sparsity** | Zeros in intermediate feature maps, arising primarily from ReLU. |
| **Weight sparsity** | Zeros in learned model parameters, primarily created by pruning. |
| **Effectual operation** | A multiply-accumulate that changes the output (neither operand is zero). |
| **Ineffectual operation** | A multiply-accumulate where at least one operand is zero; the result is always 0 or unchanged. |
| **Pruning** | The process of setting model weights to zero (or removing them) to reduce computation and storage. |
| **Magnitude-based pruning** | Remove weights with the smallest absolute values; the most common scoring method. |
| **Energy-aware pruning (EAP)** | Incorporate the energy cost of weight access (determined by its location in the memory hierarchy) into the pruning score. |
| **Saliency** | A measure of how much a weight's removal impacts training error (from Optimal Brain Damage). |
| **Fine-tuning** | Retraining the surviving weights after pruning to recover lost accuracy. |
| **Splicing** | Allowing previously pruned weights to be restored during fine-tuning if they become important again. |
| **Scheduling** | Deciding how many weights to prune in each iteration of the pruning loop. |
| **Unstructured sparsity** | Zeros scattered at arbitrary locations (individual weights). Highest flexibility, hardest for hardware. |
| **Structured sparsity** | Zeros in regular patterns (whole rows, channels, or filters). Easier for hardware, higher accuracy cost. |
| **Pattern-based pruning** | A middle-ground approach: prune based on a small set of structured zero patterns within filters. |
| **NetAdapt** | A platform-aware DNN adaptation algorithm that uses empirical latency/energy measurements to guide pruning. |
| **Cnvlutin** | A hardware architecture (ISCA 2016) that skips zero-activation multiplications in convolutions. |
| **SnaPEA** | A predictor (ISCA 2018) that terminates partial-sum computation early when the ReLU output is predictably zero. |
| **Mixture of Experts (MoE)** | A model architecture that activates only a subset of "expert" sub-networks per input; a form of dynamic coarse-grained sparsity. |
| **GNN** | Graph Neural Network — operates on sparse adjacency matrices, a structural form of sparsity. |

---

## Takeaways

- DNN tensors contain large fractions of zeros: ~**75% of AlexNet's convolutional activations** are zero after ReLU; weight pruning can create **50–80% weight sparsity** without significant accuracy loss.
- Sparsity yields two hardware benefits: **reduced data movement/storage** (skip fetching zeros) and **reduced computation** (skip multiply-accumulates with a zero operand).
- The **effectual/ineffectual operation** framework precisely defines the opportunity: actual hardware captures only the fraction of that opportunity that its complexity budget allows.
- **Pruning** is a four-stage iterative loop: **scoring → grouping → ranking → fine-tuning**. Scheduling controls how aggressively to prune per iteration.
- **Energy-aware scoring** (EAP) outperforms magnitude-based scoring by **1.7×** on AlexNet energy because it targets the actual objective, not a proxy.
- **MAC count is a poor proxy** for real latency (NetAdapt): always measure on the actual platform.
- **Unstructured sparsity** gives the most compression but is the hardest for hardware; **structured sparsity** trades some accuracy for hardware regularity.
- Practical DNN sparsity is **30–70%** — standard software libraries do not help; **specialized hardware is required**.
- Starting from a well-designed efficient model matters as much as pruning: an unpruned efficient model can beat a pruned inefficient one at the same compute budget.

---

## Connections to Later Lectures

- **L08–L10 (Sparse Architectures):** The direct sequel. L07 establishes the *source and magnitude* of sparsity; L08–L10 show how hardware (Eyeriss v2, SCNN, ExTensor, and others) is designed to detect, compress, and exploit it efficiently.
- **L01 Pyramid of Concerns — Format layer:** Weight and activation sparsity are manifestations of the **Format** layer. Choosing whether to use a dense or compressed representation, and at what granularity, is a Format-layer decision that interacts with the Mapping and Architecture layers above and below it.
- **L01 Energy hierarchy:** The entire motivation for energy-aware pruning (EAP) and platform-aware adaptation (NetAdapt) traces back to the L01 insight that DRAM access costs ~200× an ALU operation. Skipping a zero-weight fetch from DRAM saves 200× the energy of the skipped multiply.
- **L05–L06 (Mapping / Partitioning):** The energy cost of a weight access in EAP depends on the dataflow (mapping): different loop orderings cause the same weight to be read from DRAM, the global buffer, or the register file. Pruning scores that ignore this are suboptimal.
- **L12 (Precision):** Reduced precision (quantization) and sparsity are complementary compression techniques: both reduce the bit-width and count of values that need to be stored and moved. Many production accelerators combine them.

---

## Appendix — Slide-to-Section Map

| Slides | Section |
|---|---|
| L07-1 | Title |
| L07-2 … L07-8 | Ch.1 — What Sparsity Buys (and What It Costs) |
| L07-9 … L07-23 | Ch.2 — Activation Sparsity: Where Natural Zeros Come From |
| L07-24 … L07-51 | Ch.3 — Weight Sparsity: Pruning DNN Models |
| L07-52 … L07-53 | Ch.4 — The Hardware Gap and the Road Ahead |
