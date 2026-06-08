# L01 - Introduction and Applications

> **Course:** 6.5930/1 - Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer and Vivienne Sze, MIT EECS
> **Lecture date:** February 2, 2026
> **Slides:** `Lecture/L01-Intro_and_Applications.pdf`, 53 pages
>
> This chapter reconstructs the teaching narrative behind Lecture 1. It is organized as a self-contained textbook chapter, not as a slide summary. Slide references are used as source anchors; copied slide figures are intentionally not embedded.

---

## TL;DR

Deep learning became practical because three things arrived together: abundant data, GPU-accelerated compute, and better machine-learning techniques. Lecture 1 focuses on the second ingredient. DNNs create a hardware problem because their compute demand, memory traffic, deployment scale, and energy cost grow faster than general-purpose processors improve.

The course answer is not simply "build more MAC units." A useful DNN accelerator is a coordinated design of computation, mapping, data representation, memory hierarchy, interconnect, and physical technology. Lecture 1 introduces the recurring mental model: arithmetic is cheap compared with moving data far through the memory hierarchy. If a datum can be read once from DRAM and reused many times near a processing element, the accelerator can save far more energy than it would save by merely making the multiplier slightly faster.

The main framework is the TeAAL Pyramid of Concerns: architecture constrains compute, mapping, format, and binding choices. The rest of the course teaches those layers in detail: DNN computations and Einsums, memory and metrics, mapping and dataflow, sparsity, precision, compute-in-memory, and modeling tools for evaluating design tradeoffs.

---

## What Problem This Lecture Solves

Lecture 1 answers a deceptively simple question:

**Why does deep learning need a hardware architecture course of its own?**

The answer has four parts.

1. DNN computation is no longer a small software workload running on a generic machine. It is a dominant consumer of compute, money, and energy.
2. The historical path of relying on Moore's Law and Dennard scaling no longer provides enough performance or energy improvement.
3. DNN workloads have structure: tensor operations, massive parallelism, reuse, approximate numerical tolerance, and data attributes such as sparsity. Hardware can exploit that structure.
4. Once hardware becomes specialized, we need a disciplined vocabulary for comparing designs. Otherwise every accelerator looks unique and incomparable.

This lecture therefore sets the course agenda: learn to read a DNN accelerator by asking what it computes, how it maps work to hardware, how it represents data, how it binds abstract operations to concrete resources, and what architecture constrains those choices.

---

## Why This Lecture Matters

If you only remember that "AI needs lots of GPUs," you will miss the architectural lesson. The important point is that DNN hardware is often limited by **movement**, **utilization**, and **constraints**, not only by peak arithmetic throughput.

A GPU, TPU, mobile neural engine, sparse accelerator, or compute-in-memory array may all advertise impressive TOPS. But a workload only achieves useful throughput when the right data arrives at the right compute units at the right time. If the PE array waits for memory, if only half the PEs are active, or if compression metadata costs more than the skipped zeros save, the advertised peak is not the actual result.

Lecture 1 gives you the first version of the course's central question:

**For this workload, where is the cost really paid: arithmetic, data movement, storage capacity, bandwidth, control irregularity, or lost utilization?**

---

## Prerequisites and Mental Model

You do not need to know accelerator design yet, but the chapter assumes four basic ideas.

- A **DNN** is a computation graph of layers, many of which reduce to tensor operations such as matrix multiplication, convolution, and attention.
- A **MAC** (multiply-accumulate) computes a product and adds it into an accumulated result. DNN layers perform many MACs.
- A **memory hierarchy** has small fast storage near compute and larger slower storage farther away: register file, on-chip buffers, and DRAM.
- A **hardware architecture** is not just a set of arithmetic units. It is the complete organization of compute units, storage, interconnect, control, and scheduling assumptions.

The simplest mental model for this lecture is:

**A DNN accelerator is a factory for tensor operations. The machines in the factory are the PEs, but the factory is productive only if materials move efficiently through storage and interconnect.**

This is a teaching interpretation, but it captures why later lectures spend so much time on dataflow, mapping, sparsity formats, and cost modeling.

---

## Learning Objectives

After studying this chapter, you should be able to:

- Explain why data, GPUs, and new ML techniques jointly enabled modern AI.
- Distinguish training and inference in terms of cost per iteration, frequency, and deployed energy.
- Explain why the slowdown of Moore's Law and Dennard scaling motivates domain-specific hardware.
- Describe why on-device inference matters for privacy, latency, communication, power, and thermal limits.
- Use the TeAAL Pyramid of Concerns to separate architecture, compute, mapping, format, and binding decisions.
- Describe the canonical DNN accelerator template: DRAM, global buffer, NoC, PE array, RF, and ALU.
- Explain why data movement can dominate energy, using the normalized hierarchy from Lecture 1 slide 43.
- Interpret a roofline-style bound as a sequence of constraints that lowers achievable throughput below theoretical peak.
- State the course vocabulary for analyzing a new accelerator: order, partitioning, dataflow, memory movement, data-attribute optimizations, co-design, and flexibility.

---

## Main Narrative

### 1. The Three Ingredients of Modern AI

Lecture 1 begins with three ingredients: big data availability, GPU acceleration, and new ML techniques. The slides give concrete data-scale examples: hundreds of millions of images uploaded per day, hundreds of hours of video uploaded per minute, and petabytes of customer data processed hourly (Lecture 1 slide 2).

Those examples are not just motivational trivia. They show why deep learning could improve when model capacity increased: there was enough data to train larger models, enough compute to process the data, and enough algorithmic progress to make the training useful.

The lecture then quotes Ilya Sutskever's 2017 statement that compute has been the "oxygen" of deep learning (slide 3). The pedagogical point is that compute is not a passive resource. It shapes what models researchers can try, how quickly they can iterate, and whether a deployed system can afford to run at scale.

### 2. Why GPUs Entered the Story

GPUs were originally attractive because DNN workloads contain large, regular, parallel tensor operations. A matrix multiplication can expose thousands of independent products and sums. That is a poor match for a small number of general-purpose CPU cores but a natural match for many parallel arithmetic lanes.

Modern GPUs then became more DNN-specific. Lecture 1 notes that GPUs added hardware for matrix multiplication, reduced precision formats, and sparsity support; NVIDIA introduced Tensor Cores in 2017 (slide 4). This matters because even "general" GPUs became partly domain-specialized once DNNs became economically important.

Teaching interpretation: GPUs are a bridge between general-purpose computing and fixed-function accelerators. They remain programmable, but they increasingly include structures that assume tensor-heavy workloads.

### 3. The Compute and Energy Crisis

Lecture 1 uses several scale examples to show that DNN compute is expensive.

- The compute demand from AlexNet to AlphaGo Zero increased by about $300{,}000\times$ in the OpenAI AI-and-compute curve reproduced on slide 9.
- GPT-3 is described as a 96-layer, 175-billion-parameter model requiring about $3.14 \times 10^{23}$ FLOPs for training (slide 12, citing Brown et al. and Lambda Labs).
- The slide estimates that training GPT-3 on one Tesla V100 would take about 355 years and cost about $4.6 million on the lowest-cost cloud GPU provider available to that source (slide 12).
- Data-center energy is presented as a growing electricity concern: slide 8 cites Goldman Sachs, April 2024, for the estimate that data centers accounted for about 3% of US electricity demand in 2022 and could grow to about 8% by 2030.

These numbers should be read as source-anchored motivation, not as universal constants. Hardware, pricing, models, and datacenter assumptions change quickly. The stable lesson is the trend: compute and energy are first-order design constraints.

### 4. Training Versus Inference

Training and inference stress hardware differently.

**Training** adjusts model parameters. It usually performs forward passes, backward passes, gradient accumulation, and optimizer updates. It has high cost per iteration but is performed less frequently than inference.

**Inference** runs an already trained model to produce outputs. It is cheaper per request but may happen billions or trillions of times. A small inefficiency in inference can become a large datacenter or battery cost when multiplied by deployment scale.

Lecture 1 supports this distinction with two cited deployment observations:

- At Google, across 2019-2021, about three-fifths of ML energy use was inference and two-fifths was training (slide 15, citing Patterson, Computer 2022).
- At Meta, the rough power-capacity breakdown for AI infrastructure was 10:20:70 for experimentation, training, and inference (slide 15, citing Wu, MLSys 2022).

The implication is subtle: even if training gets headlines because it is expensive, inference hardware often determines long-term energy and deployment feasibility.

### 5. Why On-Device Computing Matters

The cloud is powerful, but cloud inference has costs that are not only financial.

- **Communication:** sending sensor data to a datacenter consumes bandwidth and energy.
- **Privacy:** raw data may be sensitive.
- **Latency:** a round trip to the cloud can be too slow for interactive or safety-critical systems.

Lecture 1 uses self-driving cars to make this concrete. Slides 19-20 state that cameras and radar can generate about 6 GB every 30 seconds, that self-driving car prototypes used about 2,500 W of compute, and that an autonomous-vehicle scenario with 10 DNN inferences at 60 Hz on 10 cameras leads to 21.6 million inferences per hour per vehicle. The arithmetic is:

$10 \text{ DNNs} \times 60 \text{ frames/s} \times 10 \text{ cameras} \times 3600 \text{ s/hour} = 21{,}600{,}000$ inferences/hour.

For one million vehicles, the same workload becomes:

$21.6 \times 10^6 \times 10^6 = 21.6 \times 10^{12}$ inferences/hour.

This is why the lecture calls the vehicle a "data center on wheels" (slide 20). The workload has datacenter scale, but the power, cooling, latency, and reliability constraints are vehicle constraints.

### 6. Why General-Purpose CPUs Are Not Enough

Slides 23-25 walk through increasingly complex CPU pipelines: simple in-order, out-of-order, and out-of-order simultaneous multithreading. These diagrams are not included to teach CPU microarchitecture in detail. They make a contrast.

A general-purpose CPU must handle branches, pointer chasing, interrupts, unpredictable memory accesses, many instruction types, and many programs. It therefore spends area and energy on structures such as branch prediction, register renaming, dependency prediction, speculative execution, retirement policy, and cache replacement.

DNN tensor kernels are often more regular. They have loop nests, repeated tensor accesses, and predictable reuse. A domain-specific accelerator can remove some general-purpose overhead and spend resources on:

- many simple PEs,
- local storage near PEs,
- an interconnect matched to tensor movement,
- schedules that reuse weights, activations, and partial sums,
- support for reduced precision or sparsity when the workload permits it.

This does not mean CPUs are bad. It means the CPU is optimized for flexibility, while the DNN accelerator trades some flexibility for energy efficiency and throughput on a narrower workload domain.

### 7. The End of "Free" Scaling

Lecture 1 explicitly connects specialization to the slowdown of Moore's Law and Dennard scaling (slide 22). Moore's Law refers to historical transistor-density scaling; Dennard scaling refers to the historical expectation that power density would remain manageable as transistors shrank.

When these trends slowed, architects could no longer expect each new process generation to automatically deliver large speed and energy improvements for the same general-purpose design. The response was a shift toward domain-specific architectures: if physics no longer gives enough improvement for free, architecture must exploit workload structure.

### 8. Every Accelerator Is Unique, So We Need a Framework

Slides 26-27 show many accelerator designs: Eyeriss, Eyeriss v2, SCNN, ExTensor, Gamma, spZip, ISOSceles, RAELLA, Highlight, Overbooking, Trapezoid, and FuseMax. The visual lesson is that accelerators do not share a single obvious shape.

Without a framework, a student might try to memorize each chip. That fails quickly. The better approach is to ask a fixed set of questions:

- What computation does it accelerate?
- What data does it try to keep stationary or nearby?
- Which loops are parallelized, tiled, or reordered?
- How is sparse or compressed data represented?
- How are abstract work units assigned to concrete PEs and buffers?
- What architecture constraints make those choices good or bad?

The next section names this framework.

### 9. The TeAAL Pyramid of Concerns

Lecture 1 slide 28 introduces the TeAAL Pyramid of Concerns, attributed on the slide to Nayak et al., MICRO 2023. The pyramid separates accelerator reasoning into layers:

| Concern | Question | Example decision |
|---|---|---|
| **Compute** | What mathematical operation is being evaluated? | Matrix multiplication, convolution, attention, an Einsum |
| **Mapping** | How is the operation scheduled? | Loop order, tiling, parallelism, dataflow |
| **Format** | How is data represented? | Dense tensor, compressed sparse row, run-length encoding |
| **Binding** | Which concrete resource performs each abstract action? | Assign tile $T_{ij}$ to PE group 3 and buffer bank 1 |
| **Architecture** | What hardware constrains all of the above? | PE count, NoC topology, buffer sizes, memory bandwidth |

The architecture is shown as a constraint on the other concerns. For example, a mapping that keeps a whole weight tile stationary is only possible if local storage is large enough. A sparse format that saves DRAM traffic may be unattractive if the architecture lacks hardware to decode metadata efficiently.

The important habit is to avoid mixing layers too early. "This accelerator is fast" is not an explanation. A better explanation is: "The compute formulation reduces redundant work; the mapping reuses the key tensor; the format compresses zeros; the binding keeps enough PEs active; and the architecture provides the storage and bandwidth needed for that schedule."

### 10. FuseMax as a Pyramid Example

Slides 29-34 use FuseMax, an attention accelerator cited as Nayak et al., MICRO 2024, to demonstrate how improvements can occur at different layers of the pyramid.

The slides identify four types of changes:

- improved computation that reduces data movement,
- changed architecture that enables a better mapping,
- better mapping that exploits capabilities exposed by architecture changes,
- improved binding that improves resource utilization.

The PE-utilization slides then show a sequence from baseline to enhanced computation, improved architecture/mapping, and improved binding (slides 30-33), followed by a speedup-on-attention slide (slide 34). This chapter does not reproduce the figures or exact plotted values. The source-based claim is only that Lecture 1 uses FuseMax to illustrate cross-layer co-design.

The teaching lesson is general: a high-performance accelerator is rarely the result of one isolated trick. It is often a stack of compatible choices across compute, architecture, mapping, and binding.

### 11. The Canonical DNN Accelerator Template

Slide 43 introduces a typical DNN accelerator:

**DRAM $\rightarrow$ global buffer $\rightarrow$ NoC $\rightarrow$ PE array $\rightarrow$ local RF and ALU.**

The terms matter:

- **DRAM** stores large tensors but is far from the compute and expensive to access.
- A **global buffer** is on-chip storage shared by many PEs.
- A **NoC** (network-on-chip) moves data between buffers and PEs.
- A **PE** (processing element) performs arithmetic and often has a small local register file or local buffer.
- The **RF** is tiny compared with DRAM, but it is close to the ALU and cheap to access.

The architecture is called **spatial** because many PEs exist at once, and different pieces of the tensor computation can be assigned to different physical places. This is unlike a single scalar pipeline that reuses the same execution unit over time.

### 12. The Central Energy Lesson: Data Movement Dominates

Slide 43 gives a normalized energy hierarchy measured from a commercial 65 nm process. The slide uses an ALU access or operation as a reference of $1\times$ and reports approximate relative costs:

| Source or movement level | Approximate normalized energy |
|---|---:|
| ALU / local RF reference | $1\times$ |
| Neighbor PE movement | $2\times$ |
| Global buffer access | $6\times$ |
| DRAM access | $200\times$ |

Do not overinterpret these as timeless physical constants. They are slide-anchored illustrative values for a process and modeling setup. Their purpose is to teach the ordering: farther and larger memories cost much more energy than nearby small memories.

The architectural implication is enormous. Suppose a weight is used by 16 MACs.

- If the accelerator reads the weight from DRAM each time, the weight traffic costs roughly $16 \times 200 = 3200$ normalized energy units.
- If it reads the weight once from DRAM and reuses it from a local RF for the next 15 uses, the weight traffic costs roughly $200 + 15 \times 1 = 215$ units.

The arithmetic work is the same 16 MACs. The energy changes because the mapping changed where the data lives between uses.

This is the first version of the dataflow argument that later lectures make precise: a good accelerator schedule tries to maximize useful reuse in cheap storage and minimize repeated movement through expensive storage.

### 13. Design Choices in a DNN Accelerator

Slide 44 lists the design choices that will recur in labs and later lectures:

- **PE array:** number of PEs and NoC connections.
- **Memory hierarchy:** number of levels, capacity per level, and data layout.
- **Scheduling:** operation ordering, dataflow, tiling, parallelism, and fusion.
- **Sparsity handling:** gating, skipping, and representation format.
- **Technology:** implementation choices for compute, memory, and interconnect, including emerging devices.

The key point is that these choices interact. A large PE array is useful only if mapping and memory bandwidth keep PEs active. A compressed sparse format is useful only if it saves more movement and arithmetic than it adds in metadata and control. A new memory technology is useful only if it improves the bottleneck that the workload actually has.

### 14. Roofline as a Way to See Inefficiency

Slide 45 introduces a roofline-style evaluation method. A conventional roofline model relates achievable throughput to compute intensity. Compute intensity is often expressed as:

$\text{compute intensity} = \frac{\text{useful operations}}{\text{data moved}}$.

In this lecture's setting, useful operations may be MACs, and data moved may be bytes, words, or tensor elements depending on the model. High compute intensity means each fetched datum supports many operations. Low compute intensity means the machine must move many data items per operation.

The slide describes a sequence of tightening constraints:

1. maximum workload parallelism,
2. maximum dataflow parallelism,
3. active PEs under finite PE count,
4. active PEs under fixed PE-array dimensions,
5. active PEs under fixed storage capacity,
6. lower utilization due to insufficient average bandwidth,
7. lower utilization due to insufficient instantaneous bandwidth.

The teaching interpretation is that peak performance is only the first ceiling. Real performance is lower because the workload, mapping, array shape, storage, and bandwidth each impose constraints.

### 15. Modeling Instead of Building Every Design

Slide 46 explains why the course emphasizes architectural modeling. RTL implementation is too slow for broad design-space exploration. The course therefore uses modeling tools such as AccelForge and Accelergy to evaluate architectural decisions before detailed hardware implementation.

This matters pedagogically because it lets the course ask "what if?" questions:

- What if the global buffer were larger?
- What if the dataflow kept weights stationary instead of outputs?
- What if sparse activations were compressed?
- What if bandwidth were the bottleneck rather than PE count?

The goal is not to replace RTL forever. The goal is to use fast models to identify promising designs and understand tradeoffs before spending effort on detailed implementation.

---

## Worked Examples

### Example 1: Counting Vehicle Inferences

Slide 20 describes 10 DNN inferences at 60 Hz on 10 cameras. The number of inferences per hour is:

$10 \times 60 \times 10 \times 3600 = 21{,}600{,}000$.

This example teaches scale. A single inference may feel small, but deployment frequency can dominate. If one million vehicles ran this workload for one hour, the fleet would perform:

$21{,}600{,}000 \times 1{,}000{,}000 = 21.6 \times 10^{12}$ inferences.

Hardware implication: an energy saving of even a small amount per inference can become significant at fleet or datacenter scale.

### Example 2: Why Reuse Beats Repeated DRAM Reads

Use the slide-43 normalized energy numbers as a teaching model. Suppose one activation value is used in eight MACs.

Repeated DRAM strategy:

$E_{\text{repeated DRAM}} \approx 8 \times 200 = 1600$ normalized units.

Read once and reuse locally:

$E_{\text{reuse}} \approx 200 + 7 \times 1 = 207$ normalized units.

The approximate reduction is:

$\frac{1600}{207} \approx 7.7\times$.

This is not a claim that every accelerator gets a $7.7\times$ energy reduction. It is a small numerical example showing why mapping and storage locality matter.

### Example 3: Reading Compute Intensity

Suppose a tile performs 4096 MACs and moves 1024 data words from a higher memory level. Its compute intensity is:

$\text{CI} = \frac{4096 \text{ MACs}}{1024 \text{ words}} = 4 \text{ MACs/word}$.

If a better mapping performs the same 4096 MACs while moving only 256 words, then:

$\text{CI} = \frac{4096}{256} = 16 \text{ MACs/word}$.

Hardware implication: the arithmetic did not change, but the second mapping gives each moved word more useful work. In roofline terms, that can move the workload away from a bandwidth limit.

---

## Key Equations and How to Read Them

### Fleet-Scale Inference Count

$N_{\text{inf/hour}} = N_{\text{models}} \times f_{\text{frames/s}} \times N_{\text{cameras}} \times 3600$.

This equation counts how deployment scale multiplies a workload. In slide 20's vehicle example, $N_{\text{models}} = 10$, $f_{\text{frames/s}} = 60$, and $N_{\text{cameras}} = 10$, giving $21.6$ million inferences per hour per vehicle.

### Compute Intensity

$\text{compute intensity} = \frac{\text{useful operations}}{\text{data moved}}$.

This measures how much computation each moved datum supports. Higher compute intensity generally makes it easier to keep a machine busy with a fixed bandwidth, but only if enough parallelism and storage are available.

### Simple Data-Movement Energy Model

$E_{\text{move}} \approx \sum_{\ell} n_{\ell} e_{\ell}$.

Here $n_{\ell}$ is the number of accesses or movements at memory level $\ell$, and $e_{\ell}$ is the energy cost per access at that level. The model is intentionally simple. It teaches why reducing the number of expensive DRAM accesses can matter more than reducing a small number of cheap local accesses.

---

## Hardware Implications

- **Energy:** moving data from DRAM can dominate MAC energy, so local reuse is central.
- **Bandwidth:** a PE array can be idle even when it has enough arithmetic units, because memory cannot feed it fast enough.
- **Latency:** on-device inference avoids network round trips and is necessary for some interactive and safety-critical systems.
- **Area:** general-purpose CPU structures buy flexibility, while accelerators spend more area on regular arrays, buffers, and interconnect.
- **Utilization:** peak TOPS is meaningful only when enough PEs are active over time.
- **Memory capacity:** a dataflow may be theoretically good but impossible if stationary tiles do not fit in local or global buffers.
- **Programmability:** specialization improves efficiency but can make software, compilers, and mapping tools harder.
- **Correctness and robustness:** reduced precision, sparsity, and approximate methods require algorithm-hardware co-design so efficiency does not silently damage model quality.
- **Scalability:** datacenter and fleet-scale inference turn tiny per-query inefficiencies into large system costs.

---

## Common Misconceptions

### Misconception: A DNN accelerator is just a large MAC array.

A MAC array is important, but it is not sufficient. Energy and throughput depend on how weights, activations, and partial sums move through the memory hierarchy, how many PEs are active, and whether the mapping exposes reuse.

### Misconception: Training is always the main energy problem because it is expensive.

Training has high cost per iteration, but inference can dominate deployed energy because it happens far more often. Lecture 1 supports this with Google and Meta energy/power breakdowns on slide 15.

### Misconception: TOPS tells you which accelerator is best.

TOPS is a peak arithmetic metric. It does not tell you whether the workload has enough parallelism, whether data can be delivered on time, whether storage capacity is sufficient, or whether sparse metadata creates overhead.

### Misconception: Specialization means no flexibility.

Specialization is a spectrum. GPUs with Tensor Cores are still programmable, but they include tensor-specific hardware. Mobile neural engines, TPUs, and sparse accelerators choose different points in the flexibility-efficiency tradeoff.

### Misconception: A sparse or compressed format automatically saves energy.

Format choices help only when the saved arithmetic and data movement exceed the overhead of metadata, decoding, irregular access, and load imbalance. This becomes a major theme in the sparsity lectures.

---

## Connections

### Connection to Previous Knowledge

This lecture builds on basic computer architecture and machine-learning ideas. From architecture, it uses the idea that compute, memory, and interconnect have different costs. From ML, it uses the idea that DNNs are tensor programs with training and inference phases.

### Connections to Later Lectures

- **L02-L04:** The course formalizes DNN components and tensor computations, including Einsums and Transformers.
- **L03:** The energy hierarchy and key metrics become more precise.
- **L05-L06:** The mapping layer becomes dataflow, tiling, partitioning, and parallel scheduling.
- **L07-L10:** The format layer becomes sparsity, sparse representations, and sparse accelerator architecture.
- **L11-L13:** Advanced technologies, precision, and motion/data-movement calculation revisit the same energy argument from different angles.

The cross-lecture thread is simple: every later optimization should be read as an attempt to improve the ratio between useful computation and expensive movement or underutilized hardware.

---

## Paper and Source Bridge

Lecture 1 is mostly a course-motivation lecture rather than a paper lecture, but it points to several sources that establish the course's framing.

**Local PDF availability note:** the repository now includes local PDFs for TeAAL and FuseMax, so those two bridges are paper-verified as well as slide-anchored.

### TeAAL, Nayak et al., MICRO 2023

- **Bibliographic identity:** *TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators*, MICRO 2023. Local PDF: `papers/TeAAL.pdf`.
- **Problem addressed:** sparse tensor accelerators are often described either with verbose RTL or incomplete diagrams, making designs difficult to compare precisely.
- **Core idea used here:** TeAAL represents accelerator designs as cascades of mapped Einsums plus transformations on fibertrees; the paper's abstraction hierarchy is the source behind the course's concern-layer framing.
- **Lecture relevance:** slide 28 uses this pyramid as the course's organizing framework. The paper makes that slide concrete by separating computation, mapping, format, architecture, and binding specifications.
- **Key claims used here:** TeAAL uses Einsums to express tensor computations while leaving iteration order to mapping (Section 2.2); mapping includes loop order, rank partitioning, and work scheduling (Section 2.3); the language also models formats, architectures, and bindings through a simulator generator (Sections 3-4).
- **What students should remember:** when analyzing a new accelerator, do not start by memorizing the chip diagram. First separate the concern layers, then ask which layer the design actually changes.
- **Limitation:** Lecture 1 uses TeAAL at a high level; the sparse fibertree details become relevant in later sparse-architecture lectures.

### FuseMax, Nayak et al., MICRO 2024

- **Bibliographic identity:** *FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design*, 2024. Local PDF: `papers/FuseMAX.pdf`.
- **Problem addressed:** attention has tensor products and softmax steps with different compute and memory behavior; prior spatial designs can reduce bandwidth but still suffer under-utilization, especially around softmax.
- **Core idea used here:** FuseMax describes attention as cascades of extended Einsums, then co-designs mapping, architecture, and binding so 1D and 2D PE arrays stay highly utilized.
- **Lecture relevance:** slides 29-34 use FuseMax to show that speedup can come from coordinated compute, architecture, mapping, and binding changes rather than a single layer of the pyramid.
- **Key claims used here:** the paper states FuseMax targets nearly 100% compute utilization and sequence-length-independent on-chip buffer requirements (Abstract); it uses a 1-pass attention cascade and a novel mapping/binding on a spatial architecture (Sections IV-V); it reports 6.7x average attention speedup over FLAT and 5.3x end-to-end inference speedup over FLAT under its evaluation setup (Abstract and Section VI).
- **What students should remember:** accelerator gains often come from a compatible stack of choices. FuseMax is useful in L01 because it demonstrates the pyramid as a design method, not because students need all attention details yet.
- **Limitation:** The speedup numbers are paper-specific and depend on the modeled workloads, baselines, and hardware assumptions; L01 uses them only to illustrate cross-layer co-design.

### Efficient Processing of Deep Neural Networks, Sze and Emer et al.

- **Problem addressed:** how to design and evaluate efficient DNN processing systems.
- **Core idea used here:** DNN accelerator efficiency depends heavily on data movement, memory hierarchy, mapping, and workload characteristics.
- **Lecture relevance:** slides 41 and 45 point to the course textbook/readings and Chapter 6 for roofline-style evaluation.
- **What students should remember:** the lecture's vocabulary is aligned with a broader methodology for evaluating DNN hardware, not just with one slide deck.

---

## Standalone Study Guide

### What to Master Before Moving On

- Explain the difference between training and inference without using only "training is learning, inference is using."
- Derive slide 20's vehicle inference count.
- Explain why a DRAM access can be more important to optimize than a MAC.
- Use the TeAAL Pyramid to classify a design change as compute, mapping, format, binding, or architecture.
- Explain why peak performance must be tightened by workload parallelism, dataflow parallelism, PE count, array shape, storage, and bandwidth.

### Self-Check Questions

1. Why did GPUs become central to deep learning before fully custom accelerators became common?
2. Why can inference dominate total energy even if a single inference is much cheaper than one training step?
3. In the vehicle example, what changes if the camera count doubles but frame rate stays fixed?
4. Why does the slowdown of Dennard scaling matter for DNN hardware?
5. Give one example of a compute-layer change and one mapping-layer change.
6. Why might a sparse format fail to improve energy?
7. What does a roofline model hide, and why is it still useful?
8. If a PE array has high peak TOPS but low utilization, which concern layers might be responsible?

### Exercises

1. **Conceptual:** Explain why "move less data" is a more useful design principle than "perform fewer operations" for some DNN accelerators. Give one case where each principle matters.
2. **Small calculation:** A value is reused 32 times. Compare repeated DRAM reads with one DRAM read plus 31 local reads using slide 43's $200\times$ and $1\times$ costs.
3. **Design tradeoff:** A mobile accelerator must run under a 1 W budget. List three architectural choices that could reduce energy and one way each choice might reduce flexibility.
4. **Pyramid classification:** Classify these changes: using INT8 instead of FP16, changing loop order, increasing global-buffer size, assigning tiles to different PE groups, and using compressed sparse row format.
5. **Paper bridge:** Read the abstract of a DNN accelerator paper. Identify its compute, mapping, format, binding, and architecture claims. Mark which claims are quantitative and need source anchors.
6. **Open-ended reasoning:** Choose a workload such as convolution, attention, or recommendation inference. Predict whether its bottleneck is likely arithmetic, memory bandwidth, storage capacity, or irregularity. State your assumptions.

---

## Key Terms

### Deep Neural Network (DNN)

A model made of many layers of numerical transformations. In this course, DNNs matter as hardware workloads: they generate tensor operations, memory traffic, and repeated inference at scale.

### Training

The process of adjusting model parameters. Training usually has high cost per iteration because it includes forward computation, backward computation, and parameter updates.

### Inference

The process of running a trained model to produce outputs. Inference is usually cheaper per run than training but can dominate total energy because it happens many more times.

### MAC (Multiply-Accumulate)

The operation $a \times b + c$. MACs are central to matrix multiplication, convolution, and many other DNN kernels, but MAC count alone does not determine energy.

### Processing Element (PE)

A small compute unit, often containing an ALU and local storage. DNN accelerators commonly arrange many PEs into a spatial array.

### Register File (RF) / Local Buffer

Small storage close to a PE. It has limited capacity but low access energy, making it valuable for reuse.

### Global Buffer

On-chip storage shared by multiple PEs. It is larger than local PE storage but more expensive to access.

### DRAM

Off-chip memory with high capacity and high access energy. Reducing repeated DRAM traffic is a central accelerator goal.

### Network-on-Chip (NoC)

The interconnect that moves data among buffers and PEs. NoC design affects bandwidth, energy, and utilization.

### Domain-Specific Hardware

Hardware optimized for a class of workloads rather than arbitrary programs. It trades some flexibility for better performance or energy efficiency on that domain.

### TeAAL Pyramid of Concerns

A framework that separates compute, mapping, format, binding, and architecture. It helps compare accelerators without reducing them to a single block diagram.

### Compute

The mathematical operation being evaluated, such as an Einsum, convolution, matrix multiplication, or attention step.

### Mapping

The schedule that places computation and data movement onto hardware. It includes loop order, tiling, parallelism, and dataflow.

### Format

The representation of data, especially dense versus compressed or sparse forms. Format affects memory traffic and decoding overhead.

### Binding

The assignment of abstract work and data to concrete hardware resources such as PE IDs, buffer banks, and time slots.

### Roofline Model

A performance model that relates throughput to compute intensity and machine limits. Lecture 1 uses it as a way to tighten theoretical peak performance into a more realistic bound.

### Compute Intensity

The ratio $\text{useful operations}/\text{data moved}$. Higher compute intensity means each moved datum does more work.

### Utilization

The fraction of hardware resources doing useful work. Low utilization can erase the benefit of a large PE array.

---

## Takeaways

- Modern AI is enabled by data, compute, and ML techniques, but compute is the resource this course teaches you to architect.
- DNN hardware matters because compute demand, inference scale, energy, and cost are system-level constraints.
- Moore's Law and Dennard scaling no longer provide enough automatic improvement, so architects exploit workload structure.
- Specialized hardware is not only a faster multiplier; it is a design of compute, storage, movement, mapping, format, and binding.
- The TeAAL Pyramid gives a reusable framework for reading new accelerators.
- The canonical accelerator template is DRAM, global buffer, NoC, PE array, local RF, and ALU.
- Data movement can dominate energy; slide 43's DRAM example is the first anchor for the course's focus on reuse.
- Roofline-style evaluation teaches that peak performance is reduced by workload, mapping, storage, array, and bandwidth constraints.

---

## Connections

Lecture 1 is the roadmap for the course. It introduces terms that become precise later:

- **Workload and DNN components:** L02-L04 explain what the models compute.
- **Metrics and memory:** L03 develops the energy, bandwidth, and evaluation vocabulary.
- **Einsum:** L03-L04 provide the notation for tensor computations.
- **Mapping and dataflow:** L05-L06 turn "reuse data near compute" into loop schedules and spatial mappings.
- **Sparsity:** L07-L10 study format and data-attribute-specific optimization.
- **Precision and advanced technologies:** L11-L13 explore new ways to reduce arithmetic and movement costs.

This lecture has no previous course lecture to build on. Its role is to establish why the rest of the topics belong together.

---

## Appendix

### Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L01-1 | Title and metadata | Course identity |
| L01-2-L01-3 | Three ingredients of modern AI | Expanded into motivation and compute framing |
| L01-4-L01-7 | GPU, datacenter, specialized, and mobile DNN hardware | Used to explain the specialization spectrum |
| L01-8-L01-17 | Energy, compute demand, ChatGPT, training/inference, GPU investment | Expanded into compute/energy crisis and training-vs-inference discussion |
| L01-18-L01-21 | On-device processing and self-driving cars | Expanded with inference-count worked example |
| L01-22-L01-25 | Moore/Dennard slowdown and CPU pipelines | Used to contrast general-purpose flexibility with domain-specific efficiency |
| L01-26-L01-27 | Accelerator galleries | Used to motivate the need for a framework |
| L01-28-L01-29 | TeAAL pyramid and FuseMax enhancements | Expanded into concern-layer framework and paper/source bridge |
| L01-30-L01-34 | FuseMax utilization and speedup | Referenced qualitatively; figures and exact plotted values not reproduced |
| L01-35 | Challenges and opportunities | Integrated into hardware implications and course motivation |
| L01-36-L01-39 | Class overview, outline, takeaways, objective | Integrated into learning objectives and connections |
| L01-40-L01-42 | Staff, requirements, labs | Mentioned only where relevant for course structure |
| L01-43-L01-45 | Accelerator template, design choices, roofline | Expanded into energy hierarchy, design choices, and roofline sections |
| L01-46-L01-50 | Architectural modeling and design project | Used to explain modeling-first workflow |
| L01-51-L01-53 | Grading, late policy, prerequisites | Not central to the technical chapter; left as slide-only logistics |

## Source Notes

- The lecture ordering and quantitative motivation come from `Lecture/L01-Intro_and_Applications.pdf`.
- The AI ingredients, compute-as-oxygen quote, GPU specialization examples, datacenter/custom/mobile hardware examples, and course logistics are slide-derived from L01-2 through L01-7 and L01-36 through L01-53.
- The data-center electricity estimate is slide-derived from L01-8, which cites Goldman Sachs, April 2024.
- The $300{,}000\times$ compute-growth claim is slide-derived from L01-9, which cites OpenAI's AI-and-compute discussion and Strubell, ACL 2019.
- GPT-3 training figures are slide-derived from L01-12, which cites Brown, NeurIPS 2020, and a Lambda Labs explainer.
- Training/inference energy and power breakdowns are slide-derived from L01-15, which cites Patterson, Computer 2022, and Wu, MLSys 2022.
- The autonomous-vehicle inference example is slide-derived from L01-19 and L01-20; the arithmetic expansion in this chapter is original teaching work.
- The TeAAL Pyramid discussion is anchored in L01-28 and the local `papers/TeAAL.pdf`, especially its Sections 2.2, 2.3, and 3-4.
- The FuseMax discussion is anchored in L01-29 through L01-34 and the local `papers/FuseMAX.pdf`, especially its Abstract, Sections IV-V, and Section VI.
- The normalized energy hierarchy is slide-derived from L01-43 and is used as a pedagogical model, not as a universal technology constant.
- Worked examples that combine slide numbers with simple arithmetic are original teaching examples.

## Uncertainty Notes

- This chapter reconstructs likely lecture narration from slides and cited source anchors. The live lecture may have emphasized different examples or caveats.
- Some slide-cited sources are not present as local papers in this repository, so this chapter uses the lecture slides as the immediate source anchor rather than independently verifying every external article or estimate.
- Fast-changing cost claims, such as cloud GPU pricing and model training costs, should be treated as historical motivation tied to the cited slide date, not as current market estimates.
- The normalized energy ratios are process- and model-dependent. They are reliable as a qualitative hierarchy for this lecture, but not as exact constants for every accelerator technology.
