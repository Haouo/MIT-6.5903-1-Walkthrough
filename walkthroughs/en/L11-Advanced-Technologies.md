# L11 — Advanced Technologies

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze
> **Lecture date:** March 9, 2026 · **Slides:** 81 · **Source:** [`Lecture/L11-Advanced_Tech.pdf`](../../Lecture/L11-Advanced_Tech.pdf)
>
> This chapter reconstructs the lecture narrative from slides and local papers. It avoids reproducing slide or paper figures; any visual content is described in words.

---

## TL;DR

Most of this course has treated memory as a hierarchy around a digital MAC array. Lecture 11 asks what happens when that boundary starts to move. **Near-memory processing** keeps compute and memory distinct but shortens the path between them, using technologies such as eDRAM and 3D-stacked DRAM. **Compute-in-memory (CiM)** goes further: the memory device or array participates directly in computation.

The lecture's main arc is: data movement is expensive, so designers first move memory closer to compute, then move simple compute closer to memory, then ask whether memory itself can perform MACs. Analog CiM crossbars use Ohm's Law for multiplication and Kirchhoff's Current Law for accumulation, but practical systems are dominated by peripheral costs, especially ADCs. The **Titanium Law** expresses ADC energy as a product of energy per conversion, conversions per MAC, MACs per DNN, and utilization penalty. **RAELLA** and **CiMLoop**, presented in the slides, show why advanced technologies require cross-layer co-design across device, circuit, architecture, mapping, and workload.

---

## What Problem This Lecture Solves

The problem is not "how do we invent a faster multiplier?" It is:

> When data movement dominates energy and performance, which parts of the memory-compute boundary should be redrawn, and what new bottlenecks appear after we redraw it?

Near-memory designs reduce wire distance and increase bandwidth. CiM designs try to eliminate some reads entirely. But neither approach makes computation free. eDRAM consumes area and has refresh/circuit constraints; 3D memory introduces thermal and logic-die area limits; analog CiM needs DACs, ADCs, calibration, precision slicing, and robust modeling. The lecture teaches how to evaluate these technologies without being dazzled by peak TOPS.

---

## Why This Lecture Matters

Advanced technologies are seductive because they sound like escape hatches from the memory wall. A hardware architect needs a colder question: which cost moved, and where did it reappear?

For example, a resistive crossbar can compute many products "in place," but the result is analog current. If the rest of the accelerator is digital, that current must pass through an ADC. The saved weight reads may be replaced by conversion energy, limited precision, and array utilization loss. This is why L11 is not a device survey. It is a lesson in **cross-layer accounting**.

---

## Prerequisites and Mental Model

You should be comfortable with:

- The energy hierarchy from L01-L03: off-chip memory is far more expensive than local arithmetic.
- Dataflow from L05: keeping one operand stationary can reduce movement.
- Sparsity from L07-L10: reducing MACs helps only when overheads are counted.
- Matrix-vector multiplication: $y_j = \sum_i x_i w_{ij}$.

The mental model for L11 is a single matrix-vector multiply. In a digital accelerator, weights are read from memory and delivered to MACs. In near-memory processing, the weight memory is physically closer or has much higher bandwidth. In CiM, the stored weight itself helps produce the product.

---

## Learning Objectives

After this lecture, you should be able to:

- Distinguish **near-memory processing** from **compute-in-memory**.
- Explain why eDRAM and 3D-stacked DRAM reduce different parts of the memory bottleneck.
- Describe how an analog crossbar computes a dot product using voltage, conductance, and current.
- Explain why ADCs and DACs can dominate practical CiM systems.
- Read the **Titanium Law** and identify which factor a proposed technique changes.
- Explain why pruning, low precision, and array size interact differently in CiM than in a digital accelerator.
- Describe what **RAELLA** changes about the ADC input distribution.
- Explain why a modeling tool such as **CiMLoop** is necessary for fair comparison across devices and circuit styles.
- Use DaDianNao, Neurocube, Tetris, and TPU as anchored examples of memory-centric design choices.

---

## Main Textbook-Style Narrative

### 1. Start With the Memory Cost, Not the Device Hype

Lecture 11 slide 5 repeats the quantitative motivation from Horowitz, ISSCC 2014: an 8-bit add is listed at 0.03 pJ, a 32-bit SRAM read from an 8 KB SRAM at 5 pJ, and a 32-bit DRAM read at 640 pJ. These are slide-derived numbers. The exact values depend on process and assumptions, but the ordering is the durable idea: moving data from far away is often more expensive than operating on it.

This explains the lecture sequence:

1. Put more memory on chip with eDRAM.
2. Put DRAM in the package with 3D stacking.
3. Put simple compute in the logic layer or memory periphery.
4. Let the memory array participate directly in computation.

### 2. Near-Memory Processing: Still Digital, But Closer

**eDRAM** is a density move. Lecture 11 slides 6-8 state that eDRAM is denser than SRAM and use DaDianNao as the example. DaDianNao stores many synaptic weights in on-chip eDRAM, reducing expensive off-chip memory traffic. The paper reports that a 10 MB SRAM at 28 nm would occupy 20.73 mm2 while same-size eDRAM has 2.85x higher storage density, and that a 256-bit eDRAM read at 28 nm is 0.0192 nJ versus 6.18 nJ for Micron DDR3, a 321x energy ratio (DaDianNao Section V-A). These are paper-derived claims.

**3D-stacked DRAM** is a bandwidth and distance move. HMC/HBM-style stacks place DRAM dies near or above a logic die, connected by TSVs. Tetris reports that 3D memory provides 160-250 GB/s bandwidth with 3-5x lower access energy than DDR3, then uses that substrate to rebalance accelerator area and move accumulation closer to memory (Tetris Section 2.4 and Section 3). Neurocube uses HMC-style memory with programmable sequence generators in vault controllers to drive neural-network data movement (Neurocube Sections III-V).

Teaching interpretation: eDRAM asks "can enough weights fit close to compute?" 3D memory asks "can memory bandwidth scale with PE count?" Both preserve mostly digital arithmetic. They do not yet use memory cells as multipliers.

### 3. Compute-In-Memory: Use the Array as the Datapath

In an analog resistive crossbar, each weight is stored as conductance $G$, each input is applied as voltage $V$, and the device current is:

$$
I = V G.
$$

For one column, currents from many rows sum:

$$
I_{\text{col}} = \sum_i V_i G_i.
$$

This is a dot product. Ohm's Law supplies multiplication; Kirchhoff's Current Law supplies addition. The crossbar is naturally **weight-stationary**: weights remain in the memory array while input vectors are applied repeatedly.

The ideal picture is powerful, but incomplete. A digital DNN accelerator still needs digital activations and outputs, so practical CiM systems require:

- DACs or pulse encoders to present inputs to the array.
- ADCs to convert column currents back to digital values.
- Bit slicing when a device stores fewer bits than the model weight requires.
- Calibration and margins for nonlinearity, device variation, temperature, voltage, and noise.
- Mapping decisions that keep arrays utilized.

### 4. The ADC Is the CiM Tax Collector

Lecture 11 slides 29-52 focus on the ADC bottleneck and introduce the **Titanium Law**:

$$
\frac{E_{\text{ADC}}}{\text{DNN}}
=
\frac{E_{\text{convert}}}{\text{convert}}
\cdot
\frac{\text{converts}}{\text{MAC}}
\cdot
\frac{\text{MACs}}{\text{DNN}}
\cdot
\frac{1}{\text{utilization}}.
$$

Read each factor as a lever:

- $E_{\text{convert}}/\text{convert}$ rises steeply with ADC resolution.
- $\text{converts}/\text{MAC}$ rises when weights or inputs are sliced across multiple cycles/devices.
- $\text{MACs}/\text{DNN}$ is reduced by smaller models, pruning, or algorithmic changes.
- $1/\text{utilization}$ rises when the array is partly empty.

The painful part is that levers fight. Reducing ADC resolution can require more slices, increasing conversions per MAC. Increasing array rows may improve utilization for some shapes but increases analog summation range and required ADC resolution. Pruning reduces MACs, but may leave CiM arrays underfilled.

### 5. RAELLA: Change the Distribution Seen by the ADC

Lecture 11 slides 43-52 present RAELLA as a response to the Titanium Law. The slide-derived explanation is that RAELLA reduces the analog input range seen by the ADC without retraining the DNN. It uses:

- **Center + offset encoding:** subtract a column center so the analog array computes smaller residuals.
- **Adaptive weight slicing:** only spend extra conversions on computations likely to exceed ADC range.
- **Dynamic input slicing:** speculate with coarse input slices and recover only when needed.

Lecture 11 slide 52 reports a 1024x reduction of input to ADC, 3.9x energy-efficiency improvement, and 1.8x throughput improvement versus an iso-area ISAAC baseline. Because the RAELLA PDF was not in the provided local paper list for this worker, these claims are treated as slide-derived.

### 6. CiM Across SRAM, DRAM, NVM, and Photonics

The lecture then widens the substrate view:

- **SRAM CiM** can use current-mode or charge-sharing circuits. It is process-compatible but constrained by SRAM cell area and circuit nonidealities.
- **DRAM CiM** can exploit charge sharing for in-array bitwise operations. It has density advantages but faces refresh, timing, and peripheral constraints.
- **NVM / ReRAM / memristor CiM** naturally stores weights as resistance/conductance and is dense/nonvolatile, but precision and variation are major issues.
- **Photonics** uses light propagation, modulation, and interference to implement linear algebra primitives. Lecture 11 uses it as an emerging example where data-movement physics differs from CMOS wires.

The important teaching point is that these are not interchangeable "faster MAC" technologies. Each changes the cost model and therefore changes the best mapping and model design.

### 7. CiMLoop: Modeling the Whole Stack

Lecture 11 slides 62-75 present CiMLoop as a Timeloop/Accelergy-style tool extended for CiM. The key modeling requirement is **data-value dependence**: analog energy can depend on the actual values being processed, not only the operation count. A high-conductance device under a high input voltage dissipates more energy than a low-conductance device under a small voltage.

The slide-derived claims are that CiMLoop captures cross-stack interactions with error within 8%, uses statistical models more than 1000x faster than prior accurate simulation, and can compare designs normalized to the same technology/device/ADC assumptions. The teaching interpretation is that advanced technologies make modeling more important, not less. When device physics enters the datapath, a simple MAC count is no longer a safe proxy.

---

## Worked Examples

### Example 1: Why Moving Weights Matters

Suppose a layer performs 1 million 8-bit additions and requires 1 million 32-bit DRAM reads. Using the slide-derived Horowitz numbers, the adds cost approximately $1{,}000{,}000 \times 0.03$ pJ = 30,000 pJ, while the DRAM reads cost $1{,}000{,}000 \times 640$ pJ = 640,000,000 pJ. The reads dominate.

The exact numbers should not be overgeneralized, but the design lesson is robust: a technology that reduces far memory reads can matter more than a slightly more efficient arithmetic unit.

### Example 2: Crossbar Dot Product

Let three inputs be voltages $V = [1, 2, 1]$ and three stored conductances be $G = [3, 0.5, 2]$ in arbitrary normalized units. The column current is:

$$
I_{\text{col}} = 1 \cdot 3 + 2 \cdot 0.5 + 1 \cdot 2 = 6.
$$

This is the same computation as a dot product. The hardware implication is that the array did not read three weights out to a digital multiplier. But the output current still needs sensing, range control, and usually ADC conversion.

### Example 3: Reading the Titanium Law

Assume a design reduces ADC energy per conversion by 4x by using a lower-resolution ADC, but it now needs 3x as many conversions per MAC because of extra slicing. Ignoring other factors, ADC energy changes by:

$$
\frac{1}{4} \times 3 = 0.75.
$$

That is only a 1.33x ADC-energy improvement, not 4x. This is the reason the Titanium Law is useful: it prevents one knob from being advertised without its coupled cost.

---

## Key Equations and How to Read Them

Analog multiplication:

$$
I = V G.
$$

The input activation is represented by voltage $V$, the weight by conductance $G$, and the product by current $I$.

Analog accumulation:

$$
I_{\text{col}} = \sum_i V_i G_i.
$$

All currents on a bitline add physically, producing a dot product.

Titanium Law:

$$
\frac{E_{\text{ADC}}}{\text{DNN}}
=
\frac{E_{\text{convert}}}{\text{convert}}
\cdot
\frac{\text{converts}}{\text{MAC}}
\cdot
\frac{\text{MACs}}{\text{DNN}}
\cdot
\frac{1}{\text{utilization}}.
$$

This is not just an equation for ADCs. It is a checklist for CiM claims: which term improved, which term got worse, and what happened to end-to-end accuracy and throughput?

---

## Hardware Implications

- **Bandwidth:** 3D memory can provide much higher bandwidth than conventional off-chip DRAM, but the logic die area and thermal envelope limit how much compute can be placed nearby.
- **Area:** eDRAM and SRAM trade density, latency, refresh, and integration complexity. Large buffers can dominate accelerator area.
- **ADC/DAC overhead:** CiM's analog compute core can be efficient while peripheral conversion dominates total energy.
- **Precision:** Device precision, ADC resolution, weight slicing, input slicing, and model accuracy form one coupled design space.
- **Utilization:** A huge CiM array is inefficient if layer shapes cannot fill rows and columns.
- **Programmability:** Near-memory and CiM designs often expose unusual mapping constraints to compilers and modeling tools.
- **Correctness:** Analog nonidealities are not just performance issues; they can change numerical results and therefore model accuracy.

---

## Common Misconceptions

### Misconception: Compute-in-memory makes MACs free.

The array-level multiply-accumulate can be cheap, but DACs, ADCs, sense amplifiers, calibration, slicing, and digital accumulation can dominate.

### Misconception: More rows in a crossbar are always better.

More rows can increase parallelism, but they also increase analog summation range, required ADC resolution, wire parasitics, and utilization risk.

### Misconception: Pruning always helps CiM.

Pruning reduces MACs/DNN, but sparse weights may underfill dense CiM arrays. The Titanium Law makes this visible through the utilization factor.

### Misconception: TOPS is enough to compare advanced technologies.

TOPS ignores precision, accuracy, conversion overhead, memory traffic, array utilization, batching assumptions, and technology normalization.

---

## Connections to Previous and Later Lectures

- **L01-L03:** L11 is the physical-technology answer to the energy hierarchy introduced early in the course.
- **L05 Dataflow:** Weight-stationary mapping appears again because CiM arrays naturally keep weights in place.
- **L07-L10 Sparsity:** Sparse models reduce MAC count but can hurt utilization in array-based CiM. Sparse benefits are technology-dependent.
- **L12 Reduced Precision:** Precision is central to CiM because ADC bits, device bits, and model accuracy are coupled.
- **Labs and final projects:** CiMLoop appears as the modeling bridge from lecture concepts to design-space exploration.

---

## Paper Bridge: DaDianNao

### Bibliographic Identity

- **Title:** DaDianNao: A Machine-Learning Supercomputer
- **Authors:** Yunji Chen et al.
- **Year / venue:** MICRO 2014
- **Local PDF:** [`papers/L15_DaDianNao_Chen_MICRO2014.pdf`](../../papers/L15_DaDianNao_Chen_MICRO2014.pdf)
- **Used in lecture:** Lecture 11 eDRAM and near-memory motivation

### Problem Addressed

DaDianNao addresses the memory bandwidth and energy cost of large neural networks, especially layers with many synaptic weights. It observes that keeping weights in external memory forces high-bandwidth, high-energy transfers.

### Core Idea

Place large eDRAM banks near neural functional units and move neuron values rather than repeatedly moving synaptic weights. A node contains many tiles with local eDRAM and compute, plus central eDRAM and interconnect.

### Relevance to This Lecture

DaDianNao is the eDRAM example for near-memory design. It keeps compute digital but changes the memory hierarchy so that weights are stored close to compute.

### Key Claims Used in This Chapter

- Off-chip memory bandwidth can become the bottleneck for large neural-network layers; see DaDianNao Sections I and IV.
- The paper states that off-chip memory accesses can increase total energy by about 10x for the discussed setting; see Section IV.
- It reports 2.85x higher density for eDRAM than SRAM and a 321x energy ratio between a 256-bit eDRAM read and Micron DDR3 in its 28 nm assumptions; see Section V-A.
- A node contains 36 MB of eDRAM; see the architecture parameter discussion around Table II.
- The paper reports 150.31x average energy reduction for a 64-chip system over the evaluated baseline; see the abstract and evaluation discussion.

### What Students Should Remember

1. DaDianNao is near-memory, not compute-in-memory.
2. The main move is storing many weights close enough that external memory traffic drops.
3. eDRAM density helps, but the design still pays area, wire, and integration costs.

### Limitations and Assumptions

The paper targets neural networks and technology assumptions from its era. Its quantitative ratios should be used as historical anchored evidence, not universal constants.

### Suggested Insertion Points

Use DaDianNao when explaining eDRAM and the first step from ordinary memory hierarchy toward memory-centric accelerator design.

---

## Paper Bridge: Neurocube

### Bibliographic Identity

- **Title:** Neurocube: A Programmable Digital Neuromorphic Architecture with High-Density 3D Memory
- **Authors:** Donghyuk Kim et al.
- **Year / venue:** ISCA 2016
- **Local PDF:** [`papers/L15_Neurocube_Kim_ISCA2016.pdf`](../../papers/L15_Neurocube_Kim_ISCA2016.pdf)
- **Used in lecture:** Lecture 11 3D-stacked memory / HMC discussion

### Problem Addressed

Neurocube addresses memory capacity and bandwidth limits for neural networks by using 3D high-density memory integrated with a logic tier.

### Core Idea

Integrate processing elements and programmable neurosequence generators (PNGs) into the HMC logic layer. The memory system drives known neural-network data movement patterns through programmable state machines.

### Relevance to This Lecture

Neurocube shows near-memory processing in a 3D memory stack. It is not analog CiM; it is a digital memory-centric architecture that uses memory organization and programmable controllers to reduce movement overhead.

### Key Claims Used in This Chapter

- HMC provides multiple vaults and highly parallel access; see Neurocube Section II-B and Table I.
- The architecture uses programmable neurosequence generators associated with vault controllers; see Sections III-V.
- General neural-network nested loops can be mapped to finite-state machines in the PNG; see Figure 8 and its surrounding text.
- The paper reports example throughput for scene-labeling inference in 28 nm and 15 nm designs; see Section VI.

### What Students Should Remember

1. 3D memory offers bandwidth and capacity, but useful performance requires mapping and scheduling.
2. Programmable memory-side controllers can exploit static neural-network access patterns.
3. Near-memory processing can remain fully digital while still changing the architecture.

### Limitations and Assumptions

Neurocube is framed around neuromorphic/neural-network workloads and HMC assumptions. It is best used as a memory-centric design example, not as evidence for analog CiM.

### Suggested Insertion Points

Use Neurocube when explaining why 3D memory is more than a wider DRAM bus: the logic layer can participate in scheduling and data movement.

---

## Paper Bridge: Tetris

### Bibliographic Identity

- **Title:** Tetris: Scalable and Efficient Neural Network Acceleration with 3D Memory
- **Authors:** Mingyu Gao et al.
- **Year / venue:** ASPLOS 2017
- **Local PDF:** [`papers/L15_TETRIS_Gao_ASPLOS2017.pdf`](../../papers/L15_TETRIS_Gao_ASPLOS2017.pdf)
- **Used in lecture:** Lecture 11 3D memory and near-memory accumulation

### Problem Addressed

Tetris asks how a neural-network accelerator should be redesigned when 3D memory provides high bandwidth and lower access energy than conventional off-chip DRAM.

### Core Idea

Use 3D memory to rebalance accelerator area away from large on-chip SRAM buffers and toward PE arrays, and move simple accumulation operations closer to DRAM banks to reduce output-feature-map traffic.

### Relevance to This Lecture

Tetris is the bridge between near-memory bandwidth and near-memory computation. It shows that simply attaching 3D DRAM is not enough; dataflow scheduling, buffer sizing, and where accumulation happens must be reconsidered.

### Key Claims Used in This Chapter

- The abstract reports 4.1x performance improvement and 1.5x energy reduction over conventional low-power DRAM systems.
- Section 2.4 states that 3D memory can provide 160-250 GB/s bandwidth with 3-5x lower access energy than DDR3.
- Section 3 discusses rebalancing PE/buffer area and placing engines per vault.
- Section 4 discusses dataflow scheduling and in-memory accumulation to reduce ofmap traffic.

### What Students Should Remember

1. 3D memory changes the optimal accelerator balance between buffers and PEs.
2. Near-memory accumulation saves traffic only when dataflow is scheduled to exploit it.
3. Bandwidth alone does not solve utilization, area, or scheduling.

### Limitations and Assumptions

Tetris is a digital 3D-memory accelerator, not an analog CiM design. Its speedup depends on evaluated networks, area budgets, vault organization, and dataflow schedules.

### Suggested Insertion Points

Use Tetris when explaining why advanced memory technology must be paired with architecture and mapping changes.

---

## Paper Bridge: TPU as a Digital Baseline

### Bibliographic Identity

- **Title:** In-Datacenter Performance Analysis of a Tensor Processing Unit
- **Authors:** Norman P. Jouppi et al.
- **Year / venue:** ISCA 2017
- **Local PDF:** [`papers/L12_TPU_Jouppi_ISCA2017.pdf`](../../papers/L12_TPU_Jouppi_ISCA2017.pdf)
- **Used in lecture:** Contextual baseline for memory-aware digital acceleration

### Problem Addressed

The TPU paper analyzes a deployed digital inference accelerator and shows how matrix units, on-chip memory, and weight-memory bandwidth determine datacenter inference performance.

### Core Idea

The TPU uses a 256 x 256 systolic matrix unit with 65,536 8-bit MACs, large software-managed on-chip memory, and a deterministic execution model. It is not an advanced memory device, but it is a useful contrast: it attacks data movement with a digital systolic array and explicit memory management.

### Relevance to This Lecture

TPU provides a baseline for what advanced technologies are competing against. A CiM design must outperform not a naive MAC array, but a carefully engineered digital accelerator with locality, systolic data reuse, and roofline-aware performance constraints.

### Key Claims Used in This Chapter

- The abstract states the TPU has a 65,536 8-bit MAC matrix unit, 92 TOPS peak, and 28 MiB on-chip memory.
- Section 2 describes the matrix unit, unified buffer, weight FIFO, and systolic execution.
- Section 4 applies a roofline model and notes a ridge point around 1350 operations per byte of weight memory fetched.
- The paper reports 15x-30x speedup over contemporary GPU/CPU inference systems and much higher TOPS/Watt in its evaluated datacenter workloads; see the abstract.

### What Students Should Remember

1. Digital accelerators already exploit locality aggressively.
2. Roofline reasoning remains useful even when comparing to advanced technologies.
3. Advanced technology claims should be compared against strong digital baselines.

### Limitations and Assumptions

The TPU paper evaluates specific Google datacenter workloads from its era. It is a baseline and modeling contrast, not evidence for CiM device behavior.

### Suggested Insertion Points

Use TPU when students need a grounded digital reference point for systolic arrays, memory bandwidth, and roofline limits.

---

## Standalone Study Guide

### What to Master Before Moving On

- Explain the difference between moving memory closer and computing inside memory.
- Derive the crossbar dot product from $I=VG$ and current summation.
- Use the Titanium Law to identify which cost a CiM technique changes.
- Explain why ADCs turn an elegant analog primitive into a system-level tradeoff.
- Compare eDRAM, 3D DRAM, analog CiM, and photonics by cost model rather than hype.

### Self-Check Questions

1. Why is DaDianNao near-memory rather than compute-in-memory?
2. What does 3D memory improve, and what constraints does it introduce?
3. In a resistive crossbar, what represents the input and what represents the weight?
4. Why can lowering ADC resolution increase conversions per MAC?
5. Why might pruning hurt CiM utilization?
6. Why does CiMLoop need data-value-dependent modeling?

### Exercises

1. Use the Titanium Law to compare two designs: Design A has 2x lower energy per conversion but 2x more conversions per MAC; Design B keeps conversions fixed but improves utilization from 50% to 80%.
2. For a $4 \times 4$ crossbar, write the dot product computed by one column and identify where DAC and ADC conversion occur.
3. Choose one near-memory paper bridge in this chapter and explain which memory movement it reduces: weights, activations, partial sums, or off-chip transfers.
4. Explain why a sparse, heavily pruned model may be good for a digital sparse accelerator but awkward for a dense analog crossbar.
5. Paper-reading bridge: read Tetris Section 2.4 and summarize why 3D memory changes PE/buffer area allocation.

---

## Key Terms

| Term | Meaning |
|---|---|
| **Near-memory processing** | Keeping compute digital but placing it physically closer to memory or using high-bandwidth memory packaging. |
| **Compute-in-memory (CiM)** | Using a memory array or memory device as part of the computation itself. |
| **eDRAM** | Embedded DRAM; denser than SRAM and useful for larger on-chip storage, with integration and refresh tradeoffs. |
| **3D-stacked DRAM** | DRAM dies stacked with a logic die and connected by TSVs, increasing bandwidth and reducing distance. |
| **HMC / HBM** | 3D or 2.5D high-bandwidth memory technologies used to reduce memory bottlenecks. |
| **Analog crossbar** | A grid of programmable conductances that computes dot products using voltages and currents. |
| **Conductance** | Reciprocal of resistance; in resistive CiM it represents a stored weight. |
| **DAC** | Digital-to-analog converter; presents digital inputs as analog signals. |
| **ADC** | Analog-to-digital converter; converts analog sums back to digital values. |
| **Weight slicing** | Representing a multi-bit weight across multiple devices or cycles. |
| **Input slicing** | Representing a multi-bit input across multiple temporal or analog steps. |
| **Titanium Law** | A product expression for ADC energy per DNN inference: conversion energy, conversions per MAC, MACs per DNN, and utilization penalty. |
| **RAELLA** | Slide-presented CiM technique that reshapes analog values entering ADCs without retraining. |
| **CiMLoop** | Slide-presented modeling framework for cross-stack CiM design exploration. |
| **Data-value dependence** | The property that analog energy depends on actual processed values, not just operation count. |
| **Photonic computing** | Computing with optical signals, often using modulation/interference for linear algebra. |

---

## Takeaways

- Advanced technologies are responses to data movement, not magic replacements for architecture.
- eDRAM and 3D memory keep computation digital but reduce storage distance or increase bandwidth.
- Analog CiM computes dot products elegantly, but ADC/DAC, slicing, variation, and utilization dominate practical design.
- The Titanium Law is a compact way to check whether a CiM proposal improved one term while worsening another.
- RAELLA's slide-level lesson is distribution reshaping: reduce the analog range that ADCs must resolve.
- CiMLoop's slide-level lesson is modeling discipline: advanced substrates require value-aware, cross-layer evaluation.
- Strong digital baselines such as TPU matter; advanced technology should be compared against memory-aware architectures, not strawman MAC arrays.

---

## Connections

L11 connects the whole course to physical implementation choices. L01-L03 taught that memory movement dominates. L05 taught that dataflow chooses what moves. L07-L10 taught that sparsity changes the dynamic work and metadata traffic. L11 asks what happens when the storage device, circuit interface, and package are redesigned. L12 then returns to precision, which is central because CiM device bits and ADC bits directly shape accuracy, energy, and throughput.

---

## Appendix — Slide-to-Section Map

| Slides | Chapter section | Notes |
|---|---|---|
| 1 | Title | Lecture framing. |
| 2-9 | Memory cost and near-memory | eDRAM, 3D DRAM, DaDianNao, Neurocube, Tetris. |
| 10-27 | Analog crossbar | Conventional processing vs. CiM, Ohm/Kirchhoff MAC, weight-stationary mapping, practical limits. |
| 28-52 | ADC bottleneck | ISAAC context, Titanium Law, RAELLA techniques and slide-reported results. |
| 53-60 | CiM substrates | SRAM, DRAM, NVM/memristor discussion. |
| 60-75 | CiMLoop | Device/circuit/architecture/mapping/workload stack and modeling claims. |
| 75-80 | Photonics | Emerging substrate and CiMLoop photonics modeling. |
| 81 | Summary/references | Used for source attribution. |

---

## Source Notes

- Lecture flow follows `Lecture/L11-Advanced_Tech.pdf`, slides 1-81.
- Memory energy numbers for add/SRAM/DRAM are slide-derived from Lecture 11 slide 5, which attributes them to Horowitz, ISSCC 2014.
- RAELLA, Titanium Law, CiMLoop, and photonics discussion are slide-derived from Lecture 11 slides 29-80. The local paper PDFs for RAELLA and CiMLoop were not in this worker's specified inputs, so this chapter does not independently verify their paper details.
- DaDianNao claims are derived from `papers/L15_DaDianNao_Chen_MICRO2014.pdf`, especially Sections IV-V and the evaluation discussion.
- Neurocube claims are derived from `papers/L15_Neurocube_Kim_ISCA2016.pdf`, especially Sections II-VI.
- Tetris claims are derived from `papers/L15_TETRIS_Gao_ASPLOS2017.pdf`, especially Sections 2-4 and 6.
- TPU baseline claims are derived from `papers/L12_TPU_Jouppi_ISCA2017.pdf`, especially the abstract, Sections 2 and 4.
- Worked examples are original teaching examples based on slide-stated concepts.

## Uncertainty Notes

- The live lecture may have emphasized RAELLA, CiMLoop, and photonics with details not recoverable from the slide text alone.
- Numerical technology ratios are process- and assumption-dependent. This chapter treats them as anchored evidence from slides or papers, not universal constants.
- Existing repository assets under `assets/L11/` may be copyright-sensitive slide captures. This chapter no longer embeds them, but this worker did not delete assets outside the owned walkthrough files.
