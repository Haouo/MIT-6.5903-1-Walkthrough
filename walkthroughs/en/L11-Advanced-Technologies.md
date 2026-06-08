# L11 — Advanced Technologies

> **Course:** 6.5930/1 — Hardware Architectures for Deep Learning
> **Instructors:** Joel Emer & Vivienne Sze (MIT EECS)
> **Lecture date:** March 9, 2026 · **Slides:** 81 · **Source:** [`Lecture/L11-Advanced_Tech.pdf`](../../Lecture/L11-Advanced_Tech.pdf)
>
> *This is a conceptual walkthrough that reconstructs the lecture's narrative from the slides. It is organized by idea, not slide-by-slide. Each section cites the slide range it draws from so you can follow along with the original deck.*

---

## TL;DR

The dominant energy cost in a DNN accelerator is not arithmetic — it is **moving data**. Every technique in the course has been a way to keep data closer to compute. This lecture takes that insight to its logical extreme: what if the computation moved *into* the memory itself? **Compute-in-Memory (CiM)** uses the physical properties of memory devices — resistance, voltage, charge — to perform multiply-accumulate operations directly inside the storage array, eliminating the most expensive data movement entirely. The lecture covers (1) the memory-technology landscape from SRAM to 3D-stacked DRAM that motivates CiM, (2) the analog crossbar as the core CiM primitive and its governing tradeoffs, (3) the **Titanium Law** quantifying ADC overhead as the fundamental CiM energy bottleneck, (4) **RAELLA** as a case study in escaping that bottleneck without retraining, (5) CiM variants across SRAM, DRAM, and non-volatile memory, and (6) **CiMLoop** as the modeling tool that enables principled design-space exploration — including optical computing as an emerging frontier. The unifying lesson is that cross-layer co-design (device, circuit, architecture, mapping, workload) is unavoidable when the physics of the compute substrate enters the loop.

---

## Learning Objectives

After this lecture you should be able to:

- Explain the **memory taxonomy** (SRAM, eDRAM, 3D-stacked DRAM, NVM) and how each addresses a different aspect of the data-movement bottleneck.
- Describe how an **analog crossbar** performs a matrix-vector multiply using Ohm's Law and Kirchhoff's Current Law.
- State the **four factors of the Titanium Law** and explain how each knob trades off against the others.
- Explain why **ADC overhead** (energy and area) is the dominant cost in most CiM designs, and what design strategies address it.
- Describe the three techniques that **RAELLA** uses to reduce ADC input range without retraining.
- Articulate why **CiM requires cross-layer co-design** (device ↔ circuit ↔ architecture ↔ mapping ↔ workload) and how CiMLoop models that full stack.
- Name at least one **beyond-SRAM CiM substrate** (DRAM, ReRAM/memristor, SRAM, photonics) and its distinguishing property.

---

## Chapter 1 — Why bring compute closer to memory?

> *Slides: L11-2 … L11-9*

### The memory-technology landscape

Every practical DNN system contains a hierarchy of memory technologies, each occupying a different point in the density-vs-cost trade space:

![Advanced storage technology taxonomy — near-memory and in-memory computing](../../assets/L11/L11-p02-storage-taxonomy.png)

The lecture organizes these into two broad strategies for attacking the data-movement problem:

- **Processing/Compute Near Memory (near-data processing):** move the compute *closer* to memory, but keep them physically separate.
  - *Embedded DRAM (eDRAM)* — higher density than SRAM (2.85× denser than 6T SRAM) so more storage fits on-chip, avoiding expensive off-chip DRAM accesses. DaDianNao used 36 MB of eDRAM to hold fully-connected-layer weights, achieving 321× better energy than DDR3.
  - *3D-stacked DRAM* — stack a logic die under multiple DRAM dies (Hybrid Memory Cube / High Bandwidth Memory). NeuroCube demonstrated 6.25× higher bandwidth than DDR3; Tetris combined HMC with the Eyeriss spatial array to get 1.5× energy reduction and 4.1× higher throughput vs. 2D DRAM.

- **Processing/Compute In Memory (in-memory computing):** integrate the computation *into* or *using* the memory array itself. This is the focus of the rest of the lecture.

### Why the cost of off-chip DRAM access is the forcing function

The slide on memory access cost is the quantitative grounding for everything that follows:

![Memory access cost hierarchy — DRAM read at 640 pJ vs. 8b add at 0.03 pJ](../../assets/L11/L11-p05-memory-cost-hierarchy.png)

The numbers (from Horowitz, ISSCC 2014) are stark: a 32-bit DRAM read costs **640 pJ**, while an 8-bit integer addition costs only **0.03 pJ**. That is a ratio of more than **20,000:1** between the most expensive memory access and the cheapest arithmetic operation. Even a 32-bit SRAM read at 8 KB costs ~5 pJ — 167× an 8b add. The conclusion is unavoidable: data movement is the bottleneck, and anything that reduces it is valuable.

> **Why it matters:** The energy hierarchy established in L01 (RF 1× → DRAM 200×) is now grounded in measured silicon numbers. CiM is a direct architectural response: if reads from the weight array are the biggest cost, what if those reads never happened — because the compute happened inside the array?

---

## Chapter 2 — The analog crossbar: CiM's core primitive

> *Slides: L11-10 … L11-27*

### Conventional processing vs. compute-in-memory

The contrast is sharpest at the system level:

![Conventional processing vs. compute-in-memory — data flow comparison](../../assets/L11/L11-p10-cim-vs-conventional.png)

In a *conventional* accelerator, weights sit in a memory array and must be read out over a high-bandwidth bus to a separate MAC unit. The read bus is the bottleneck: many bytes must travel many wire lengths, paying capacitive charging energy at every hop. In a *CiM* accelerator, weights remain in the array and input activations are delivered to the array periphery as voltages (via a DAC). The computation happens inside the array; only a single low-bandwidth output (a current sum) exits, converted to digital by an ADC. The read bandwidth inside the chip drops dramatically.

### Ohm's Law as a multiplier, Kirchhoff as an adder

The physical mechanism is elegant:

![Analog MAC principle — voltage × conductance = current, currents sum on bit-line](../../assets/L11/L11-p11-analog-mac-principle.png)

- Each **weight** is stored as the *conductance* G of a device (resistor, memristor, or transistor). Conductance is 1/resistance.
- Each **input activation** is delivered as a *voltage* V on the word line.
- By Ohm's Law, the current through the device is I = V × G. This is a **multiplication**.
- All currents on the same bit-line sum by Kirchhoff's Current Law: I_total = Σ Vᵢ × Gᵢ. This is an **accumulation** (a dot product).

Thus, a single crossbar column computes an entire dot product *in one step*, using the physics of the circuit — no digital adder tree required. This is why CiM proponents call it a fundamentally different compute paradigm.

### Weight-stationary dataflow in the CiM array

The natural dataflow for a CiM array is weight-stationary: weights are written into the array once and stay there while many input vectors are streamed through.

![Weight-stationary CiM dataflow — loop nest and array organization](../../assets/L11/L11-p14-weight-stationary-cim.png)

The mapping fits neatly onto the loop-nest view introduced in earlier lectures: the M (output channel / row in weight matrix) and CHW (input channel-height-width / column in weight matrix) dimensions tile onto the rows and columns of the array. Benefits include: reduced weight data-movement (weights never move after programming), higher memory read bandwidth (multiple weights accessed in parallel along a row), higher throughput (many dot-product computations run simultaneously), and lower input-activation delivery cost (activations are delivered once to many rows simultaneously).

### Design considerations and practical limits

Analog computing introduces a set of practical constraints that *reduce* the idealized gains:

1. **Non-linearity and device variation (PVT):** Analog values are sensitive to process, voltage, and temperature variations. This limits achievable precision.
2. **Number of storage elements per weight (weight slicing):** Each device typically stores only 1–4 bits of precision. A full 8-bit weight requires multiple devices ("bit slices"), multiplying array area and ADC conversion count.
3. **Array size limits:** Word-line and bit-line resistance and capacitance grow with array dimension. Large arrays degrade robustness and sensing margin. Utilization drops when the workload does not fill the full array.
4. **Number of rows activated in parallel:** Limited by the ADC's resolvable range. More parallel rows → larger analog sum → higher required ADC resolution → exponentially more energy.
5. **Number of columns activated in parallel:** Limited by ADC area (one ADC per column is expensive).
6. **Input delivery time:** DAC non-linearity forces time-encoded inputs (pulse-width modulation), requiring multiple cycles per input and reducing throughput.
7. **Temporal accumulation:** Single-bit or two-bit device operations require multi-cycle temporal accumulation to build up a multi-bit result, further reducing throughput.

These constraints mean that the raw physical speedup of analog compute is significantly eroded by the overhead of the digital interfaces on both sides of the array.

> **Why it matters:** Every "gain" claimed for CiM must be weighed against these overheads. The ADC in particular — which must convert the analog column current back to a digital number — turns out to be the dominant cost. The next chapter quantifies this precisely.

---

## Chapter 3 — The Titanium Law and the ADC bottleneck

> *Slides: L11-29 … L11-52*

### The energy breakdown of a CiM accelerator

When you account for the full system, the energy distribution is striking:

![CiM accelerator energy breakdown — ADC dominates](../../assets/L11/L11-p29-cim-energy-breakdown.png)

The ADC (Analog-to-Digital Converter) consumes a significant fraction of total system energy — in many designs, *more* than the analog crossbar computation itself. DAC energy and other analog processing add to the overhead. The promised efficiency of CiM is real, but it is largely consumed by the conversion interfaces.

### The Titanium Law: a closed-form expression for ADC energy

The lecture introduces a key analytical result from Andrulis, ISCA 2023:

![The Titanium Law — ADC energy as product of four factors](../../assets/L11/L11-p30-titanium-law.png)

The total ADC energy per DNN inference is the product of four terms:

```
ADC Energy      Energy       Converts     MACs        1
──────────── = ────────── × ────────── × ──────── × ──────────
    DNN         Convert       MAC          DNN       Utilization
```

- **Energy/Convert:** Energy per ADC conversion — increases *exponentially* with ADC resolution (number of bits). This is the steepest term.
- **Converts/MAC:** Number of ADC conversions per MAC — determined by weight slicing and input slicing (how many bit-slices are needed to represent each value at the required precision).
- **MACs/DNN:** Total number of MAC operations in the DNN — set by the workload, not the hardware.
- **1/Utilization:** Array utilization penalty — always ≥ 1, worsens when the workload cannot fill the array dimensions.

The law reveals the fundamental tension: reducing ADC resolution (to save Energy/Convert) requires more slices (raising Converts/MAC), while increasing array size (to reduce 1/Utilization) increases the analog sum range (raising required ADC resolution). Every knob tightens another constraint.

### Applying the law: why ISAAC's tradeoffs are hard to escape

The ISAAC design (Shafiee, ISCA 2016) used 128 rows of 2-bit memristors. Applying the Titanium Law:

- Increasing rows to 1024 reduces 1/Utilization but forces higher ADC resolution (11-bit), so ADC energy dominates even more.
- Decreasing bits per weight slice (1-bit per memristor) reduces Energy/Convert but increases Converts/MAC — ADC energy again rises.

Both directions worsen the total ADC energy relative to the analog crossbar energy. This is not a flaw in ISAAC; it is a fundamental tension in the design space.

### Two prior escape routes and their costs

Two strategies have been used to work around the Titanium Law, each paying a price:

1. **Weight pruning (weight-count-limited designs):** Reduce MACs/DNN by pruning the network. This lowers the "MACs/DNN" factor. The cost: pruned networks may sacrifice accuracy, and the reduced weight count can reduce array utilization.
2. **Low-resolution ADC (sum-fidelity-limited designs):** Use a lower-resolution ADC, reducing Energy/Convert. The cost: the ADC cannot accurately represent all output values; accuracy must be recovered by retraining the DNN to tolerate low-resolution readouts.

Both approaches **require retraining the DNN** to preserve accuracy — which is expensive and couples the hardware design directly to the model.

### RAELLA: reshaping distributions without retraining

RAELLA (Andrulis, ISCA 2023) escapes the bottleneck through three complementary techniques applied *at inference time*, without modifying the DNN:

![RAELLA distribution reshaping — three techniques to reduce ADC input range](../../assets/L11/L11-p43-raella-reshape.png)

**Technique 1 — Center + Offset Encoding (shift the distribution mean):**
Each weight column is decomposed into a *center* (mean) and *offset* (residual). The center is computed digitally at high resolution (cheap, since it is a scalar multiply). The analog array computes only the offsets, whose values cluster near zero — requiring far less ADC range. This shifts the distribution of analog results toward zero, reducing the required ADC dynamic range.

**Technique 2 — Adaptive Weight Slicing (split only large-result computations):**
Instead of always slicing weights into fixed-precision pieces, RAELLA monitors whether a given computation's result falls outside the ADC range. Only out-of-range columns are re-run with finer weight slices. This keeps Converts/MAC low for the majority of computations while handling outliers.

**Technique 3 — Dynamic Input Slicing (speculate and recover):**
RAELLA first *speculates* with a coarse input slice. If the result is in range, no further action is needed. If out of range, it recovers by re-running with finer input slices. This amortizes the extra ADC conversion cost over only the fraction of inputs that need it.

Together, these three techniques achieve a **1024× reduction in analog input to ADC**, enabling lower-resolution ADCs with no accuracy loss:

![RAELLA results — 3.9× energy improvement and 1.8× throughput vs. iso-area ISAAC](../../assets/L11/L11-p52-raella-results.png)

Compared to a iso-area ISAAC baseline, RAELLA achieves **3.9× energy efficiency improvement** and **1.8× throughput improvement** while **maintaining DNN accuracy without retraining**.

> **Why it matters:** The Titanium Law is not just a descriptive formula — it is a design compass. RAELLA shows that understanding the law precisely enough allows you to reshape the *input* to the bottleneck term (Energy/Convert via ADC resolution) rather than just tuning the knobs that are in tension with each other. This is cross-layer co-design at its most concrete.

---

## Chapter 4 — CiM across substrate technologies

> *Slides: L11-53 … L11-60*

### Designing DNNs for CiM — a different optimization landscape

An important insight: the *best* DNN for a digital accelerator may not be the best for a CiM accelerator. The tradeoffs differ:

- **Weight count vs. utilization:** Pruning weights is desirable on digital hardware (fewer MACs), but on CiM it can hurt array utilization (fewer rows filled). CiM arrays perform best with dense weight matrices.
- **Filter shape:** CiM is weight-stationary and prefers *fewer activations* relative to weights, so shallower networks with larger filters may be better suited than deeper networks with tiny filters.
- **Robustness to non-idealities:** Quantization-aware training is more important for CiM because device variability introduces noise that the DNN must tolerate.

### CiM using SRAM bit cells

SRAM-based CiM uses the access transistor's I-V relationship to perform multiplication. Two implementations:

- **Current-mode:** Word-line voltage modulates the transistor current; binary weights are stored in the bit-cell state. Bit-line current sums give the partial sum. Limited by transistor non-linearity.
- **Charge-sharing:** Uses an explicit capacitor to store charge. XNOR logic on the bit-cell performs binary multiplication; charge sharing on the bit-line performs addition (Vf = ½(V1 + V2), a scaled sum). Better linearity and matching than current-mode.

SRAM CiM is attractive because it uses the standard SRAM process — no exotic devices or process modifications required.

### CiM using DRAM

DRAM-based CiM (Ambit, MICRO 2017) uses charge sharing to perform **bitwise AND and OR** operations without moving data out of the array:

- Activating three rows simultaneously causes charge sharing that resolves to AND or OR depending on the pre-charge voltage (VDD/2 − δ for AND, VDD/2 + δ for OR).
- Multi-bit multiplication requires multiple cycles of temporal accumulation, but the operation runs at the full DRAM bus width — massive parallelism across rows.

### CiM using non-volatile memory (memristors)

Non-volatile memories (ReRAM/RRAM, Phase-Change Memory, STT-RAM) store data as resistance states that persist without power. Memristors are particularly attractive because:

- **Resistance is directly programmable** — weights are encoded as conductance values, which is the natural CiM representation.
- **Higher density** than SRAM (no need for the 6-transistor bit cell).
- **Non-volatile** — weights survive power-off, enabling instant-on inference without reloading.

The challenge is limited precision per device (1–4 bits typically) and device-to-device variation, which amplifies the need for weight slicing and careful calibration.

> **Why it matters:** No single substrate is dominant. The right choice depends on the target application (power, area, latency, precision), the DNN architecture, and the manufacturing process available. This diversity motivates a modeling framework that can compare across all of them.

---

## Chapter 5 — CiMLoop: modeling the full stack

> *Slides: L11-60 … L11-75*

### The CiM design space is enormous

The lecture enumerates the design choices at each level of the CiM stack:

![The CiM stack — devices, circuits, architecture, mapping, workload](../../assets/L11/L11-p62-cim-stack.png)

Every level presents multiple options:
- **Devices:** SRAM, DRAM, ReRAM, STT-RAM, photonic elements, superconducting circuits.
- **Circuits:** DAC type (R-2R, pulse-train, capacitor-based), ADC type (flash, SAR, integrating), MAC circuit (current-mode, charge-sharing, digital XNOR), sparsity/and-logic controllers.
- **Architecture:** Array dimensions, number of arrays, banking, periphery organization.
- **Mapping:** Which dimensions tile onto rows/columns, loop order, weight stationary vs. output stationary, batch size.
- **Workload:** DNN layer type, shape, sparsity, precision.

With so many interacting choices, hand-analysis cannot navigate the space systematically. The cross-layer dependencies (data values affect device energy, which depends on the encoding, which depends on the mapping, which depends on the architecture...) make any decoupled analysis inaccurate.

### CiMLoop: flexible, accurate, and fast

![CiMLoop overview — flexible, accurate, and fast CiM modeling tool](../../assets/L11/L11-p65-cimloop-overview.png)

CiMLoop (Andrulis, ISPASS 2024, **Best Paper Award**) is the Timeloop+Accelergy-based tool extended to handle the cross-stack interactions of CiM. Its three distinguishing properties:

1. **Flexibility:** User-defined specifications describe any device, circuit, or architecture component. The library includes models for 6T SRAM, 8T SRAM, DRAM, ReRAM, multiple ADC architectures, DAC architectures, and photonic components. Users can add new models via a plug-in interface.

2. **Accuracy (data-value-dependent modeling):** Most prior tools (Timeloop, NeuroSim) assume fixed energy per operation. CiMLoop recognizes that in analog, energy depends on the *actual values* being processed: a higher-conductance memristor dissipates more power for the same input voltage. CiMLoop captures the chain: value → binary representation → encoding → bit assignment to components → per-component energy. The result is within **8% error** of value-by-value simulation.

3. **Speed (statistical modeling):** Accurate value-dependent simulation would require evaluating >10¹² value combinations. CiMLoop instead computes *data distributions* (histograms) and applies statistical models — achieving **>1000× speedup** compared to NeuroSim while matching its accuracy. Compared to Timeloop (fast but inaccurate), CiMLoop is the same speed but 10× more accurate.

### Apples-to-apples comparison and design-space exploration

With a common modeling framework, designs from different papers (different technology nodes, ADC types, memory devices) can be compared fairly by normalizing to the same technology, ADC, and device:

![Design space exploration with CiMLoop — array size vs. DNN shape](../../assets/L11/L11-p74-design-space-exploration.png)

CiMLoop also enables exploration of how array size (an architecture decision) interacts with DNN layer shapes (a workload property) — a question that is impossible to answer with decoupled tools. This kind of joint optimization is the kind of work proposed for the course's final project.

### CiMLoop enabling collaborations — and photonic computing

CiMLoop has been used at MIT to model not just conventional CiM but also:

- **Resistive memory CiM** (collaboration with Jesus del Alamo's group)
- **Superconducting electronics CiM** (collaboration with Karl Berggren and Neil Gershenfeld — started as a 6.5930/1 final project!)
- **Optical / photonic computing** (collaboration with Dirk Englund's group)

The photonic frontier is worth pausing on:

![Compute with light — optical matrix multiplication](../../assets/L11/L11-p77-compute-with-light.png)

Photons have properties that make them attractive for DNN compute:

- **Distance-independent energy:** Moving a photon across a chip consumes (nearly) the same energy regardless of distance — unlike electrons, where wire resistance creates ohmic loss proportional to wire length.
- **Passive multiplication:** An optical modulator scales a light signal's intensity by a weight value without active amplification.
- **No electromagnetic interference:** Photons do not interact with each other in a linear medium, enabling dense wavelength-multiplexed interconnects.

A 2017 Nature Photonics paper (Shen et al.) demonstrated matrix multiplication in the optical domain using a Mach-Zehnder interferometer network. Lightmatter's Envise chip, based on this principle, was reported to run BERT inference 5× faster than an NVIDIA A100 at 1/6 the power.

CiMLoop's library includes models for photonic components (waveguides, microring resonators, photodiodes, modulator drivers), enabling the same principled co-design methodology to be applied to photonic DNN accelerators.

> **Why it matters:** The modeling framework is not a product of one design point — it is infrastructure that makes the entire field of CiM-based DNN acceleration more reproducible and comparable. Without it, every new paper compares against baselines on different technology nodes with different assumptions, making progress hard to measure.

---

## Standalone Study Guide

### What to master before moving on

- Explain why compute-in-memory attacks data movement by moving MACs toward storage.
- Describe the analog crossbar multiply-accumulate primitive and why ADC/DAC overhead matters.
- State the Titanium Law: ADC cost can dominate analog CiM energy as resolution and array size grow.
- Explain RAELLA as arithmetic reform, not simply a better memory cell.
- Use CiMLoop as the modeling bridge across devices, circuits, architectures, mappings, and workloads.

### Self-check questions

1. What part of a crossbar performs multiplication, and what part performs accumulation?
2. Why can analog CiM lose its energy advantage after accounting for ADCs?
3. Why is an apples-to-apples modeling framework necessary when comparing CiM designs?

### Exercises

1. Trace one vector-matrix multiply through a resistive crossbar, including input delivery and output conversion.
2. List three device-level nonidealities and explain how each can appear as model accuracy loss.
3. Compare SRAM CiM, DRAM CiM, ReRAM CiM, and photonic computing along precision, density, and programmability axes.

### Common traps

- Treating analog CiM as "free MACs." Data conversion, input drivers, and peripheral circuits are often dominant.
- Comparing papers by peak TOPS without normalizing precision, accuracy, technology node, and array size.
- Forgetting that advanced technologies move constraints rather than eliminating them.

---

## Key Terms

| Term | Gloss |
|---|---|
| **Compute-in-Memory (CiM)** | Using the physical properties of memory cells to perform arithmetic (e.g., MAC) inside the storage array, eliminating read-data movement. |
| **Near-data processing** | Moving computation physically close to memory (but not inside it) to reduce wire lengths and capacitive energy. Includes eDRAM and 3D-stacked DRAM. |
| **eDRAM** | Embedded DRAM integrated on the same die as logic; 2.85× denser than SRAM but pricier than off-chip DRAM. |
| **HMC / HBM** (Hybrid Memory Cube / High Bandwidth Memory) | 3D-stacked DRAM technologies that place memory dies directly above a logic die, multiplying bandwidth and reducing access energy. |
| **Analog crossbar** | A 2D array of programmable resistances (weights) with word-line voltages (inputs) producing bit-line currents (partial sums) via Ohm's Law. |
| **Memristor / ReRAM / RRAM** | Non-volatile resistive memory device whose resistance is programmable; used as the weight element in analog CiM crossbars. |
| **DAC** (Digital-to-Analog Converter) | Converts digital input activations to analog voltages/currents for delivery to the crossbar word lines. |
| **ADC** (Analog-to-Digital Converter) | Converts analog bit-line currents back to digital partial sums. The dominant energy/area overhead in most CiM designs. |
| **Weight slicing** | Representing a multi-bit weight across multiple devices (bit slices), each storing fewer bits; increases ADC conversion count. |
| **Input slicing** | Decomposing a multi-bit input activation into bit-serial cycles; increases compute time. |
| **Titanium Law** | Formula expressing total ADC energy as Energy/Convert × Converts/MAC × MACs/DNN × 1/Utilization; each factor trades off against the others. |
| **RAELLA** | An inference-time technique (center+offset encoding, adaptive weight slicing, dynamic input slicing) that reduces ADC input range 1024× without DNN retraining. |
| **CiMLoop** | MIT's Timeloop+Accelergy-based CiM modeling tool; flexible user-defined specs, data-value-dependent energy accuracy (within 8%), 1000× faster than prior accurate tools. |
| **Pulse-width modulation (PWM)** | Encoding an analog value in the time duration of a pulse; used for input delivery to CiM when voltage modulation is impractical. |
| **Charge sharing** | A technique in DRAM/SRAM CiM where simultaneous activation of multiple bit-line rows causes charge averaging, performing bitwise logic without digital gates. |
| **Data-value-dependence** | The property that analog component energy depends on the *actual numerical values* being processed, not just the number of operations. |
| **Photonic computing** | Using optical signals (photons) in waveguides and modulators to perform matrix multiplication with distance-independent energy and no electromagnetic crosstalk. |
| **CiM stack** | The five levels of cross-layer co-design: devices → circuits → architecture → mapping → workload. |

---

## Takeaways

- The **fundamental motivation for CiM** is the same energy hierarchy established in L01: DRAM access costs ~200× an ALU operation. CiM eliminates the read entirely by moving computation into the array.
- **Analog crossbars** implement multiply-accumulate using Ohm's Law (weight = conductance, input = voltage, product = current) and Kirchhoff's Current Law (accumulation = current sum). This is physically elegant but introduces sensitivity to device variation, non-linearity, and limited precision.
- The **ADC is the dominant cost** in practical CiM designs, not the crossbar computation. ADC energy scales exponentially with resolution; this is quantified by the **Titanium Law** (four-factor product involving Energy/Convert, Converts/MAC, MACs/DNN, and 1/Utilization).
- The Titanium Law reveals a **fundamental tension**: reducing ADC resolution requires more bit-slices (more conversions); increasing array rows reduces utilization overhead but increases ADC resolution. Prior escape routes (pruning, low-resolution ADC) both require retraining.
- **RAELLA** escapes without retraining by reshaping the *distribution* of analog values going into the ADC — center+offset encoding, adaptive weight slicing, and dynamic input slicing — achieving 3.9× energy improvement and 1.8× throughput improvement vs. iso-area ISAAC.
- **CiM spans multiple substrates**: SRAM (process-compatible, modest gains), DRAM (bitwise logic via charge sharing), and NVM/memristors (density and non-volatility, but precision challenges). **No single substrate dominates** all applications.
- **Cross-layer co-design is mandatory**: device physics, circuit topology, array architecture, dataflow mapping, and DNN workload shape all interact. The **CiMLoop** modeling tool captures these interactions (data-value-dependent energy, within 8% error, 1000× faster than prior accurate tools) and enables apples-to-apples comparison and design-space exploration.
- **Photonic computing** is an emerging substrate where multiplication is passive and inter-PE energy is distance-independent — a potentially radical departure from the energy scaling of CMOS, modeled by CiMLoop's photonic component library.

---

## Connections to Later Lectures

- **L01 energy hierarchy (DRAM 200×):** CiM is the most aggressive architectural response to that hierarchy — it eliminates the weight read cost entirely. L11 is the culmination of the energy-reduction arc that started on the first day.
- **L05–L06 dataflows:** The weight-stationary dataflow introduced in earlier lectures maps directly onto the CiM weight-stationary array. The loop-nest view (M rows × CHW columns) is the same formalism applied to a new substrate.
- **L07–L10 sparsity:** Sparse weights reduce MACs/DNN (one Titanium Law factor), but they also reduce array utilization (another factor). The interaction is non-trivial — sparsity is less straightforwardly beneficial for CiM than for digital architectures.
- **L12 — Reduced Precision:** Weight/input quantization (the Format layer of the TeAAL pyramid) directly determines precision per device slice, Converts/MAC, and ADC resolution — all Titanium Law factors. Precision co-design with CiM is a natural joint problem.
- **Lab 5:** The course lab on CiM uses CiMLoop directly to explore the design space introduced in this lecture. Concepts here (array size, bit slicing, ADC cost) are the parameters students will tune.
- **Final Project:** Design-space exploration across array size and DNN shapes (shown in the CiMLoop slides) is explicitly proposed as a final project topic. Photonic modeling with CiMLoop is also an available extension.

---

## Appendix — Slide-to-Section Map

| Slides | Section |
|---|---|
| L11-1 | Title |
| L11-2 … L11-9 | Ch.1 — Memory technology landscape; near-data processing (eDRAM, 3D DRAM); memory cost quantification |
| L11-10 … L11-27 | Ch.2 — Analog crossbar principle; weight-stationary CiM dataflow; practical design constraints (device precision, array size, ADC, input delivery) |
| L11-28 … L11-52 | Ch.3 — ISAAC case study; CiM energy breakdown; Titanium Law; prior escape routes; RAELLA techniques and results |
| L11-53 … L11-59 | Ch.4 — DNN/CiM co-design; SRAM CiM (current-mode, charge-sharing); DRAM CiM (Ambit charge-sharing AND/OR); research landscape |
| L11-60 … L11-75 | Ch.5 — CiM stack; CiMLoop (flexibility, accuracy, speed); apples-to-apples comparison; design-space exploration |
| L11-75 … L11-80 | Ch.5 — CiMLoop collaborations; photonic computing frontier |
| L11-81 | Summary and references |
