# L11 — 進階技術（Advanced Technologies）

> **課程：** 6.5930/1 — 深度學習硬體架構（Hardware Architectures for Deep Learning）
> **講師：** Joel Emer 與 Vivienne Sze
> **講授日期：** 2026 年 3 月 9 日 · **投影片：** 81 頁 · **來源：** [`Lecture/L11-Advanced_Tech.pdf`](../../Lecture/L11-Advanced_Tech.pdf)
>
> 本章依據投影片與本地論文 PDFs 重建 lecture narrative。為了 copyright safety，本章不重製投影片或論文圖，而用文字描述必要視覺內容。

---

## 一句話總結（TL;DR）

本課程多數時間都把 memory 視為 digital MAC array 周圍的 hierarchy。Lecture 11 問的是：當這條 compute-memory 邊界開始移動，會發生什麼？**鄰近記憶體運算（near-memory processing）** 仍然讓 compute 與 memory 分離，但縮短兩者距離，例如 eDRAM 與 3D-stacked DRAM。**記憶體內運算（compute-in-memory, CiM）** 更進一步：memory device 或 array 本身參與 computation。

本講主線是：data movement 很昂貴，因此設計者先把 memory 移近 compute，再把簡單 compute 移近 memory，最後問 memory 本身能否做 MAC。Analog CiM crossbar 用 Ohm's Law 做 multiplication、用 Kirchhoff's Current Law 做 accumulation，但 practical systems 往往被 peripheral costs 主導，尤其是 ADC。**Titanium Law** 把 ADC energy 表達為 energy per conversion、conversions per MAC、MACs per DNN 與 utilization penalty 的乘積。Slides 中的 **RAELLA** 與 **CiMLoop** 展示為什麼 advanced technologies 必須跨 device、circuit、architecture、mapping 與 workload 做 co-design。

---

## 本講解決什麼問題

本講問題不是「如何發明更快 multiplier？」而是：

> 當 data movement 主導 energy 與 performance 時，compute-memory 邊界的哪些部分應該重畫？重畫之後，新瓶頸會在哪裡出現？

Near-memory designs 會縮短 wire distance、增加 bandwidth。CiM designs 嘗試完全消除某些 reads。但兩者都不會讓 computation 免費。eDRAM 消耗 area 並有 refresh/circuit constraints；3D memory 有 thermal 與 logic-die area limits；analog CiM 需要 DAC、ADC、calibration、precision slicing 與可靠 modeling。這講是在教你如何評估這些技術，而不是被 peak TOPS 閃瞎。

---

## 為什麼本講重要

Advanced technologies 很有誘惑力，因為它們聽起來像 memory wall 的逃生門。Hardware architect 需要問更冷靜的問題：哪個成本被移走了？它又在哪裡重現？

例如 resistive crossbar 可以「就地」計算很多 products，但結果是 analog current。如果 accelerator 其餘部分是 digital，current 必須經過 ADC。省下的 weight reads 可能換成 conversion energy、limited precision 與 array utilization loss。這就是為什麼 L11 不是 device survey，而是**跨層 accounting** 的課。

---

## 先備知識與心智模型

你應該熟悉：

- L01-L03 的 energy hierarchy：off-chip memory 遠比 local arithmetic 昂貴。
- L05 的 dataflow：讓某個 operand stationary 可以減少 movement。
- L07-L10 的 sparsity：減少 MACs 只有在 overhead 被計入時才有意義。
- Matrix-vector multiplication：$y_j = \sum_i x_i w_{ij}$。

L11 的心智模型是一個 matrix-vector multiply。在 digital accelerator 中，weights 從 memory 讀出後送到 MACs。在 near-memory processing 中，weight memory 物理上更近或 bandwidth 更高。在 CiM 中，stored weight 本身幫忙產生 product。

---

## 學習目標（Learning Objectives）

讀完本講後，你應該能夠：

- 區分**鄰近記憶體運算（near-memory processing）**與**記憶體內運算（compute-in-memory）**。
- 解釋 eDRAM 與 3D-stacked DRAM 分別降低 memory bottleneck 的哪一部分。
- 描述 analog crossbar 如何用 voltage、conductance 與 current 計算 dot product。
- 解釋為什麼 ADC 與 DAC 可能主導 practical CiM systems。
- 讀懂 **Titanium Law**，並判斷某項技術改變了哪個 factor。
- 解釋 pruning、low precision 與 array size 在 CiM 中為何與 digital accelerator 不同。
- 描述 **RAELLA** 對 ADC input distribution 做了什麼改變。
- 說明為什麼 **CiMLoop** 這類 modeling tool 對跨 devices 與 circuit styles 的公平比較不可或缺。
- 用 DaDianNao、Neurocube、Tetris 與 TPU 作為 memory-centric design choices 的 anchored examples。

---

## 主要教材式敘事

### 1. 從 Memory Cost 開始，而不是從 Device Hype 開始

Lecture 11 slide 5 重複 Horowitz, ISSCC 2014 的量化動機：8-bit add 是 0.03 pJ，32-bit SRAM read from 8 KB SRAM 是 5 pJ，32-bit DRAM read 是 640 pJ。這些是 slide-derived numbers。精確數值依 process 與 assumptions 而變，但排序是穩定教訓：從遠處搬資料通常比對資料做運算更昂貴。

這解釋了本講順序：

1. 用 eDRAM 把更多 memory 放到 chip 上。
2. 用 3D stacking 把 DRAM 放進 package。
3. 把簡單 compute 放在 logic layer 或 memory periphery。
4. 讓 memory array 直接參與 computation。

### 2. Near-Memory Processing：仍是 Digital，但更靠近

**eDRAM** 是 density move。Lecture 11 slides 6-8 說 eDRAM 比 SRAM dense，並以 DaDianNao 為例。DaDianNao 把大量 synaptic weights 存在 on-chip eDRAM，降低昂貴 off-chip memory traffic。Paper 報告在 28 nm 下 10 MB SRAM 需要 20.73 mm2，同容量 eDRAM 有 2.85x higher storage density；256-bit eDRAM read 為 0.0192 nJ，而 Micron DDR3 為 6.18 nJ，energy ratio 為 321x（DaDianNao Section V-A）。這些是 paper-derived claims。

**3D-stacked DRAM** 是 bandwidth 與 distance move。HMC/HBM 類堆疊把 DRAM dies 放在 logic die 附近或上方，用 TSVs 連接。Tetris 報告 3D memory 提供 160-250 GB/s bandwidth，且 access energy 比 DDR3 低 3-5x，接著用這個 substrate 重新平衡 accelerator area 並把 accumulation 移近 memory（Tetris Section 2.4 and Section 3）。Neurocube 使用 HMC-style memory，讓 vault controllers 中的 programmable sequence generators 驅動 neural-network data movement（Neurocube Sections III-V）。

教學解讀：eDRAM 問「足夠多 weights 能不能放近 compute？」3D memory 問「memory bandwidth 能不能跟 PE count 一起 scale？」兩者大多仍保留 digital arithmetic，還沒有讓 memory cells 成為 multipliers。

### 3. Compute-In-Memory：讓 Array 成為 Datapath

在 analog resistive crossbar 中，每個 weight 存為 conductance $G$，每個 input 以 voltage $V$ 施加，device current 是：

$$
I = V G.
$$

對一個 column 而言，多個 rows 的 currents 會相加：

$$
I_{\text{col}} = \sum_i V_i G_i.
$$

這就是 dot product。Ohm's Law 提供 multiplication；Kirchhoff's Current Law 提供 addition。Crossbar 天然是 **weight-stationary**：weights 留在 memory array，input vectors 重複施加。

理想圖像很強，但不完整。Digital DNN accelerator 仍需要 digital activations 與 outputs，因此 practical CiM systems 需要：

- DACs 或 pulse encoders，把 inputs 呈現給 array。
- ADCs，把 column currents 轉回 digital values。
- Bit slicing，因為 device 可存 bits 少於 model weight 精度。
- Calibration 與 margins，以處理 nonlinearity、device variation、temperature、voltage 與 noise。
- Mapping decisions，讓 arrays 保持 utilized。

### 4. ADC 是 CiM 的稅吏

Lecture 11 slides 29-52 聚焦 ADC bottleneck，並介紹 **Titanium Law**：

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

每個 factor 都是一個 lever：

- $E_{\text{convert}}/\text{convert}$ 會隨 ADC resolution 急遽上升。
- $\text{converts}/\text{MAC}$ 會在 weights 或 inputs 被切成多個 cycles/devices 時上升。
- $\text{MACs}/\text{DNN}$ 由 smaller models、pruning 或 algorithmic changes 降低。
- $1/\text{utilization}$ 在 array 部分空著時上升。

痛點在於 levers 彼此打架。降低 ADC resolution 可能需要更多 slices，使 conversions per MAC 上升。增加 array rows 可能改善某些 shapes 的 utilization，但也提高 analog summation range 與所需 ADC resolution。Pruning 降低 MACs，卻可能讓 CiM arrays 填不滿。

### 5. RAELLA：改變 ADC 看見的 Distribution

Lecture 11 slides 43-52 以 RAELLA 作為 Titanium Law 的回應。Slide-derived explanation 是：RAELLA 在不 retrain DNN 的情況下，降低 ADC 看到的 analog input range。它使用：

- **Center + offset encoding：** 減去 column center，讓 analog array 計算較小 residuals。
- **Adaptive weight slicing：** 只對可能超過 ADC range 的 computations 花額外 conversions。
- **Dynamic input slicing：** 先用粗 input slices speculate，只有需要時才 recover。

Lecture 11 slide 52 報告 input to ADC 降低 1024x，energy efficiency 相對 iso-area ISAAC 提升 3.9x，throughput 提升 1.8x。由於 RAELLA PDF 不在本 worker 指定 local paper list 中，這些 claims 在本章標記為 slide-derived。

### 6. CiM 橫跨 SRAM、DRAM、NVM 與 Photonics

接著本講把 substrate 視野打開：

- **SRAM CiM** 可用 current-mode 或 charge-sharing circuits。它 process-compatible，但受 SRAM cell area 與 circuit nonidealities 限制。
- **DRAM CiM** 可利用 charge sharing 做 in-array bitwise operations。它有 density 優勢，但面對 refresh、timing 與 peripheral constraints。
- **NVM / ReRAM / memristor CiM** 天然以 resistance/conductance 存 weights，dense 且 nonvolatile，但 precision 與 variation 是大問題。
- **Photonics** 用 light propagation、modulation 與 interference 實作 linear algebra primitives。Lecture 11 用它作為 emerging example，展示 data-movement physics 如何不同於 CMOS wires。

重要教訓是：這些不是可互換的「更快 MAC」技術。每一種都改變 cost model，因此也改變最佳 mapping 與 model design。

### 7. CiMLoop：對整個 Stack 建模

Lecture 11 slides 62-75 將 CiMLoop 呈現為 Timeloop/Accelergy-style tool 的 CiM 延伸。關鍵 modeling requirement 是 **data-value dependence**：analog energy 可能取決於實際被處理的 values，而不只是 operation count。高 conductance device 在高 input voltage 下耗散的 energy，比低 conductance device 在低 voltage 下更多。

Slide-derived claims 是：CiMLoop 可捕捉 cross-stack interactions，error within 8%；用 statistical models 比 prior accurate simulation 快超過 1000x；並可把 designs normalize 到相同 technology/device/ADC assumptions 比較。教學解讀是：advanced technologies 讓 modeling 更重要，而不是更不重要。當 device physics 進入 datapath，單純 MAC count 不再是安全 proxy。

---

## Worked Examples

### 範例 1：為什麼移動 Weights 很重要

假設某 layer 做 1 million 次 8-bit additions，並需要 1 million 次 32-bit DRAM reads。用 slide-derived Horowitz numbers，adds 約花 $1{,}000{,}000 \times 0.03$ pJ = 30,000 pJ，而 DRAM reads 約花 $1{,}000{,}000 \times 640$ pJ = 640,000,000 pJ。Reads 主導。

精確數字不應過度泛化，但 design lesson 穩定：能減少遠端 memory reads 的 technology，可能比稍微更有效率的 arithmetic unit 更重要。

### 範例 2：Crossbar Dot Product

令三個 inputs voltages 為 $V = [1, 2, 1]$，三個 stored conductances 為 $G = [3, 0.5, 2]$，使用任意 normalized units。Column current 是：

$$
I_{\text{col}} = 1 \cdot 3 + 2 \cdot 0.5 + 1 \cdot 2 = 6.
$$

這與 dot product 是同一個 computation。硬體意義是 array 沒有把三個 weights 讀出送到 digital multiplier；但 output current 仍需要 sensing、range control，通常也需要 ADC conversion。

### 範例 3：讀 Titanium Law

假設某 design 用 lower-resolution ADC 讓每次 ADC conversion energy 降低 4x，但因為 extra slicing，現在每個 MAC 需要 3x conversions。忽略其他 factors，ADC energy 變化為：

$$
\frac{1}{4} \times 3 = 0.75.
$$

這只有 1.33x ADC-energy improvement，不是 4x。Titanium Law 的價值就是防止某個 knob 被單獨宣傳，而其 coupled cost 被藏起來。

---

## 關鍵方程式與讀法

Analog multiplication：

$$
I = V G.
$$

Input activation 由 voltage $V$ 表示，weight 由 conductance $G$ 表示，product 由 current $I$ 表示。

Analog accumulation：

$$
I_{\text{col}} = \sum_i V_i G_i.
$$

同一 bitline 上的 currents 物理相加，形成 dot product。

Titanium Law：

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

這不只是 ADC equation，而是檢查 CiM claims 的 checklist：哪個 term 改善？哪個 term 變差？End-to-end accuracy 與 throughput 如何？

---

## 硬體含義

- **Bandwidth：** 3D memory 可比 conventional off-chip DRAM 提供更高 bandwidth，但 logic die area 與 thermal envelope 限制可放置的 nearby compute。
- **Area：** eDRAM 與 SRAM 在 density、latency、refresh 與 integration complexity 間取捨。Large buffers 可能主導 accelerator area。
- **ADC/DAC overhead：** CiM 的 analog compute core 可很有效率，但 peripheral conversion 可能主導 total energy。
- **Precision：** Device precision、ADC resolution、weight slicing、input slicing 與 model accuracy 是一個 coupled design space。
- **Utilization：** 巨大 CiM array 若 layer shapes 無法填滿 rows/columns，就會低效。
- **Programmability：** Near-memory 與 CiM designs 常把不尋常 mapping constraints 暴露給 compilers 與 modeling tools。
- **Correctness：** Analog nonidealities 不只是 performance issues；它們會改變 numerical results，進而影響 model accuracy。

---

## 常見誤解

### 誤解：Compute-in-memory 讓 MACs 免費。

Array-level multiply-accumulate 可能很便宜，但 DACs、ADCs、sense amplifiers、calibration、slicing 與 digital accumulation 可能主導成本。

### 誤解：Crossbar rows 越多永遠越好。

更多 rows 可能增加 parallelism，但也會增加 analog summation range、所需 ADC resolution、wire parasitics 與 utilization risk。

### 誤解：Pruning 永遠幫助 CiM。

Pruning 降低 MACs/DNN，但 sparse weights 可能讓 dense CiM arrays 填不滿。Titanium Law 透過 utilization factor 讓這件事可見。

### 誤解：TOPS 足以比較 advanced technologies。

TOPS 忽略 precision、accuracy、conversion overhead、memory traffic、array utilization、batching assumptions 與 technology normalization。

---

## 與前後講次的連結

- **L01-L03：** L11 是早期 energy hierarchy 的 physical-technology answer。
- **L05 Dataflow：** Weight-stationary mapping 再次出現，因為 CiM arrays 天然把 weights 留在原地。
- **L07-L10 Sparsity：** Sparse models 降低 MAC count，但在 array-based CiM 中可能傷害 utilization。Sparsity benefits 是 technology-dependent。
- **L12 Reduced Precision：** Precision 是 CiM 中央議題，因為 ADC bits、device bits 與 model accuracy 耦合。
- **Labs and final projects：** CiMLoop 是從 lecture concepts 走向 design-space exploration 的 modeling bridge。

---

## Paper Bridge: DaDianNao

### Bibliographic Identity

- **Title:** DaDianNao: A Machine-Learning Supercomputer
- **Authors:** Yunji Chen et al.
- **Year / venue:** MICRO 2014
- **Local PDF:** [`papers/L15_DaDianNao_Chen_MICRO2014.pdf`](../../papers/L15_DaDianNao_Chen_MICRO2014.pdf)
- **Used in lecture:** Lecture 11 eDRAM and near-memory motivation

### Problem Addressed

DaDianNao 處理大型 neural networks 的 memory bandwidth 與 energy cost，尤其是具有大量 synaptic weights 的 layers。它觀察到若 weights 留在 external memory，就需要 high-bandwidth、high-energy transfers。

### Core Idea

把 large eDRAM banks 放在 neural functional units 附近，並移動 neuron values，而不是反覆移動 synaptic weights。Node 包含多個 tiles，每個 tile 有 local eDRAM 與 compute，另有 central eDRAM 與 interconnect。

### Relevance to This Lecture

DaDianNao 是 eDRAM near-memory design 的例子。它讓 computation 保持 digital，但改變 memory hierarchy，使 weights 儲存在 compute 附近。

### Key Claims Used in This Chapter

- Off-chip memory bandwidth 會成為大型 neural-network layers 的瓶頸；見 DaDianNao Sections I and IV。
- Paper 指出在其討論情境中 off-chip memory accesses 可讓 total energy 增加約 10x；見 Section IV。
- Paper 報告 eDRAM density 比 SRAM 高 2.85x，且在 28 nm assumptions 下 256-bit eDRAM read 與 Micron DDR3 的 energy ratio 為 321x；見 Section V-A。
- 一個 node 包含 36 MB eDRAM；見 Table II 附近 architecture parameter discussion。
- Paper 報告 64-chip system 相對 evaluated baseline 有 150.31x average energy reduction；見 abstract 與 evaluation discussion。

### What Students Should Remember

1. DaDianNao 是 near-memory，不是 compute-in-memory。
2. 主要動作是讓大量 weights 足夠靠近 compute，以降低 external memory traffic。
3. eDRAM density 有幫助，但設計仍付 area、wire 與 integration costs。

### Limitations and Assumptions

該 paper 針對其時代的 neural networks 與 technology assumptions。量化 ratios 應當作 historical anchored evidence，而非 universal constants。

### Suggested Insertion Points

解釋 eDRAM，以及從 ordinary memory hierarchy 走向 memory-centric accelerator design 的第一步時引用 DaDianNao。

---

## Paper Bridge: Neurocube

### Bibliographic Identity

- **Title:** Neurocube: A Programmable Digital Neuromorphic Architecture with High-Density 3D Memory
- **Authors:** Donghyuk Kim et al.
- **Year / venue:** ISCA 2016
- **Local PDF:** [`papers/L15_Neurocube_Kim_ISCA2016.pdf`](../../papers/L15_Neurocube_Kim_ISCA2016.pdf)
- **Used in lecture:** Lecture 11 3D-stacked memory / HMC discussion

### Problem Addressed

Neurocube 用 3D high-density memory 與 logic tier integration 處理 neural networks 的 memory capacity 與 bandwidth limits。

### Core Idea

在 HMC logic layer 中整合 processing elements 與 programmable neurosequence generators（PNGs）。Memory system 透過 programmable state machines 驅動已知的 neural-network data movement patterns。

### Relevance to This Lecture

Neurocube 展示 3D memory stack 中的 near-memory processing。它不是 analog CiM，而是使用 memory organization 與 programmable controllers 降低 movement overhead 的 digital memory-centric architecture。

### Key Claims Used in This Chapter

- HMC 提供多個 vaults 與 highly parallel access；見 Neurocube Section II-B and Table I。
- Architecture 使用與 vault controllers 相關的 programmable neurosequence generators；見 Sections III-V。
- General neural-network nested loops 可映射到 PNG 中的 finite-state machines；見 Figure 8 及其周邊文字。
- Paper 報告 28 nm 與 15 nm designs 在 scene-labeling inference 的 example throughput；見 Section VI。

### What Students Should Remember

1. 3D memory 提供 bandwidth 與 capacity，但 useful performance 需要 mapping 與 scheduling。
2. Programmable memory-side controllers 可利用 neural-network static access patterns。
3. Near-memory processing 可以維持 fully digital，同時改變 architecture。

### Limitations and Assumptions

Neurocube 以 neuromorphic/neural-network workloads 與 HMC assumptions 為背景。它適合作為 memory-centric design example，而不是 analog CiM 證據。

### Suggested Insertion Points

解釋 3D memory 為什麼不只是更寬的 DRAM bus，而是 logic layer 可參與 scheduling 與 data movement 時引用 Neurocube。

---

## Paper Bridge: Tetris

### Bibliographic Identity

- **Title:** Tetris: Scalable and Efficient Neural Network Acceleration with 3D Memory
- **Authors:** Mingyu Gao et al.
- **Year / venue:** ASPLOS 2017
- **Local PDF:** [`papers/L15_TETRIS_Gao_ASPLOS2017.pdf`](../../papers/L15_TETRIS_Gao_ASPLOS2017.pdf)
- **Used in lecture:** Lecture 11 3D memory and near-memory accumulation

### Problem Addressed

Tetris 問的是：當 3D memory 提供比 conventional off-chip DRAM 更高 bandwidth 與更低 access energy 時，neural-network accelerator 應該如何重新設計？

### Core Idea

使用 3D memory 將 accelerator area 從 large on-chip SRAM buffers 重新平衡到 PE arrays，並把 simple accumulation operations 移近 DRAM banks，以降低 output-feature-map traffic。

### Relevance to This Lecture

Tetris 是 near-memory bandwidth 與 near-memory computation 之間的橋樑。它說明單純接上 3D DRAM 不夠；dataflow scheduling、buffer sizing 與 accumulation location 都必須重新思考。

### Key Claims Used in This Chapter

- Abstract 報告相對 conventional low-power DRAM systems 有 4.1x performance improvement 與 1.5x energy reduction。
- Section 2.4 指出 3D memory 可提供 160-250 GB/s bandwidth，access energy 比 DDR3 低 3-5x。
- Section 3 討論 PE/buffer area rebalancing 與 per-vault engines。
- Section 4 討論 dataflow scheduling 與 in-memory accumulation 以降低 ofmap traffic。

### What Students Should Remember

1. 3D memory 改變 buffers 與 PEs 之間的最佳 accelerator balance。
2. Near-memory accumulation 只有在 dataflow scheduling 也配合時才省 traffic。
3. Bandwidth alone 不會解決 utilization、area 或 scheduling。

### Limitations and Assumptions

Tetris 是 digital 3D-memory accelerator，不是 analog CiM design。Speedup 依賴 evaluated networks、area budgets、vault organization 與 dataflow schedules。

### Suggested Insertion Points

解釋 advanced memory technology 為什麼必須搭配 architecture 與 mapping changes 時引用 Tetris。

---

## Paper Bridge: TPU as a Digital Baseline

### Bibliographic Identity

- **Title:** In-Datacenter Performance Analysis of a Tensor Processing Unit
- **Authors:** Norman P. Jouppi et al.
- **Year / venue:** ISCA 2017
- **Local PDF:** [`papers/L12_TPU_Jouppi_ISCA2017.pdf`](../../papers/L12_TPU_Jouppi_ISCA2017.pdf)
- **Used in lecture:** Memory-aware digital acceleration 的 contextual baseline

### Problem Addressed

TPU paper 分析已部署 digital inference accelerator，展示 matrix units、on-chip memory 與 weight-memory bandwidth 如何決定 datacenter inference performance。

### Core Idea

TPU 使用 256 x 256 systolic matrix unit，含 65,536 個 8-bit MACs、大型 software-managed on-chip memory，以及 deterministic execution model。它不是 advanced memory device，但很適合當對照：它以 digital systolic array 與 explicit memory management 攻擊 data movement。

### Relevance to This Lecture

TPU 提供 advanced technologies 要競爭的 baseline。CiM design 需要打敗的不是 naive MAC array，而是具備 locality、systolic data reuse 與 roofline-aware performance constraints 的成熟 digital accelerator。

### Key Claims Used in This Chapter

- Abstract 指出 TPU 有 65,536 8-bit MAC matrix unit、92 TOPS peak 與 28 MiB on-chip memory。
- Section 2 描述 matrix unit、unified buffer、weight FIFO 與 systolic execution。
- Section 4 使用 roofline model，並指出 weight memory fetched 的 ridge point 約 1350 operations per byte。
- Paper 報告在 evaluated datacenter workloads 中相對 contemporary GPU/CPU inference systems 有 15x-30x speedup 與顯著更高 TOPS/Watt；見 abstract。

### What Students Should Remember

1. Digital accelerators 已經非常積極利用 locality。
2. Roofline reasoning 在比較 advanced technologies 時仍有用。
3. Advanced technology claims 應該與 strong digital baselines 比較。

### Limitations and Assumptions

TPU paper 評估特定時代的 Google datacenter workloads。它是 baseline 與 modeling contrast，不是 CiM device behavior 的證據。

### Suggested Insertion Points

當學生需要 systolic arrays、memory bandwidth 與 roofline limits 的 grounded digital reference point 時引用 TPU。

---

## 獨立學習指南

### 進入下一講前必須掌握

- 解釋 moving memory closer 與 computing inside memory 的差異。
- 從 $I=VG$ 與 current summation 推導 crossbar dot product。
- 用 Titanium Law 判斷某 CiM 技術改變哪個 cost。
- 解釋 ADCs 如何把優雅的 analog primitive 變成 system-level tradeoff。
- 依 cost model 而不是 hype 比較 eDRAM、3D DRAM、analog CiM 與 photonics。

### 自我檢核問題

1. 為什麼 DaDianNao 是 near-memory 而不是 compute-in-memory？
2. 3D memory 改善什麼？又引入哪些 constraints？
3. 在 resistive crossbar 中，什麼代表 input？什麼代表 weight？
4. 為什麼降低 ADC resolution 可能增加 conversions per MAC？
5. 為什麼 pruning 可能傷害 CiM utilization？
6. 為什麼 CiMLoop 需要 data-value-dependent modeling？

### 練習

1. 用 Titanium Law 比較兩個 designs：Design A 每次 conversion energy 降低 2x，但 conversions per MAC 增加 2x；Design B 保持 conversions 不變，但 utilization 從 50% 提升到 80%。
2. 對一個 $4 \times 4$ crossbar，寫出其中一個 column 計算的 dot product，並標出 DAC 與 ADC conversion 發生在哪裡。
3. 選本章一個 near-memory paper bridge，說明它降低哪種 memory movement：weights、activations、partial sums，或 off-chip transfers。
4. 解釋為什麼 sparse、heavily pruned model 可能適合 digital sparse accelerator，卻不適合 dense analog crossbar。
5. Paper-reading bridge：閱讀 Tetris Section 2.4，總結 3D memory 為什麼改變 PE/buffer area allocation。

---

## 關鍵詞彙（Key Terms）

| 詞彙 | 意義 |
|---|---|
| **鄰近記憶體運算（Near-memory processing）** | 讓 compute 保持 digital，但物理上靠近 memory，或使用 high-bandwidth memory packaging。 |
| **記憶體內運算（Compute-in-memory, CiM）** | 使用 memory array 或 memory device 本身作為 computation 的一部分。 |
| **eDRAM** | Embedded DRAM；比 SRAM dense，適合較大 on-chip storage，但有 integration 與 refresh tradeoffs。 |
| **3D-stacked DRAM** | DRAM dies 與 logic die 堆疊並用 TSVs 連接，提高 bandwidth 並縮短距離。 |
| **HMC / HBM** | 用於降低 memory bottleneck 的 3D 或 2.5D high-bandwidth memory technologies。 |
| **Analog crossbar** | 由 programmable conductances 組成的 grid，用 voltages 與 currents 計算 dot products。 |
| **Conductance（電導）** | Resistance 的倒數；在 resistive CiM 中代表 stored weight。 |
| **DAC** | Digital-to-analog converter；把 digital inputs 轉成 analog signals。 |
| **ADC** | Analog-to-digital converter；把 analog sums 轉回 digital values。 |
| **Weight slicing** | 將 multi-bit weight 分散到多個 devices 或 cycles 表示。 |
| **Input slicing** | 將 multi-bit input 分散到多個 temporal 或 analog steps 表示。 |
| **Titanium Law** | ADC energy per DNN inference 的乘積式：conversion energy、conversions per MAC、MACs per DNN 與 utilization penalty。 |
| **RAELLA** | Slides 中介紹的 CiM technique，用不需 retraining 的方式重塑進入 ADC 的 analog values。 |
| **CiMLoop** | Slides 中介紹的 cross-stack CiM design exploration modeling framework。 |
| **Data-value dependence** | Analog energy 取決於實際處理 values，而不只是 operation count 的特性。 |
| **Photonic computing** | 使用 optical signals 進行 computation，常用 modulation/interference 實作 linear algebra。 |

---

## 重點回顧（Takeaways）

- Advanced technologies 是 data movement 問題的回應，不是 architecture 的魔法替代品。
- eDRAM 與 3D memory 讓 computation 保持 digital，但降低 storage distance 或增加 bandwidth。
- Analog CiM 優雅地計算 dot products，但 ADC/DAC、slicing、variation 與 utilization 主導 practical design。
- Titanium Law 是檢查 CiM proposal 是否一邊改善、一邊惡化另一項成本的簡潔工具。
- RAELLA 的 slide-level lesson 是 distribution reshaping：降低 ADC 必須解析的 analog range。
- CiMLoop 的 slide-level lesson 是 modeling discipline：advanced substrates 需要 value-aware、cross-layer evaluation。
- TPU 這類 strong digital baseline 很重要；advanced technology 應該與 memory-aware architectures 比較，而非 naive MAC arrays。

---

## 連結（Connections）

L11 把整門課連到 physical implementation choices。L01-L03 說 memory movement 主導成本；L05 說 dataflow 決定什麼會移動；L07-L10 說 sparsity 改變 dynamic work 與 metadata traffic；L11 問如果 storage device、circuit interface 與 package 被重新設計會如何。L12 接著回到 precision，因為 CiM device bits 與 ADC bits 會直接塑造 accuracy、energy 與 throughput。

---

## 附錄 — 投影片對照表

| Slides | 章節 | Notes |
|---|---|---|
| 1 | Title | Lecture framing。 |
| 2-9 | Memory cost and near-memory | eDRAM、3D DRAM、DaDianNao、Neurocube、Tetris。 |
| 10-27 | Analog crossbar | Conventional processing vs. CiM、Ohm/Kirchhoff MAC、weight-stationary mapping、practical limits。 |
| 28-52 | ADC bottleneck | ISAAC context、Titanium Law、RAELLA techniques 與 slide-reported results。 |
| 53-60 | CiM substrates | SRAM、DRAM、NVM/memristor discussion。 |
| 60-75 | CiMLoop | Device/circuit/architecture/mapping/workload stack 與 modeling claims。 |
| 75-80 | Photonics | Emerging substrate 與 CiMLoop photonics modeling。 |
| 81 | Summary/references | 用於 source attribution。 |

---

## Source Notes

- Lecture flow follows `Lecture/L11-Advanced_Tech.pdf`, slides 1-81。
- Add/SRAM/DRAM 的 memory energy numbers 是 Lecture 11 slide 5 的 slide-derived claims，該 slide attribution 為 Horowitz, ISSCC 2014。
- RAELLA、Titanium Law、CiMLoop 與 photonics discussion 來自 Lecture 11 slides 29-80。RAELLA 與 CiMLoop 的 local paper PDFs 不在本 worker 指定 inputs 中，因此本章沒有獨立驗證其 paper details。
- DaDianNao claims derived from `papers/L15_DaDianNao_Chen_MICRO2014.pdf`，尤其 Sections IV-V 與 evaluation discussion。
- Neurocube claims derived from `papers/L15_Neurocube_Kim_ISCA2016.pdf`，尤其 Sections II-VI。
- Tetris claims derived from `papers/L15_TETRIS_Gao_ASPLOS2017.pdf`，尤其 Sections 2-4 and 6。
- TPU baseline claims derived from `papers/L12_TPU_Jouppi_ISCA2017.pdf`，尤其 abstract、Sections 2 and 4。
- Worked examples 是根據 slide-stated concepts 建立的 original teaching examples。

## Uncertainty Notes

- Live lecture 可能對 RAELLA、CiMLoop 與 photonics 有投影片文字無法恢復的額外細節。
- Numerical technology ratios 依 process 與 assumptions 而變。本章將它們視為 slides 或 papers 中的 anchored evidence，而非 universal constants。
- Repository 既有 `assets/L11/` 可能包含 copyright-sensitive slide captures。本章不再 embed 它們，但 Worker C 未刪除 owned walkthrough files 以外的 assets。
