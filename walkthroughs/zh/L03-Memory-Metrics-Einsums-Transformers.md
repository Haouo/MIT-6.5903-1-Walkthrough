# L03 - 記憶體、評估指標、Einsum 與 Transformer（Memory, Metrics, Einsums, and Transformers）

> **課程：** 6.5930/1 - 深度學習硬體架構（Hardware Architectures for Deep Learning）
> **講師：** Joel Emer 與 Vivienne Sze（MIT EECS）
> **講授日期：** 2026 年 2 月 9 日。**投影片：** 127 頁。**來源：** [`Lecture/L03-Memory+Metrics+Einsums+Transformers.pdf`](../../Lecture/L03-Memory+Metrics+Einsums+Transformers.pdf)
>
> 本章依據公開投影片重建缺少的課堂講解。它不是投影片摘要，而是給無法觀看 lecture video 的讀者使用的自學章節。文中以投影片 PDF 頁碼作為來源錨點；公式與例子則重新撰寫，以利獨立學習。

---

## 一句話總結（TL;DR）

L03 把四個會貫穿後續課程的觀念接在一起。

第一，DNN 硬體常常受限於**資料搬移（data movement）**，而不是算術本身。投影片引用 Horowitz 的能耗表：一次 32-bit DRAM 讀取是 640 pJ，一次 32-bit 浮點乘法是 3.7 pJ。這個投影片陳述的比例約為 170 倍。因此硬體架構師的工作不只是增加 MAC units，而是安排記憶體階層與計算順序，讓 weights、activations、partial sums 盡量少搬動。

第二，加速器評估需要一組**評估指標（metrics）**，不能只看 GOPS/W 這種單一數字。Accuracy、throughput、latency、energy、power、cost、flexibility、scalability 回答的是不同問題。一個設計可能在某個指標看起來很好，卻在實際應用上失敗。

第三，本課程需要一種記法，能說清楚 DNN 計算**要算什麼**，但不先規定**怎麼排程**。這個記法就是 Einsum。像 $Z_{m,n} = A_{m,k} B_{n,k}$ 這樣的 Einsum 定義了 tensor contraction 與 iteration space，但沒有規定 loop order、tiling、data placement 或 parallelization。

第四，Transformer self-attention 不是另一種完全無關的 workload。它是一串 tensor contractions：input projection、$QK^T$、softmax、$AV$、output projection。硬體上最重要的警告是：標準 self-attention 會產生一個大小為 $M \times M$ 的 attention matrix，其中 $M$ 是 sequence length。

---

## 本講要解決的問題

前面講次介紹了 deep learning workload 與 accelerator 的動機。L03 補上比較與映射這些 workload 所需的共同語言。

問題在於，「一個 DNN layer」對硬體設計來說太模糊。硬體架構師需要知道：

- 有哪些 tensors？
- 哪些 indices 會保留在 output？
- 哪些 indices 會被 reduction？
- 有多少 bytes 在 memory hierarchy 之間移動？
- 正在最佳化哪個 metric？
- 報告中的結果是真實系統結果，還是 weight count、peak TOPS 這類 proxy？

L03 用三座橋來解決這件事：

| 橋接 | 從 | 到 | 為什麼重要 |
|---|---|---|---|
| Memory hierarchy | 物理記憶體技術 | DNN data movement cost | 解釋 local reuse 為什麼有價值 |
| Evaluation metrics | Model/application goals | Hardware comparison | 避免被單一數字誤導 |
| Einsum notation | Neural-network layers | Loop nests 與 mapping | 讓後續課程可以精確討論 dataflow |

本講最後把同一套 Einsum 語言用在 Transformer attention，因為現代 DNN accelerator 不可能只處理 CNN。

---

## 為什麼本講重要

初學者常有一個太簡化的心智模型：「神經網路主要是矩陣乘法，所以最好的 accelerator 就是乘法單元最多的 accelerator。」L03 修正這個模型。

乘法單元當然重要，但只有在系統能餵資料給它們時才有用。若每個 MAC 都從 DRAM 抓 operands，系統花在搬資料的能量會遠高於乘法本身。若 benchmark 只報 peak TOPS，卻不報 PE utilization、batch size、off-chip bandwidth 或 accuracy，那個結果可能不能描述一個可部署系統。

同樣的修正也出現在記法上。像 $O = AB$ 這樣的公式不夠。硬體設計需要知道 $A$ 與 $B$ 是 weights、activations 還是 intermediate tensors；reduced rank 能不能 tile；intermediate 是否 materialize；loop order 是否暴露 reuse。

因此 L03 是「語言與測量」的講次。L05 與 L06 會問如何把 Einsum map 到硬體。L07 之後會問 sparsity、precision 與 specialized architectures 如何改變同一套成本模型。

---

## 先備知識與心智模型

你需要帶著三個觀念閱讀本章。

第一，DNN layer 是 tensor computation。Tensor 是有命名維度的陣列，例如 channel、height、width、batch、sequence position 或 embedding dimension。

第二，DNN accelerator 有 memory hierarchy。一個有用的簡化圖像是：

```text
DRAM -> global buffer / SRAM -> PE-local RF or registers -> MAC units
```

資料離 MAC unit 越遠，通常需要越多 energy 與 latency。這是來自 memory slides 的硬體原理，不只是 software locality 口號。

第三，許多 DNN 計算都是 reductions。Dot product、matrix multiplication、convolution、attention score 都會計算 products，然後沿著至少一個 rank 加總。Einsum 讓課程能乾淨地命名這些 ranks。

本講的心智模型是：

```text
mathematical layer -> Einsum -> possible loop orders -> memory traffic -> metrics
```

Einsum 固定數學合約。後續 mapping 會選擇 loop order 與 data placement。Metrics 告訴我們產生的系統是否有用。

---

## 學習目標

讀完本章後，你應該能夠：

- 解釋為什麼大型記憶體通常比小型近端記憶體更慢、每次存取能耗更高。
- 使用投影片中的 Horowitz table 比較 arithmetic、SRAM access 與 DRAM access 的能耗。
- 定義 accuracy、throughput、latency、energy、power、hardware cost、flexibility、scalability 這些 accelerator metrics。
- 解釋為什麼 weights 與 operation counts 是 proxy metrics，而不是 energy 或 latency 的直接測量。
- 讀懂 Einsum，並指出 output ranks、input ranks 與 reduction ranks。
- 將 Einsum 的 Operational Definition 理解為對 iteration space 的遍歷。
- 透過 flattening ranks，把 fully connected layer 轉成 matrix-vector 或 matrix-matrix multiplication。
- 解釋 Toeplitz/im2col 如何把 convolution 轉成 matrix multiplication，以及為什麼這個轉換會重複資料。
- 追蹤 self-attention 從 $I$、$Q$、$K$、$V$ 到 $QK$、softmax、$A$、$AV$、$Z$ 的 cascade。
- 解釋 $M \times M$ attention matrix 對硬體的影響。
- 區分投影片直接陳述、paper/source-derived claims、standard background 與 teaching interpretation。

---

## 1. 記憶體是第一級設計限制

**來源錨點：** PDF 頁 2-19。能耗表在投影片中歸因於 Horowitz, ISSCC 2014。

### 直覺

記憶體不是被動容器，而是一個電路。記憶體越大，通常線越長、電容越大、周邊電路越多，每次存取能耗也越高。投影片用物理規則 $E = C V^2$ 表達這件事：在固定電壓下，電容越大，能量越高。

這就是 memory hierarchy 存在的原因。我們把小、快、低能耗的 storage 放在 compute 附近，把大容量但較昂貴的 storage 放在遠處。

### 精確意義

本講比較四種 storage technologies。

| Storage | 本講中的典型角色 | 優點 | 代價 |
|---|---|---|---|
| Latches / flip-flops | 很小的 local state、pipeline registers | 極低 latency，靠近 logic | 密度低，每 bit 需要很多 transistors |
| SRAM | Register files、buffers、on-chip memories | On-chip、可重用、比 DRAM 快 | 面積比 DRAM 大；peripheral circuits 很重要 |
| DRAM | Main memory，通常 off-chip | 容量大，每 bit 成本低 | 高 access energy 與 latency |
| Flash | Persistent storage | 非揮發、密度高 | 寫入對 compute 使用來說昂貴且慢 |

對 DNN 最重要的不是每個 accelerator 都必須有完全相同的階層，而是 capacity、latency、bandwidth、density、energy 彼此拉扯。Local buffer 只有在 computation 會在資料被 evict 前重用它時才有幫助。

### 量化錨點

投影片表格給出以下能耗：

| Operation | Energy |
|---|---:|
| 8-bit add | 0.03 pJ |
| 32-bit add | 0.1 pJ |
| 32-bit floating-point add | 0.9 pJ |
| 8-bit multiply | 0.2 pJ |
| 32-bit multiply | 3.1 pJ |
| 32-bit floating-point multiply | 3.7 pJ |
| 32-bit SRAM read from an 8 kB SRAM | 5 pJ |
| 32-bit DRAM read | 640 pJ |

兩個比例值得仔細看：

- 在這張表中，一次 32-bit DRAM read 是一次 32-bit SRAM read 的 $640 / 5 = 128$ 倍。
- 在這張表中，一次 32-bit DRAM read 是一次 32-bit floating-point multiply 的 $640 / 3.7 \approx 173$ 倍。

這些是特定技術脈絡下的投影片數值，不是普世常數。但它們足以支持架構方向：減少 large-memory traffic。

### Worked example：一個被重用的 word 為什麼重要

假設某個 weight 會被 16 次 MAC 使用。若系統每次 MAC 都從 DRAM 讀這個 weight，依 Horowitz table，這個 32-bit 值的 weight traffic 是 $16 \times 640 = 10{,}240$ pJ。

若系統只從 DRAM 讀一次，然後把它留在 local storage，DRAM 部分是 $640$ pJ。即使每次 local reuse 都用類似 SRAM 的 $5$ pJ，16 次 local reads 也只是 $80$ pJ，總計 $720$ pJ。

這個 toy example 不是完整 accelerator energy model。它故意忽略 address generation、interconnect、control 與 writes。它的目的只是展示 reuse opportunity 的量級。

### 硬體意義

Memory hierarchy 改變 mapping problem。若某個 loop order 讓 weight 在 PE-local register file 中被重用 16 次，它可能比另一個反覆 evict/refetch 同一 weight 的 loop order 便宜很多。Arithmetic 相同，traffic 不同。

### 常見誤解

**誤解：** Data movement 昂貴只是因為 DRAM 慢。

**修正：** Latency 很重要，但 energy 也同樣核心。Large memories 與 off-chip links 具有高 capacitance。投影片先把 energy cost 連到 capacitance 與 voltage，再用 Horowitz table 顯示 memory movement 可以主導 arithmetic。

---

## 2. Efficient models 不會自動變成 efficient hardware

**來源錨點：** PDF 頁 20-39。

本講在 metrics 之前放了一段 efficient-CNN interlude。這不是離題，而是在鋪陳一個警告：model-design papers 常用 number of weights 與 number of operations 當作「complexity」，但這些只是 indirect metrics。

### Model 端試圖減少什麼

投影片列出幾種 CNN designer 降低表面 complexity 的做法：

- 用堆疊的小 filters 取代一個大的 spatial filter，例如用兩個 $3 \times 3$ filters 取代一個 $5 \times 5$ filter。
- 用 $1 \times 1$ bottleneck convolutions 在昂貴 layer 前降低 channel count。
- 用 grouped 或 depthwise convolutions，讓每個 filter 只看一部分 channels。
- 跨 layers reuse feature maps，例如 DenseNet-style connectivity。
- 用 NAS 自動搜尋 architectures。

MobileNet 投影片給了一個緊湊公式。Standard convolution 的 work 正比於 $H W C R S M$。Depthwise-separable 版本有 depthwise work $H W C R S$ 加上 pointwise work $H W C M$，所以 work 正比於 $H W C (R S + M)$。因此 standard-to-depthwise-separable MAC ratio 是：

$$
\frac{H W C R S M}{H W C (R S + M)} = \frac{R S M}{R S + M}.
$$

這是真實的 algorithmic reduction，來源錨點是 MobileNets 投影片。但投影片接著警告：operations 變少不會自動代表 latency 或 energy 變低。

### 為什麼硬體答案更細

Operation count 至少忽略五件硬體事實：

- 減少後的 operation 是否還有足夠 parallelism 填滿 PE array。
- 較小 tensors 是否仍保有良好 reuse。
- grouped/depthwise layers 是否造成尷尬的 memory access patterns。
- 是否出現 metadata、layout conversion 或 kernel launch overhead。
- 硬體是為 dense GEMM-like work 設計，還是能處理許多小而不規則的 kernels。

這是根據 PDF 頁 36-37 的投影片警告與頁 40-56 的 metrics section 所做的教學詮釋。投影片不是說 efficient CNN techniques 不好，而是說 algorithmic complexity 必須透過真實 mapping 與 measurement 連到 hardware cost。

### 常見誤解

**誤解：** 如果一個 model 有 50 倍更少 parameters，它就應該用 50 倍更少 energy。

**修正：** Parameter count 估的是 weights 的 storage，但 energy 還取決於 activations、outputs、partial sums、reuse、memory level、data layout、utilization 與 batch size。後續 metrics 投影片用 AlexNet 與 SqueezeNet 作為警告：proxy metrics 可能誤導。

---

## 3. Evaluation Metrics：必須報告什麼

**來源錨點：** PDF 頁 40-56。

### 直覺

一個 metric 只回答一個問題，不可能回答所有問題。

例如，throughput 問「每秒處理多少 inputs？」Latency 問「單一 input 等多久？」Energy 問「每 joule 做多少工作？」Accuracy 問「結果是否有用？」一個系統可能贏其中一項，卻輸掉另一項。

投影片用 ring oscillator 例子把這點講得很清楚：一個快速切換但不做有用 DNN inference 的電路，也能製造出看似很高的 TOPS/W。Peak arithmetic 除以 peak power 不夠。

### 指標集合

| Metric | 它問什麼 | 必須附上的脈絡 |
|---|---|---|
| Accuracy | Model 是否解決任務？ | Dataset、task difficulty、training/evaluation procedure |
| Throughput | 每秒完成多少 operations 或 inferences？ | 實際 model、PE count、utilization、batch size |
| Latency | 從 input 到 output 多久？ | Batch size 與 end-to-end path |
| Energy and power | 每次 inference 消耗多少 energy？執行時 power 多高？ | Model、memory traffic、off-chip bandwidth、measurement/simulation method |
| Hardware cost | Implementation 多昂貴？ | Area、process node、on-chip storage、PE count、external interfaces |
| Flexibility | 多少 workloads 能有效率地跑？ | Models、shapes、precisions、sparsity、layer types 的範圍 |
| Scalability | 增加資源後會如何？ | Scaling variable 與 bottleneck，例如 PEs 或 memory bandwidth |

### Throughput、latency 與 batch size

Throughput 與 latency 相關，但不是同一件事。Batch size 64 可能提高 throughput，因為同一組 weights 可被更多 inputs reuse，但它也可能增加單一 input 的等待時間。因此投影片明確說 low latency 有額外限制：small batch size。

### Energy 與 off-chip memory

Metrics 投影片特別強調 off-chip memory access。若 paper 只報 chip power，它可能把 accelerator core 外部的 DRAM traffic energy 藏起來。一個有很多 multipliers 但 local storage 不足的設計，可能在 chip 上看似便宜，卻把成本推給 memory system。

### Worked example：operational intensity

Operational intensity 常讀成 $\text{ops}/\text{byte}$。Metrics references 提到 Roofline paper；本章把這個概念當作標準背景使用。

假設一個 tiny kernel 做 1024 次 MAC。這個 toy example 中把一次 MAC 算成一次 operation，且 kernel 從 DRAM 讀 2048 bytes，則 operational intensity 是：

$$
\text{OI} = \frac{1024\ \text{ops}}{2048\ \text{bytes}} = 0.5\ \text{ops/byte}.
$$

若另一個 loop order 透過 local reuse，讓同樣 1024 次 MAC 只需從 DRAM 讀 512 bytes，則：

$$
\text{OI} = \frac{1024}{512} = 2\ \text{ops/byte}.
$$

Arithmetic 沒變，memory traffic 變了。較高 operational intensity 通常表示每個 fetched byte 支撐更多 computation，這正是 local reuse 想達成的事。

### Source bridge：MLPerf、Accelergy 與 AccelForge

投影片指向三種 evaluation infrastructure。

| Source/tool | 解決的問題 | 與 L03 的關係 |
|---|---|---|
| MLPerf | 跨 models 與 platforms 的 standardized benchmarking | 用共同 workloads 與 divisions 降低 cherry-picking |
| Accelergy, Wu et al., ICCAD 2019 | Architecture-level energy estimation | 把 components 與 actions 連到 estimated energy |
| AccelForge | DNN mapping 與 performance simulation | 產生可餵給 energy estimator 的 action counts |

本章把它們當作 source bridges，而不是完整 paper summaries。重點是：公平的 accelerator evaluation 需要 workload shape、architecture description、mapping、action counts 與 energy modeling，而不是 peak arithmetic alone。

### 常見誤解

**誤解：** GOPS/W 就等於某個有用應用的 energy efficiency。

**修正：** GOPS/W 只有在 operations 是有用的、workload 被指明、utilization 被測量、accuracy 被保留、memory-system energy 被納入時才有意義。否則很容易最佳化一個 ratio，而不是最佳化 application。

---

## 4. Einsum：Tensor computation 的合約

**來源錨點：** PDF 頁 57-72。

### 直覺

Einsum 把兩個問題分開：

- 哪些值必須被 multiplied、added 或 otherwise combined？
- 機器應該用什麼順序執行這些 operations？

第一個問題是 Einsum。第二個問題是 mapping。

例如 $Z_{m,n} = A_{m,k} B_{n,k}$ 表示每個 output element $Z_{m,n}$ 都會沿著 $k$ 加總 products。它沒有說 $m$、$n$ 或 $k$ 哪個 loop 先跑，也沒有說 $m$ 與 $n$ 是否要 parallelize。

### Einsum 的 Operational Definition

投影片用 operational definition 定義 Einsum：

1. 遍歷 rank variables 的所有合法值。這個集合就是 **iteration space**。
2. 在每個點，用該點的 rank-variable values 計算 right-hand side。
3. 將結果 assign 到 left-hand-side tensor。
4. 如果 target location 已經有值，就 reduce 進去，通常是加總。

以 $Z_{m,n} = A_{m,k} B_{n,k}$ 來說，合法點是 triples $(m,n,k)$。Output location 由 $(m,n)$ 識別。因為 $k$ 只出現在 right-hand side，多個 iteration points 會貢獻到同一個 $Z_{m,n}$，所以 $k$ 是 reduction rank。

### 精確詞彙

| Term | Meaning |
|---|---|
| Rank variable | Einsum 中的 index，例如 $m$、$n$、$k$ |
| Rank name | 維度標籤，例如 $M$、$N$、$K$ |
| Rank shape | 維度大小，例如 $M=64$ |
| Uncontracted rank | 出現在左右兩側的 rank；保留在 output 中 |
| Contracted rank | 出現在 right-hand side 但不在 left-hand side；會被 reduction |
| Iteration space | 所有 rank-variable ranges 的 Cartesian product |

### 常見模式

| Einsum | 讀法 | Reduced rank |
|---|---|---|
| $Z_{m,n} = A_{m,k} B_{k,n}$ | Matrix-matrix multiply | $k$ |
| $Z_m = A_{k,m} B_k$ | Matrix-vector multiply | $k$ |
| $Z_{m,n} = A_m B_n$ | Cartesian product | 無 |
| $Z_m = A_m B_m$ | Element-wise multiply | 無 |
| $Z_m = A_m + B_m$ | Element-wise addition | 無 |

Variables 的名字本身不特殊。$Z_{p,q} = A_{p,r} B_{q,r}$ 仍是 matrix-matrix-style contraction，因為 $r$ 被 reduce，$p,q$ 被保留。

### Partitioning 與 flattening

投影片介紹 rank split：若 $i = i_1 I_0 + i_0$，一個原始 rank $i$ 可以用兩個 ranks $(i_1,i_0)$ 表示。這就是 partitioning。

Flattening 是反向操作。一對 $(i_1,i_0)$ 可以被視為一個 flattened coordinate $i$。這很重要，因為後續講次會把 tiling 描述成 rank partitioning，而不是 ad hoc code trick。

### Worked example：讀一個 Einsum

考慮：

$$
Y_{b,m} = X_{b,k} W_{m,k}.
$$

Rank variables 是 $b$、$m$、$k$。Output ranks 是 $b$ 與 $m$。Rank $k$ 只出現在 right-hand side，所以它被 reduce。若 $B=2$、$M=3$、$K=4$，iteration space 有 $2 \times 3 \times 4 = 24$ 個點，output 有 $2 \times 3 = 6$ 個 elements。每個 output element 累加 4 個 products。

### 硬體意義

Einsum 讓硬體架構師可以用 workload-independent 的方式討論 data reuse。若 $k$ 是 reduced，left-hand-side tensor 的 partial sums 必須被累加。若 $m$ 或 $n$ 是 uncontracted，這些 ranks 可能暴露 parallel output elements。若某個 rank 被 partition，tile size 可以選到符合 buffer capacity。

### 常見誤解

**誤解：** Einsum 只是矩陣乘法的短記法。

**修正：** Matrix multiplication 只是其中一個 Einsum pattern。這個記法更廣：它描述 tensor contractions、element-wise operations、Cartesian products、convolution index relationships 與 attention projections，而且不預先承諾 loop order。

---

## 5. Fully connected 與 convolution layers 作為 matrix multiplication

**來源錨點：** PDF 頁 73-109。

### Fully connected layer

Fully connected layer 可以看成 filter 覆蓋整個 input spatial extent 的 convolution。投影片方程式是：

$$
O_m = I_{c,h,w} F_{m,c,h,w}.
$$

重複的 ranks $c,h,w$ 會被 reduce。若要把它轉成 matrix-vector multiply，可將 $(c,h,w)$ flatten 成單一 rank $chw$：

$$
O_m = I_{chw} F_{m,chw}.
$$

加入 batch size $N$ 後，input 多了 batch rank $n$：

$$
O_{n,m} = I_{n,chw} F_{m,chw}.
$$

這就是 Einsum 形式的 matrix-matrix multiplication。Output 有 ranks $(n,m)$，而 $chw$ 是 reduction rank。

### 為什麼 flattening 不只是記法

Flattening 改變我們看待 memory layout 的方式。若 tensor 的儲存方式讓連續的 $chw$ elements 彼此 contiguous，matrix-vector view 可以很有效率。若 layout 不合，硬體可能需要 strided accesses 或 layout conversion。數學 contraction 相同，但 memory behavior 可能不同。

### Convolution 與 Toeplitz/im2col

對 1-D convolution：

$$
O_q = I_{q+s} F_s.
$$

Input index 不是單純的 $q$ 或 $s$，而是 $q+s$。這種 coupling 是 convolution 與普通 matrix multiplication 的差異。

投影片把轉換拆成兩步：

$$
T_{q,s} = I_{q+s}
$$

然後：

$$
O_q = T_{q,s} F_s.
$$

第一步建立由 shifted input windows 組成的 Toeplitz/im2col matrix。第二步是 matrix multiplication。

對有 batch 的 2-D convolution，投影片陳述的 matrix dimensions 是：

$$
\text{Filters }[M \times C R S] \times \text{Input-Toeplitz }[C R S \times P Q N] = \text{Output }[M \times P Q N],
$$

其中在投影片顯示的 no-padding、unit-stride 情況下，$P = H - R + 1$ 且 $Q = W - S + 1$。

### Worked example：tiny 1-D Toeplitz conversion

令 input 為 $I=[1,2,3,4,5]$，filter 為 $F=[a,b,c]$。Valid 1-D convolution 有三個 output positions：

$$
O_0 = 1a + 2b + 3c,
$$

$$
O_1 = 2a + 3b + 4c,
$$

$$
O_2 = 3a + 4b + 5c.
$$

Toeplitz matrix 是：

```text
T = [1 2 3
     2 3 4
     3 4 5]
```

於是 $O = T F$。重要的硬體觀察是 input values 會在 $T$ 中重複。值 3 出現在三個 windows。Materializing $T$ 會讓 operation 看起來像 GEMM，但若 implementation 真的寫出所有重複 entries，memory traffic 可能增加。

### 硬體意義

這個轉換解釋了為什麼 GEMM engines 可以跑 FC 與 CONV layers。它也揭示一個取捨：im2col 可以產生規則的 matrix multiplication，但規則性可能來自 data duplication。後續 mapping lectures 會問 accelerator 能否不完整 materialize Toeplitz matrix，也能 exploit convolutional reuse。

### 常見誤解

**誤解：** 既然 convolution 可以轉成 matrix multiplication，convolution hardware 只需要 generic GEMM block。

**修正：** GEMM 是強大的 abstraction，但轉換可能複製 input data 並改變 memory traffic。Convolution-aware mapping 可以直接利用 overlap；naive im2col implementation 則可能花能量搬重複資料。

---

## 6. Transformer self-attention 作為 Einsums

**來源錨點：** PDF 頁 110-127。

### 心智模型

Self-attention 接收長度為 $M$ 的 token sequence。每個 token 一開始是維度 $D$ 的 embedding vector。這個 block 計算三種 projections：

- Query $Q$：這個位置想找什麼。
- Key $K$：每個位置提供什麼可被匹配的特徵。
- Value $V$：若被 attend 到，每個位置貢獻什麼資訊。

Query position 與 key position 的 attention score 來自 dot product。Softmax 把 scores 轉成 weights。Value vectors 依這些 weights 混合。最後 output projection 把結果轉回 model embedding space。

### Rank dictionary

| Rank | Meaning |
|---|---|
| $M$ | Keys、values 與 self-attention positions 的 sequence length |
| $P$ | 投影片用於 query/output position 的 sequence-length alias |
| $R$ | Non-self-attention 中的 query sequence length |
| $C$ | Vocabulary size |
| $D$ | Input/global embedding dimension，常稱 $d_{\text{model}}$ |
| $E$ | Query/key projection dimension，常稱 $d_k$ |
| $F$ | Value projection dimension，常稱 $d_v$ |
| $G$ | Output embedding dimension |
| $B$ | Batch size |
| $H$ | Number of attention heads |

投影片用 $M$ 與 $P$ 作為 sequence length 的 aliases，讓 attention matrix 的兩個 axes 可以分開命名。

### Single-head attention cascade

只有第一層需要把 raw one-hot input $IR$ embedding 起來：

$$
I_{m,d} = IR_{m,c} W^I_{c,d}.
$$

接著 input 被投影到 key、query、value spaces：

$$
K_{m,e} = I_{m,d} W^K_{d,e},
$$

$$
Q_{m,e} = I_{m,d} W^Q_{d,e},
$$

$$
V_{m,f} = I_{m,d} W^V_{d,f}.
$$

Softmax 前的 score matrix 是：

$$
QK_{m,p} = Q_{p,e} K_{m,e}.
$$

這裡 $e$ 被 reduce。對每個 query position $p$，score 會把該 query 與每個 key position $m$ 比較。投影片註明某些 constant scaling steps 沒畫出來，因此本章不把 scaling 當作投影片陳述的方程式處理。

投影片 convention 下的 softmax components 是：

$$
SN_{m,p} = \exp(QK_{m,p}),
$$

$$
SD_p = \sum_m SN_{m,p},
$$

$$
A_{m,p} = SN_{m,p} / SD_p.
$$

接著 values 被混合：

$$
AV_{p,f} = A_{m,p} V_{m,f}.
$$

最後 output projection 是：

$$
Z_{p,g} = AV_{p,f} W^Z_{f,g}.
$$

### Worked example：attention shapes

假設 single head、無 batch，$M=4$、$D=8$、$E=2$、$F=3$。

| Tensor | Shape | 原因 |
|---|---|---|
| $I$ | $4 \times 8$ | 四個 tokens，每個是八維 embedding |
| $W^Q$ 與 $W^K$ | $8 \times 2$ | 從 $D$ 投影到 $E$ |
| $Q$ 與 $K$ | $4 \times 2$ | 每個 token 一個 2-D query/key vector |
| $QK$ | $4 \times 4$ | 每個 query position 都與每個 key position 比較 |
| $W^V$ | $8 \times 3$ | 從 $D$ 投影到 $F$ |
| $V$ | $4 \times 3$ | 每個 token 一個 value vector |
| $AV$ | $4 \times 3$ | 每個 query position 一個 mixed value vector |

Quadratic tensor 是 $QK$ 或 $A$：此例有 $M^2 = 16$ 個 entries。若 $M$ 加倍，這個 intermediate 約成長 4 倍，還沒算 batch 或 heads。

### Batched 與 multi-headed attention

Batching 加入 rank $b$：

$$
QK_{b,m,p} = Q_{b,p,e} K_{b,m,e}.
$$

Multi-headed attention 加入 head rank $h$：

$$
QK_{b,h,m,p} = Q_{b,h,p,e} K_{b,h,m,e}.
$$

每個 head 產生 $AV_{b,h,p,f}$。接著 head 與 value dimensions 會 concat 或 flatten，再做 output projection。依投影片 convention：

$$
C_{b,p,hF+f} = AV_{b,h,p,f},
$$

再接 output projection，例如：

$$
Z_{b,p,d} = C_{b,p,g} W^Z_{g,d}.
$$

### 硬體意義

Attention block 裡有兩種不同成本型態。Projection layers 很像 GEMM，成本大致隨 $M D E$、$M D F$、$M F G$ 成長。Attention score 與 value-mixing steps 則會產生與消耗形狀受 $M^2$ 控制的 tensors。這表示即使 embedding dimensions 中等，sequence length 也可能主導 memory capacity 與 bandwidth。

硬體問題因此不只是「accelerator 能不能做 matrix multiplication？」還包括「它是否 materialize $M \times M$ attention matrix？這個 matrix 存在哪一層？softmax 能否與前後 operations fusion？」

### 常見誤解

**誤解：** Transformer attention 與 CNN 所用的 tensor computations 完全不同。

**修正：** Data dependencies 不同，但本講把 attention 表達成與 FC/CONV 相同的 Einsum cascade。新的挑戰是 intermediates 的 shape 與 lifetime，尤其是 $M \times M$ attention matrix 與 softmax normalization。

---

## 7. 關鍵方程式與閱讀方式

### Energy ratio

$$
\frac{E_{\text{DRAM read, 32b}}}{E_{\text{FP multiply, 32b}}} = \frac{640}{3.7} \approx 173.
$$

這要讀成「減少 DRAM access 的來源錨定動機」，不是 universal technology constant。

### Operational intensity

$$
\text{Operational intensity} = \frac{\text{operations}}{\text{bytes moved from the measured memory level}}.
$$

Denominator 必須被指明。DRAM operational intensity 與 SRAM operational intensity 是不同測量。

### Matrix multiplication Einsum

$$
Z_{m,n} = A_{m,k} B_{k,n}.
$$

Ranks $m$ 與 $n$ 保留在 output。Rank $k$ 被 reduce。Mapping 可以用許多合法 loop orders 執行它。

### Fully connected layer

$$
O_m = I_{c,h,w} F_{m,c,h,w}
$$

變成：

$$
O_m = I_{chw} F_{m,chw}.
$$

Flattening 把 input dimensions 變成一個 reduction rank。

### Convolution

$$
O_{n,m,p,q} = I_{n,c,U p + r,U q + s} F_{m,c,r,s}.
$$

Ranks $c,r,s$ 被 reduce。Output ranks $n,m,p,q$ 保留。Input spatial coordinates 由 output location 與 filter offset 推導而來。

### Attention score

$$
QK_{m,p} = Q_{p,e} K_{m,e}.
$$

Rank $e$ 被 reduce。結果比較每個 query position $p$ 與每個 key position $m$。

---

## 8. 硬體意義

- **Energy：** DRAM traffic 可能主導 arithmetic energy；local reuse 是第一級設計目標。
- **Bandwidth：** PE array 要達到 peak throughput，memory system 必須夠快供應 operands。
- **Latency：** Batching 可能改善 throughput 但傷害 latency；low-latency applications 需要 small-batch evaluation。
- **Area：** 更多 local SRAM/RF 可降低 traffic，但會消耗 area；若 array 太大，access energy 也可能提高。
- **Utilization：** Depthwise convolution 這類 model shapes 可能減少 MACs，但也可能減少固定 array 可利用的 dense parallel work。
- **Interconnect：** Tensor contractions 建立 producer-consumer relationships；intermediates 存在哪一層會決定 network traffic。
- **Correctness：** Einsum reductions 需要正確累加。改變 loop order 只有在 reduction semantics 被保留時才合法。
- **Programmability：** Flexible hardware 必須處理 CNNs、FC layers、attention、varying precision、sparsity 與 layer shapes。

---

## 9. 常見誤解

### 誤解：Data movement 是 compute 之後的次要細節。

在 accelerator design 中，compute 與 movement 不能分開。投影片的能耗表顯示，一次 DRAM read 可能比一次 multiply 貴好幾個數量級。

### 誤解：MACs 更少一定代表 latency 更低。

MACs 更少只有在 hardware bottleneck 是 arithmetic，而且剩餘工作能良好 map 到硬體時才會有比例效果。若 bottleneck 是 memory bandwidth、layout conversion、synchronization 或 poor utilization，latency 不一定等比例下降。

### 誤解：Accuracy 是 algorithm metric，不是 hardware metric。

Hardware choices 會限制 precision、sparsity support、model size 與 memory capacity。若這些選擇改變 model 或 numerical behavior，accuracy 就必須納入 evaluation。

### 誤解：Einsum 告訴 accelerator 該如何跑 loops。

Einsum 告訴我們要算什麼。Mapping 才選擇 loop order、tiling、storage placement 與 parallelism。

### 誤解：im2col 總是有效率，因為它把 convolution 變成 GEMM。

im2col 暴露規則的 GEMM structure，但可能複製 input data。有效率的 implementations 常避免 materialize 完整 expanded matrix。

### 誤解：Attention 只需要 matrix multiplication acceleration。

Matrix multiplication 很核心，但 $M \times M$ score/attention matrix、softmax、memory capacity 與 fusion opportunities 同樣重要。

---

## 10. 重點回顧（Takeaways）

- 投影片的 Horowitz table 給出 32-bit DRAM read = 640 pJ、32-bit floating-point multiply = 3.7 pJ，這是 data-movement-aware design 的主要動機。
- Memory hierarchy 存在，是因為沒有一種 memory 可以同時大、快、高密度、便宜、低能耗。
- Efficient model techniques 降低 weights 與 OPs 這些 proxy costs，但硬體效率取決於 mapping、locality、utilization 與 memory traffic。
- 公平的 accelerator evaluation 需要 accuracy、throughput、latency、energy、power、cost、flexibility、scalability，且每個 metric 必須附足夠脈絡。
- Einsum 是本課程對 tensor computations 的 mathematical contract。它固定 output、input 與 reduction ranks，但不固定 loop order。
- FC layers 透過 flattening ranks 與加入 batch 變成 matrix-vector 或 matrix-matrix multiplication。
- Convolution 透過 Toeplitz/im2col conversion 變成 matrix multiplication，但這個轉換可能重複資料。
- Transformer self-attention 是一串 Einsum cascade，且含有 $M \times M$ attention intermediate。

---

## 11. 與前後講次的連結（Connections）

- **L01：** L01 的 accelerator motivation 在這裡透過 memory energy 與 evaluation metrics 變成量化論點。
- **L02：** CNN layer shapes 與 efficient model designs 在 L03 重新出現，用來說明 proxy metrics 可能誤導。
- **L04：** Einsum formalism 會成為描述更多 DNN operations 與 tensor transformations 的基礎。
- **L05-L06：** Mapping 以 Einsum 作為 input，選擇 loop order、partitioning、data placement 與 parallelism。
- **Sparsity lectures：** Sparsity 改變的是同一個 Einsum framework 內部的 tensors，但會增加 metadata、irregularity 與 load-balancing issues。
- **Precision lectures：** Bit width 會改變 arithmetic energy 與 storage traffic，但必須連同 accuracy 與 system metrics 評估。
- **Attention accelerators：** 本章 attention equations 解釋為什麼後續系統會嘗試 fusion 或避免 materialize $QK$ 與 $A$。

---

## 12. 來源橋接（Source Bridge）

### Paper Bridge: Computing's Energy Problem

**Bibliographic identity：** Mark Horowitz，*Computing's Energy Problem (and what we can do about it)*，ISSCC 2014。Local PDF：`papers/L07_ComputingsEnergyProblem_Horowitz_ISSCC2014.pdf`。

**Problem addressed：** Technology scaling 已無法自動提供足夠 energy improvement。本文問的是：當 power/energy 成為一級限制時，computing systems 要如何繼續改善？

**Core idea：** 高 energy efficiency 需要 low-energy operations 與 **extreme locality**。Paper 強調 memory-system energy 可能壓過 efficient computation，而 specialization 在能減少 movement 與 overhead 時有價值。

**Relevance to L03：** 這是本講 memory-vs-compute argument 的 paper-level 支撐。它說明為什麼 L03 把 memory hierarchy、action counts 與 operational intensity 當作架構概念，而不是帳務細節。

**Key claims used in this chapter：**

- DRAM accesses 比 internal cache accesses 或 simple functional operations 貴數個數量級。來源：Section 5, "Don't Forget the Memory Energy"。
- Figure 1.1.9 給出 operations 與 memory accesses 的 rough energy costs；lecture energy table 是這個概念的投影片呈現。
- 高 energy efficiency 需要 data locality，讓一次昂貴 memory access 支援多次 operations。來源：Sections 5-6。

**What students should remember：** Memory hierarchy 不是次要細節；它正是 DNN accelerators 重視 reuse、tiling、dataflow 的原因。

**Limitations and assumptions：** 數值 energy values 具 technology specificity。它們應作為 order-of-magnitude motivation，而不是 universal constants。

### Paper Bridge: AlexNet

**Bibliographic identity：** Alex Krizhevsky、Ilya Sutskever、Geoffrey Hinton，*ImageNet Classification with Deep Convolutional Neural Networks*，NeurIPS 2012。Local PDF：`papers/L03_AlexNet_Krizhevsky_NeurIPS2012.pdf`。

**Problem addressed：** 在 ImageNet-scale data 上訓練大型 supervised CNN；其 model capacity 與 computation 都超過早期小型 CNN examples。

**Core idea：** 結合大型 convolutional/fully connected layers、ReLU activations、GPU implementation、data augmentation 與 dropout，使 deep CNN 能在 ImageNet scale 上訓練。

**Relevance to L03：** AlexNet 把「DNN 是 tensor program」連到「DNN 是 hardware workload」。它展示 convolution、FC layers、memory capacity、GPU parallelism 與 regularization 如何同時影響設計。

**Key claims used in this chapter：**

- Model 有八個 learned layers 與約 60M parameters。來源：Abstract 與 Section 3.5。
- Paper 明確討論 GPU memory limits 與 two GTX 580 GPUs training。來源：Introduction 與 Section 3.2。
- ReLU 用來比 saturating nonlinearities 更快訓練。來源：Section 3.1。
- Dropout 用於降低 FC layers overfitting。來源：Section 4。

**What students should remember：** AlexNet 不只是「一個 CNN」。它是 dataset scale、model size、GPU memory 與 implementation choices 一起塑造 architecture 的早期例子。

**Limitations and assumptions：** AlexNet 架構具有歷史重要性，但後來模型改變了 accuracy/efficiency tradeoff。本章把它當作 workload history 與 hardware motivation，不當作現代建議設計。

### Paper Bridge: Deep Residual Learning

**Bibliographic identity：** Kaiming He、Xiangyu Zhang、Shaoqing Ren、Jian Sun，*Deep Residual Learning for Image Recognition*，CVPR 2016。Local PDF：`papers/L03_ResNet_He_CVPR2016.pdf`。

**Problem addressed：** 單純讓 plain networks 更深可能讓 optimization 變差；paper 稱之為 **degradation problem**，且不只是 overfitting。

**Core idea：** 學 residual function $F(x)=H(x)-x$，再透過 shortcut 把 input 加回來，使 block 計算 $F(x)+x$。Identity shortcuts 讓 very deep networks 學 identity mapping 附近的 perturbations。

**Relevance to L03：** ResNet 支撐本章警告：model complexity 不能只化約成 layer count 或 MAC count。Graph structure 會引入 elementwise additions 與 skip paths，進而影響 memory traffic、buffering 與 fusion。

**Key claims used in this chapter：**

- Degradation problem 於 Introduction 引入。來源：Section 1。
- Residual formulation $F(x)+x$ 與 identity shortcut 在 residual-learning section 定義。來源：Section 3。
- Paper 報告 18/34-layer comparisons 與 deeper bottleneck ResNets。來源：Tables 1-4。

**What students should remember：** Skip connections 數學上簡單，但硬體上可見：tensor 必須被保存或重新載入以供後續 addition 使用。

**Limitations and assumptions：** L03 不把 ResNet accuracy numbers 當作 current benchmark claims；它使用 paper 來解釋現代 layer graph 為何包含 non-chain data dependencies。

### Paper Bridge: MobileNets

**Bibliographic identity：** Andrew Howard et al.，*MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*，2017。Local PDF：`papers/L03_MobileNet_Howard_2017.pdf`。

**Problem addressed：** Mobile 與 embedded systems 需要 computation 與 model size 較低、但仍保有實用 accuracy 的 CNNs。

**Core idea：** 用 **depthwise separable convolution** 取代 standard convolution：每個 input channel 各自做 depthwise spatial filter，接著用 $1 \times 1$ pointwise convolution 混合 channels。Paper 也引入 width multiplier 與 resolution multiplier。

**Relevance to L03：** MobileNet 是本章 efficient-CNN warning 的 paper 支撐：減少 MACs 很有用，但 hardware 還要看 work 被移到哪裡。在 MobileNet 中，許多 computation 轉移到 dense $1 \times 1$ convolution，具有不同 reuse 與 bandwidth behavior。

**Key claims used in this chapter：**

- Standard convolution cost 在 paper notation 中是 $D_K^2 M N D_F^2$。來源：Section 3.1。
- Depthwise separable convolution 分離 filtering 與 channel mixing，降低 computation 與 model size。來源：Section 3.1。
- 對 $3 \times 3$ filters，paper 陳述約 8-9x computation reduction，accuracy 只小幅下降。來源：Section 3.1。
- Table 2 指出 MobileNet 多數 computation 在 $1 \times 1$ convolution。來源：Table 2。

**What students should remember：** Lower-MAC model 可能只是移動 bottleneck。Depthwise layers 算術上便宜，但 pointwise layers 可能主導 compute 與 memory traffic。

**Limitations and assumptions：** Paper 的 accuracy/latency tradeoffs 取決於 training setup 與 hardware context。本章使用 operator decomposition，而不是精確 benchmark ranking，作為長期概念。

### Paper Bridge: EfficientNet

**Bibliographic identity：** Mingxing Tan 與 Quoc Le，*EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*，ICML 2019。Local PDF：`papers/L03_EfficientNet_Tan_ICML2019.pdf`。

**Problem addressed：** 只沿 depth、width 或 input resolution 單一方向 scaling CNN，會有 diminishing returns 且需要手動調參。

**Core idea：** Compound scaling 用係數 $\phi$ 共同 scaling depth、width、resolution：$d=\alpha^\phi$、$w=\beta^\phi$、$r=\gamma^\phi$，並受 resource constraint 限制。

**Relevance to L03：** EfficientNet 強化本章訊息：model-level efficiency 是多維的。Depth、width、resolution 會以不同方式改變 arithmetic、activation sizes、memory footprint 與 hardware utilization。

**Key claims used in this chapter：**

- Single-dimension scaling 會提高 accuracy，但模型變大後收益趨於飽和。來源：Section 3.2 與 Figure 3。
- Compound scaling 平衡 depth、width、resolution。來源：Section 3.3 與 Equation 3。
- Paper 用 accuracy、parameters、FLOPs、latency 比較 scaled EfficientNets 與其他 ConvNets。來源：Tables 2 與 4。

**What students should remember：** "Efficient" 不是單一 scalar。模型可以在 depth、channel width、resolution、parameter count、activation size 與 latency 之間做不同 tradeoff。

**Limitations and assumptions：** EfficientNet 的結果依賴 architecture search、training recipe 與 evaluation hardware。L03 使用它解釋 scaling dimensions 與 proxy metrics，不做 current leaderboard claim。

### Source Bridge: Evaluation Tools and Benchmarks

**Bibliographic identity：** Slides 引用 MLPerf、Accelergy、Timeloop、AccelForge 作為 benchmark 與 modeling infrastructure 例子。

**Relevance to L03：** 這些來源支撐 realistic accelerator evaluation 必須指定 workload、mapping、precision、batch size、memory hierarchy 與 action counts。

**Key claims used in this chapter：** Required metric specifications 清單取自 L03 PDF page 48；attention equations 與 rank names 取自 PDF pages 115-127。

**Limitations：** 此 bridge 是 slide-anchored。對應 original papers 並非全部存在於本 repository 的 local PDFs。

---

## 13. 獨立學習指南（Standalone Study Guide）

### 必須掌握什麼

- 能解釋 memory hierarchy，而不是只說「data movement is expensive」。
- 記住 qualitative ordering：local register-like storage 小而便宜；DRAM 大但 access 昂貴。
- 練習讀 Einsum，標出 output ranks 與 reduction ranks。
- 練習把 FC 與 convolution 轉成 matrix multiplication，同時指出哪些資料被重複。
- 追蹤 self-attention shapes，從 $I$ 到 $QK$ 與 $AV$。

### 自我檢核問題

1. 為什麼 small local buffer 只有在有 reuse 時才會降低 energy？
2. 為什麼 latency claim 必須附 batch size？
3. 在 $Z_{m,n} = A_{m,k}B_{k,n}$ 中，哪個 rank 被 reduce？哪些 ranks 保留？
4. 為什麼 im2col 即使 enable GEMM，也可能增加 memory traffic？
5. 在投影片 convention 中，$QK_{m,p}$ 儲存什麼？
6. 為什麼 depthwise convolution 即使 MACs 較少，也可能讓 dense accelerator 難以有效利用？
7. 哪個 metric 可以揭露「低 chip power 但大量 off-chip bandwidth」的設計？

### 練習

1. **Conceptual：** 解釋為什麼 ring oscillator 可以產生誤導性的 TOPS/W-style number。
2. **Small calculation：** 用 Horowitz table 計算 32-bit SRAM read 與 8-bit multiply 的 energy ratio。
3. **Einsum reading：** 對 $Y_{b,p,f} = A_{b,m,p} V_{b,m,f}$，指出所有 output ranks 與 reduction ranks。
4. **Convolution：** 對 input $[2,4,6,8]$ 與 length-2 filter 建立 Toeplitz matrix。
5. **Attention shapes：** 若 $B=2$、$H=8$、$M=128$、$E=64$，$QK$ 的 shape 是什麼？
6. **Design tradeoff：** 用 energy、bandwidth、programmability 比較 im2col-based convolution implementation 與 direct convolution implementation。
7. **Paper/source bridge：** 閱讀 Accelergy 投影片，解釋為什麼估 energy 前需要 action counts。

---

## 14. 關鍵詞彙（Key Terms）

### Memory hierarchy（記憶體階層）

多層 storage system，把小、快、低能耗 memories 放在 compute 附近，把大型 memories 放在遠處。在 accelerator design 中，它存在的理由是 exploit reuse 並減少 expensive traffic。

### Data movement（資料搬移）

Weights、activations、partial sums 或 intermediates 在 memory levels 或 PEs 之間的移動。它影響 energy、bandwidth、latency 與 utilization。

### Operational intensity（操作強度）

在指定 memory boundary 上的 $\text{operations}/\text{bytes moved}$。它衡量每個 fetched byte 支撐多少 computation。

### Throughput（吞吐量）

單位時間完成的 work，例如 inferences/s 或 operations/s。必須在真實 workload 與 utilization 脈絡下報告。

### Latency（延遲）

從 input 到 output 的 elapsed time。Batch size 是必要脈絡，因為 large batch 可改善 throughput，但可能增加單一 input 等待時間。

### PE utilization（PE 利用率）

Processing elements 實際做 useful work 的比例。Peak TOPS 假設 high utilization；真實 workloads 未必達到。

### Einsum

命名 output、input 與 reduction ranks 的 tensor expression，不指定 traversal order。它是後續 mapping lectures 使用的 mathematical contract。

### Rank variable

Einsum 中的 index variable，例如 $m$、$n$、$k$。

### Contracted rank（收縮維度）

出現在 right-hand side 但不在 left-hand side 的 rank。它會被 reduction，通常是 summation。

### Uncontracted rank（非收縮維度）

出現在 output 中並被保留的 rank。

### Partitioning（切分）

把一個 rank 拆成多個 ranks，例如 $i = i_1 I_0 + i_0$。後續講次用這個觀念描述 tiling。

### Flattening（展平）

把多個 ranks 折成一個 rank，例如把 $(c,h,w)$ 折成 $chw$。

### Toeplitz / im2col

一種把 convolution 表示為 matrix multiplication 的轉換，方法是把 shifted input windows 收集成 matrix。它可暴露 GEMM structure，但可能複製資料。

### Query, Key, Value（查詢、鍵、值）

Attention 中的三種 projections。Queries 表示要找什麼，keys 用來匹配，values 提供被 attention weights 混合的資訊。

### Attention matrix（注意力矩陣）

投影片 convention 中 softmax-normalized score tensor $A_{m,p}$。它隨 sequence length 以平方成長。

---

## 15. 附錄 - 投影片對照表（Slide-to-Section Map）

| PDF 頁 | PDF 中的 slide label | 章節 | 備註 |
|---|---|---|---|
| 1 | L02-1 | Title | PDF 內部把早期頁標成 L02，但 repository 視為 L03 |
| 2-19 | L02-2 to L02-19 | 記憶體是第一級設計限制 | 加入 reuse example 與 hardware implications |
| 20-39 | L02-20 to L02-39 | Efficient models 不會自動變成 efficient hardware | 當作 metrics bridge，不作完整 CNN-model survey |
| 40-56 | L02-40 to L02-56 | Evaluation metrics | 加入 operational-intensity example 與 tool bridge |
| 57-72 | Extended Einsums 2/37 to 18/37 | Einsum contract | 加入 vocabulary 與 worked reading example |
| 73-90 | Kernel Computation 1 to 18 | Fully connected as matrix multiplication | 解釋 flattening 與 batch |
| 91-109 | Kernel Computation 19 to 37 | Convolution as matrix multiplication | 解釋 Toeplitz/im2col 與 data repetition |
| 110-127 | Attention Einsums 20/37 to 37/37 | Transformer self-attention as Einsums | 加入 shape example 與 hardware implications |

---

## 16. 來源註記（Source Notes）

- 投影片直接陳述的 claims 包括 Horowitz energy table values、memory hierarchy tradeoffs、key metric list、metric specification checklist、MobileNets MAC ratio、FC/CONV conversion dimensions，以及 attention rank/equation cascade。
- Paper/source-derived claims 限於投影片引用的來源：Horowitz ISSCC 2014、MobileNets 2017、MLPerf、Accelergy 與相關 evaluation tools。
- Standard background explanations 包括 operational intensity、matrix multiplication reading、batch-vs-latency interpretation，以及 query/key/value roles 的說明。
- Teaching interpretations 包括 small energy-reuse example、operational-intensity toy calculation 與 attention-shape example。
- 本次重寫沒有嵌入 slide figures 或 paper figures。Repository 中既有的 `assets/L03/*.png` 仍保留，但本章不使用。

---

## 17. 不確定性註記（Uncertainty Notes）

- Public PDF 的許多早期頁面標成 L02，但 repository 把它作為 L03。本章引用 PDF 頁碼以避免混淆。
- Live lecture 可能對 efficient-CNN examples 有不同強調。本章主要把 PDF 頁 20-39 用來支持 proxy-metric warning。
- Horowitz energy values 依技術而變；應作為 order-of-magnitude guidance，而非 universal constants。
- Attention softmax axis 依投影片 convention：$p$ 是 query/output position，normalization 沿 $m$ 加總。
