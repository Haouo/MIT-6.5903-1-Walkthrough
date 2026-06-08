# L01 - 導論與應用（Introduction and Applications）

> **課程：** 6.5930/1 - 深度學習硬體架構（Hardware Architectures for Deep Learning）
> **講師：** Joel Emer 與 Vivienne Sze，MIT EECS
> **講授日期：** 2026 年 2 月 2 日
> **投影片：** `Lecture/L01-Intro_and_Applications.pdf`，53 頁
>
> 本章依據 Lecture 1 投影片重建講課時會補上的教學脈絡。它是自學用的 textbook-style chapter，不是逐頁投影片摘要。章中以投影片頁碼作為來源錨點；為了降低版權風險，本文刻意不嵌入投影片截圖。

---

## TL;DR

深度學習之所以變得可行，是因為三件事同時成熟：大量資料、GPU 加速的運算能力，以及更好的機器學習技術。Lecture 1 聚焦在第二件事。DNN 造成硬體問題，是因為它的運算需求、記憶體流量、部署規模與能源成本，成長速度都超過通用處理器自然進步的速度。

本課程的答案不是單純「放更多 MAC 單元」。有用的 DNN 加速器（DNN accelerator）是一組協調過的設計：運算本身、映射（mapping）、資料表示法（format）、記憶體階層、互連網路與實作技術都要一起考慮。Lecture 1 先建立一個會貫穿整門課的心智模型：比起算術本身，把資料搬得很遠通常更貴。如果一筆資料能從 DRAM 讀一次，之後在 processing element 附近重用很多次，省下的能量可能遠大於把 multiplier 做快一點所省下的能量。

本講最重要的框架是 TeAAL 關注點金字塔（Pyramid of Concerns）：architecture 會約束 compute、mapping、format 與 binding。後續課程會逐層展開：DNN 運算與 Einsum、記憶體與 metrics、mapping 與 dataflow、sparsity、precision、compute-in-memory，以及用建模工具評估設計取捨。

---

## 本講解決什麼問題

Lecture 1 回答一個看似簡單的問題：

**為什麼深度學習需要一門專門的硬體架構課？**

答案有四層。

1. DNN 運算已經不再只是跑在通用機器上的小型軟體工作負載。它已經成為消耗運算、金錢與能源的主要來源。
2. 過去依賴 Moore's Law 與 Dennard scaling 的路線，已經不能提供足夠的效能與能效改善。
3. DNN 工作負載有可利用的結構：tensor operations、大量平行性、資料重用、對近似數值的容忍度，以及 sparsity 之類的資料屬性。硬體可以利用這些結構。
4. 一旦硬體開始特化，我們就需要有紀律的詞彙來比較設計。否則每個 accelerator 都看起來獨一無二，無法比較。

因此本講設定整門課的主軸：讀一個 DNN accelerator 時，要問它算什麼、如何把工作映射到硬體、資料如何表示、抽象操作如何綁定到具體資源，以及 architecture 對這些選擇施加什麼限制。

---

## 為什麼本講重要

如果你只記得「AI 需要很多 GPU」，就會錯過真正的架構重點。重點是 DNN hardware 常常受限於**資料搬移（movement）**、**利用率（utilization）**與**硬體限制（constraints）**，而不只是 peak arithmetic throughput。

GPU、TPU、mobile neural engine、sparse accelerator 或 compute-in-memory array 都可能標榜很高的 TOPS。但一個 workload 只有在正確資料於正確時間抵達正確 compute units 時，才會達到有用吞吐量。如果 PE array 在等 memory、如果只有一半 PE 真的活躍、如果 compression metadata 的成本比跳過 zero 省下的成本還高，那麼 advertised peak 就不是實際結果。

Lecture 1 給出本課程第一版核心問題：

**對這個 workload 而言，真正付出成本的地方在哪裡：算術、資料搬移、儲存容量、頻寬、控制不規則性，還是損失的 utilization？**

---

## 先備知識與心智模型

你還不需要懂 accelerator design，但本章會用到四個基本概念。

- **DNN（Deep Neural Network）** 是由許多 layer 組成的 computation graph，其中許多 layer 可化為 matrix multiplication、convolution、attention 等 tensor operations。
- **MAC（multiply-accumulate）** 會計算一個乘積並累加到結果中。DNN layer 會執行大量 MAC。
- **記憶體階層（memory hierarchy）** 有靠近 compute 的小而快儲存，也有較遠、較大、較慢的儲存：register file、on-chip buffer、DRAM。
- **硬體架構（hardware architecture）** 不只是 arithmetic units 的集合，而是 compute units、storage、interconnect、control 與 scheduling assumptions 的整體組織。

本講最簡單的心智模型是：

**DNN accelerator 是一座 tensor operations 工廠。工廠裡的機器是 PE，但只有在材料能有效率地穿過 storage 與 interconnect 時，工廠才有生產力。**

這是教學詮釋（teaching interpretation），但它能說明為什麼後面課程會花很多時間討論 dataflow、mapping、sparsity format 與 cost modeling。

---

## 學習目標

讀完本章後，你應該能夠：

- 說明 data、GPU 與新的 ML techniques 如何共同促成現代 AI。
- 從每次迭代成本、執行頻率與部署後能源的角度，區分 training 與 inference。
- 說明 Moore's Law 與 Dennard scaling 趨緩為何推動 domain-specific hardware。
- 說明 on-device inference 為何與 privacy、latency、communication、power、thermal limits 有關。
- 用 TeAAL Pyramid of Concerns 區分 architecture、compute、mapping、format 與 binding 決策。
- 描述典型 DNN accelerator template：DRAM、global buffer、NoC、PE array、RF 與 ALU。
- 根據 Lecture 1 slide 43 的 normalized hierarchy，解釋為何 data movement 可能主導 energy。
- 將 roofline-style bound 理解成一連串把 achievable throughput 從 theoretical peak 往下壓的限制。
- 說出本課程分析新 accelerator 的詞彙：order、partitioning、dataflow、memory movement、data-attribute optimizations、co-design 與 flexibility。

---

## 主要敘事

### 1. 現代 AI 的三個要素

Lecture 1 一開始列出三個要素：big data availability、GPU acceleration、new ML techniques。投影片給了具體的資料規模例子：每天上傳數億張影像、每分鐘上傳數百小時影片、每小時處理 petabytes 等級的 customer data（Lecture 1 slide 2）。

這些例子不是單純的動機故事。它們說明為什麼 model capacity 增加時，deep learning 真的有機會變好：有足夠資料可訓練大模型，有足夠 compute 可處理資料，也有足夠 algorithmic progress 讓訓練結果有用。

接著本講引用 Ilya Sutskever 在 2017 年的說法：compute 是 deep learning 的「oxygen」（slide 3）。教學重點是 compute 不是被動資源。它會形塑研究者能嘗試什麼模型、迭代速度有多快，以及 deployed system 是否付得起大規模執行成本。

### 2. 為什麼 GPU 進入故事

GPU 之所以吸引人，是因為 DNN workload 包含大型、規律、可平行的 tensor operations。一次 matrix multiplication 可以暴露成千上萬個獨立的 product 與 sum。這與少數 general-purpose CPU cores 不太匹配，卻很適合大量 parallel arithmetic lanes。

現代 GPU 接著變得更 DNN-specific。Lecture 1 指出 GPU 加入 matrix multiplication 專用硬體、reduced precision formats 與 sparsity support；NVIDIA 在 2017 年引入 Tensor Core（slide 4）。這件事重要，因為即使是「通用」GPU，一旦 DNN 變得具有經濟價值，也開始內建部分 domain-specialized 結構。

教學詮釋：GPU 是 general-purpose computing 與 fixed-function accelerator 之間的橋樑。它仍可程式化，但越來越多結構假設 workload 會以 tensor 為核心。

### 3. Compute 與 energy crisis

Lecture 1 用多個 scale example 說明 DNN compute 很昂貴。

- Slide 9 重現 OpenAI AI-and-compute curve，指出從 AlexNet 到 AlphaGo Zero 的 compute demand 增加約 $300{,}000\times$。
- GPT-3 被描述為 96-layer、175-billion-parameter model，training 需要約 $3.14 \times 10^{23}$ FLOPs（slide 12，引用 Brown et al. 與 Lambda Labs）。
- Slide 12 估計在單張 Tesla V100 上訓練 GPT-3 約需 355 年，使用該來源中最低價 cloud GPU provider 約需 $4.6 million。
- Slide 8 引用 Goldman Sachs, April 2024，說明 data centers 在 2022 年約占 US electricity demand 的 3%，到 2030 年可能成長到約 8%。

這些數字應該被讀成有來源錨點的動機，而不是永恆常數。硬體、價格、模型與 datacenter assumptions 都會快速改變。穩定的教訓是趨勢：compute 與 energy 是一階設計限制。

### 4. Training 與 inference

Training 與 inference 對硬體施加的壓力不同。

**Training（訓練）** 會調整 model parameters，通常包含 forward passes、backward passes、gradient accumulation 與 optimizer updates。它每次 iteration 成本很高，但執行頻率通常低於 inference。

**Inference（推論）** 是執行已訓練好的 model 來產生 output。單次 request 比 training 便宜，但可能發生數十億或數兆次。inference 中一點小 inefficiency，乘上部署規模後就會變成龐大的 datacenter 或 battery cost。

Lecture 1 用兩個 deployment observations 支持這個區分：

- 在 Google，2019-2021 年間約有三分之五的 ML energy use 用於 inference，五分之二用於 training（slide 15，引用 Patterson, Computer 2022）。
- 在 Meta，AI infrastructure 的 rough power-capacity breakdown 是 experimentation、training、inference 約 10:20:70（slide 15，引用 Wu, MLSys 2022）。

其含意很細：即使 training 因為昂貴而常登上新聞，inference hardware 往往決定長期 energy 與 deployment feasibility。

### 5. 為什麼 on-device computing 重要

Cloud 很強，但 cloud inference 的成本不只金錢。

- **Communication（通訊）：** 把 sensor data 送到 datacenter 會消耗 bandwidth 與 energy。
- **Privacy（隱私）：** raw data 可能很敏感。
- **Latency（延遲）：** 往返 cloud 的時間對互動式或 safety-critical system 可能太慢。

Lecture 1 用 self-driving car 讓這點具體化。Slides 19-20 指出 cameras 與 radar 每 30 秒可產生約 6 GB data，self-driving car prototypes 使用約 2,500 W compute，而一個 autonomous-vehicle scenario 若在 10 cameras 上以 60 Hz 跑 10 個 DNN inferences，則每台車每小時會有 21.6 million inferences。算式是：

$10 \text{ DNNs} \times 60 \text{ frames/s} \times 10 \text{ cameras} \times 3600 \text{ s/hour} = 21{,}600{,}000$ inferences/hour。

若有一百萬台車，同樣 workload 會變成：

$21.6 \times 10^6 \times 10^6 = 21.6 \times 10^{12}$ inferences/hour。

這就是為什麼本講稱車輛為「data center on wheels」（slide 20）。它有 datacenter scale 的 workload，卻受 vehicle 的 power、cooling、latency 與 reliability constraints 限制。

### 6. 為什麼 general-purpose CPU 不夠

Slides 23-25 展示越來越複雜的 CPU pipeline：simple in-order、out-of-order、out-of-order simultaneous multithreading。這些圖不是為了在本講詳細教 CPU microarchitecture，而是為了建立對比。

General-purpose CPU 必須處理 branches、pointer chasing、interrupts、不可預測 memory accesses、許多 instruction types 與許多 programs。因此它會把 area 與 energy 花在 branch prediction、register renaming、dependency prediction、speculative execution、retirement policy、cache replacement 等結構上。

DNN tensor kernels 通常更規律。它們有 loop nests、重複的 tensor accesses 與可預測 reuse。Domain-specific accelerator 可以拿掉部分 general-purpose overhead，把資源花在：

- 大量簡單 PE，
- 靠近 PE 的 local storage，
- 與 tensor movement 匹配的 interconnect，
- 能重用 weights、activations、partial sums 的 schedules，
- 當 workload 允許時，支援 reduced precision 或 sparsity。

這不是說 CPU 不好，而是 CPU 最佳化的是 flexibility；DNN accelerator 則用部分 flexibility 換取特定 workload domain 上的 energy efficiency 與 throughput。

### 7. 「免費」scaling 的結束

Lecture 1 明確把 specialization 連到 Moore's Law 與 Dennard scaling 的趨緩（slide 22）。Moore's Law 指歷史上的 transistor-density scaling；Dennard scaling 指過去電晶體微縮時 power density 仍可管理的期待。

當這些趨勢放慢，architects 就不能再期待每一代製程自動讓相同 general-purpose design 得到大量速度與 energy 改善。回應方式就是 domain-specific architectures：如果物理不再免費給足改善，architecture 就必須利用 workload structure。

### 8. 每個 accelerator 都獨特，所以需要框架

Slides 26-27 展示許多 accelerator designs：Eyeriss、Eyeriss v2、SCNN、ExTensor、Gamma、spZip、ISOSceles、RAELLA、Highlight、Overbooking、Trapezoid、FuseMax。視覺上的教訓是 accelerators 並沒有單一明顯形狀。

若沒有框架，學生可能會試著死背每顆 chip。這很快會失敗。更好的方式是固定問一組問題：

- 它加速的是什麼 computation？
- 它試圖讓哪些 data stationary 或留在附近？
- 哪些 loops 被 parallelized、tiled 或 reordered？
- Sparse 或 compressed data 如何表示？
- 抽象 work units 如何指派到具體 PEs 與 buffers？
- 哪些 architecture constraints 讓這些選擇變好或變差？

下一節會命名這個框架。

### 9. TeAAL Pyramid of Concerns

Lecture 1 slide 28 引入 TeAAL Pyramid of Concerns，投影片將其歸因於 Nayak et al., MICRO 2023。這個 pyramid 把 accelerator reasoning 拆成幾個層級：

| 關注點 | 問題 | 例子 |
|---|---|---|
| **Compute** | 要評估什麼數學運算？ | Matrix multiplication、convolution、attention、Einsum |
| **Mapping** | 運算如何排程？ | Loop order、tiling、parallelism、dataflow |
| **Format** | 資料如何表示？ | Dense tensor、compressed sparse row、run-length encoding |
| **Binding** | 每個抽象動作由哪個具體資源執行？ | 把 tile $T_{ij}$ 指派給 PE group 3 與 buffer bank 1 |
| **Architecture** | 哪些硬體限制約束以上選擇？ | PE count、NoC topology、buffer sizes、memory bandwidth |

投影片把 architecture 畫成對其他 concerns 的約束。例如，一個讓整個 weight tile stationary 的 mapping，只有在 local storage 夠大時才可能成立。Sparse format 若能省 DRAM traffic，但 architecture 缺乏有效率的 metadata decoding 硬體，也可能不划算。

重要習慣是不要太早混淆層級。「這個 accelerator 很快」不是解釋。更好的解釋是：「compute formulation 減少 redundant work；mapping 重用關鍵 tensor；format 壓縮 zeros；binding 讓足夠多 PE 保持 active；architecture 提供該 schedule 需要的 storage 與 bandwidth。」

### 10. FuseMax 作為 pyramid 範例

Slides 29-34 用 FuseMax，一個投影片標示為 Nayak et al., MICRO 2024 的 attention accelerator，展示改善可出現在 pyramid 不同層級。

投影片辨識出四種 change：

- improved computation，降低 data movement，
- changed architecture，讓 better mapping 成為可能，
- better mapping，利用 architecture changes 暴露出的 capabilities，
- improved binding，改善 resource utilization。

接著 PE-utilization slides 從 baseline、enhanced computation、improved architecture/mapping 到 improved binding（slides 30-33），最後是 speedup-on-attention slide（slide 34）。本章不重製圖，也不引用圖上的精確數值。基於來源的 claim 只是：Lecture 1 使用 FuseMax 來說明 cross-layer co-design。

教學重點是一般性的：高效能 accelerator 很少只靠單一技巧，通常是一串彼此相容的 compute、architecture、mapping 與 binding 選擇。

### 11. 典型 DNN accelerator template

Slide 43 引入典型 DNN accelerator：

**DRAM $\rightarrow$ global buffer $\rightarrow$ NoC $\rightarrow$ PE array $\rightarrow$ local RF and ALU。**

這些詞很重要：

- **DRAM** 儲存大型 tensors，但離 compute 遠，access energy 高。
- **Global buffer** 是許多 PE 共享的 on-chip storage。
- **NoC（network-on-chip）** 在 buffers 與 PEs 之間搬資料。
- **PE（processing element）** 執行 arithmetic，通常有小型 local register file 或 local buffer。
- **RF（register file）** 比 DRAM 小很多，但靠近 ALU，access energy 低。

這種 architecture 被稱為 **spatial**，因為許多 PE 同時存在，tensor computation 的不同部分可以被指派到不同物理位置。這不同於單一 scalar pipeline 反覆在時間上重用同一個 execution unit。

### 12. 核心 energy lesson：data movement dominates

Slide 43 給出從 commercial 65 nm process 量測而來的 normalized energy hierarchy。投影片以 ALU access 或 operation 作為 $1\times$ reference，並給出近似相對成本：

| 來源或搬移層級 | 近似 normalized energy |
|---|---:|
| ALU / local RF reference | $1\times$ |
| Neighbor PE movement | $2\times$ |
| Global buffer access | $6\times$ |
| DRAM access | $200\times$ |

不要把這些數字過度解讀成永恆物理常數。它們是特定 process 與 modeling setup 下、投影片用來教學的數值。目的在於教排序：越遠、越大的 memory，energy cost 通常越高。

架構含意非常大。假設一個 weight 被 16 個 MAC 使用。

- 如果 accelerator 每次都從 DRAM 讀取該 weight，weight traffic 約需 $16 \times 200 = 3200$ normalized energy units。
- 如果它從 DRAM 讀一次，接下來 15 次從 local RF 重用，weight traffic 約需 $200 + 15 \times 1 = 215$ units。

Arithmetic work 仍是同樣的 16 個 MAC。Energy 改變，是因為 mapping 改變了資料在兩次使用之間住在哪裡。

這是後續 dataflow 論證的第一版：好的 accelerator schedule 會試圖在便宜 storage 中最大化有用 reuse，並最小化昂貴 storage 的重複搬移。

### 13. DNN accelerator 的設計選擇

Slide 44 列出 labs 與後續 lectures 會反覆出現的 design choices：

- **PE array：** PE 數量與 NoC 連線。
- **Memory hierarchy：** 層數、各層 capacity、data layout。
- **Scheduling：** operation ordering、dataflow、tiling、parallelism、fusion。
- **Sparsity handling：** gating、skipping、representation format。
- **Technology：** compute、memory、interconnect 的實作選擇，包括 emerging devices。

重點是這些選擇會互動。大型 PE array 只有在 mapping 與 memory bandwidth 能讓 PE 保持 active 時才有用。Compressed sparse format 只有在節省的 movement 與 arithmetic 超過 metadata/control 成本時才有用。新 memory technology 只有在改善 workload 真正 bottleneck 時才有用。

### 14. 用 roofline 看 inefficiency

Slide 45 引入 roofline-style evaluation method。傳統 roofline model 會把 achievable throughput 與 compute intensity 連在一起。Compute intensity 常寫成：

$\text{compute intensity} = \frac{\text{useful operations}}{\text{data moved}}$。

在本講脈絡中，useful operations 可能是 MACs；data moved 可依模型寫成 bytes、words 或 tensor elements。高 compute intensity 代表每筆搬來的資料支持很多 operations。低 compute intensity 代表每做一點 operation 就要搬很多資料。

投影片描述一串 tightening constraints：

1. maximum workload parallelism，
2. maximum dataflow parallelism，
3. finite PE count 下的 active PEs，
4. fixed PE-array dimensions 下的 active PEs，
5. fixed storage capacity 下的 active PEs，
6. insufficient average bandwidth 造成的 utilization 降低，
7. insufficient instantaneous bandwidth 造成的 utilization 降低。

教學詮釋是：peak performance 只是第一個天花板。真實 performance 會更低，因為 workload、mapping、array shape、storage 與 bandwidth 都會施加限制。

### 15. 先 modeling，而不是每個 design 都實作

Slide 46 說明為什麼本課強調 architectural modeling。RTL implementation 對廣泛 design-space exploration 太慢。因此課程使用 AccelForge 與 Accelergy 這類 modeling tools，在詳細硬體實作前先評估 architectural decisions。

這在教學上重要，因為它讓課程能問「what if？」：

- 如果 global buffer 變大會怎樣？
- 如果 dataflow 讓 weights stationary，而不是 outputs stationary，會怎樣？
- 如果 sparse activations 被 compressed，會怎樣？
- 如果 bottleneck 是 bandwidth 而不是 PE count，會怎樣？

目標不是永遠取代 RTL，而是先用快速模型找出有希望的 designs，並在投入詳細實作前理解 tradeoffs。

---

## Worked Examples

### Example 1：計算車輛推論量

Slide 20 描述在 10 cameras 上以 60 Hz 執行 10 個 DNN inferences。每小時推論數是：

$10 \times 60 \times 10 \times 3600 = 21{,}600{,}000$。

這個例子教的是 scale。單次 inference 可能看起來很小，但 deployment frequency 會主導成本。如果一百萬台車各跑一小時，整個 fleet 會執行：

$21{,}600{,}000 \times 1{,}000{,}000 = 21.6 \times 10^{12}$ inferences。

硬體含意：即使每次 inference 只省一點 energy，乘上 fleet 或 datacenter scale 後都可能變得巨大。

### Example 2：為什麼 reuse 勝過重複 DRAM reads

用 slide 43 的 normalized energy numbers 作為教學模型。假設一個 activation value 被 8 個 MAC 使用。

重複從 DRAM 讀：

$E_{\text{repeated DRAM}} \approx 8 \times 200 = 1600$ normalized units。

讀一次並在 local 重用：

$E_{\text{reuse}} \approx 200 + 7 \times 1 = 207$ normalized units。

近似降低比例是：

$\frac{1600}{207} \approx 7.7\times$。

這不是宣稱每個 accelerator 都會得到 $7.7\times$ energy reduction。它是一個小型數值例子，用來說明 mapping 與 storage locality 為何重要。

### Example 3：讀懂 compute intensity

假設一個 tile 執行 4096 個 MAC，並從較高 memory level 搬 1024 個 data words。它的 compute intensity 是：

$\text{CI} = \frac{4096 \text{ MACs}}{1024 \text{ words}} = 4 \text{ MACs/word}$。

如果更好的 mapping 執行同樣 4096 個 MAC，但只搬 256 個 words，則：

$\text{CI} = \frac{4096}{256} = 16 \text{ MACs/word}$。

硬體含意：arithmetic 沒變，但第二種 mapping 讓每個 moved word 支持更多 useful work。以 roofline 來說，這可能讓 workload 遠離 bandwidth limit。

---

## 關鍵方程式與如何閱讀

### Fleet-scale inference count

$N_{\text{inf/hour}} = N_{\text{models}} \times f_{\text{frames/s}} \times N_{\text{cameras}} \times 3600$。

這個式子計算 deployment scale 如何放大 workload。在 slide 20 的車輛例子中，$N_{\text{models}} = 10$、$f_{\text{frames/s}} = 60$、$N_{\text{cameras}} = 10$，所以每台車每小時是 21.6 million inferences。

### Compute intensity

$\text{compute intensity} = \frac{\text{useful operations}}{\text{data moved}}$。

它衡量每筆搬動資料支援多少 computation。較高 compute intensity 通常較容易在固定 bandwidth 下讓 machine 保持 busy，但前提是 parallelism 與 storage 也足夠。

### Simple data-movement energy model

$E_{\text{move}} \approx \sum_{\ell} n_{\ell} e_{\ell}$。

其中 $n_{\ell}$ 是 memory level $\ell$ 的 accesses 或 movements 數量，$e_{\ell}$ 是該層每次 access 的 energy cost。這個模型刻意簡化；它的用途是教你為什麼減少昂貴 DRAM accesses，可能比減少少量便宜 local accesses 更重要。

---

## 硬體含意

- **Energy：** 從 DRAM 搬資料可能主導 MAC energy，因此 local reuse 是核心。
- **Bandwidth：** PE array 即使有足夠 arithmetic units，也可能因 memory 餵不夠快而 idle。
- **Latency：** on-device inference 避免 network round trip，對互動式與 safety-critical systems 很重要。
- **Area：** general-purpose CPU structures 購買 flexibility；accelerator 則把更多 area 花在規律 arrays、buffers 與 interconnect。
- **Utilization：** peak TOPS 只有在足夠 PE 長時間 active 時才有意義。
- **Memory capacity：** 某個 dataflow 理論上很好，但若 stationary tile 放不進 local/global buffers，就不可行。
- **Programmability：** specialization 改善效率，但可能讓 software、compiler 與 mapping tools 更困難。
- **Correctness and robustness：** reduced precision、sparsity、approximate methods 需要 algorithm-hardware co-design，避免效率提升偷偷破壞 model quality。
- **Scalability：** datacenter 與 fleet-scale inference 會把微小 per-query inefficiency 放大成大型 system cost。

---

## 常見誤解

### 誤解：DNN accelerator 就是一個很大的 MAC array。

MAC array 很重要，但不夠。Energy 與 throughput 取決於 weights、activations、partial sums 如何通過 memory hierarchy、多少 PE 真的 active，以及 mapping 是否暴露 reuse。

### 誤解：training 很貴，所以它永遠是主要 energy problem。

Training 每次 iteration 成本高，但 inference 可能主導 deployed energy，因為它發生得多得多。Lecture 1 slide 15 用 Google 與 Meta 的 energy/power breakdown 支持這點。

### 誤解：TOPS 可以告訴你哪個 accelerator 最好。

TOPS 是 peak arithmetic metric。它不告訴你 workload 是否有足夠 parallelism、data 是否能準時送到、storage capacity 是否足夠，或 sparse metadata 是否帶來額外 overhead。

### 誤解：specialization 代表沒有 flexibility。

Specialization 是一個光譜。帶 Tensor Cores 的 GPU 仍然可程式化，但包含 tensor-specific hardware。Mobile neural engines、TPUs 與 sparse accelerators 則選擇不同的 flexibility-efficiency tradeoff。

### 誤解：sparse 或 compressed format 會自動省 energy。

Format choices 只有在節省的 arithmetic 與 data movement 超過 metadata、decoding、irregular access 與 load imbalance 成本時才有幫助。這會成為 sparsity lectures 的主要主題。

---

## 連結

### 與既有知識的連結

本講建立在基本 computer architecture 與 machine learning 概念上。從 architecture 來看，它使用 compute、memory、interconnect 有不同成本的概念。從 ML 來看，它使用 DNN 是 tensor programs 且有 training/inference phases 的概念。

### 與後續講次的連結

- **L02-L04：** 課程會形式化 DNN components 與 tensor computations，包括 Einsums 與 Transformers。
- **L03：** Energy hierarchy 與 key metrics 會更精確。
- **L05-L06：** Mapping layer 會變成 dataflow、tiling、partitioning 與 parallel scheduling。
- **L07-L10：** Format layer 會變成 sparsity、sparse representations 與 sparse accelerator architecture。
- **L11-L13：** Advanced technologies、precision 與 motion/data-movement calculation 會從不同角度回到同一個 energy argument。

跨講次主線很簡單：後面每個 optimization 都應該被理解為試圖改善 useful computation 與 expensive movement 或 underutilized hardware 之間的比例。

---

## 論文與來源橋接（Paper and Source Bridge）

Lecture 1 主要是 course-motivation lecture，不是 paper lecture，但它指向幾個建立課程框架的重要來源。

**Local PDF availability note：** 目前 repository 已有 TeAAL 與 FuseMax 的 local PDFs，因此下方兩個 bridge 同時是 paper-verified 與 slide-anchored。

### TeAAL, Nayak et al., MICRO 2023

- **Bibliographic identity：** *TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators*，MICRO 2023。Local PDF：`papers/TeAAL.pdf`。
- **Problem addressed：** Sparse tensor accelerators 常被 verbose RTL 或 incomplete diagrams 描述，導致很難精確比較 designs。
- **Core idea used here：** TeAAL 將 accelerator designs 表示為 mapped Einsums cascades 加上 fibertrees 上的 transformations；paper 的 abstraction hierarchy 是課程 concern-layer framing 的來源。
- **Lecture relevance：** slide 28 使用這個 pyramid 作為課程組織框架。Paper 使該 slide 具體化：分離 computation、mapping、format、architecture、binding specifications。
- **Key claims used here：** TeAAL 用 Einsums 表達 tensor computations，並把 iteration order 留給 mapping（Section 2.2）；mapping 包含 loop order、rank partitioning、work scheduling（Section 2.3）；language 也透過 simulator generator 建模 formats、architectures、bindings（Sections 3-4）。
- **What students should remember：** 分析新 accelerator 時，不要先死背 chip diagram，而要先分離 concern layers，接著問設計真正改了哪一層。
- **Limitation：** Lecture 1 只高層次使用 TeAAL；sparse fibertree details 會在後續 sparse-architecture lectures 才重要。

### FuseMax, Nayak et al., MICRO 2024

- **Bibliographic identity：** *FuseMax: Leveraging Extended Einsums to Optimize Attention Accelerator Design*，2024。Local PDF：`papers/FuseMAX.pdf`。
- **Problem addressed：** Attention 同時有 tensor products 與 softmax steps，兩者 compute/memory behavior 不同；既有 spatial designs 可降低 bandwidth，卻仍可能在 softmax 周圍 under-utilized。
- **Core idea used here：** FuseMax 將 attention 描述為 extended Einsums cascades，再 co-design mapping、architecture、binding，使 1D 與 2D PE arrays 維持高 utilization。
- **Lecture relevance：** slides 29-34 使用 FuseMax 展示 speedup 可來自 coordinated compute、architecture、mapping、binding changes，而不是 pyramid 中單一層。
- **Key claims used here：** Paper 在 Abstract 中表示 FuseMax 目標是 nearly 100% compute utilization 與 sequence-length-independent on-chip buffer requirements；它使用 1-pass attention cascade 與 spatial architecture 上的新 mapping/binding（Sections IV-V）；在其 evaluation setup 下報告相對 FLAT 的 6.7x average attention speedup 與 5.3x end-to-end inference speedup（Abstract 與 Section VI）。
- **What students should remember：** Accelerator gains 常來自一組相容選擇。L01 使用 FuseMax 是為了展示 pyramid 作為 design method，而不是要求學生此時掌握所有 attention details。
- **Limitation：** Speedup numbers 具 paper-specific 性質，取決於 modeled workloads、baselines 與 hardware assumptions；L01 只用它們說明 cross-layer co-design。

### Efficient Processing of Deep Neural Networks, Sze and Emer et al.

- **Problem addressed：** 如何設計與評估高效率 DNN processing systems。
- **Core idea used here：** DNN accelerator efficiency 高度取決於 data movement、memory hierarchy、mapping 與 workload characteristics。
- **Lecture relevance：** slides 41 與 45 指向 course textbook/readings 與 Chapter 6 的 roofline-style evaluation。
- **What students should remember：** 本講詞彙連到一套更廣泛的 DNN hardware evaluation 方法，而不只是單一投影片 deck。

---

## 獨立學習指南

### 進入下一講前要掌握

- 不只用「training 是學習，inference 是使用」來解釋兩者差異，而要說出成本與頻率差異。
- 推導 slide 20 的 vehicle inference count。
- 解釋為什麼 DRAM access 可能比 MAC 更值得最佳化。
- 用 TeAAL Pyramid 將 design change 分類為 compute、mapping、format、binding 或 architecture。
- 解釋為什麼 peak performance 會被 workload parallelism、dataflow parallelism、PE count、array shape、storage 與 bandwidth 逐步收緊。

### 自我檢核問題

1. 為什麼 GPU 在 fully custom accelerator 普及前，會先成為 deep learning 的核心硬體？
2. 為什麼單次 inference 遠比 training step 便宜，inference 仍可能主導 total energy？
3. 在車輛例子中，如果 camera count 加倍但 frame rate 不變，推論數如何變化？
4. Dennard scaling 趨緩為什麼對 DNN hardware 重要？
5. 各舉一個 compute-layer change 與 mapping-layer change。
6. Sparse format 為什麼可能無法改善 energy？
7. Roofline model 隱藏了什麼？為什麼它仍然有用？
8. 如果 PE array 有很高 peak TOPS 但 utilization 很低，可能是哪幾個 concern layers 出問題？

### 練習

1. **Conceptual：** 說明為什麼對某些 DNN accelerators 而言，「少搬資料」比「少做 operations」更有用。各舉一個兩者重要的情境。
2. **Small calculation：** 一個 value 被 reuse 32 次。使用 slide 43 的 $200\times$ 與 $1\times$ 成本，比較重複 DRAM reads 與一次 DRAM read 加 31 次 local reads。
3. **Design tradeoff：** 一個 mobile accelerator 必須在 1 W budget 內執行。列出三個能降低 energy 的 architecture choices，並說明每個 choice 可能如何降低 flexibility。
4. **Pyramid classification：** 分類以下 changes：使用 INT8 取代 FP16、改變 loop order、增加 global-buffer size、把 tiles 指派到不同 PE groups、使用 compressed sparse row format。
5. **Paper bridge：** 閱讀一篇 DNN accelerator paper 的 abstract，辨識它的 compute、mapping、format、binding 與 architecture claims。標出哪些 quantitative claims 需要 source anchors。
6. **Open-ended reasoning：** 選一個 workload，例如 convolution、attention 或 recommendation inference。預測它的 bottleneck 可能是 arithmetic、memory bandwidth、storage capacity 或 irregularity，並說明你的 assumptions。

---

## 關鍵詞彙

### Deep Neural Network (DNN，深度神經網路)

由多層 numerical transformations 組成的 model。在本課中，DNN 重要之處是它作為 hardware workload：它產生 tensor operations、memory traffic，以及大規模重複 inference。

### Training（訓練）

調整 model parameters 的過程。Training 通常包含 forward computation、backward computation 與 parameter updates，因此每次 iteration 成本高。

### Inference（推論）

執行已訓練 model 以產生 outputs 的過程。Inference 單次通常比 training 便宜，但因為發生次數多，可能主導 total energy。

### MAC (Multiply-Accumulate，乘加)

運算 $a \times b + c$。MAC 是 matrix multiplication、convolution 與許多 DNN kernels 的核心，但 MAC count alone 不決定 energy。

### Processing Element (PE，處理單元)

小型 compute unit，通常包含 ALU 與 local storage。DNN accelerators 常把許多 PE 排成 spatial array。

### Register File (RF) / Local Buffer（暫存器檔／本地緩衝）

靠近 PE 的小型儲存。容量有限，但 access energy 低，因此對 reuse 很有價值。

### Global Buffer（全域緩衝）

由多個 PE 共享的 on-chip storage。它比 local PE storage 大，但 access cost 也較高。

### DRAM

高容量但 access energy 高的 off-chip memory。減少重複 DRAM traffic 是 accelerator 的核心目標。

### Network-on-Chip (NoC，晶片上網路)

在 buffers 與 PEs 之間搬資料的 interconnect。NoC design 會影響 bandwidth、energy 與 utilization。

### Domain-Specific Hardware（領域專屬硬體）

針對某一類 workload 而非任意 programs 最佳化的硬體。它用部分 flexibility 換取該 domain 上更好的 performance 或 energy efficiency。

### TeAAL Pyramid of Concerns（TeAAL 關注點金字塔）

把 compute、mapping、format、binding 與 architecture 分離的框架。它讓我們能比較 accelerators，而不是只看單一 block diagram。

### Compute（運算）

被評估的數學運算，例如 Einsum、convolution、matrix multiplication 或 attention step。

### Mapping（映射）

把 computation 與 data movement 放到硬體上的 schedule，包含 loop order、tiling、parallelism 與 dataflow。

### Format（格式）

資料表示法，特別是 dense、compressed 或 sparse forms。Format 會影響 memory traffic 與 decoding overhead。

### Binding（綁定）

把抽象 work 與 data 指派給具體 hardware resources，例如 PE IDs、buffer banks 與 time slots。

### Roofline Model（屋頂線模型）

把 throughput 與 compute intensity、machine limits 連在一起的 performance model。Lecture 1 用它來說明 theoretical peak performance 如何被收緊成較 realistic 的 bound。

### Compute Intensity（運算強度）

比例 $\text{useful operations}/\text{data moved}$。較高 compute intensity 代表每筆 moved datum 支持更多 useful work。

### Utilization（利用率）

Hardware resources 正在做 useful work 的比例。低 utilization 會抵消大型 PE array 的好處。

---

## 重點回顧

- 現代 AI 由 data、compute 與 ML techniques 共同促成，但 compute 是本課教你如何 architect 的資源。
- DNN hardware 重要，因為 compute demand、inference scale、energy 與 cost 都是 system-level constraints。
- Moore's Law 與 Dennard scaling 不再提供足夠自動改善，因此 architects 必須利用 workload structure。
- Specialized hardware 不只是更快的 multiplier，而是 compute、storage、movement、mapping、format 與 binding 的整體設計。
- TeAAL Pyramid 提供可重用框架，用來閱讀新的 accelerators。
- 典型 accelerator template 是 DRAM、global buffer、NoC、PE array、local RF 與 ALU。
- Data movement 可能主導 energy；slide 43 的 DRAM example 是本課聚焦 reuse 的第一個錨點。
- Roofline-style evaluation 說明 peak performance 會被 workload、mapping、storage、array 與 bandwidth constraints 降低。

---

## 連結

Lecture 1 是整門課的 roadmap。它引入的詞彙會在後續講次精確化：

- **Workload and DNN components：** L02-L04 解釋 models 實際算什麼。
- **Metrics and memory：** L03 發展 energy、bandwidth 與 evaluation vocabulary。
- **Einsum：** L03-L04 提供 tensor computations 的記法。
- **Mapping and dataflow：** L05-L06 把「讓資料靠近 compute 重用」轉成 loop schedules 與 spatial mappings。
- **Sparsity：** L07-L10 研究 format 與 data-attribute-specific optimization。
- **Precision and advanced technologies：** L11-L13 探索降低 arithmetic 與 movement costs 的其他方式。

本講沒有前一個 course lecture 可連接。它的角色是建立為什麼後續主題屬於同一門課。

---

## 附錄

### 投影片對照表（Slide-to-Section Map）

| 投影片範圍 | 本章章節 | 備註 |
|---|---|---|
| L01-1 | 標題與 metadata | 課程身份 |
| L01-2-L01-3 | 現代 AI 的三個要素 | 擴寫為 motivation 與 compute framing |
| L01-4-L01-7 | GPU、datacenter、specialized、mobile DNN hardware | 用來解釋 specialization spectrum |
| L01-8-L01-17 | Energy、compute demand、ChatGPT、training/inference、GPU investment | 擴寫為 compute/energy crisis 與 training-vs-inference |
| L01-18-L01-21 | On-device processing 與 self-driving cars | 加入 inference-count worked example |
| L01-22-L01-25 | Moore/Dennard slowdown 與 CPU pipelines | 用來對比 general-purpose flexibility 與 domain-specific efficiency |
| L01-26-L01-27 | Accelerator galleries | 用來動機化 framework 的必要性 |
| L01-28-L01-29 | TeAAL pyramid 與 FuseMax enhancements | 擴寫為 concern-layer framework 與 paper/source bridge |
| L01-30-L01-34 | FuseMax utilization 與 speedup | 只做定性引用；不重製 figures 或精確 plotted values |
| L01-35 | Challenges and opportunities | 整合到 hardware implications 與 course motivation |
| L01-36-L01-39 | Class overview、outline、takeaways、objective | 整合到 learning objectives 與 connections |
| L01-40-L01-42 | Staff、requirements、labs | 僅在 course structure 相關處簡述 |
| L01-43-L01-45 | Accelerator template、design choices、roofline | 擴寫為 energy hierarchy、design choices、roofline sections |
| L01-46-L01-50 | Architectural modeling 與 design project | 用來說明 modeling-first workflow |
| L01-51-L01-53 | Grading、late policy、prerequisites | 非技術章節核心；保留為投影片課務資訊 |

## 來源註記（Source Notes）

- 本章的 lecture ordering 與 quantitative motivation 來自 `Lecture/L01-Intro_and_Applications.pdf`。
- AI ingredients、compute-as-oxygen quote、GPU specialization examples、datacenter/custom/mobile hardware examples 與 course logistics，皆由 L01-2 到 L01-7 以及 L01-36 到 L01-53 的投影片衍生。
- Data-center electricity estimate 由 L01-8 衍生；該頁引用 Goldman Sachs, April 2024。
- $300{,}000\times$ compute-growth claim 由 L01-9 衍生；該頁引用 OpenAI AI-and-compute discussion 與 Strubell, ACL 2019。
- GPT-3 training figures 由 L01-12 衍生；該頁引用 Brown, NeurIPS 2020 與 Lambda Labs explainer。
- Training/inference energy 與 power breakdowns 由 L01-15 衍生；該頁引用 Patterson, Computer 2022 與 Wu, MLSys 2022。
- Autonomous-vehicle inference example 由 L01-19 與 L01-20 衍生；本章的 arithmetic expansion 是原創教學內容。
- TeAAL Pyramid discussion 以 L01-28 與本地 `papers/TeAAL.pdf` 為依據，尤其是 Sections 2.2、2.3、3-4。
- FuseMax discussion 以 L01-29 到 L01-34 與本地 `papers/FuseMAX.pdf` 為依據，尤其是 Abstract、Sections IV-V、Section VI。
- Normalized energy hierarchy 由 L01-43 衍生，本文將其作為 pedagogical model，而非 universal technology constant。
- 將投影片數字與簡單算術結合的 worked examples，是本章原創教學例子。

## 不確定性註記（Uncertainty Notes）

- 本章依據投影片與 cited source anchors 重建可能的 lecture narration。現場講課可能強調不同例子或 caveats。
- 有些投影片引用來源並不存在於本 repository 的 local papers 中，因此本章以 lecture slides 作為 immediate source anchor，未獨立查證每個 external article 或 estimate。
- Cloud GPU pricing、model training cost 這類快速變動的成本 claim，應視為 tied to the cited slide date 的歷史動機，不應視為當前市場估計。
- Normalized energy ratios 依 process 與 model 而變。它們可作為本講的 qualitative hierarchy，但不是每種 accelerator technology 的精確常數。
