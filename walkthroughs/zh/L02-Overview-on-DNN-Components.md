# L02 — DNN 元件概論（Overview on DNN Components）

> **課程：** 6.5930/1 — 深度學習硬體架構（Hardware Architectures for Deep Learning）
> **講師：** Joel Emer 與 Vivienne Sze（MIT EECS）
> **講授日期：** 2026 年 2 月 4 日 · **投影片：** 102 頁 · **來源：** [`Lecture/L02-Overview_on_DNN_components.pdf`](../../Lecture/L02-Overview_on_DNN_components.pdf)
>
> 本章重建投影片背後的教學脈絡。投影片提供順序與來源錨點；本文則以沒有課堂影片的自學讀者為對象，補上動機、定義、推理、例子與硬體意義。

---

## TL;DR

L02 教的是後續整門課會反覆使用的兩套語言。

第一套是**工作負載描述語言（workload-description language）**：張量（tensor）、秩（rank）、Einsum、迭代空間（iteration space）、運算強度（compute intensity, CI）與屋頂線模型（Roofline Model）。這些概念讓硬體架構師能精確說明一個 DNN 層到底算什麼、需要多少算術、理想上要搬多少資料，以及某個實作是受限於記憶體頻寬還是運算平行度。核心觀念是：Einsum 指定**算什麼**，而迴圈順序或映射（mapping）指定**怎麼走訪**這個迭代空間。

第二套是 **CNN 元件語言（CNN-component language）**：CONV、activation、NORM、POOL、FC、feature map、filter、stride、padding、channel、batch size，以及標準迴圈變數 $N, C, H, W, R, S, M, P, Q, U$。其中最重要的工作負載是卷積（convolution）：每個輸出激活值都是對局部感受野與輸入通道做加總，而完整一層需要 $N \times M \times P \times Q \times C \times R \times S$ 次 multiply-accumulate。

這兩套語言放在同一講不是偶然。當 CONV 被寫成 Einsum 後，硬體設計者才能討論資料重用、記憶體流量、駐留性（stationarity）、mapping、tiling 與 parallelism。後面幾講會一直回到這座橋。

---

## 這一講解決什麼問題

L01 說明了為什麼 DNN acceleration 重要：AI 的算力需求高、資料搬移很貴，而且通用處理器不一定能有效利用 DNN 的結構。L02 接著問下一個問題：**加速器到底要跑的 workload 是什麼？**

這個問題比「跑一個 CNN」精確得多。硬體架構師需要知道：

- 有哪些 tensor，以及每個 tensor 的 shape 是什麼；
- 哪些 index 是 output index，哪些 index 是 reduction index；
- layer 需要多少 multiply-accumulate；
- 哪些 value 會在多次運算中被重用；
- stride 與 padding 如何改變 output shape；
- 某個 loop order 是否會反覆把 partial sum 搬進搬出記憶體；
- 增加 MAC units 是否真的有用，或真正瓶頸其實是 memory traffic。

因此本講先建立一套 modeling discipline：用 tensor algebra 描述 workload，用 compute 與 traffic metrics 評估它，然後才談硬體選擇。接著這套語言被套用到 CNN components，尤其是 convolution layer 與 fully connected layer。

**來源說明：** 這個順序依據 Lecture 02 slides 2-44 的 accelerator design methodology，以及 slides 45-102 的 CNN components。

---

## 為什麼這一講重要

很多 CNN 入門說明著重在機器學習觀點：filters 偵測邊緣，深層偵測高階概念，FC layers 產生 class scores。這個觀點有用，但對本課不夠。

對硬體架構師而言，DNN layer 也是一個有結構的 data-movement problem。Convolutional layer 不只是有很多 multiplication；它還有 weight 的重複使用、相鄰 activation 的重複使用，以及 partial sum 的反覆更新。這些 reuse patterns 決定 accelerator 的能量花在 arithmetic，還是花在 DRAM、SRAM、interconnect、register file 與 processing element 之間搬資料。

本講更深的訊息是：

> DNN component 不只是神經網路概念。它是一個 tensor 上的 iteration space，而如何走訪這個 iteration space 會決定 memory traffic、reuse、utilization 與 bottleneck。

這個觀念直接銜接 L03-L04 的 tensor algebra 與 memory metrics，也銜接 L05-L06 的 mapping/dataflow 與 partitioning。

---

## 先備知識與心智模型

你應該熟悉：

- vector 與 matrix；
- matrix-vector multiply 與 matrix-matrix multiply；
- 基本 CNN 詞彙，例如 input image、feature map、filter、layer；
- memory access 可能比 arithmetic 貴很多的概念。

本章的心智模型是：

1. Tensor 是多維 value box。
2. Einsum 描述一組 iteration-space points。
3. 每個 point 通常做一次 multiply，並貢獻到某個 output。
4. 如果某個 index 出現在右側但沒有出現在左側，計算會沿著該 index 做 reduction。
5. 硬體效率取決於每搬動一個 value 能換到多少有用 arithmetic。

你也可以把它想成 nested loop nest。Einsum 說明有哪些 loop 必須存在，但不規定 loop order。Mapping 則決定 loop order、tiling 與 parallelism。後面出現的 output-stationary、weight-stationary 等 dataflow 名稱，本質上就是在描述哪些 value 要留在 PE 附近。

---

## 學習目標

讀完本章後，你應該能夠：

1. 說明 TeAAL 五步 accelerator design methodology，並區分 architecture、workload、mapping、format、binding。
2. 定義 tensor rank、rank shape、tensor size、iteration space、free index 與 reduction index。
3. 讀懂 $Z_m = A_{k,m} \times B_k$ 這類 Einsum，並推導 iteration-space size、multiplication count 與 reduction behavior。
4. 計算本講 matrix-vector example 的 best-case 與 achieved compute intensity。
5. 用 Roofline reasoning 說明什麼時候增加 compute parallelism 有用，什麼時候 memory bandwidth 才是瓶頸。
6. 說明 CONV、activation、NORM、POOL、FC 在 CNN 中的角色。
7. 推導含 stride 與 optional padding 的 2-D convolution output dimensions。
8. 說明 CNN decoder-ring variables $N, C, H, W, R, S, M, P, Q, U$ 的意思。
9. 把 CONV layer 寫成 Einsum，並指出 output indices 與 reduction indices。
10. 說明為什麼 FC layer 是 CONV 的 special case，以及 batching 為什麼會把 FC 變成 matrix-matrix multiplication。
11. 把這些 DNN components 連回硬體問題：data reuse、partial sums、memory traffic、bandwidth、latency、utilization 與 mapping。

---

## 主要敘事：從 Workload 到 Hardware

### TeAAL 的五步方法論

L02 不是先從 neural-network layer 開始，而是先從 design method 開始。這是有意義的：DNN accelerator 不是先選一個 PE array，然後希望 workload 剛好合適。本課要建立的是從 workload 到 hardware 的可重複流程。

投影片用 TeAAL 作為組織框架。五個步驟是：

1. **描述架構（Describe the architecture）。** 選擇 processing elements (PEs)、ALUs、register files、global buffers、DRAM 等 hardware components，組成 accelerator specification。
2. **開發工作負載（Develop the workload）。** 用 cascade of Einsums 描述 computation，並指定 mapping、format 與 binding。
3. **評估工作負載（Evaluate the workload）。** 計算 operations、memory traffic 與 compute intensity。
4. **比較實作（Compare implementations）。** 正規化硬體參數後比較不同 designs。
5. **最佳化設計（Optimize the design）。** 修改 architecture、mapping、format 或 binding，再重新評估。

這不是只走一次的 checklist，而是循環。Memory traffic 太差可能代表需要換 mapping；某個 mapping 暴露更多 reuse，可能需要更多 local storage；sparse format 可能降低 data movement，但增加 metadata 與 control complexity。

**來源說明：** 五步設計流程直接來自 Lecture 02 slides 4-8 與 21-44，這些 slides 引用 TeAAL/HiFiber 與 Nayak, MICRO 2023。

### 關注點分離（Separation of Concerns）

這套方法論最重要的部分，是分開**算什麼**與**怎麼算**。

投影片把 workload concerns 分成四層，由最精簡到最細：

| 關注點 | 回答的問題 | 例子 |
|---|---|---|
| Cascade of Einsums | Workload 包含哪些 computations？ | CONV 接 activation 與 pooling |
| Mapping | Iteration space 以什麼順序走訪？ | Loop order、tiling、parallelism |
| Format | Data 怎麼表示？ | Dense、sparse、compressed |
| Binding | 哪個硬體資源負責哪部分？ | 哪個 PE、buffer 或 memory level 放某個 tensor tile |

這個分離很重要，因為只有 model 先把 concerns 分開，設計空間才容易被探索。例如 CONV Einsum 可以保持不變，但 mapping 可以從 output-stationary 換成 weight-stationary；同樣地，sparse format 可以在不改變 layer mathematical definition 的情況下加入。

**教學詮釋：** Slides 8 與 28 的 pyramid 是提醒：不要太早把 computation definition 和某個 loop order 綁死。若一開始就綁死，後面比較 mapping 或 accelerator design 會變困難。

---

## Tensors、Ranks 與 Einsums

### Tensor Terminology

**Tensor（張量）**是多維 array。本課把 tensor 的一個 dimension 稱為 **rank（秩）**。這和某些數學語境中 rank 的意思不同；在本課，rank 就是具名維度。

例子：

| 物件 | Rank 數 | Shape example | Size |
|---|---:|---|---:|
| Scalar | 0 | `[]` | $1$ |
| Vector | 1 | $[K]$ | $K$ |
| Matrix | 2 | $[M, K]$ | $M \times K$ |
| 3-D activation tensor | 3 | $[C, H, W]$ | $C \times H \times W$ |
| Batched activation tensor | 4 | $[N, C, H, W]$ | $N \times C \times H \times W$ |

**Rank shape** 是某個 rank 上有多少 elements。**Tensor size** 是所有 rank shapes 的乘積。

**來源說明：** Lecture 02 slides 9-11 介紹 tensors、ranks、rank shapes 與 tensor size。

### Einsum 的意思

**Einsum（Einstein summation notation）**是一種精簡描述 tensor algebra 的方法。它告訴我們哪些 operands 要相乘、哪個 output element 被更新，以及哪些 indices 被 reduce。

矩陣乘法可寫成：

$$Z_{m,n} = A_{k,m} \times B_{k,n}.$$

Index $k$ 出現在右側但不在左側，所以 $k$ 是 **reduction index**。明寫出來就是：

$$Z_{m,n} = \sum_k A_{k,m}B_{k,n}.$$

Matrix-vector multiplication 可寫成：

$$Z_m = A_{k,m} \times B_k.$$

這裡 $m$ 是 **free index**，因為它出現在 output 中；$k$ 是 **reduction index**，因為它只出現在右側。意思是：對每個 output coordinate $m$，沿著所有 $k$ 加總。

投影片給出 Einsum 的 operational definition：

1. 從所有 unique indices 的合法值建立 iteration space。
2. 在每個 point，依照 indices 讀取 operand values。
3. 將選到的 values 相乘。
4. 如果有 reduction index，就累加到 output。

對 $Z_m = A_{k,m} \times B_k$ 而言，iteration space 是 $K \times M$。每個 point $(k,m)$ 做一次 multiplication；具有相同 $m$ 的 points 會 reduce 到同一個 $Z_m$。

### 小型 Worked Example：Matrix-Vector Einsum

假設 $K=3$、$M=2$，output 是 $Z_0, Z_1$。Iteration space 有 $3 \times 2 = 6$ 個 points：

```text
(k,m): (0,0), (1,0), (2,0), (0,1), (1,1), (2,1)
```

輸出 equations 是：

$$Z_0 = A_{0,0}B_0 + A_{1,0}B_1 + A_{2,0}B_2,$$

$$Z_1 = A_{0,1}B_0 + A_{1,1}B_1 + A_{2,1}B_2.$$

共有 $K \times M = 6$ 次 multiplication。若每個 output 要把 $K$ 個 products 合併，則 additions 是 $(K-1) \times M = 4$。實際 addition count 會依初始化與 accumulation 是否分開計算而略有差異，所以本講在 compute intensity 中主要看 multiplication count 與 memory traffic。

**硬體意義：** 同一個 $B_0$ 會被 $m=0$ 與 $m=1$ 使用。如果 PE 載入一次 $B_0$，並在與 $A_{0,0}$、$A_{0,1}$ 相乘時把它留在 register 中，就能降低 memory traffic。這是後面稱為 **stationarity（駐留性）** 的概念第一次出現。

---

## Compute Intensity 與 Roofline Reasoning

### Best-Case Compute Intensity

L02 將 compute intensity 定義為：

$$\mathrm{CI} = \frac{\text{multiplications}}{\text{values accessed}}.$$

投影片刻意使用 **multiplications per value**，而不是常見的 FLOPs/byte。這避免兩個歧義：MAC 到底算一個 operation 還是兩個，以及 value bitwidth 是多少。

對 $Z_m = A_{k,m} \times B_k$：

- Multiplications：$K \times M$。
- Best-case traffic：每個 $A_{k,m}$ load 一次，每個 $B_k$ load 一次，每個 $Z_m$ store 一次。
- Best-case values accessed：$K \times M + K + M$。

因此：

$$\mathrm{CI}_{\text{best}} = \frac{K \times M}{K \times M + K + M}.$$

投影片例子使用 $K=250$、$M=100$：

$$\mathrm{CI}_{\text{best}} = \frac{250 \times 100}{250 \times 100 + 250 + 100} \approx 0.99\ \text{multiplications/value}.$$

這是這個簡化 memory model 下的 upper bound。它假設 implementation 能充分利用 reuse，避免額外 traffic。

**來源說明：** Lecture 02 slides 23-26 定義 compute intensity 並推導 matrix-vector example 的 best-case traffic；slide 41 給出 $K=250, M=100$ 的數值例子。

### Achieved Compute Intensity 取決於 Mapping

Best-case CI 不會自動達成。實際 loop order 與 storage behavior 會決定 achieved traffic。

本講使用的 loop order 是：

```text
for k in range(K):
    keep B[k] in a register
    for m in range(M):
        load A[k,m]
        load current Z[m] partial sum when needed
        update Z[m]
        store Z[m]
```

在這個 processing order 與簡化 storage model 下，achieved traffic 是：

- $K \times M$ 次載入 $A_{k,m}$；
- $K$ 次載入 $B_k$；
- $(K-1) \times M$ 次載入 $Z_m$ partial sums；
- $K \times M$ 次儲存 $Z_m$。

所以：

$$\text{traffic}_{\text{achieved}} = 3KM - M + K,$$

而：

$$\mathrm{CI}_{\text{achieved}} = \frac{K \times M}{3KM - M + K}.$$

對 $K=250$、$M=100$：

$$\mathrm{CI}_{\text{achieved}} = \frac{250 \times 100}{3 \times 250 \times 100 - 100 + 250} \approx 0.33\ \text{multiplications/value}.$$

$0.99$ 與 $0.33$ 的差距不是 matrix-vector multiplication 的數學本質，而是這個 implementation 的結果。這個 loop order 把 $B_k$ 留住，但會在 $k$ loop 之間反覆 reload/store partial sums $Z_m$。

**常見誤解：**「Einsum 決定 memory traffic。」不對。Einsum 決定 mathematical iteration space；mapping 與 hardware storage choices 才決定 achieved traffic。

### Roofline Model

**Roofline Model（屋頂線模型）**把 compute intensity 連到 throughput。最簡化的形式是：

$$\text{achievable throughput} \le \min(\text{peak compute throughput},\ \mathrm{CI} \times \text{memory bandwidth}).$$

斜線是 memory-bandwidth limit，水平線是 compute limit。低 CI workload 會落在斜線上，稱為 **memory-bound**；高 CI workload 才有機會碰到水平屋頂，變成 **compute-bound**。

架構上的教訓很直接：

- 如果 workload 是 memory-bound，增加 MAC lanes 不一定會提升 throughput。
- 如果 workload 是 compute-bound，改善 memory bandwidth 可能不是主要瓶頸。
- 如果實測 implementation 離 roof 很遠，原因可能是 stalls、instruction overhead、不佳 mapping、buffering 不足或 utilization 問題。

**來源說明：** Lecture 02 slides 42-43 介紹 Roofline Model，並引用 Williams, Waterman, and Patterson, CACM 2009。投影片也指出 roofline 可以針對 memory hierarchy 的每一層畫出來，雖然常見圖通常以 DRAM 為主。

---

## DNN Workloads 與 CNN Components

### 為什麼 Einsum 之後接 CNN

本講從 Einsum 轉到 CNN 不是跳題，而是把抽象 workload language 套到真實 DNN family 上。

CNN 可用在 computer vision、speech spectrogram、gameplay 與 medical imaging。Modern deep CNN 可能有約 5 到 1000 層，逐步把低階 input features 轉換為高階 features，最後變成 class scores。

常見 components：

| Component | 做什麼 | 硬體角度 |
|---|---|---|
| CONV | 在 feature maps 的局部區域套用 learned filters | 主要 MAC count 與豐富 reuse patterns |
| Activation | 套用 ReLU 等 pointwise nonlinearity | 通常是簡單 elementwise logic，常與 CONV/FC output 融合 |
| NORM | 正規化 activations，以穩定 training 或 inference behavior | 許多情況下 data movement/control 比 arithmetic 更重要 |
| POOL | 對 local spatial regions 做 downsampling | 降低後續 work；max/average reductions over local windows |
| FC | 將所有 input neurons 連到 output neurons | Flatten 後是 dense matrix-vector 或 matrix-matrix multiply |

投影片指出，在典型 CNN 中，convolutions 佔 overall computation 超過 90%，並主導 runtime 與 energy consumption。這就是為什麼本講大部分 DNN-component 內容都聚焦在 CONV。

**來源說明：** CNN applications 與 components 出現在 Lecture 02 slides 45-52；CONV 超過 90% computation 的說法在 slide 52。

### CONV Layer：直覺與精確意思

Convolutional layer 將 learned filter 滑過 input feature map。在每個 output position，layer 將 filter weights 與 filter 覆蓋到的 input values 相乘，然後加總。這個 sum 就是一個 output activation。

單通道 2-D convolution 中：

- input feature map shape：$H \times W$；
- filter shape：$R \times S$；
- output feature map shape：$P \times Q$。

對一個 output location $(p,q)$，計算是：

$$O_{p,q} = \sum_{r=0}^{R-1}\sum_{s=0}^{S-1} I_{Up+r,\ Uq+s}F_{r,s},$$

其中 $U$ 是 stride。$Up+r$ 選出 filter 覆蓋的 input row，$Uq+s$ 選出 input column。

Filter support $R \times S$ 也可稱為該 layer 中一個 output activation 的 **receptive field（感受野）**。若 $R=S=3$，每個 output activation 在還沒考慮 channels 前，需要 9 個 input values 與 9 個 weights。

### Worked Example：5-by-5 Input 與 3-by-3 Filter

Lecture 02 使用 $5 \times 5$ input、$3 \times 3$ filter、stride $U=1$ 的例子。沒有 padding 時，filter 可以從 rows $0,1,2$ 與 columns $0,1,2$ 開始，因此 output 是 $3 \times 3$。

投影片使用的 no-padding output-size 公式是：

$$P = \frac{H - R + U}{U}, \qquad Q = \frac{W - S + U}{U},$$

當除法剛好整除時可直接使用。更一般地，許多 frameworks 對 valid convolution 使用 floor-style rule：

$$P = \left\lfloor \frac{H - R}{U} \right\rfloor + 1, \qquad Q = \left\lfloor \frac{W - S}{U} \right\rfloor + 1.$$

對 $H=W=5$、$R=S=3$、$U=1$：

$$P=Q=\left\lfloor \frac{5-3}{1} \right\rfloor + 1 = 3.$$

每個 output activation 需要 $R \times S = 9$ 次 multiplication。整層有 $P \times Q = 9$ 個 output activations，所以這個單通道例子共有 $9 \times 9 = 81$ 次 multiplication。

**硬體意義：** 這 81 次 multiplication 不應該需要 81 次獨立 filter loads。如果 hardware 能把同一個 $3 \times 3$ filter 重用於 9 個 output positions，就能降低 costly memory access。同樣地，相鄰 output positions 的 sliding windows 會重疊，因此也能重用 input pixels。

### Stride

**Stride（步幅）**是 filter 在相鄰 output positions 之間移動的距離。Stride $U=1$ 評估每個合法 window；stride $U=2$ 跳過每隔一個 starting position；stride $U=3$ 跳更多。

同樣使用 $5 \times 5$ input 與 $3 \times 3$ filter：

- $U=1$ 給 $P=Q=3$，所以有 9 個 output activations；
- $U=2$ 給 $P=Q=2$，所以有 4 個 output activations；
- $U=3$ 給 $P=Q=1$，所以有 1 個 output activation。

Stride 因此是一種 downsampling mechanism。它減少 output size 與後續 computation，但也改變被 sample 的 input positions。從硬體角度看，較大的 stride 會減少 output partial sums 的數量，但也可能降低相鄰 windows 之間的 input overlap reuse。

**來源說明：** Lecture 02 slides 64-71 顯示 stride 1、stride 2、stride 3 的 output sizes，並說明 stride 大於 1 等價於對 stride-1 output feature map 做 downsampling。

### Zero Padding

沒有 padding 時，convolution 會縮小 spatial dimensions。$5 \times 5$ input 搭配 $3 \times 3$ filter 與 stride 1 會產生 $3 \times 3$ output。若多層反覆縮小，feature map 很快會塌縮。

**Zero padding（零填補）**在 input 邊界加上 zeros，讓 filter 可以放在邊緣附近。對 $3 \times 3$ filter，在四周各 padding 1 row/column，等效 input 變成 $7 \times 7$，再做 stride-1 valid convolution 會得到 $5 \times 5$ output。

對 symmetric padding $A_h$ rows 與 $A_w$ columns，常見公式是：

$$P = \left\lfloor \frac{H + 2A_h - R}{U} \right\rfloor + 1,\qquad
Q = \left\lfloor \frac{W + 2A_w - S}{U} \right\rfloor + 1.$$

對 odd filter sizes 與 stride $U=1$，選 $A_h=(R-1)/2$、$A_w=(S-1)/2$ 可以保持 spatial size 不變。

**重要 caveat：** 不同 frameworks 對 padding modes 的定義可能略有不同。Lecture 02 明確提到 PyTorch examples，也提醒 padding 不一定會被明寫，但常可由 feature-map sizes 推回來。

### Depth 與 Receptive Field Growth

CNN 越深，後面 layer 的一個 output activation 會依賴原始 input 中更大的區域。單層 $3 \times 3$ convolution 看到 $3 \times 3$ patch；第二層 $3 \times 3$ convolution 會組合第一層相鄰 outputs，而第一層每個 output 已經依賴 $3 \times 3$ patch。因此 effective receptive field 會隨 depth 增長。

這也用硬體語言解釋了常見 CNN 敘事：

- early layers 操作大的 spatial maps，通常 channels 較少；
- later layers 操作較小 spatial maps，但 channels 往往更多；
- deeper features 更 semantic，因為每個 activation 聚合了更大 input region 的資訊。

**來源說明：** Lecture 02 slides 47-48 與 75 連結 CNN depth、low-level/high-level features 與 receptive-field growth。

---

## Multichannel CONV 與 CNN Decoder Ring

### 從單通道到多通道

前面的 2-D 例子隱藏了 channel dimension。真實 CNN layer 通常有多個 input channels 與多個 output channels。

對 batched CONV layer：

| Tensor | Shape | 意義 |
|---|---|---|
| Input activations $I$ | $N \times C \times H \times W$ | $N$ 個 input feature maps，每個有 $C$ 個 channels |
| Filter weights $F$ | $M \times C \times R \times S$ | $M$ 個 filters，每個跨越全部 $C$ 個 input channels |
| Bias $B$ | $M$ | 每個 output channel 一個 bias |
| Output activations $O$ | $N \times M \times P \times Q$ | $N$ 個 output feature maps，每個有 $M$ 個 channels |

一個 filter 產生一個 output channel。若某層有 $M$ 個 output channels，就有 $M$ 個 filters。下一層的 input-channel count 通常就是前一層的 output-channel count。

### Decoder-Ring Variables

Lecture 02 定義了一組後續課程反覆使用的 notation：

| Symbol | Meaning |
|---|---|
| $N$ | batch size，input/output feature maps 的數量 |
| $C$ | input channels 數量 |
| $H$ | input feature-map height |
| $W$ | input feature-map width |
| $R$ | filter height |
| $S$ | filter width |
| $M$ | output channels 數量 / filters 數量 |
| $P$ | output feature-map height |
| $Q$ | output feature-map width |
| $U$ | convolution stride |

這些變數不只是 notation。它們是 loop bounds。當你看到 $N \times M \times P \times Q \times C \times R \times S$，你應該看到七個 loops，以及一個可以 reorder、tile、parallelize 的巨大 design space。

**來源說明：** Lecture 02 slides 76-80 介紹 many input channels、many output channels、batch size 與 CNN decoder ring。

### CONV Einsum

完整 dense CONV layer 可寫為：

$$O_{n,m,p,q} = B_m + I_{n,c,U p+r,U q+s} \times F_{m,c,r,s}.$$

對 $c,r,s$ 的 reduction 是 implicit。展開後是：

$$O_{n,m,p,q} = B_m +
\sum_{c=0}^{C-1}\sum_{r=0}^{R-1}\sum_{s=0}^{S-1}
I_{n,c,U p+r,U q+s}F_{m,c,r,s}.$$

Free output indices 是 $n,m,p,q$。Reduction indices 是 $c,r,s$。

總 multiplication count 是：

$$N \times M \times P \times Q \times C \times R \times S.$$

這個 equation 是本課最重要的式子之一。它說明 work 如何隨 batch size、channel count、output spatial size 與 filter size 成長，也顯示 reuse 來自哪裡：

- weight $F_{m,c,r,s}$ 可跨 $N \times P \times Q$ 個 output positions 重用；
- input activation 可被多個 filters $M$ 與重疊 spatial windows 使用；
- output partial sum $O_{n,m,p,q}$ 在完成前會被更新 $C \times R \times S$ 次。

### Naive Loop Nest

投影片給出 naive seven-loop implementation。可讀的 pseudocode 是：

```text
for n in [0, N):
  for m in [0, M):
    for q in [0, Q):
      for p in [0, P):
        O[n,m,p,q] = B[m]
        for c in [0, C):
          for r in [0, R):
            for s in [0, S):
              O[n,m,p,q] += I[n,c,U*p+r,U*q+s] * F[m,c,r,s]
        O[n,m,p,q] = Activation(O[n,m,p,q])
```

Loop nest 會強制一個 order；Einsum 不會。這個差異就是通往 mapping 與 dataflow 的橋。

**硬體意義：** 在這個 loop order 中，inner $c,r,s$ loops 執行時，output partial sum 很自然可以留在 local storage。這類似 output-stationary behavior。不同 loop order 則可能讓 weights stationary 或 input activations stationary。

---

## Fully Connected Layers

### FC 是 CONV 的 Special Case

Fully connected layer 將每個 input neuron 連到每個 output neuron。從 CONV 角度看，FC layer 是 filter 覆蓋整個 input feature map 的 convolution：

$$R=H,\qquad S=W.$$

沒有 spatial sliding。每個 output channel 由一個跨越所有 $C \times H \times W$ input values 的 filter 產生。

對單一 input example：

$$O_m = I_{c,h,w} \times F_{m,c,h,w},$$

也就是：

$$O_m = \sum_{c=0}^{C-1}\sum_{h=0}^{H-1}\sum_{w=0}^{W-1} I_{c,h,w}F_{m,c,h,w}.$$

### Flattening 成 Matrix-Vector Multiplication

$C,H,W$ 三個 ranks 可以 flatten 成一個 rank $CHW$：

$$chw = H W c + W h + w.$$

因此：

$$O_m = I_{chw} \times F_{m,chw}.$$

這就是 matrix-vector multiplication。Weight tensor 變成 shape $M \times CHW$ 的 matrix，input 變成長度 $CHW$ 的 vector，output 是長度 $M$ 的 vector。

例如 $C=2$、$H=2$、$W=2$、$M=3$，則 $CHW=8$。FC layer 用 $3 \times 8$ weight matrix 乘上 8-element input vector，產生 3-element output vector，總共 $M \times CHW = 24$ 次 multiplication。

**硬體意義：** Batch size 1 時，每個 weight 通常只服務一個 input example。即使 FC 的 loop structure 比 CONV 簡單，它仍可能因為 weight traffic 而 memory-bandwidth intensive。

### Batching 把 FC 變成 Matrix-Matrix Multiplication

若 batch size 是 $N$，input 是 $N \times CHW$，output 是 $N \times M$：

$$O_{n,m} = I_{n,chw} \times F_{m,chw}.$$

這等價於 matrix-matrix multiplication，reduction 發生在 $chw$。以典型 matrix multiplication notation：

$$C_{m,n} = A_{m,k} \times B_{k,n},$$

其中 $k$ 對應 $chw$。

Batching 會改善 weight reuse：同一個 weight matrix $F$ 可跨 $N$ 個 input examples 重用。這也是 large-batch FC layers 在 dense linear algebra hardware 上常能取得高 arithmetic utilization 的原因之一。

**來源說明：** Lecture 02 slides 85-102 推導 FC 是 CONV variant、flattening 成 matrix-vector multiplication，以及 batch size $N$ 讓 computation 變成 matrix-matrix multiplication。

---

## Worked Examples

### Example 1：CONV Shape 與 Work Count

假設某 layer 有：

- $N=1$；
- $C=3$；
- $H=W=32$；
- $R=S=3$；
- $M=16$；
- stride $U=1$；
- padding $A_h=A_w=1$。

Output size 是：

$$P=Q=\left\lfloor \frac{32 + 2 - 3}{1} \right\rfloor + 1 = 32.$$

Output tensor 有：

$$N \times M \times P \times Q = 1 \times 16 \times 32 \times 32 = 16{,}384$$

個 output activations。

每個 output activation 需要 reduce：

$$C \times R \times S = 3 \times 3 \times 3 = 27$$

個 products。

因此整層執行：

$$1 \times 16 \times 32 \times 32 \times 3 \times 3 \times 3 = 442{,}368$$

次 multiplication。

硬體意義：filter tensor 只有 $M \times C \times R \times S = 432$ 個 weights，但這些 weights 支撐 442,368 次 multiplication。好的 accelerator 會避免從昂貴記憶體反覆抓取同樣的 432 個 weights。

### Example 2：Output Partial Sum Lifetime

對同一層，單一 output value $O_{0,5,10,12}$ 不是由一次 multiplication 產生，而是累加 $C \times R \times S = 27$ 個 products：

$$O_{0,5,10,12} = B_5 + \sum_{c=0}^{2}\sum_{r=0}^{2}\sum_{s=0}^{2} I_{0,c,10+r,12+s}F_{5,c,r,s}.$$

在 27 個 products 全部累加前，這個 output 是 **partial sum（部分和）**。如果 partial sum 留在 register 或 local buffer，hardware 就能避免反覆 load/store；如果每個 product 後都 spill 到 DRAM，memory traffic 會暴增。

這就是後續講次非常重視 output-stationary dataflow 的原因。

### Example 3：FC Batch Reuse

假設 FC layer 有 $CHW=1024$ 個 input values 與 $M=100$ 個 outputs。對單一 input example：

$$1024 \times 100 = 102{,}400$$

次 multiplication，並使用 $1024 \times 100 = 102{,}400$ 個 weights。

若 batch size 是 $N=16$，同一個 weight matrix 可跨 16 個 examples 重用。Multiplication count 變成：

$$16 \times 1024 \times 100 = 1{,}638{,}400.$$

如果 weight matrix 或其 tiles 能留在 compute units 附近，weight traffic 可在 batch 上被攤銷。這說明 batch size 會改變 FC layer 的硬體行為，即使 mathematical layer 沒有改變。

---

## Key Equations and How to Read Them

### Matrix-Vector Einsum

$$Z_m = A_{k,m} \times B_k.$$

讀法：對每個 $m$，沿 $k$ 加總。Output index 是 $m$；reduction index 是 $k$。

### Best-Case Compute Intensity

$$\mathrm{CI}_{\text{best}} = \frac{K \times M}{K \times M + K + M}.$$

讀法：分子是 work；分母是在本講簡化 model 下最少需要 accessed 的 values。

### Achieved Compute Intensity for the Lecture's Loop Order

$$\mathrm{CI}_{\text{achieved}} = \frac{K \times M}{3KM - M + K}.$$

讀法：同一個 mathematical work 因為 implementation 反覆搬 partial sums，所以 CI 較低。

### No-Padding CONV Output Shape

$$P = \left\lfloor \frac{H-R}{U} \right\rfloor + 1,\qquad
Q = \left\lfloor \frac{W-S}{U} \right\rfloor + 1.$$

讀法：計算 filter 的合法 top-left positions 有幾個。

### Padded CONV Output Shape

$$P = \left\lfloor \frac{H + 2A_h - R}{U} \right\rfloor + 1,\qquad
Q = \left\lfloor \frac{W + 2A_w - S}{U} \right\rfloor + 1.$$

讀法：padding 先增加 effective input size，再套用同樣的 legal-window count。

### CONV Einsum

$$O_{n,m,p,q} = B_m +
\sum_{c=0}^{C-1}\sum_{r=0}^{R-1}\sum_{s=0}^{S-1}
I_{n,c,U p+r,U q+s}F_{m,c,r,s}.$$

讀法：每個 output activation 是一個 bias 加上沿 input channels 與 filter positions 的 reduction。

### CONV Work Count

$$\text{multiplications} = N \times M \times P \times Q \times C \times R \times S.$$

讀法：output activations 數量乘以每個 output activation 的 products 數量。

### FC Flattening

$$O_m = I_{c,h,w} \times F_{m,c,h,w}
\quad \Longrightarrow \quad
O_m = I_{chw} \times F_{m,chw}.$$

讀法：把 input-channel 與 spatial ranks flatten 成一個 reduction rank。

---

## Hardware Implications

### Data Reuse 是核心資源

本講的 CI examples 與 CONV equations 指向同一個硬體問題：當 values 能在 compute 附近被重用，效率就會提高。

CONV 有三種主要 reuse opportunities：

- **Weight reuse：** 一個 $F_{m,c,r,s}$ 可貢獻到許多 output positions 與 batch elements。
- **Input reuse：** 一個 input activation 可被多個 overlapping windows 與多個 filters 使用。
- **Output reuse：** 一個 partial sum 在成為 final output 前會被更新很多次。

Accelerator 的 mapping 決定哪一種 reuse 最容易被利用。

### Partial Sums 也是 Data

初學者常只把 input activations 與 weights 算成 memory traffic，忽略 partial sums。這會漏掉大問題。若 mapping 無法讓 partial sum 留在 local storage，它可能被反覆 read/write。本講前半段的 achieved-CI example 正是用來展示這點。

### More Parallelism 不一定等於 More Throughput

Roofline Model 解釋了原因。如果 implementation 落在 memory-bandwidth slope 上，更多 MAC lanes 可能只是閒置等資料。真正的修正可能是 tiling、不同 loop order、更多 local buffering、更好的 reuse、compression，或不同 dataflow。

### Shape Parameters 也是 Hardware Parameters

改變 $R,S,C,M,P,Q,N,U$ 不只是改變 neural-network accuracy，也會改變 loop bounds、buffer capacity needs、interconnect traffic 與 reuse。例如：

- 增加 $M$ 會增加 output channels 與 filter count；
- 增加 $C$ 會增加每個 output 的 reduction work；
- 增加 $R,S$ 會增加 receptive field 與每個 output 的 products；
- 增加 $U$ 會降低 output spatial size；
- 增加 $N$ 可能改善 weight reuse，但增加 activation 與 output storage。

### FC 與 CONV 對 Memory 的壓力不同

CONV 因為 sliding windows 有豐富 spatial reuse。FC flatten 後有較簡單的 dense matrix structure，但 batch size 1 時 weights 可能幾乎沒有 reuse。Batch size 變大後，FC 變成 matrix-matrix multiplication，可跨 examples 重用 weights。

---

## Common Misconceptions

### 誤解：Einsum 就是 Loop Nest

Einsum 定義 mathematical computation 與 iteration space。Loop nest 選擇 traversal order。同一個 Einsum 可以由多種 loop nests 實作，而 memory traffic 可能非常不同。

### 誤解：Compute Intensity 是 Layer 的固定屬性

Best-case CI 是某個 traffic model 下的 theoretical upper bound。Achieved CI 取決於 mapping、buffering 與 hardware behavior。同一層在不同 accelerator 上可以有不同 achieved CI。

### 誤解：Convolution 一定要數學上翻轉 Filter

許多 deep-learning libraries 實作的是 cross-correlation，但仍稱為 convolution。對本硬體課而言，重點不是 signal-processing convention，而是 sliding-window multiply-accumulate pattern 與其 data reuse。

### 誤解：Stride 只是在減少 Compute

Stride 會減少 output size 與 compute，但也改變 sampling 與 overlap reuse。較大的 stride 會減少 partial sums 數量，同時也可能減少相鄰 windows 共享 input data 的程度。

### 誤解：FC 和 CONV 完全不同

FC 是 $R=H$、$S=W$ 的 CONV special case。差異不是新的 arithmetic，而是 shape 與 reuse pattern。

### 誤解：Activation、NORM、POOL 在架構上不重要

它們通常不像 CONV 一樣主導 MAC count，但仍會影響 fusion、buffering、memory traffic、precision 與 control。好的 accelerator 常把它們放在 CONV/FC datapath 附近處理，以避免額外 memory round trips。

---

## 論文與來源橋接（Paper and Source Bridge）

### Local PDF Note

目前 repository 有 Batch Normalization 與 TeAAL 的 local PDFs，但沒有 Roofline CACM 2009 原始 paper。因此 BatchNorm 與 TeAAL 是 paper-verified bridges；Roofline 仍是 slide-anchored。

### Paper Bridge: Batch Normalization

**Bibliographic identity：** Sergey Ioffe 與 Christian Szegedy，*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*，ICML 2015。Local PDF：`papers/L02_BatchNorm_Ioffe_ICML2015.pdf`。

**Problem addressed：** Deep networks 難以訓練的一個原因，是某一層 input distribution 會隨著前面層的參數更新而改變。論文把這個現象稱為 **internal covariate shift**，並提出在 training 中穩定 layer inputs。

**Core idea：** 對每個 mini-batch，用 mini-batch mean 與 variance normalize activations，接著套用 learned scale 與 shift parameters。以論文記號來說，normalized activation 會再經過 $\gamma$ 與 $\beta$，讓 layer 仍能表示有用的 scale，必要時也能接近 identity。

**Relevance to this lecture：** L02 介紹 CONV 與 FC 之外的 DNN components。BatchNorm 是很好的例子：它的重點不只是 MAC count，而是 training behavior、inference-time computation、fusion opportunities，以及鄰近 layers 周圍的 memory traffic。

**Key claims used here：**

- 論文將 internal covariate shift 定義為 training 時 network parameters 改變而造成 network activations distribution 改變。來源：Section 2。
- Mini-batch transform 會計算 mini-batch mean/variance、normalize activation，然後套用 learned scale 與 shift。來源：Algorithm 1。
- Inference 時 BatchNorm 不應依賴當前 mini-batch；論文使用 population statistics 與 fixed linear transform。來源：Section 3 與 Algorithm 2。
- 對 convolutional layers，論文在 feature map 上跨 mini-batch examples 與 spatial locations 做 normalization。來源：Section 3.2。

**What students should remember：** BatchNorm 不只是另一個 arithmetic layer。它改變 training/inference 的區分，也常成為 hardware/compiler fusion target，因為反覆 materialize BN outputs 可能浪費 memory bandwidth。

**Limitations and assumptions：** BatchNorm 的 training benefit 是 paper-derived，會依 optimizer/model 而變。對本章 accelerator design 來說，重要的不是精確 training-speed claim，而是 non-CONV components 會影響 scheduling、fusion、storage 與 inference datapaths。

**Suggested insertion points：** 讀完 activation、normalization、pooling layers 討論後閱讀本 bridge。它說明為什麼「非 CONV layers」仍然有架構意義。

### Source Bridge: TeAAL and HiFiber

**Bibliographic identity：** *TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators*，MICRO 2023。Local PDF：`papers/TeAAL.pdf`。

**Problem addressed：** Accelerator design 需要精簡的 workload descriptions 與精確的 implementation descriptions。若沒有 separation of concerns，就很難系統化比較 mappings、formats 與 hardware bindings。

**Core idea：** 用 Einsums 描述 tensor algebra workloads，並分開描述 mapping、format 與 binding。

**Relevance to this lecture：** L02 使用 TeAAL 引入本課 methodology，並說明為什麼 Einsum 是 workload modeling tools 的 input format。

**Key claims used here：**

- TeAAL 將 computations 表達為 extended Einsums，並把 iteration order 留給 mapping。來源：paper Section 2.2。
- Mapping 包含 loop order、rank partitioning、work scheduling。來源：paper Section 2.3。
- TeAAL specifications 包含 computation、mapping、format、architecture、binding information，可 lowering 成 performance models。來源：paper Sections 3-4。
- Lecture 的 describe architecture、develop workload、evaluate、compare、optimize design flow，是這種 specification discipline 的課程版表達。來源：Lecture 02 slides 4-8, 21, 44。

**What students should remember：** TeAAL 不是單純軟體細節，而是用來訓練架構思考的 discipline：先指定 computation，再探索 mappings 與 hardware。

### Source Bridge: Roofline Model

**Bibliographic identity：** Lecture 02 slides 引用 Williams, Waterman, and Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, Communications of the ACM, 2009。目前 repository 沒有 local PDF。

**Problem addressed：** Designers 需要簡單方法判斷 performance 是受限於 memory bandwidth 還是 compute throughput。

**Core idea：** 將 achievable throughput 畫成 compute intensity 的函數。上限由 bandwidth slope 或 compute roof 決定。

**Relevance to this lecture：** 這個 model 說明為什麼 CI 重要：低 CI implementation 不能只靠更多 compute lanes 提速，除非降低 memory traffic 或增加 bandwidth。

**Key claims used here：**

- Roofline 用 memory bandwidth、compute parallelism 與 compute intensity visualizes throughput（Lecture 02 slides 42-43）。
- 當 memory-bound 時，增加 lanes 不一定增加 throughput（Lecture 02 slide 43）。

**What students should remember：** Roofline 不只是一張圖，而是 design diagnostic：它告訴你應該先優化哪個 resource。

---

## 連結

### 與 L01 的連結

L01 用 AI demand、energy cost 與 specialized architectures 的需求來 motivate DNN accelerators。L02 給出讓這個 motivation 變精確的詞彙：tensors、Einsums、compute intensity、memory traffic 與 CNN layer shapes。

### 與 L03 和 L04 的連結

L03 和 L04 會延伸 tensor-algebra language。本章的 matrix-vector 與 CONV Einsums 是最早例子。後續 lectures 會把類似 reasoning 套用到更複雜的 operations，包括 transformer 與 attention。

### 與 L05 和 L06 的連結

L05 和 L06 關注 mapping、dataflow 與 partitioning。本講的 seven-loop CONV nest 就是要被 mapped 的對象。Output-stationary、weight-stationary、input-stationary dataflows 是對同一問題的不同回答：哪個 tensor 應該留在 PEs 附近？

### 與 L07-L10 的連結

Dense work count $NMPQCRS$ 會成為 sparsity 的 baseline。Sparse accelerators 試圖跳過 weights 或 activations 中的 zeros，但必須付出 metadata、irregular traversal 與 load balancing 的成本。

### 與 L11-L13 的連結

FC-as-matrix-multiply 的觀點會在 advanced technologies 與 precision 中變重要。Matrix-vector 與 matrix-matrix kernels 常被用來示範 reduced precision 與 compute-in-memory ideas。

---

## 獨立學習指南

### 沒有影片時該怎麼讀

1. 先確認你能用文字解釋 $Z_m = A_{k,m} \times B_k$。指出 $m$ 是 free，$k$ 是 reduced。
2. 重新推導 best-case 與 achieved CI formulas。不要背最後數字，要理解每個 data-movement term 為什麼出現。
3. 畫一個 $5 \times 5$ input 與 $3 \times 3$ filter。分別數 stride 1 與 stride 2 有幾個合法 filter positions。
4. 不要先硬背 CNN decoder-ring variables；先理解每個 variable 屬於哪個 tensor。
5. 寫出 CONV Einsum，指出 output indices $n,m,p,q$ 與 reduction indices $c,r,s$。
6. 說明 FC 是 $R=H,S=W$ 的 CONV，然後把 $C,H,W$ flatten 成 $CHW$。
7. 對每個 equation 問三件事：哪些 values 被重用？partial sums 可以住在哪裡？如果 spill 到 DRAM 會發生什麼？

### 自我檢核問題

1. 在 $Z_m = A_{k,m} \times B_k$ 中，為什麼 $k$ 是 reduction index，而 $m$ 是 free index？
2. 為什麼 matrix-vector example 的 iteration-space size 是 $K \times M$？
3. Best-case CI denominator 中有哪些 data movement terms？為什麼？
4. 為什麼 achieved-CI example 會包含 $(K-1) \times M$ 次 $Z_m$ loads？
5. 如果 implementation 落在 Roofline Model 的 memory-bandwidth slope 上，代表什麼？
6. 對 $7 \times 7$ input、$3 \times 3$ filter、stride $1$、no padding，$P$ 與 $Q$ 是多少？
7. 對 $5 \times 5$ input、$3 \times 3$ filter、stride $1$、padding $1$，為什麼 output 仍是 $5 \times 5$？
8. 在 CONV Einsum 中，哪些 indices 定義 output tensor shape？
9. 為什麼每個 CONV output activation 需要 $C \times R \times S$ 個 products？
10. 為什麼 batching 會讓 FC 更接近 matrix-matrix multiplication？

### 練習

1. **Conceptual：** 用 Roofline vocabulary 解釋為什麼「更多 parallelism」不是完整的 accelerator optimization strategy。
2. **Calculation：** 對 $Z_m = A_{k,m} \times B_k$，令 $K=64$、$M=32$。用本講公式計算 $\mathrm{CI}_{\text{best}}$ 與 $\mathrm{CI}_{\text{achieved}}$。
3. **Shape reasoning：** 某 CONV layer 有 $N=4$、$C=16$、$H=W=28$、$R=S=3$、$M=32$、stride $U=1$、padding $1$。計算 $P$、$Q$、output tensor size 與 multiplication count。
4. **Data reuse：** 對 exercise 3 的 layer，分別列出 weights、inputs 與 output partial sums 的一個 reuse opportunity。
5. **Design tradeoff：** 假設 mapping 讓 weights stationary，但常常 spill output partial sums。哪個 traffic term 可能變大？這會如何影響 CI？
6. **FC bridge：** 對 $C=8$、$H=W=4$、$M=10$、batch size $N=1$，寫出 FC matrix-vector shape；再對 $N=16$ 寫出 matrix-matrix multiplication shape。
7. **Source bridge：** 閱讀 Lecture 02 slides 42-43。用自己的話說明：即使 CI 足以 compute-bound，為什麼 workload 仍可能離 roof 很遠？

---

## 關鍵詞彙

| 詞彙 | 定義 |
|---|---|
| Tensor（張量） | 多維 values array。在本課中用來描述 activations、weights、outputs 與其他 DNN data。 |
| Rank（秩） | Tensor 的一個具名維度，例如 $C$、$H$ 或 $W$。 |
| Rank shape | 某個 rank 上的 element 數量。 |
| Tensor size | 所有 rank shapes 的乘積。 |
| Einsum | Tensor algebra 的精簡記法；指定 computation，但不固定 loop order。 |
| Free index | 出現在 output 中的 index；用來命名 output coordinates。 |
| Reduction index | 出現在右側但不在 output 中的 index；計算會沿它加總。 |
| Iteration space（迭代空間） | Einsum 中所有合法 index values 的 Cartesian product；其 size 代表 loop work。 |
| Mapping（映射） | Iteration space 的 traversal policy，包含 loop order、tiling 與 parallelism。 |
| Format（格式） | Data representation，例如 dense 或 sparse encoding。 |
| Binding（綁定） | 將 mapped computation 與 data 指派到 hardware resources。 |
| Compute intensity (CI)（運算強度） | 每 accessed value 可換得的 multiplications 數；反映 data movement 的 amortization。 |
| Best-case CI | 在 minimum-traffic assumption 下的 theoretical upper-bound CI。 |
| Achieved CI | 特定 implementation、mapping 與 storage behavior 產生的 CI。 |
| Roofline Model（屋頂線模型） | 比較 compute intensity、memory bandwidth 與 peak compute throughput 的 throughput model。 |
| CNN | Convolutional neural network，主要由 convolutional layers 與 auxiliary layers 組成。 |
| CONV layer（卷積層） | 在 local input regions 上套用 learned filters，並加總 products 產生 output feature maps。 |
| Feature map (fmap)（特徵圖） | Activation tensor，常被視為多個 spatial maps channels。 |
| Filter / kernel（濾波器／卷積核） | CONV layer 使用的 learned weight tensor。 |
| Receptive field（感受野） | 對一個 output activation 有貢獻的 input region。 |
| Stride ($U$)（步幅） | 相鄰 filter positions 之間的 step size。 |
| Zero padding（零填補） | 在 input boundaries 周圍加 zeros，以控制 output size。 |
| Channel ($C$ or $M$)（通道） | Feature dimension；本課 CONV notation 中 $C$ 是 input channels，$M$ 是 output channels。 |
| Batch size ($N$)（批次大小） | 一起處理的 examples 數量。 |
| Partial sum (psum)（部分和） | Reduction products 全部加完前的 intermediate output accumulation。 |
| Activation | Pointwise nonlinear operation，常接在 CONV 或 FC 後。 |
| NORM | Normalization layer，用來控制 activation statistics。 |
| POOL | 對 local spatial regions 做 downsampling 的 layer。 |
| FC layer（全連接層） | Fully connected layer；等價於 $R=H$、$S=W$ 的 CONV，flatten 後是 matrix-vector/matrix-matrix multiplication。 |

---

## 重點回顧

1. L02 不只是 CNN overview；它建立了整門課會使用的 tensor-algebra language。
2. Einsum 指定 computation；mapping 指定 traversal。混淆兩者會遮蔽真正的 accelerator design space。
3. Compute intensity 將 workload structure 連到 memory traffic 與 throughput。
4. Best-case CI 與 achieved CI 的差距，是未被利用的 reuse 的具體量化。
5. Roofline reasoning 告訴架構師應該先優化 bandwidth/reuse，還是 compute parallelism。
6. CONV 主導典型 CNN computation，核心 loop structure 是 $N,M,P,Q,C,R,S$。
7. Stride 與 padding 不是表面上的 neural-network details；它們改變 output shape、work、reuse 與 buffer needs。
8. FC 是 CONV 的 special case，batched 後成為 matrix-matrix multiplication。
9. Partial sums 是一級硬體資料；無法保持 local 可能讓 traffic 主導成本。
10. 本講為 dataflow、partitioning、sparsity、precision 與 advanced accelerator technologies 打地基。

---

## 附錄

### Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L02-1-L02-2 | Title and outline | 用來建立本講兩段式結構 |
| L02-3-L02-8 | 主要敘事：從 Workload 到 Hardware | TeAAL methodology 與 separation of concerns |
| L02-9-L02-20 | Tensors、Ranks 與 Einsums | 補充 free/reduction index 解釋 |
| L02-21-L02-26 | Compute Intensity 與 Roofline Reasoning | Best-case CI derivation |
| L02-27-L02-40 | Compute Intensity 與 Roofline Reasoning | Mapping-dependent achieved traffic 與 stationarity |
| L02-41 | Compute Intensity 與 Roofline Reasoning | $K=250$, $M=100$ 的數值 CI example |
| L02-42-L02-43 | Compute Intensity 與 Roofline Reasoning；Paper and Source Bridge | Roofline Model |
| L02-44 | 主要敘事 | Methodology recap |
| L02-45-L02-52 | DNN Workloads 與 CNN Components | CNN applications、layer types、CONV dominance |
| L02-53-L02-64 | CONV Layer | Single-channel convolution 與 stride-1 worked example |
| L02-65-L02-71 | CONV Layer | Stride examples 與 downsampling interpretation |
| L02-72-L02-74 | CONV Layer | Zero padding 與 framework caveats |
| L02-75 | CONV Layer | Depth 與 receptive-field growth |
| L02-76-L02-80 | Multichannel CONV 與 Decoder Ring | $N,C,H,W,R,S,M,P,Q,U$ |
| L02-81-L02-83 | CONV Einsum and Naive Loop Nest | 補充 output/reduction index interpretation |
| L02-84-L02-91 | Fully Connected Layers | FC as full-input CONV and flattening |
| L02-92-L02-102 | Fully Connected Layers | Matrix-vector 與 matrix-matrix views |

## 來源註記（Source Notes）

- 本章的 lecture ordering 與 terminology 依據 Lecture 02 slides。
- TeAAL methodology 與 separation-of-concerns 討論依據 Lecture 02 slides 3-8, 21, 27-28, and 44，以及本地 `papers/TeAAL.pdf`，尤其是 Sections 2.2、2.3、3-4。
- Tensor、rank、Einsum 與 ODE 說明依據 Lecture 02 slides 9-20。
- Compute-intensity formulas 與 $K=250, M=100$ 數值依據 Lecture 02 slides 23-26 and 38-41。
- Roofline discussion 依據 Lecture 02 slides 42-43，這些 slides 引用 Williams, Waterman, and Patterson, CACM 2009。
- CNN component taxonomy 與 CONV dominance claim 依據 Lecture 02 slides 45-52。
- CONV shape、stride、padding、decoder-ring 與 Einsum material 依據 Lecture 02 slides 53-83。
- FC-to-matrix-vector 與 FC-to-matrix-matrix derivations 依據 Lecture 02 slides 85-102。
- 未直接出現在 slides 的 worked examples 是根據 slide equations 製作的原創教學例子。

## 不確定性註記（Uncertainty Notes）

- 本章根據 slides 與 source anchors 重建可能的口頭說明；實際 live lecture 可能對某些例子的強調不同。
- 本章使用 standard floor-form convolution output formulas 作為背景說明。Lecture 02 在 slides 64, 69, 70, and 74 使用較簡化的 exact-division formula。
- 本章沒有嵌入 slide images。Repo 中現有 local assets 可能仍包含 slide-derived images；公開發布前應由 repository-level copyright audit 決定保留、移除或替換。
