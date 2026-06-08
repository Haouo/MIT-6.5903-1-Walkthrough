# L05 - 映射：資料流（Mapping: Dataflows）

> **課程：** 6.5930/1 - 深度學習硬體架構（Hardware Architectures for Deep Learning）
> **講師：** Joel Emer 與 Vivienne Sze（MIT EECS）
> **講授日期：** 2026 年 2 月 17 日。**投影片：** 111 頁。**來源：** [`Lecture/L05-Mapping.pdf`](../../Lecture/L05-Mapping.pdf)
>
> 本章依據公開投影片重建缺少的課堂講解。它不是投影片摘要，而是給無法觀看 lecture video 的讀者使用的自學章節。文中以投影片頁碼作為來源錨點，方便對照原始順序。

---

## 一句話總結（TL;DR）

映射（mapping）是一組選擇，用來把抽象的張量運算變成某個加速器上實際可執行的排程。L03 與 L04 告訴我們「要算什麼」：也就是 Einsum。L05 問的是另一件事：迴圈應該用什麼順序跑、資料應該住在哪一層記憶體、哪一種運算元應該停在 MAC 單元附近。

本講的核心概念是**駐留性（stationarity）**。一個稠密卷積會反覆碰到三種資料：權重（weights）、輸入激活值（input activations）、部分和（partial sums）。資料流（dataflow）選擇其中一種資料留在低成本本地儲存中，讓其他資料流過。**輸出駐留（Output Stationary, OS）**讓部分和留在本地。**權重駐留（Weight Stationary, WS）**讓權重留在本地。**輸入駐留（Input Stationary, IS）**讓輸入激活值留在本地。三者計算同一個數學結果，但造成的記憶體流量、互連壓力、緩衝區需求與 PE 利用率都不同。

投影片的量化動機很直接：在本講使用的 Eyeriss 65 nm 能耗表中，一次 DRAM 存取約是一個 MAC 的 200 倍能耗，而暫存器檔（register file, RF）存取約是 1 倍。因此，好的資料流主要不是為了減少算術，而是把昂貴的資料搬移換成本地再利用。

---

## 本講要解決的問題

Einsum 是數學合約。對一維卷積而言：

```text
O[q] = sum_s I[q+s] * F[s]
```

這個式子說明哪些乘積必須累加到每個輸出中。它沒有說加速器應該先把某個輸出的所有 filter tap 算完，還是先拿同一個 filter weight 掃過所有輸出位置，或先以輸入位置為外層迴圈。這些選擇不改變最後的數值結果，卻會改變哪些資料從 DRAM 讀出、哪些資料留在 RF、哪些資料必須在 PE 之間通訊。

L05 要解決的是映射中的**資料流**問題，也就是 loop order 的問題：

> 給定一個固定的 spatial accelerator 與固定的 DNN layer，我們要如何安排 loop nest，讓經常被重用的資料停留在 MAC 附近？

這個問題重要，因為 DNN accelerator 往往不是被乘法本身限制，而是被供應乘法器的資料搬移限制。投影片 L05-10 到 L05-19 從 MAC 周圍的讀寫次數建立這個論點。最壞情況下，AlexNet 有 7.24 億次 MAC，若每個 MAC 的四次讀寫都到 DRAM，會需要 28.96 億次 DRAM 存取；投影片也給出最佳情況可降到 6,100 萬次 DRAM 存取。這些數字是針對 AlexNet 的投影片主張，應作為再利用動機，而不是普世保證。

---

## 為什麼本講重要

初學者常有一個太簡化的模型：「PE 越多就越快、越好。」L05 修正這個模型。大型 PE array 只有在能被有效餵資料時才有價值。如果每個 MAC 都從 DRAM 讀權重、輸入、部分和，再把更新後的部分和寫回 DRAM，那麼陣列主要消耗能量在搬資料，而不是算術。

硬體架構師看這個問題時，會問：

- **能耗（energy）：** 存取發生在 DRAM、global buffer、NoC，還是 RF？
- **頻寬（bandwidth）：** 記憶體系統能不能快到讓 PE 不空等？
- **延遲（latency）：** 排程是否產生很長的 reduction 或序列化搬移？
- **面積（area）：** 這個 dataflow 需要多少 RF、global buffer 與 interconnect？
- **利用率（utilization）：** layer 維度是否配得上 array 暴露出的 parallel ranks？
- **可程式性（programmability）：** accelerator 能不能跨 layer 改 mapping，還是被鎖在單一 dataflow？

這就是為什麼 mapping 位於 TeAAL pyramid 的 algorithm 與 architecture 之間。Algorithm 說要算什麼；architecture 提供儲存、運算與通訊資源；mapping 則試圖讓 workload 與硬體資源互相配合。

---

## 先備知識與心智模型

閱讀本章前，應帶著三個前面講次的概念。

第一，L02 介紹過 convolution 的 rank。對稠密 2-D convolution：

```text
O[m][p][q] = sum_c sum_r sum_s I[c][p+r][q+s] * F[m][c][r][s]
```

其中 `m` 是輸出通道，`c` 是輸入通道，`p,q` 是輸出空間位置，`r,s` 是 filter 空間位置。許多 dataflow 本質上就是把這些 rank 對應的 loop 重新排列。

第二，L03 與 L04 的 Einsum 固定算術，但不固定遍歷順序。同一個方程式可以用許多合法的 loop nest 評估，因為 reduction ranks 上的乘加可以用不同順序排程，只要最終累加結果正確。

第三，L01 建立了記憶體階層的重要性。L05 補上具體正規化成本：ALU 1x、RF 1x、相鄰 PE 經 NoC 2x、global buffer 6x、DRAM 200x。這些值在本章歸因於 Chen、Emer、Sze 的 Eyeriss 論文（ISCA 2016）Table IV，並對應投影片 L05-23 到 L05-24。

本講可以用一個 MAC 單元來想像：

```text
weight  ----\
input   ----- MAC ----> updated partial sum
psum    ----/
```

每次 dense convolution MAC 都消耗一個 weight、一個 input activation、一個舊 partial sum，並產生更新後的 partial sum。Dataflow 問的是：哪一種值應該留在 MAC 附近，讓接下來多次 MAC 可以重用它？

---

## 學習目標

讀完本章後，你應該能夠：

- 定義 mapping 的五個面向：partitioning、dataflow、data placement、compute placement、partition sizing。
- 說明 local reuse 與 local accumulation 為什麼能降低 DRAM traffic。
- 從 loop nest 判斷它是 OS、WS 還是 IS。
- 解釋 outer loops 如何編碼 stationarity。
- 用小型 convolution 範例算出：相同 MAC 數可以有不同 memory traffic。
- 說明 spatial dataflow 為什麼隱含具體 physical interconnect pattern。
- 解釋為什麼 L05 的 energy comparison 不會導出單一普世贏家。
- 將 LoopTree 讀成 loop order 加 storage placement 的表示法。
- 區分投影片直接陳述、論文推導、標準背景解釋與本章教學詮釋。

---

## 1. 映射有五個面向

**來源錨點：** 投影片 L05-2 到 L05-7。

Mapping 不是單一決策。投影片 L05-3 到 L05-5 將它分成五個面向。

| 面向 | 選擇什麼 | 對 loop nest 的影響 | 硬體意義 |
|---|---|---|---|
| Partitioning（切分） | 如何把 tensor 分成 tile | 增加 partitioned ranks 與 tile loops | 決定資料能否放入 buffer 或分散到 PEs |
| Dataflow（資料流） | loop 的順序 | 重新排列 loop ranks | 決定哪個 operand 變化最慢、能停在本地 |
| Data placement（資料放置） | 每個 tensor tile 放在哪個 memory level | 增加 storage annotations | 控制存取命中 RF、global buffer 或 DRAM |
| Compute placement（運算放置） | 哪些 loop 是 temporal，哪些是 parallel | 使用一般 loop 或 `parallel_for` | 控制 PE utilization 與 spatial sharing |
| Partition sizing（切分大小） | 具體 tile size 與 loop bounds | 設定數值範圍 | 在 capacity、bandwidth、parallelism 之間取捨 |

本講聚焦在 **dataflow**。這個焦點刻意縮小：真實 mapping 需要五個面向全部一起決定，但 loop order 是第一個槓桿，因為它決定自然的 reuse pattern。L06 會再深入 partitioning。

### 直覺

想像一個 tensor tile 被放入很小的 RF。若接下來許多 MAC 都用同一個 tile，載入它很值得。若下一個 MAC 馬上需要不同 tile，RF 幫助就很有限。Dataflow 的任務就是讓接下來許多 MAC 盡量重用相同資料。

### 精確意義

在本講中，**dataflow** 指 computation 的 loop order，也包括哪些 ranks 被 parallelize。它不只是圖上箭頭的幾何方向，而是透過 loop nest 表達的 compute、storage、communication policy。

### 常見誤解

**誤解：** Mapping 跟 hardware architecture 是同一件事。

**修正：** Architecture 提供資源：PE、buffer、NoC link、memory port。Mapping 決定某個 computation 如何使用這些資源。同一個硬體有時能跑多種 mapping；同一種 mapping 也可能在不同硬體上有不同成本。

---

## 2. 為什麼記憶體存取是瓶頸

**來源錨點：** 投影片 L05-8 到 L05-27。

對一維卷積：

```text
O[q] = sum_s I[q+s] * F[s]
```

總共有 `Q * S` 次 MAC。最壞情況下，每次 MAC 需要：

- 讀一個 filter weight；
- 讀一個 input activation；
- 讀一個舊 partial sum；
- 寫一個更新後的 partial sum。

也就是每個 MAC 四次 memory operation。投影片 L05-11 將這個最壞情況套到 AlexNet：7.24 億次 MAC 若全部在 DRAM 讀寫，會變成 28.96 億次 DRAM access。

重點不是真實 accelerator 一定會這樣做，而是 memory hierarchy 要避免的 baseline 正是這種情況。

### 兩個機會

投影片 L05-13 到 L05-19 指出兩個機會。

**Data reuse（資料再利用）：** 一個值取一次，供多個 MAC 使用。CNN 有幾種結構性 reuse：

| Reuse 類型 | 出現位置 | 被重用的資料 | 為什麼會發生 |
|---|---|---|---|
| Convolutional reuse（卷積再利用） | CONV | input activations 與 filter weights | sliding windows 彼此重疊 |
| Fmap reuse（feature-map reuse） | CONV 與 FC | input activations | 一個 activation 會乘上多個 filters |
| Filter reuse（濾波器再利用） | CONV 與 FC，batch > 1 | filter weights | 同一個 weight 用於多個 input examples |

**Local accumulation（本地累加）：** partial sum 留在本地儲存直到成為 final output。若沒有 local accumulation，每次中間更新都可能變成 DRAM read/write。若在 RF 或 local buffer 累加，只有最後完整 output 需要寫出。

### Computational intensity

投影片 L05-12 問：額外 local memory level 什麼時候有幫助？答案是：

```text
computational intensity > 1
```

在本講可以讀成：

```text
某個被取入的值服務的 MAC 數 / 被取入的相異值數 > 1
```

如果一個 fetched word 只服務一次 MAC，把它放本地幫助不大。如果一個 fetched word 服務 10、100、甚至 500 次 MAC，本地儲存就能大幅降低 DRAM traffic。投影片 L05-17 指出，在有利 reuse 下，AlexNet CONV layers 的 filter/fmap DRAM reads 最多可降 500x。

### 硬體意義

Memory hierarchy 不是事後加上的 cache 而已。它的容量、頻寬、位置決定哪些 reuse pattern 能被利用。L05 的 spatial architecture 模型中，DRAM 餵給 100-500 kB global buffer，再餵給 200-1000 個 PE 的 array，每個 PE 有 0.5-1.0 kB RF。這些容量是投影片範例，不是所有 accelerator 的必要規格。

---

## 3. 駐留性：讀懂 dataflow 的規則

**來源錨點：** 投影片 L05-28 到 L05-35，以及 L05-49 到 L05-55。

本講最有用的閱讀規則是：

> 哪個 tensor 的識別 ranks 被放在 outer loops，哪個 tensor 就變化最慢，也最具 stationarity。

比較兩個合法的一維卷積 loop nest。

```text
# Output Stationary
for q in [0, Q):
  for s in [0, S):
    o[q] += i[q+s] * f[s]
```

對固定外層 `q`，同一個 output partial sum `o[q]` 會針對所有 `s` 被更新。Output 留在本地。

```text
# Weight Stationary
for s in [0, S):
  for q in [0, Q):
    o[q] += i[q+s] * f[s]
```

對固定外層 `s`，同一個 weight `f[s]` 會用於所有 `q`。Weight 留在本地。

Loop nest 做的是同一組 MAC；它沒有做同一種 memory traffic。

### 重要細節

Stationary 不代表「永遠不動」。它代表某個值在有用時間窗內保持 resident。WS accelerator 最終還是要載入新 weights；OS accelerator 最終還是要寫出 final outputs。差別在於移動之前發生了多少 reuse。

---

## 4. 範例演練：相同 MAC，不同 traffic

**教學詮釋：** 本例為本章原創，用 Eyeriss Table IV 的能耗比值作為背景，但 tiny tensor 是為教學而建構。

使用 valid 1-D convolution：

```text
W = 5 inputs
S = 3 filter taps
Q = W - S + 1 = 3 outputs
```

總共有 `Q * S = 9` 次 MAC。輸出為：

```text
O[0] = I[0]F[0] + I[1]F[1] + I[2]F[2]
O[1] = I[1]F[0] + I[2]F[1] + I[3]F[2]
O[2] = I[2]F[0] + I[3]F[1] + I[4]F[2]
```

先只看 partial-sum operand。

| 策略 | Psum reads | Psum writes | 約略 psum energy |
|---|---:|---:|---:|
| No local reuse | 9 次 DRAM | 9 次 DRAM | `(9 + 9) * 200 = 3600` |
| Output stationary | 9 次 RF | 6 次 RF 中間寫 + 3 次 DRAM 最終寫 | `(9 + 6) * 1 + 3 * 200 = 615` |

算術完全相同，partial-sum traffic 完全不同。OS 避免把 intermediate partial sums 反覆送到 DRAM。這就是投影片說 memory access is the bottleneck 的具體機制。

### 這個例子沒有主張什麼

它沒有計算 weight 或 activation traffic。完整模型必須計算三種 operand 在所有 memory levels 的 access。L13 會用 ISL 形式化這件事。本例刻意很小，是為了讓 stationarity 機制清楚可見。

---

## 5. 輸出駐留（Output Stationary, OS）

**來源錨點：** 投影片 L05-28 到 L05-48。

**定義：** 在 Output Stationary（OS）中，output partial sums 留在本地，weights 與 input activations 則流過。

對一維卷積：

```text
for q in [0, Q):
  for s in [0, S):
    o[q] += i[q+s] * f[s]
```

對每個 `q`，accelerator 先完成該 output 的所有 `S` 個 contribution，再移到下一個 output。如果 partial sum 放得進 PE 的 RF，中間 psum read/write 就會是 local access。

對 dense 2-D convolution，投影片給的 loop nest 是：

```text
for p in [0, P):
  for q in [0, Q):
    for r in [0, R):
      for s in [0, S):
        parallel-for c in [0, C):
          parallel-for m in [0, M):
            o[m][p][q] += i[c][p+r][q+s] * f[m][c][r][s]
```

Output spatial ranks `p,q` 是外層 temporal loops。Channel ranks `c,m` 在投影片範例中被 parallelize。硬體意義是，多個 PE 針對 output channels 與 input channels 貢獻運算，同時相關 output partial sums 在本地累加。

### 直覺

OS 是「先把這個 output 做完，再把它趕出本地」的策略。它有吸引力，因為 partial sums 天生 write-heavy：每次 MAC 都會更新一個 partial sum。如果這些更新都 spill 到 DRAM，能耗會很差。

### 硬體意義

投影片 L05-29 到 L05-31 引用 ShiDianNao 與 KU Leuven 等 OS 例子。共同模式是 activations 與 weights 送進 array，而 partial sums 在 PE 內或 PE 附近累加。這偏好能 broadcast/multicast operands，且能提供低成本 local accumulation 的硬體。

### 常見誤解

**誤解：** OS 代表 outputs 完全不移動。

**修正：** Final outputs 還是要離開 PE 或 local buffer。OS 減少的是 intermediate partial sums 的搬移，不是取消 final output storage。

---

## 6. 權重駐留（Weight Stationary, WS）

**來源錨點：** 投影片 L05-48 到 L05-84。

**定義：** 在 Weight Stationary（WS）中，weights 留在本地，input activations 與 partial sums 移動。

對一維卷積：

```text
for s in [0, S):
  for q in [0, Q):
    o[q] += i[q+s] * f[s]
```

對固定 `s`，同一個 weight `f[s]` 用於所有 output positions `q`。如果 weight 存在 PE RF，一次 weight fetch 就能服務多次 MAC。

投影片 L05-54 與 L05-55 也展示 parallel WS：

```text
parallel_for s in [0, S):
  for q in [0, Q):
    o[q] += i[q+s] * f[s]
```

此時不同 PE 持有不同 weights。Activations 被 multicast 到那些 PEs，而 resulting partial sums 必須被累加。

### NVDLA 作為 WS 例子

投影片 L05-57 到 L05-80 使用簡化的 NVDLA 例子。投影片描述的 PE array 組織為 `M * C` 個 MAC，其中 `M` 是 output channels，`C` 是 input channels。簡化 loop nest 為：

```text
for r in [0, R):
  for s in [0, S):
    for p in [0, P):
      for q in [0, Q):
        parallel-for m in [0, M):
          parallel-for c in [0, C):
            o[m][p][q] += i[c][p+r][q+s] * f[m][c][r][s]
```

最上層 loops 是 `r,s`，也就是 filter spatial ranks，因此它是 weight stationary。對固定 filter position，硬體處理許多 input/output positions，同時讓相關 weights 保持 resident。

### 硬體意義

當每個 weight 被大量重用時，WS 能降低 weight read energy。它也要求與 OS 不同的實體路徑。Activations 常需要 broadcast 或 multicast 到持有 weights 的 PEs，partial sums 可能需要 spatial accumulation 或透過 output path 移動。

投影片 L05-57 也暴露 utilization 問題：如果 layer 的 `M` 與 `C` 維度不符合 array shape，一些 MAC 可能 idle。這是 mapping 與 architecture 的交互作用，不是 convolution 數學本身的性質。

### 常見誤解

**誤解：** WS 一定最好，因為 weights 是 model parameters，理應重用。

**修正：** WS 在 weights 重用夠多、且放得進 local storage 時很強。但若 activation 或 partial-sum traffic 主導、layer shape 讓 array utilization 不佳，或 reduction traffic 很昂貴，WS 就未必最好。

---

## 7. 輸入駐留（Input Stationary, IS）

**來源錨點：** 投影片 L05-85 到 L05-90。

**定義：** 在 Input Stationary（IS）中，input activations 留在本地，weights 與 partial sums 移動。

一維卷積方程式使用 compound input index `q+s`，所以 OS 形式中沒有顯式 input loop：

```text
for q in [0, Q):
  for s in [0, S):
    w = q + s
    o[q] += i[w] * f[s]
```

若要明確表達 input stationarity，要換變數：

```text
w = q + s
q = w - s
```

然後改以 raw input position `w` 迭代：

```text
for w in [0, W):
  for s in [0, S):
    q = w - s
    if 0 <= q < Q:
      o[q] += i[w] * f[s]
```

這個 guard 很重要。沒有它，在 convolution 邊界附近會產生非法 output index。

### 為什麼 IS 會和 sparse CNN 一起出現

投影片 L05-86 說 IS 用於 sparse CNN，並引用 SCNN（Parashar et al., ISCA 2017），同時說本講不分析 dense CNN 的 IS。原因是結構性的：如果 inputs 很大且許多 weights 為零，保留 nonzero inputs 並和相關 weights 結合，可能減少對較大 memory 的讀取。後續 sparse architecture 講次會更完整回到這個想法。

### 硬體意義

IS 可能產生分散的 output updates，因為固定 input `i[w]` 會貢獻到多個 `o[w-s]` 位置。這讓 accumulator path 比簡單的「一個 output 留在這裡」OS 圖像更複雜。在 sparse settings 中，zero skipping 改變了 traffic balance，因此這個複雜度可能值得付出。

### 常見誤解

**誤解：** IS 只是把 OS 的 loop 名稱換掉。

**修正：** IS 需要圍繞 raw input coordinate 改變 traversal，並處理 `q = w - s` projection。這個 projection 會改變 output update pattern，也改變硬體累加路徑。

---

## 8. 能耗比較與設計權衡

**來源錨點：** 投影片 L05-91 到 L05-94。

介紹 OS、WS、IS 之後，投影片用三種 tiling variants 回到 OS：

| 變體 | Output channels | Output activations | 投影片註記 |
|---|---|---|---|
| OSA | single `M` | multiple `P * Q` | targeting CONV layers |
| OSB | multiple `M` | multiple `P * Q` | - |
| OSC | multiple `M` | single `P * Q` | targeting FC layers |

投影片 L05-93 在 equal total area、256 PEs、AlexNet CONV layers、batch size 16 的設定下比較 WS、OSA、OSB、OSC 與 NLR。主要教學結論不是某個單一數字，而是 WS 與 OS variants 都遠優於 no-local-reuse baseline，但沒有任何一者在所有情況都支配其他者。

### 如何解讀這個比較

Layer shape 會改變最佳 dataflow。Large spatial maps、many channels、small filters、large filters、batch size 都會改變哪個 operand 的 reuse 最有價值。因此固定 dataflow 的晶片，至少在某些 layers 上很可能不是最優。

### 硬體意義

這個比較導向 flexible 或 reconfigurable mappings。Flexibility 不是免費的：它需要更複雜的 control、更彈性的 storage，通常也需要更通用的 NoC。但另一個選項是把 silicon 固定在單一 reuse pattern 上，然後期待 workload 剛好符合它。

---

## 9. LoopTree：dataflow 加 data placement

**來源錨點：** 投影片 L05-95 到 L05-111。

Pseudocode loop nests 很適合表達順序，但不擅長表達資料存在哪裡。LoopTree 被引入來同時表示兩者。

### Workload example：matrix multiplication

投影片改用 matrix multiplication：

```text
Z[m][n] = sum_ni A[m][ni] * B[ni][n]
```

這個式子命名 tensor ranks、binary operation，以及 reduction rank `ni`。它仍然不假設 operation order。

### LoopTree 中的 dataflow

在 LoopTree 中，loop nodes 編碼順序。類似 OS 的 matrix multiplication 可能把 `m` 與 `n` 放在 `ni` 上方，表示一個 output `Z[m][n]` 沿著 `ni` 被累加。

### Partitioning 與 rank swizzling

投影片 L05-102 到 L05-105 顯示，切分 `NI` 這樣的 rank 會產生 `NI1` 與 `NI0` 等 sub-ranks。Einsum 會用這些 sub-ranks 重寫；投影片稱這個過程為 swizzling ranks。Dataflow 要在 partitioning 後指定，因為 loop tree 排列的是 partitioned ranks，而不只是原始 rank。

### Storage plan

投影片 L05-106 到 L05-111 加入 storage nodes：

- DRAM 是所有 tensors 的 backing storage。
- Global buffer 可以取入所有 weights。
- 每次 `for m` iteration 可以取入一塊 `A`。
- 每次 `for ni1` iteration 可以取入另一個 operand 的一塊。

重點是，完整 mapping 需要 loop nodes 與 storage nodes。Loop order 說值何時被使用；storage placement 說值從哪一層 memory 供應。

### 常見誤解

**誤解：** LoopTree 只是比較漂亮的 loop nest。

**修正：** LoopTree 是 mapping specification。它能在同一結構中表示 loop order、partitioned ranks 與 storage placement。這就是它對 TeAAL-style analysis 和後續 data-motion counting 有用的原因。

---

## 10. 關鍵方程式與如何閱讀

### 一維卷積

```text
O[q] = sum_s I[q+s] * F[s]
```

`q` 選 output position。`s` 選 filter tap。`q+s` 選該 tap 使用的 input position。硬體意義：compound input coordinate 是 input stationarity 需要 projection 的原因。

### 稠密二維卷積

```text
O[m][p][q] = sum_c sum_r sum_s I[c][p+r][q+s] * F[m][c][r][s]
```

`c,r,s` 是 reduction ranks。`m,p,q` 命名 output。硬體意義：固定 `m,p,q` 並遍歷 reduction 有利於 local output accumulation；固定 `r,s` 則有利於 weight reuse。

### Dataflow reading rule

```text
outer loops change slowest -> associated tensor is most stationary
```

這是教學用 shorthand。完整 mapping 還要考慮 partitioning 與 data placement。單靠 outer loop 不能保證 stationarity；如果 tile 放不下或 storage plan 讓它被 evict，它仍然無法真正留在本地。

### Energy model reminder

```text
energy roughly tracks access count * energy per access level
```

因此相同 MAC 數可能有不同 energy。也因此，量化主張必須有來源：access count 與 per-level cost 取決於 workload、technology 與 architecture。

---

## 11. 硬體意涵

**Energy：** Dataflow 改變哪些 access 是 RF、NoC、global buffer 或 DRAM access。由於引用的階層從 1x 到 200x，這可能主導算術能耗。

**Bandwidth：** Dataflow 可以降低總流量，卻對某一層 memory 造成高瞬時頻寬需求。例如 WS 可能需要 activation multicast；OS 可能需要把 weights 與 activations 送到持有 psum 的 PEs。

**Latency：** Parallel loops 可以減少 cycles，但 reductions 與 psum movement 如果不匹配 dataflow，會形成延遲。

**Area：** Stationarity 需要 storage。OS 需要 local psum capacity。WS 需要 local weight capacity。IS 需要 local input capacity，且常需要更彈性的 output update path。

**Utilization：** Mapping choices 會和 layer dimensions 互動。NVDLA 簡化的 `M * C` array 例子說明，如果 channel dimensions 不符合 exposed parallelism，MAC 可能 idle。

**Programmability：** Fixed dataflow 對匹配的 layer 可以更簡單有效。Flexible dataflow 能跨 layers 適應，但需要更多 mapping machinery。

**Correctness：** Loop reordering 只有在保留 accumulation dependencies 時才合法。Partial sums 可以流經不同 storage levels，但 final reduction 必須包含同一組 products。

---

## 12. 常見誤解

### 誤解：Dataflow 就是圖上資料移動的方向。

在本課程中，dataflow 主要是 loop-order policy，決定 stationarity、reuse 與 accumulation。箭頭圖是 dataflow 的結果，不是定義本身。

### 誤解：最好的 dataflow 是 MAC 數最少的 dataflow。

Dense OS、WS、IS 對同一 layer 執行相同 MAC。差別在 memory traffic、bandwidth 與 utilization。

### 誤解：讓一種 operand stationary 就解決所有資料搬移。

它只解決搬移的一部分。OS 仍可能搬很多 weights 與 inputs。WS 仍可能搬很多 activations 與 partial sums。IS 仍可能造成分散 output updates。

### 誤解：local memory 越多一定越好。

Local memory 只有在捕捉 reuse 時才有幫助。如果 dataflow 在 evict tile 前沒有重用它，額外 storage 可能只增加 area，收益很小。

### 誤解：某篇 dataflow comparison 的數字可以直接套到任何現代模型。

投影片比較使用 AlexNet CONV layers、batch size 16、256 PEs，以及特定 technology context 的 energy model。比精確 bar 更能安全轉移的結論是：「reuse 很重要，而且沒有固定 dataflow 永遠勝出。」

---

## 13. 重點回顧

- Mapping 是圍繞 Einsum 的硬體排程：loop order、tiling、storage placement、parallelism 與 tile sizes 都屬於 mapping。
- Dataflow 是 mapping 中的 loop-order 部分；它決定哪個 operand 最具 stationarity。
- OS 讓 partial sums 留在本地，WS 讓 weights 留在本地，IS 讓 inputs 留在本地。
- 相同 MAC count 仍可能有很不同的 energy，因為 memory accesses 可能發生在不同 hierarchy levels。
- 沒有單一 dataflow 永遠勝出；layer shape、buffer capacity、interconnect 與 PE utilization 都會影響結果。
- LoopTree 用 partitioned ranks 與 storage nodes 擴充 loop nests，讓工具能分析 mappings。

---

## 14. 與前後講次的連結

**銜接 L01：** L01 引入 memory-energy argument 與 separation-of-concerns pyramid。L05 透過 loop order 與 stationarity 讓 mapping layer 變具體。

**銜接 L02：** L02 介紹 convolution ranks。L05 把這些 ranks 變成 loop nests，並問哪些 ranks 應該在 outer、inner 或 parallel loops。

**銜接 L03-L04：** Einsum 表達 what to compute，但不承諾 order。L05 說明這個分離為什麼重要：許多合法 order 有不同 hardware cost。

**導向 L06：** L06 研究 partitioning。L05 問「哪個 operand 應該留在本地」；L06 問「tile 應該多大、partitioned ranks 如何揭露 temporal 與 spatial reuse」。

**導向 L07-L10：** Sparse architectures 改變 access counts，可能讓 input-stationary 或 hybrid schedules 更有吸引力。本講的 IS projection 會在 sparse fibers traversal 與 projection 時再次出現。

**導向 L13：** L05 用直覺 access counting。L13 用 sets、maps、timestamps、shrink/delta calculations 將 data-motion counting 形式化。

---

## 15. 論文橋接：Eyeriss（Chen、Emer、Sze，ISCA 2016）

### 文獻身分

- **標題：** *Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks*
- **作者：** Yu-Hsin Chen、Joel Emer、Vivienne Sze
- **年份／會議：** 2016，ISCA
- **用於本講：** L05 dataflow taxonomy、energy hierarchy、energy comparison；後續講次會回到 Eyeriss 與 sparse successors。

### 解決的問題

此論文處理 CNN accelerators 中 data movement 的能耗問題。高度並行的 MAC array 可以提供 throughput，但如果 weights、activations、partial sums 太常穿越昂貴 memory levels，energy efficiency 仍然很差。

### 核心想法

論文先系統化既有 dataflows，依它們降低哪種 data movement 來分類，包括 weight-stationary、output-stationary 與 no-local-reuse styles。接著提出 **Row Stationary（RS）**，試圖同時利用 filters、input feature maps 與 partial sums 的 reuse，而不是只最佳化單一 operand。

### 與本講的關聯

投影片 L05-28 將 dataflow taxonomy 歸因於 Chen, ISCA 2016。投影片 L05-93 將 energy comparison 歸因於 Chen et al., ISCA 2016。投影片 L05-23 到 L05-24 使用的 normalized memory hierarchy 與 Eyeriss energy table，尤其是 Table IV，相互對應。

### 本章使用的關鍵主張

- **Energy hierarchy：** DRAM 200x、global buffer 6x、inter-PE movement 2x、RF 1x，相對於 MAC reference。來源錨點：Eyeriss Table IV 與投影片 L05-23 到 L05-24。
- **Taxonomy：** 既有 CNN accelerator dataflows 可用「哪個 operand stationary」或「降低哪種 movement」來描述。來源錨點：Eyeriss Section V 與投影片 L05-28。
- **公平比較方法：** 比較 dataflows 時應固定硬體資源假設，例如 equal area 與 equal parallelism。來源錨點：Eyeriss evaluation methodology 與投影片 L05-93。
- **Row Stationary 結果：** 論文在 AlexNet/equal-area 方法下報告 RS 比所比較的既有 dataflows 更節能。這是 paper-derived claim，不是 L05 投影片完整展開的內容。

### 學生應記住什麼

- OS 與 WS 不只是名稱，而是有硬體後果的 reuse policies。
- 本講使用的 energy numbers 來自特定論文與 technology context。
- 公平 dataflow comparison 必須先固定 hardware assumptions，再比較 mappings。
- 「沒有普世贏家」會推動更 flexible 的 dataflows 與 mapping tools。

### 限制與假設

Eyeriss 結果與論文中的 networks、technology assumptions、area constraints、architecture model 綁在一起。精確改善倍數不應在未重新評估 workload 與 hardware 時直接推廣。本章只用此論文橋接 L05 概念，不把它當作完整 paper summary。

### 建議閱讀位置

讀完第 8 與第 9 節後再讀本橋接。它說明 taxonomy 與 energy comparison 來自何處，也說明為什麼本講自然導向更 flexible 的 dataflow design。

---

## 16. 獨立學習指南

### 必須掌握

- 將 mapping 解釋為 loop、storage、parallelism decisions 的組合。
- 給定 loop nest，辨認 stationary operand。
- 解釋為什麼 DRAM traffic 可能在 MAC count 不變時主導 energy。
- 用 tiny convolution 計算 OS 與 no local reuse 的 partial-sum traffic。
- 解釋 OS、WS、IS 隱含的硬體路徑。
- 說明 LoopTree 比 Einsum 多表達了什麼。

### 自我檢核問題

1. 在 `O[q] = sum_s I[q+s]F[s]` 中，為什麼 `for q` 在 `for s` 外面會讓 output stationary？
2. 為什麼 `for s` 在 `for q` 外面會讓 weight stationary？
3. Input-stationary loop nest 需要哪個 guard？為什麼？
4. 為什麼兩個 MAC count 相同的 mappings 可能有不同 energy？
5. Dataflow 控制 mapping 的哪個面向？它不控制哪些面向？
6. 為什麼 NVDLA 簡化的 `M * C` PE organization 可能造成 utilization loss？
7. LoopTree storage node 表達了 Einsum 沒有表達的哪些資訊？
8. 為什麼若不說 layer shape 與 hardware assumptions，「dataflow X 最好」是不完整主張？

### 練習

1. **概念題：** 對 OS、WS、IS，分別指出 stationary operand，以及最可能造成 movement pressure 的 operand。
2. **小計算：** 對 `W = 6`、`S = 3`、`Q = 4` 重做 worked example。只計算 no local reuse 與 OS 的 partial-sum traffic。
3. **Loop reading：** 給定 loop order `r, s, p, q, parallel m, parallel c`，判斷 dataflow 並解釋原因。
4. **設計權衡：** 假設 accelerator 的 RF 很小，但 global buffer 較大。OS 或 WS 的哪些假設會變脆弱？
5. **Paper bridge：** 閱讀 Eyeriss abstract 與 Table IV。本章哪些 claims 依賴 paper，而不只是投影片？
6. **開放架構推理：** 為 OS 設計一個 NoC primitive，再為 WS 設計一個。解釋為什麼兩者不一樣。

---

## 17. 關鍵詞彙（Key Terms）

### Mapping（映射）

把 tensor computation 映射到硬體上的 scheduling 與 placement decisions，包括 partitioning、dataflow、data placement、compute placement、partition sizing。硬體意義：mapping 決定 computation 的每一部分使用哪些 memory levels 與 PEs。

### Dataflow（資料流）

決定哪些 tensor values 變化慢、可在本地被重用的 loop-order policy。它不只是圖上的箭頭方向。硬體意義：dataflow 形塑 memory traffic、NoC traffic、buffer requirements 與 PE utilization。

### Stationarity（駐留性）

某個值在多次有用 MAC 期間停留在低成本 storage 的性質。硬體意義：stationarity 把反覆 DRAM 或 global-buffer access 轉成 RF 或 NoC-local access。常見混淆：stationary 不等於永久不動。

### Output Stationary, OS（輸出駐留）

Output partial sums 保持本地、沿 reduction terms 累加的 dataflow。硬體意義：降低 intermediate psum movement，偏好 local accumulation structures。

### Weight Stationary, WS（權重駐留）

Weights 保持本地、activations 與 partial sums 移動的 dataflow。硬體意義：當每個 weight 被大量重用時可降低 weight read energy，但可能需要 activation multicast 與 psum reduction。

### Input Stationary, IS（輸入駐留）

Input activations 保持本地的 dataflow。硬體意義：在某些 sparse settings 中有用，但透過 `q = w - s` projection 可能造成 scattered output updates。

### Partial Sum, Psum（部分和）

尚未成為 final output 的中間累加值。硬體意義：psums 每次 MAC 都更新，因此不好的 psum placement 可能主導 traffic。

### Local Accumulation（本地累加）

在 output 完成前，把 psum updates 留在 RF 或 local buffer。硬體意義：避免 intermediate psums 反覆進出 DRAM。

### Data Reuse（資料再利用）

一個 fetched value 服務多個 MAC。硬體意義：reuse 是小型 local memories 值得佔用 area 與 energy 的原因。

### Convolutional Reuse（卷積再利用）

卷積 sliding windows 重疊造成的 reuse。硬體意義：一個 input activation 可貢獻到多個 output windows。

### Fmap Reuse（特徵圖再利用）

Input activations 被多個 filters 或 output channels 重用。硬體意義：multicast 或 buffering 可讓一個 activation 餵給多個 MAC。

### Filter Reuse（濾波器再利用）

Weights 在 batch 的多個 images 或多個 spatial positions 中重用。硬體意義：weight storage 與 scheduling 能攤銷昂貴的 weight fetch。

### Data Placement（資料放置）

在 loop nest 的每個位置，選擇每個 tensor tile 位於哪個 memory level。硬體意義：placement 決定 logical access 是 RF、NoC、global-buffer 還是 DRAM energy。

### LoopTree

一種 mapping tree notation。Loop nodes 表達 dataflow；storage nodes 表達 data placement；partitioned ranks 表達 tiling。硬體意義：LoopTree 可被工具分析以估算 data movement。

### NLR, No Local Reuse（無本地再利用）

沒有有效 local reuse 的 baseline。硬體意義：它是警示案例，不是好的 accelerator design。

---

## 18. 附錄 - 投影片對照表（Slide-to-Section Map）

| 投影片範圍 | 章節 | 註記 |
|---|---|---|
| L05-1 | 標題與 metadata | 課程脈絡 |
| L05-2 | 本講要解決的問題 | Separation of concerns 中的 mapping |
| L05-3 到 L05-5 | 第 1 節 | Mapping 的五個面向 |
| L05-6 到 L05-7 | 學習目標與來源註記 | Goals 與 background reading |
| L05-8 到 L05-13 | 第 2 節 | 1-D convolution、MAC memory traffic、computational intensity |
| L05-14 到 L05-19 | 第 2 節 | Reuse types 與 AlexNet DRAM reduction |
| L05-20 到 L05-27 | 第 2 與第 11 節 | Spatial architecture 與 low-cost local access |
| L05-28 到 L05-35 | 第 3 與第 5 節 | Taxonomy 與 1-D OS |
| L05-36 到 L05-48 | 第 5 節 | CONV-layer OS 與 OS examples |
| L05-49 到 L05-55 | 第 3 與第 6 節 | 1-D WS 與 parallel WS |
| L05-56 到 L05-84 | 第 6 節 | nn-X、NVDLA 與 WS examples |
| L05-85 到 L05-90 | 第 7 節 | IS、coordinate projection、sparse CNN note |
| L05-91 到 L05-94 | 第 8 節 | OS variants 與 energy comparison |
| L05-95 到 L05-111 | 第 9 節 | LoopTree、partitioning、swizzling、storage plan |
| 背景補充 | 第 4、10 到 17 節 | 教學範例、誤解、關鍵詞彙、練習 |

---

## 19. 來源註記（Source Notes）

- **投影片：** Lecture ordering、mapping-aspects list、reuse taxonomy、OS/WS/IS loop nests、NVDLA simplified example、OS variants、energy comparison setup、LoopTree introduction，皆依據 `Lecture/L05-Mapping.pdf`。
- **Energy ratios：** 1x/2x/6x/200x hierarchy 顯示於投影片 L05-23 到 L05-24；本章歸因於 Chen、Emer、Sze，ISCA 2016，Table IV。
- **AlexNet access counts：** 7.24 億 MAC 與 28.96 億 worst-case DRAM accesses 來自投影片 L05-11。6,100 萬 best-case DRAM accesses 來自投影片 L05-19。
- **最多 500x reuse claim：** 投影片 L05-17 指出 AlexNet CONV layers 的 filter/fmap DRAM reads 最多可降低 500x。
- **Dataflow taxonomy：** 投影片 L05-28 引用 Chen, ISCA 2016。
- **Hardware examples：** ShiDianNao、KU Leuven、nn-X/NeuFlow、NVDLA、TPU、ISAAC、PRIME 等例子出現在投影片 L05-30 到 L05-84。本章只把它們作為簡短例子，不做完整 paper summary。
- **Input Stationary and sparse CNNs：** 投影片 L05-86 引用 SCNN，Parashar，ISCA 2017。
- **Paper bridge：** Eyeriss 討論只用來支撐本講概念：energy hierarchy、taxonomy、fair comparison、row-stationary motivation。
- **教學詮釋：** Worked examples、misconceptions、hardware implication synthesis，以及部分 cross-lecture connections，是為自學章節新增的原創解釋。

## 20. 不確定性註記（Uncertainty Notes）

- Live lecture 可能更依賴動畫講解；本章以 loop-nest reasoning 取代動畫，使讀者不看影片也能理解。
- AlexNet DRAM reduction numbers 是投影片陳述的 best-case figures。實際 reduction 取決於 layer dimensions、tile sizes、memory capacity 與 implementation。
- Eyeriss energy table 具有 technology specificity。質性的 memory hierarchy 很有用，但精確 ratios 不應被視為 universal constants。
- Row Stationary 放在 paper bridge 中，是為了回應投影片「Is it possible to do better?」的動機。L05 投影片本身沒有完整教 RS。
- 既有 `assets/L05/*.png` 檔案看起來是 extracted slide images。本次重寫沒有在章節中嵌入它們，但檔案仍留在 repository 中；公開發布前應另外做 copyright review。
