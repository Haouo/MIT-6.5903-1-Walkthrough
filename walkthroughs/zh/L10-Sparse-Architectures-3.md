# L10 — 稀疏架構三（Sparse Architectures 3）

> **課程：** 6.5930/1 — 深度學習硬體架構（Hardware Architectures for Deep Learning）
> **講師：** Joel Emer 與 Vivienne Sze
> **講授日期：** 2026 年 3 月 9 日 · **投影片：** 66 頁 · **來源：** [`Lecture/L10-Sparse_Architectures-3.pdf`](../../Lecture/L10-Sparse_Architectures-3.pdf)
>
> 本章以投影片重建缺少的講課脈絡。為了 copyright safety，本章不重貼投影片或論文圖，而用文字、小例子與必要的程式式 loop nest 說明。
>
> **追溯註記：** 這份 PDF 內部標號是 `L06-*`。本 repository 將它作為 Lecture 10，因此下文 source anchors 會同時用 Lecture 10 與 `L06-n` slide label。

---

## 一句話總結（TL;DR）

稀疏加速真正困難的時刻，是**權重（weights）與激活值（activations）同時被壓縮**的時候。如果只有一邊稀疏，硬體可以走訪稀疏 operand，然後直接索引另一個 dense operand。如果兩邊都稀疏，硬體還要解一個座標問題：哪些非零權重與非零激活值真的會在同一個輸出座標相遇？

Lecture 10 從容易情形推到 joint case。它先區分**閘控（gating）**與**跳過（skipping）**，再回顧 sparse-weight-only 與 sparse-activation-only convolution，最後比較兩種 joint sparsity 解法。**SCNN** 採用 input-stationary Cartesian product，把非零激活值與非零權重全對全相乘，再把 products scatter 到輸出累加器。**ISOSceles** 採用兩階段 **IS-OS dataflow**：input-stationary pass 先建立小型中間張量，再由 output-stationary pass 在 rank swizzling 之後讀取。核心教訓是：sparsity 不只是 format 決策；它會同時牽動 representation、loop order、coordinate generation、routing、buffer sizing 與 load balance。

---

## 本講解決什麼問題

前幾講已經說明 zeros 可以降低 arithmetic 與 memory traffic。剩下的問題是架構問題：

> 如果 weights 和 activations 都 sparse，真實 accelerator 要如何跳過無用乘法，同時不被 metadata、random access、scatter traffic 與 load imbalance 淹沒？

天真的答案是「把兩個 tensors 都壓縮，然後 loop over nonzeros」。這不夠。Convolution product 只有在 weight coordinate 與 activation coordinate 對應到合法 output coordinate 時才成立。在 dense code 裡，nested loops 自動保證合法性；在 sparse code 裡，硬體必須從 metadata 重建這件事。

所以本講解的是 mapping problem，不只是 storage problem。它問的是：哪一種 dataflow 能讓 coordinate problem 便宜到足以實作？

---

## 為什麼本講重要

Sparsity 在數學上很誘人，因為 zero product 沒有貢獻：$0 \times x = 0$。但硬體不會自動獲得這個好處。一個 zero 可能帶來：

- **只省 energy**：硬體 gating multiplier，但 cycle 仍被占用。
- **energy 與 time 都省**：硬體把那次 operation 從 schedule 移除。
- **什麼都沒省**：metadata 與 routing overhead 大於省下的 work。

對 hardware architect 而言，關鍵問題不是「tensor sparse 嗎？」而是「sparsity 出現在哪裡、coordinate stream 是否規則到能利用、需要加什麼硬體才能讓 nonzero work 持續流動？」

---

## 先備知識與心智模型

你需要記得 L07-L09 的三個想法：

- **纖維（fiber）**是 tensor 的一維切片，常表示成 coordinate-payload pairs，例如 `(coord, value)`。
- **協調遍歷（concordant traversal）**依照壓縮座標的儲存順序走訪；**不協調遍歷（discordant traversal）**則要求以 format 不自然支援的順序存取資料。
- **資料流（dataflow）**決定哪個 tensor 停在 PE 附近，哪個 tensor 經由 memory/interconnect 串流。

本講的心智模型是一個很小的 1-D convolution：

$$
o[q] = \sum_s i[q+s] \cdot f[s].
$$

如果 `f` sparse 而 `i` dense，非零 weight coordinate `s` 直接告訴硬體要讀 `q+s` 的 input。如果 `i` sparse 而 `f` dense，非零 input coordinate `w` 直接告訴硬體要讀 `w-q` 的 filter location。如果兩者都 sparse，兩種直接 lookup 都不再免費：機器必須找 coordinate matches，或先產生 products 再 scatter 到 output coordinates。

---

## 學習目標（Learning Objectives）

讀完本講後，你應該能夠：

- 區分**閘控（gating）**與**跳過（skipping）**，並解釋為什麼只有 skipping 會降低 latency。
- 將 1-D convolution loop 改寫為 sparse weights、sparse inputs 與 joint sparsity 版本。
- 解釋為什麼 sparse-weight-only 與 sparse-input-only accelerator 比 joint-sparse accelerator 容易。
- 描述 **Cambricon-X** 與 **Cnvlutin** 如何對應到投影片中的 single-sparsity cases。
- 解釋 SCNN 的 **input-stationary Cartesian-product** dataflow，以及為什麼它需要 scatter accumulator。
- 解釋為什麼 SCNN 的 useful work 會隨 activation density 與 weight density 的乘積縮放，但硬體 overhead 不會消失。
- 解釋 ISOSceles 的 **IS-OS** 拆分、中間張量為什麼出現，以及 **swizzling** 為什麼能把 discordant traversal 轉成 concordant traversal。
- 評估 sparse-accelerator claims 時能問清楚：計入的是 arithmetic、memory traffic、metadata、routing、utilization，還是 end-to-end speedup。

---

## 主要教材式敘事

### 1. Gating 不是 Skipping

Lecture 10 一開始提醒我們，zero operand 提供兩種不同機會。**Gating** 偵測到 zero 後，關閉 multiplier 或 memory read port 等 datapath 活動。Cycle 仍然存在，因此 latency 不會改善。**Skipping** 則把那次 operation 從動態 schedule 中移除；機器實際執行較少 useful cycles。

Lecture 10 slide `L06-7` 以 Eyeriss 作為 gating 例子，並報告 input activation 為 zero 時可降低 45% PE power。這是 slide-derived quantitative claim。教學解讀是：gating 是低風險的第一步，因為 dense schedule 完全保留，不需要複雜的 metadata-driven load balancing。但它的限制也很清楚：即使一半 activations 是 zero，accelerator 也不會自動用一半時間完成。

### 2. 只有 Sparse Weights：走訪 Weights，索引 Inputs

對 sparse weights 而言，compressed filter 只儲存非零 `(s, f_val)` pairs。簡單的 output-stationary loop 是：

```text
for q in [0, Q):
  for (s, f_val) in f:
    w = q + s
    o[q] += i[w] * f_val
```

Sparse tensor 以 concordant traversal 走訪。Dense input tensor 仍可直接 index。重要硬體含義是 coordinate generator 必須計算 `w = q + s`，但不需要搜尋 matching input coordinate。這就是 sparse-weight-only 設計相對乾淨的原因。

Weight-stationary 版本把前兩層 loop 對調：

```text
for (s, f_val) in f:
  for q in [0, Q):
    w = q + s
    o[q] += i[w] * f_val
```

現在 weight 可以停在 PE 附近，被多個 outputs 重用。代價是 output partial-sum traffic。若 outputs 很多、local accumulator capacity 很小，設計可能省下 weight reads，卻增加 partial-sum movement。

Lecture 10 slides `L06-22` 到 `L06-24` 介紹用 fiber splitting 做 parallel sparse-weight traversal。細節在於 compressed fiber 可以依照**壓縮 stream 中的位置**切分，而不是依 coordinate value 切分。Position splitting 讓每個 PE 取得相近的 nonzero 數量，但 coordinate range 可能不規則。Coordinate splitting 讓 spatial ownership 清楚，但 nonzeros 聚集時 load balance 可能很差。

### 3. 只有 Sparse Inputs：走訪 Activations，索引 Weights

當 input activations sparse、weights dense，角色反過來：

```text
for q in [0, Q):
  for (w, i_val) in i if q <= w < q + S:
    s = w - q
    o[q] += i_val * f[s]
```

Input fiber 是 sparse，因此只有非零 activations 產生 MAC。Dense filter 用 `s = w - q` 直接索引。Lecture 10 slides `L06-28` 到 `L06-35` 把它呈現為 sparse sliding window。Window condition 很重要：不是每個非零 activation 都貢獻到每個 output。

Cnvlutin 在 Lecture 10 slides `L06-36` 到 `L06-38` 中作為 activation skipping 的 case study。Slide-derived lesson 是：per-channel encoders 移除 zero activations，硬體用 activation coordinates 選出正確 weight。架構限制是 dense side 必須便宜地 index。如果 weights 也被壓縮，直接 `getPayload(s)` 就不再是簡單 array lookup。

### 4. Joint Sparsity：Coordinate Problem 出現

當兩個 tensors 都 sparse，loop 不能單純走訪一個 compressed stream，然後直接 index 另一個。假設：

- 非零 input coordinates 是 $i = \{0, 3, 4\}$。
- 非零 filter coordinates 是 $f = \{0, 2\}$。
- Output coordinate 由 $q = w - s$ 計算。

Cartesian products 如下：

| Input `w` | Weight `s` | Output `q = w-s` | 如果 output range 包含 `q` 是否合法？ |
|---:|---:|---:|---|
| 0 | 0 | 0 | yes |
| 0 | 2 | -2 | usually no |
| 3 | 0 | 3 | yes |
| 3 | 2 | 1 | yes |
| 4 | 0 | 4 | maybe yes |
| 4 | 2 | 2 | yes |

這個例子顯示新的工作：accelerator 必須計算 output coordinates、丟棄 illegal products，並把 legal products route 到 accumulators。Dense loop indices 原本默默替我們完成這些事。

### 5. SCNN：Cartesian Products 加 Scatter

SCNN 在 Lecture 10 slides `L06-45` 到 `L06-51` 被歸屬於 Parashar et al., ISCA 2017。它選擇 **input-stationary Cartesian-product** 策略。在每個 PE 中，一組非零 activations 與一組非零 weights 全對全相乘。如果有 $I$ 條 activation lanes 與 $F$ 條 weight lanes，PE 每一步產生 $I \times F$ 個 candidate products。

概念 loop 是：

```text
for (w, i_val) in sparse_inputs:
  for (s, f_val) in sparse_weights if product_is_legal(w, s):
    q = w - s
    scatter_add(o[q], i_val * f_val)
```

乘法步驟規則；累加步驟不規則。每個 product 可能指向不同 output coordinate，因此 SCNN 需要 scatter network 與 banked accumulators。這是 sparse hardware 的典型模式：移除 zero MACs 之後，不規則 communication 露出來了。

從 SCNN paper 來看，該架構把 weights 與 activations 都保留為 compressed-sparse form，在 PE 內做 Cartesian product，從 sparse indices 計算 output coordinates，並用 scatter accumulator array route products（SCNN Sections III-IV，特別是 paper Figures 5 and 6 的相關說明）。該 paper 報告相對 dense accelerator 有 2.7x performance improvement 與 2.3x energy reduction（SCNN Abstract and Section VI）。這些是 paper-derived claims。

### 6. ISOSceles：把工作拆成 IS 再 OS

Lecture 10 slides `L06-53` 到 `L06-66` 展示另一種 joint sparsity 解法。與其產生大量 irregular products 並立即 scatter，ISOSceles 使用 **IS-OS** decomposition。

Output-stationary joint-sparse 想法可以寫成 intersection：

```text
for q in [0, Q):
  for (coord, (f_val, i_val)) in f.project(+q) & i:
    o[q] += i_val * f_val
```

Projection 會平移 filter coordinates，使它們與 input coordinates 對齊。Intersection 只輸出同時非零的 pairs。數學上很漂亮，但如果 participating fibers 的 storage order 與 loop order 不一致，就會變成 discordant traversal。

ISOSceles 把計算拆成：

- **IS pass：** 以 input-oriented sparse streams 建立 intermediate tensor `T`。
- **OS pass：** 以 output order 走訪 `T`，累加 final outputs。

關鍵轉換是 **swizzling**，也就是重新排列 `T` 的 ranks，使第二階段能 concordantly consume intermediate tensor。教學上可以把 ISOSceles 理解為：付出小型 intermediate buffer 的代價，換掉更混亂的 scatter pattern。

Lecture 10 slide `L06-66` 報告 ISOSceles 最高 7.5x speedup、平均約 1.7x speedup。這是 slide-derived quantitative claim，投影片歸屬於 Yang et al., HPCA 2023。由於本 worker 指定的 local paper list 中沒有 ISOSceles PDF，本章把 ISOSceles 的詳細解釋標記為 slide-derived 加 teaching interpretation，而非獨立 paper-verified。

---

## Worked Examples

### 範例 1：Gating vs. Skipping

假設 PE 收到八組 activation-weight pairs，其中四個 activations 是 zero。

- 使用 **gating** 時，PE 仍消耗八個 cycle slots。四次 multiplier operation 被關閉，因此 PE dynamic energy 下降，但 latency 仍是八個 slots。
- 使用 **skipping** 時，PE 只 schedule 四個 nonzero products。Latency 可能接近四個 slots，但前提是 compressed stream、coordinate generator 與 accumulator 都跟得上。

硬體意義：gating 容易，因為 dense schedule 保持完整。Skipping 更強，因為它改變 schedule，但需要 metadata 與 load balancing。

### 範例 2：Sparse Sliding Window

令 $Q = 4$、$S = 3$，sparse input coordinates 為 $\{0, 2, 4, 5\}$，dense weights 為 $f[0], f[1], f[2]$。對 output $q = 2$，window 是 $2 \le w < 5$，所以 active sparse inputs 是 $w = 2$ 與 $w = 4$。

Weight coordinates 是：

- 對 $w = 2$，$s = w-q = 0$。
- 對 $w = 4$，$s = w-q = 2$。

因此 $o[2]$ 只收到兩個 products：$i[2]f[0]$ 與 $i[4]f[2]$。Dense loop 還會測試 $i[3]f[1]$，但 sparse traversal 因為 `i[3]` 是 zero 或不存在而避開它。

### 範例 3：Joint Density 是乘積，不是和

如果 activation density 是 $d_i = 0.3$，weight density 是 $d_f = 0.2$，在 independent-density model 下，預期 nonzero products 的比例是 $d_i d_f = 0.06$。

教學解讀：這就是 joint sparsity 看起來很有吸引力的原因。不過 accelerator 仍要付 fixed 與 semi-fixed costs：metadata reads、coordinate arithmetic、scatter/intersection logic、underutilized lanes 與 synchronization。Sparse accelerator 只有在 skipped work 大於這些 overhead 時才真的優秀。

---

## 關鍵方程式與讀法

本講反覆使用的 dense 1-D convolution 是：

$$
o[q] = \sum_s i[q+s] f[s].
$$

讀法是：對一個 output coordinate $q$，沿著 filter coordinate $s$ 滑動，對應 input coordinate $w=q+s$。

Input-stationary traversal 的 output coordinate 是：

$$
q = w - s.
$$

讀法是：一旦 coordinate $w$ 的非零 input 與 coordinate $s$ 的非零 filter entry 相遇，output coordinate 不是獨立 loop index；它必須被計算出來並用於 accumulation。

簡單 independent-density model 是：

$$
d_{\text{joint}} = d_i d_f.
$$

讀法是：如果 activations 與 weights 分別以 $d_i$、$d_f$ 的 density 獨立非零，預期只有兩者乘積比例的 work 會產生 nonzero multiplication。這個式子不包含 metadata、routing 或 load-balance overhead。

---

## 硬體含義

- **Metadata bandwidth：** Sparse formats 用 coordinate reads 取代部分 payload reads。若 coordinates 很寬或很不規則，metadata 會變成一級成本。
- **Coordinate generation：** Sparse convolution 需要 $w=q+s$ 或 $q=w-s$ 這類 arithmetic。邏輯本身不大，但必須跟上 PE rate。
- **Accumulator design：** SCNN-style scatter 需要 banked accumulators 與 conflict management。ISOSceles-style decomposition 需要 intermediate buffer 與 rank-swizzled access。
- **Load balance：** Equal nonzero counts 不必然等於 equal execution time。Products 可能 illegal，scatter conflicts 可能不同，不同 layers 的 densities 也不同。
- **Memory hierarchy：** Sparse weights、sparse activations 與 sparse outputs 會壓力到不同 memories。最佳 dataflow 取決於哪個 buffer 小、哪個 tensor 被重用、哪個 tensor 可以 stream。
- **Programmability：** Dataflow-specific sparse hardware 很有效率但較不 general。前面課程介紹的 TeAAL 很有用，因為它把 format、mapping、architecture、binding 分開思考。

---

## 常見誤解

### 誤解：Sparsity 自動帶來 speedup。

只有當機器真的 skipping work，而不只是 gating；且 metadata 與 routing overhead 沒有主導時，sparsity 才會帶來 speedup。

### 誤解：Joint sparsity 只是 sparse weights 加 sparse activations。

Joint case 多了一個 coordinate matching 問題。兩個 operands 都 compressed 時，accelerator 必須 intersection coordinate streams，或先產生 products 再 scatter 到 runtime 計算出的 output locations。

### 誤解：Cartesian-product multiplication 很浪費，因為它產生很多 products。

如果許多 products illegal 或撞到同一 accumulator bank，它確實可能浪費。但它也規則且平行。SCNN 正是利用這種 regularity 同時處理兩個 compressed streams。

### 誤解：Peak sparse speedup number 描述整個 network。

Layer densities 會變。Early layers 與 later layers 的 activation density 常不同，有些 layers 可能太 dense 或太小，無法攤平 sparse overhead。

---

## 與前後講次的連結

- **L05 Mapping/Dataflows：** L10 是 mapping 的直接應用。Output-stationary、weight-stationary、input-stationary 與 IS-OS choices 決定哪些 sparse operations 便宜。
- **L07-L09 Sparse Formats：** Fibers、concordant traversal、discordant traversal、projection 與 intersection 在本講變成具體硬體機制。
- **L11 Advanced Technologies：** Compute-in-memory 改變 weights 與 partial sums 的 movement cost，但不會讓 sparse coordinate management 消失。
- **L12 Reduced Precision：** Quantization 可能創造更多 zeros，也可能降低 metadata/payload width。Precision 與 sparsity 在 accuracy 與 hardware cost 上都會交互作用。
- **後續 accelerator modeling：** 本講的 sparse cases 正是 cost model 需要 source discipline 的地方：arithmetic count、memory traffic、metadata traffic 與 utilization 必須分開。

---

## Paper Bridge: SCNN

### Bibliographic Identity

- **Title:** SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks
- **Authors:** Angshuman Parashar et al.
- **Year / venue:** ISCA 2017
- **Local PDF:** [`papers/L17_SCNN_Parashar_ISCA2017.pdf`](../../papers/L17_SCNN_Parashar_ISCA2017.pdf)
- **Used in lecture:** Lecture 10 slides `L06-45` 到 `L06-51`

### Problem Addressed

SCNN 處理 sparse models 與 dense accelerator schedules 之間的落差。早期設計可以 gating zeros 省 energy，或只利用一邊 sparse operand。SCNN 問的是：accelerator 能否讓 weights 與 activations 都保持 compressed、跳過 zero-valued products，並仍然正確累加 convolution outputs？

### Core Idea

核心 abstraction 是 **PlanarTiled-InputStationary-CartesianProduct-sparse** dataflow。每個 PE 取一組 nonzero activations 與一組 nonzero weights，形成 Cartesian product，從 sparse indices 計算 output coordinates，再經由 scatter accumulator route products。

### Relevance to This Lecture

SCNN 是 Lecture 10 從 single-sparse skipping 走到 joint-sparse skipping 的具體架構。它說明 joint sparsity 需要的不只是 compressed storage：PE 還要計算 coordinates 並 route partial sums。

### Key Claims Used in This Chapter

- SCNN 將 sparse weights 與 sparse activations 都以 compressed form 儲存並利用兩者 sparsity；見 paper abstract 與 Sections III-IV。
- PE 對 compressed activation 與 weight vectors 形成 Cartesian products，接著計算 output coordinates 並 scatter products；見 SCNN Section III-B 與 Section IV，尤其是 Figures 5 and 6 周邊說明。
- 評估設計使用 64 PEs，每個 PE 16 multipliers，總計 1024 multipliers；見 SCNN Table IV 與 Section IV。
- Paper 報告相對 dense accelerator 有 2.7x performance improvement 與 2.3x energy reduction；見 abstract 與 Section VI。
- Paper 指出 accumulator 與 activation memories 是 PE area 的大宗；見 Table III 附近的 area discussion。

### What Students Should Remember

1. SCNN 的 multiplication 規則；accumulation 不規則。
2. 兩個 operands 都 compressed 可以省 payload movement，但引入 coordinate 與 scatter overhead。
3. Sparse speedup 取決於 density，也取決於 PE array 如何避免 bank conflicts 與 load imbalance。
4. SCNN 是一個 design point，不是 universal sparse recipe。它選擇以 scatter hardware 作為 joint skipping 的代價。

### Limitations and Assumptions

SCNN 的 published results 依賴 evaluated CNNs、pruning/activation sparsity patterns 與特定 area/performance model。未檢查 density、layer shape 與 accumulator behavior 前，不應把該 speedup 泛化到所有 networks 或 sparsity structures。

### Suggested Insertion Points

解釋 joint-sparse Cartesian products、scatter accumulators，以及 arithmetic reduction 與 end-to-end accelerator speedup 差異時引用 SCNN。

---

## Paper Bridge: State of Pruning

### Bibliographic Identity

- **Title:** What is the State of Neural Network Pruning?
- **Authors:** Davis Blalock et al.
- **Year / venue:** MLSys 2020
- **Local PDF:** [`papers/L16_StateOfPruning_Blalock_MLSys2020.pdf`](../../papers/L16_StateOfPruning_Blalock_MLSys2020.pdf)
- **Used in lecture:** 說明 sparse weights 從何而來的背景 bridge

### Problem Addressed

該 paper survey pruning literature，問 reported pruning results 是否可比較，以及能得出哪些一致結論。這對 Lecture 10 重要，因為 sparse accelerators 常假設 pruned weights 已經存在。

### Core Idea

Pruning 不是單一技術。該 paper 區分 unstructured pruning、structured pruning、scoring rules、local vs. global pruning、fine-tuning schedules、compression ratios 與 theoretical speedups。

### Relevance to This Lecture

Lecture 10 把 sparse weights 當成可用輸入。Pruning survey 提醒讀者：weight sparsity 來自 algorithmic pipeline，而且 reported compression 或 theoretical speedup 不一定轉化為 hardware speedup。

### Key Claims Used in This Chapter

- Paper 將 pruning 定義為產生 masked 或 reduced model $f(x; M \odot W')$；見 Section 2。
- Paper 區分 unstructured pruning 與 structured pruning，並說明 unstructured pruning 未必能乾淨映射到 modern libraries and hardware speedups；見 Section 2。
- Paper 指出 pruning papers 常使用不一致 metrics 與 baselines；見 Sections 3-5。
- Paper 建議報告 compression、theoretical speedup、controls 與 tradeoff curves；見 Section 6。

### What Students Should Remember

1. Sparse weights 不是免費的；它們來自 accuracy-efficiency tradeoff。
2. Unstructured sparsity 對 compression 很有吸引力，但硬體較難利用。
3. Theoretical FLOP reduction 不等於 accelerator speedup。
4. Sparse architecture paper 應清楚說明 sparsity pattern、density 與 evaluation baseline。

### Limitations and Assumptions

這份 survey 討論 pruning methodology，不是 accelerator microarchitecture。它支撐 sparse weights 的來源，但不驗證任何特定 sparse accelerator。

### Suggested Insertion Points

討論 sparse weight density 為何跨 layers 變動，以及 hardware speedup claims 為何需要謹慎評估時引用此 paper。

---

## 獨立學習指南

### 進入下一講前必須掌握

- 解釋 late detection 的 zero 與 schedule 前就被 skipped 的 zero 有何不同。
- 從 $o[q] = \sum_s i[q+s]f[s]$ 推導 sparse-weight-only 與 sparse-input-only loops。
- 解釋 joint sparsity 為什麼需要 coordinate matching。
- 比較 SCNN 的 scatter-based design 與 ISOSceles 的 intermediate-tensor design。
- 把 sparse speedup results 看成 density 加 overhead，而不是 density alone。

### 自我檢核問題

1. 為什麼 Eyeriss gating 可以省 PE power，卻不省 latency？
2. 在 sparse-weight-only traversal 中，為什麼 dense input 容易 index？
3. 在 sparse-input-only traversal 中，sparse sliding window 限制了什麼？
4. 為什麼 SCNN 需要 scatter accumulator？
5. Rank swizzling 在 IS-OS dataflow 中完成什麼事？
6. 為什麼 pruning 可能降低 theoretical MACs，卻無法帶來同比例 hardware speedup？

### 練習

1. 給定 input coordinates $\{1, 4, 5\}$ 與 filter coordinates $\{0, 2, 3\}$，列出 outputs $q \in [0, 4)$ 的所有 legal products。
2. 將 dense 1-D convolution loop 改寫成 sparse-weight-only、sparse-input-only 與 joint-sparse input-stationary loops。
3. 假設 activation density 是 $0.4$、weight density 是 $0.25$。計算 independent-density estimate 的 nonzero product density，並列出這個估計忽略的兩個 hardware costs。
4. 為四個同時產生的 SCNN products 設計一個小型 accumulator banking scheme。若兩個 products 指向同一 bank，會發生什麼？
5. Paper-reading bridge：閱讀 SCNN Section III-B，解釋為什麼 Cartesian product 必須搭配 coordinate computation。

---

## 關鍵詞彙（Key Terms）

| 詞彙 | 意義 |
|---|---|
| **閘控（Gating）** | Operand 為 zero 時關閉硬體活動；省 dynamic energy，但保留 dense schedule。 |
| **跳過（Skipping）** | 從動態 schedule 移除 zero-valued operations；可省 time 與 energy，但需要 compressed traversal。 |
| **纖維（Fiber）** | Tensor 的一維切片，常表示為 coordinate-payload pairs。 |
| **協調遍歷（Concordant traversal）** | 依 compressed fiber 的自然 coordinate order 讀取。 |
| **不協調遍歷（Discordant traversal）** | 要求的存取順序不符合 stored sparse order。 |
| **纖維投影（Fiber projection）** | 平移 coordinates，使兩個 fibers 能對齊並做 intersection。 |
| **纖維交集（Fiber intersection）** | 只輸出兩個 sparse fibers 都存在的 coordinates。 |
| **輸出駐留（Output-stationary）** | 讓 output accumulators 保持 local，inputs 與 weights 串流經過。 |
| **權重駐留（Weight-stationary）** | 讓 weights 保持 local，以最大化 weight reuse。 |
| **輸入駐留（Input-stationary）** | 讓 input activations 保持 local，weights 對多個 outputs 產生 contributions。 |
| **Cartesian product** | 一組 nonzero activations 與一組 nonzero weights 的全對全乘法。 |
| **Scatter accumulator** | 將 products route 到 runtime 計算出的 output coordinates 的累加硬體。 |
| **IS-OS dataflow** | 兩階段 sparse dataflow：input-stationary 產生 intermediate tensor，接著 output-stationary reduction。 |
| **Swizzling** | 重排 tensor ranks，使後續 traversal 變成 concordant。 |
| **Joint density** | 預期 nonzero product density；在 independence 假設下常近似為 activation density 乘 weight density。 |

---

## 重點回顧（Takeaways）

- Sparse acceleration 是 scheduling 與 communication problem，不只是 compression problem。
- Gating 有用但不降低 latency；skipping 會改變實際執行 schedule。
- Single-sparse cases 較容易，因為另一個 operand 可以保持 dense 並直接 index。
- Joint sparsity 引入 coordinate matching、legality checks 與 irregular accumulation。
- SCNN 付出 scatter hardware 來直接利用兩個 compressed operands。
- ISOSceles 付出 intermediate tensor 與 swizzled traversal，使第二階段更結構化。
- Quantitative sparse claims 必須說明 source 與 scope：layer density、architecture baseline、metadata cost、measured vs. theoretical speedup。

---

## 連結（Connections）

本講結束從 L07 開始的 sparse-architecture arc，也為 L11 做鋪墊：資料應該在哪裡移動？Advanced memory technologies 可以降低某些 movement costs，但不會讓 sparse coordinate management 消失。TeAAL 的 separation of concerns 仍然有用：format 決定儲存什麼，mapping 決定 traversal，architecture 決定可用硬體，binding 決定哪個硬體資源執行哪個 operation。

---

## 附錄 — 投影片對照表

| Slide label | 章節 | Notes |
|---|---|---|
| `L06-1` | Title and framing | PDF label 與 repository lecture number 不同。 |
| `L06-2`-`L06-7` | Gating vs. skipping | 包含 Eyeriss 45% PE power slide-derived claim。 |
| `L06-8`-`L06-25` | Sparse weights only | Output-stationary、weight-stationary、fiber splitting、Cambricon-X。 |
| `L06-26`-`L06-38` | Sparse inputs only | Sparse sliding window 與 Cnvlutin。 |
| `L06-39`-`L06-51` | SCNN and joint sparsity | 加入 SCNN paper bridge。 |
| `L06-52`-`L06-66` | IS-OS and ISOSceles | Slide-derived projection、intersection、swizzling 與 reported speedup。 |
| Background | State of pruning bridge | 補充 sparse weights 從何而來，以及 speedup metrics 為何要小心。 |

---

## Source Notes

- Lecture ordering 與 loop-nest examples follow Lecture 10 slides `L06-1` through `L06-66`。
- Eyeriss gating 45% PE power reduction stated on Lecture 10 slide `L06-7`。
- Cambricon-X 與 Cnvlutin 是 Lecture 10 slides `L06-23` 與 `L06-36` through `L06-38` 的 slide-stated architecture examples；它們的 original PDFs 不在本 worker 指定的 local paper list 中。
- SCNN details and numerical results derived from `papers/L17_SCNN_Parashar_ISCA2017.pdf`，尤其 abstract、Sections III-IV、Section VI 與 Tables III-IV。
- Pruning context derived from `papers/L16_StateOfPruning_Blalock_MLSys2020.pdf`，尤其 Sections 2-6。
- ISOSceles details and speedup numbers are slide-derived from Lecture 10 slides `L06-53` through `L06-66`；original ISOSceles PDF 不在指定 local paper inputs 中。
- Worked examples 是本章為教學目的建立的 original examples。

## Uncertainty Notes

- Live lecture 可能對 Cambricon-X、Cnvlutin 或 ISOSceles 的 implementation details 有不同強調；本章只能從 slides 重建。
- Independent-density equation $d_i d_f$ 是 teaching model，不是保證。實際 activation 與 weight sparsity 可能依 layer、channel、data distribution 相關。
- Repository 既有 `assets/L10/` 可能包含 copyright-sensitive slide captures。本章不再 embed 它們，但 Worker C 未刪除 owned walkthrough files 以外的 assets。
