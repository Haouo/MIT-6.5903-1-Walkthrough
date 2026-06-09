# L08 — 稀疏架構（Sparse Architectures）

> **課程：** 6.5930/1 — Hardware Architectures for Deep Learning
> **授課教師：** Joel Emer and Vivienne Sze
> **日期：** 2026-03-02
> **主要來源：** [`Lecture/L08 - Sparse Architectures.pdf`](../../Lecture/L08%20-%20Sparse%20Architectures.pdf)

本章根據公開投影片與本地 paper PDF 重建教學敘事。為了避免直接複製投影片或論文圖，本章以文字、小型原創例子與 ASCII 圖表說明概念。

---

## TL;DR

稀疏性（sparsity）吸引人，是因為零值 operand 會產生**無效工作（ineffectual work）**：不會改變輸出的 compute、read、write 與 interconnect traffic。困難在於，零值不只是「缺少的數值」；它們會帶來不規則控制流、可變 tile occupancy、metadata、intersection logic 與 load imbalance。

Lecture 08 介紹三種 sparse acceleration features：

- **閘控（gating）：** 偵測到零值後，讓相關硬體在該 cycle idle。省 energy，但 dense schedule 的 cycle 仍然存在。
- **跳過（skipping）：** 直接移到下一個有用 coordinate。省 energy 和 time，但需要 metadata 與 traversal hardware。
- **格式（format）：** 用 sparse representation 讓零值不被儲存與搬移。省 capacity 與 bandwidth，但 metadata 可能變成瓶頸。

接著本講從 unstructured formats 擴展到 structured sparsity、hierarchical structured sparsity（HSS）與 sparse tiling。架構上的核心教訓是：只有當省下的工作大於找出、表示、路由與平衡 nonzeros 的代價時，稀疏性才真正有幫助。

## 本講解決什麼問題

Dense DNN accelerator 假設規則陣列：每個 loop iteration 取 operands、做 MAC、更新 partial sum，然後前往下一個 coordinate。Sparse tensor 打破這個假設。activation 或 weight 若為零，對應乘法可以省掉，但 accelerator 必須在花費 energy 或 time 前知道這件事。

本講處理 sparse architecture 的第一半：

1. 如何定義哪些 operation 是有用的？
2. 硬體如何避免讀取或計算 zero operands？
3. Sparse data 應如何表示？
4. 為何 structured sparsity 讓硬體更簡單？
5. 當每個 tile 的 nonzero 數量不同時，tiling 為何變困難？

Lecture 09 會把這些概念落到具體 sparse convolution dataflows。

## 為什麼本講重要

Sparse acceleration 不是 dataflow 設計之後的小優化。它會改變 memory layout、loop order、PE utilization 與 tile sizing 的意義。Dense dataflow 問：「哪個 operand 應該留在 PE 附近？」Sparse dataflow 還要問：「哪個 operand 能便宜地告訴我下一個有用 coordinate 在哪裡？」

對硬體架構師而言，稀疏性影響：

- **Energy：** 減少 value reads、metadata-dependent reads、MACs 與 partial-sum updates。
- **Latency：** skipping 可以減少 cycle count，但 gating 不行。
- **Bandwidth：** compression 可降低 DRAM/SRAM traffic，但 metadata 也必須搬移。
- **Area：** sparse decoders、intersection units、metadata buffers 與 flexible routers 都吃面積。
- **Utilization：** nonzero 數量不同，PE 可能拿到不等量工作。
- **Programmability：** 同一個數學 tensor 會因 loop order 不同而需要不同 format。

Source note：上述動機主要來自 Lecture 08 slides 2-7 與 25-26；關於 sparse accelerator modeling 的說法與 TeAAL Section 2.3 一致。

## 先備知識與心智模型

你需要熟悉：

- MAC：\(o \leftarrow o + a \times b\)。
- Dot product：\(Z = \sum_k A_k B_k\)。
- Density \(d\)：儲存座標中非零值的比例。
- Dataflow：weights、activations、partial sums 在 memory 與 PEs 之間的 loop order 與放置方式。

核心心智模型是：

> Sparse accelerator 是 dense accelerator 加上一台**座標機器（coordinate machine）**。

Dense 部分仍然負責乘值。Coordinate machine 決定要拜訪哪些 coordinate、payload 在哪裡、結果要送到哪個 output coordinate。

## 學習目標（Learning Objectives）

讀完本講後，你應該能夠：

- 定義 effectual operation 與 ineffectual operation。
- 區分 total operations 與 total operations performed。
- 解釋 gating、skipping、format 作為三種不同 sparse acceleration features。
- 比較 single-sided 與 dual-sided intersection。
- 為小型 sparse dot product 計算 read/cycle counts。
- 以 compression efficiency 與 access efficiency 比較 uncompressed、bitmask、coordinate-payload、run-length formats。
- 解釋 metadata 為什麼可能主導 unstructured sparse storage。
- 解釋 structured sparsity 中 flexibility/efficiency 的取捨。
- 計算 hierarchical structured sparsity pattern 的 effective density。
- 概念性說明 sparse tiling、tile occupancy、overbooking、Tailors 與 Swiftiles。
- 把 sparse formats 連到後續 fibertree 與 sparse dataflow 講次。

## 教科書式主敘事

### 1. 從零值到無效工作

Lecture 08 從兩種 sparsity 來源開始：

- **Activation sparsity：** ReLU、input correlation、graph-like representations 與其他 input-dependent effects 會讓 activations 變成零。
- **Weight sparsity：** pruning 可以把訓練後的 weights 設為零。

真正有用的區分不是「零值 vs. 非零值」，而是 **effectual vs. ineffectual operation**。

對乘法而言，只有兩個 operands 都非零時，operation 才是 effectual。若任一 operand 為零，\(a \times b = 0\)，乘法不會改變 partial sum。對加法而言，加上零也 ineffectual，因為 \(x+0=x\)。

Lecture 08 使用兩個 count：

- \(N_\text{total}=N_\text{effectual}+N_\text{ineffectual}\)。
- \(N_\text{performed}=N_\text{effectual}+N_\text{unexploited ineffectual}\)。

目標不只是讓 tensor 稀疏，而是在不讓每個剩餘 operation 變太貴的前提下，降低 \(N_\text{unexploited ineffectual}\)。

### 2. 利用稀疏性的代價是不規則性

Dense tensor 很規則：第 \(k\) 個 loop iteration 通常對應到第 \(k\) 個 stored value。Sparse tensor 破壞了這件事：

- nonzero 數量會在 vectors、rows、channels、tiles、layers 之間變動。
- nonzero 位置可能要 decode metadata 後才知道。
- 分到 dense sparse tile 的 PE 可能比拿到 empty tile 的 PE 跑更久。
- Compressed tensor 可能必須先讀 metadata，才能讀 value。

這就是投影片說 exploiting sparsity makes processing irregular 的意思。Accelerator 只有在支付 metadata、decoding、coordinate arithmetic、intersection 與 scheduling 的成本後，才省得到工作。

### 3. Sparse Acceleration Features：gating、skipping、format

本講把 sparse hardware mechanisms 分成 **Sparse Acceleration Features（SAFs）**：

| SAF | 做什麼 | 省 energy？ | 省 cycles？ | cycle 前需要 sparse metadata？ |
|---|---|---:|---:|---:|
| Gating | 偵測零值，讓部分硬體 idle | 是 | 否 | 通常不需要 |
| Skipping | 直接跳到有用 nonzero coordinate | 是 | 是 | 需要 |
| Format | 儲存與搬移 payload 加 metadata，而非所有零值 | 是 | 有時 | 若要 skipping 則需要 |

關鍵轉折是：gating 可以在 dense schedule 中即時發現零值；skipping 必須事先知道要跳去哪裡。因此 representation format 不再只是軟體資料結構，而是架構設計的一部分。

### 4. Dot-product 範例

本講使用一維 dot product：

```text
A = [0, 0, c, d, 0, f]
B = [g, h, 0, j, k, l]
Z = A dot B = d*j + f*l
```

共有六個 algorithmic multiply positions，但只有 coordinates \(3\) 與 \(5\) 是 effectual。Nonzero coordinate sets 為 \(A_\text{nz}=\{2,3,5\}\) 與 \(B_\text{nz}=\{0,1,3,4,5\}\)，intersection 是 \(\{3,5\}\)。

| 策略 | Cycles | A value reads | B value reads | Computes performed | 說明 |
|---|---:|---:|---:|---:|---|
| Dense baseline | 6 | 6 | 6 | 6 | 拜訪每個 coordinate |
| Gate \(B \leftarrow A\) | 6 | 6 | 3 | 3 | A 是 leader；A 非零時才讀 B |
| Skip \(B \leftarrow A\) | 3 | 3 | 3 | 3 | 只拜訪 A 的 nonzero coordinates |
| Dual-sided skip \(A \cap B\) | 至少 2 | 2 | 2 | 2 | 只拜訪兩者皆非零的 coordinate |

Source note：這些 counts 來自 Lecture 08 slides 9-20。與投影片一致，表格不把 metadata reads 算入 value reads。

常見誤解是「跳過 A 的零值就最佳了」。不是。若 A 在 coordinate 2 非零但 B 為零，\(c \times 0\) 仍然是 ineffectual。完整 work reduction 需要找出兩個 operands 的 intersection。

### 5. Single-sided 與 dual-sided intersection

**Single-sided intersection** 選一個 leader。若 A 是 leader，硬體拜訪 A 的 nonzero coordinates，然後在同座標詢問 B。這較簡單，但收益取決於 leader 是否能預測 useful work。

**Dual-sided intersection** 把兩個 operands 都當 sparse lists，只輸出 matching coordinates。簡單 merge-style intersection 如下：

```text
A coordinates: 2, 3, 5
B coordinates: 0, 1, 3, 4, 5

compare 2 and 0 -> advance B
compare 2 and 1 -> advance B
compare 2 and 3 -> advance A
compare 3 and 3 -> emit 3
compare 5 and 4 -> advance B
compare 5 and 5 -> emit 5
```

Dual-sided skipping 很強，但比較次數取決於資料。硬體常會限制每個 cycle 能走幾步 metadata；若沒有很快找到 match，PE 可能 idle。Lecture 08 提到 ExTensor 使用 binary search over remaining coordinates，當下一個 match 很遠時可能有效。

### 6. Format：compression efficiency 與 access efficiency

Sparse format 必須回答兩個問題：

1. **Compression efficiency：** 相對 dense storage 需要多少 bits？
2. **Access efficiency：** 硬體能多便宜地找下一個 nonzero coordinate，或測試某 coordinate 是否存在？

本講四種 formats：

| Format | Metadata 想法 | 適合 density | Access behavior |
|---|---|---|---|
| Uncompressed | 無 metadata；每個 value 都存 | Dense | 直接 coordinate access，但不壓縮 |
| Bitmask | 每個 coordinate 一個 bit | 中度稀疏 | 掃描 bits 或做 bit operations |
| Coordinate payload | 每個 nonzero 存 coordinate | 高稀疏 | 直接讀下一個 nonzero |
| Run-length encoding | 存 nonzeros 之間的零值數量 | 高稀疏且 zero runs 長 | 累加 run lengths |

對 \(A=[0,0,c,d,0,f]\)，value 為 8-bit：

- Uncompressed：\(6 \times 8 = 48\) bits。
- Bitmask：\(6 \times 1 + 3 \times 8 = 30\) bits。
- Coordinate payload 使用 3-bit coordinates：\(3 \times 3 + 3 \times 8 = 33\) bits。
- RLE 使用 3-bit runs：\(3 \times 3 + 3 \times 8 = 33\) bits。

對長度 16、8-bit values、4-bit coordinates 的 vector：

| Nonzeros | Density | Uncompressed | Bitmask | Coordinate payload | 4-bit RLE |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.25% | 128 bits | 24 bits | 12 bits | 12 bits |
| 8 | 50% | 128 bits | 80 bits | 96 bits | 96 bits |
| 16 | 100% | 128 bits | 144 bits | 192 bits | 192 bits |

教訓不是「compressed 一定比較好」。在 100% density 下，此例所有 compressed formats 都比 uncompressed 更差，因為 metadata 是額外成本。

### 7. Metadata 的硬體意義

Metadata 不是靜態描述，它會改變 datapath：

- Bitmask 需要 bit reads，常搭配 popcount 或 bit-scan logic。
- Coordinate payload 需要 coordinate storage、coordinate comparison，有時還要在另一個 operand 中 random lookup。
- RLE 需要 accumulation state 來重建 absolute coordinates。
- Dual-sided skipping 需要 intersection unit。

Slide 35 引用 Han et al. 的重點是：對 unstructured sparsity，index metadata 可能約佔 storage 的一半。這是警訊，不是普世常數；它取決於 value precision、coordinate width 與 sparsity pattern。

### 8. Structured sparsity

Unstructured sparsity 允許 nonzeros 出現在任何位置。對 model design 很彈性，但硬體昂貴，因為每個 nonzero 可能都要 coordinate metadata。

**Structured sparsity** 限制 nonzero 的合法位置。硬體好處是 search space 更小，metadata 更精簡。

常見 \(G:H\) pattern 表示：每 \(H\) 個值中恰好有 \(G\) 個非零。NVIDIA 2:4 是本講標準例子：每四個值恰好兩個非零，也就是 50% density。

好處是 decode 簡單。限制也很直接：一種 \(G:H\) ratio 只支援一種 sparsity degree。若 layer 想要 30% 或 80% sparsity，固定 2:4 hardware 無法平滑地轉成等比例 savings。

### 9. Hierarchical Structured Sparsity

Hierarchical Structured Sparsity（HSS）在多個巢狀粒度上組合簡單 \(G:H\) rules。假設使用 \(3:4 \rightarrow 2:4\)：

- 外層規則：4 個 blocks 中保留 3 個 non-empty blocks。
- 內層規則：每個 surviving block 中保留 4 個值中的 2 個。

結果 density 是 \((3/4)(2/4)=3/8=37.5\%\)，sparsity 是 \(62.5\%\)。

Lecture 08 的 HSS 例子把外層 \(\{4:4,4:5,4:6,4:7\}\) 與內層 \(\{4:4,2:4,1:4\}\) 組合，得到 \(4 \times 3 = 12\) 種 sparsity degrees。硬體不需要十二種獨立 decoder，而是組合簡單的 per-rank decoders。

這是一個 format-level 技巧，但有 architecture-level 後果：model 可選更多 sparsity degrees，而硬體仍接近少數簡單 structured primitives。

### 10. Sparse tiling 與 overbooking

Dense tiling 問：「哪個矩形 tile 放得進 buffer？」Sparse tiling 問：「這個 tile 會有多少 nonzeros？」

兩個形狀相同的 sparse tiles 可能有很不同的 occupancy。若 buffer 依最 dense tile 設計，平均 tile 會浪費容量。若依 equal nonzero count 切 tile，coordinate ranges 會不規則，使另一個 operand 難以 tile。

本講比較：

- **Uniform occupancy：** nonzero count 平衡，但 shape 不規則。
- **Uniform shape：** shape 規則，但 nonzero count 變動大。

**Overbooking** 選擇比 worst-case buffer capacity 更大的 nominal tile，賭大多 tiles 因為稀疏而實際放得下。若某 tile nonzeros 太多，溢出的部分是 **bumped data**。Tailors 會把 bumped data streaming，而不是放入 buffer。Swiftiles 透過 random sampling 估計 tile occupancy，決定 overbooking 程度。

硬體含義很微妙：overbooking 增大 average tile size，提升 reuse，同時接受偶發 streamed overflow。它是對 worst-case dense tiling 假設的一種受控違反。

## Worked Examples

### 範例 1：Density 與預期 useful work

若 weight density \(d_W=0.4\)，activation density \(d_A=0.5\)，並在 toy estimate 中假設 nonzero locations 獨立，dual-sided skipping 大約會拜訪 dense multiply positions 的 \(d_W d_A = 0.2\)。

Dense loop 有 1000 次乘法時，idealized effectual count 是 \(1000 \times 0.2 = 200\)。只用 activations 當 leader 的 single-sided scheme 會拜訪 \(1000 \times 0.5 = 500\) 個 positions。它有省工作，但仍會做許多對應 weight 為零的乘法。

Teaching interpretation：independence 只是教學簡化。真實 DNN sparsity 可能依 layer、channel、input 而相關。

### 範例 2：Metadata 可能打敗 compression

假設 values 8 bits、vector length 16、coordinate 需要 4 bits。Coordinate payload size 是 \(n_\text{nz}(8+4)\)。Dense size 是 \(16 \times 8=128\) bits。

Coordinate payload 只有在 \(12n_\text{nz}<128\)，也就是 \(n_\text{nz}<10.67\) 時比 dense 小。如果 vector 有 11 個以上 nonzeros，coordinate payload 反而更大。

硬體意義：若 layer density 很高，sparse decoding 會花 area 與 bandwidth 搬 metadata，卻沒有移除多少工作。

### 範例 3：HSS effective sparsity

對 \(4:6 \rightarrow 1:4\)：

- Outer density 是 \(4/6\)。
- Inner density 是 \(1/4\)。
- Effective density 是 \((4/6)(1/4)=1/6\)。
- Effective sparsity 是 \(1-1/6=5/6\approx 83.3\%\)。

重點不是這個 pattern 一定最好，而是 nested ratios 會相乘，讓少量硬體模式覆蓋許多 sparsity degrees。

## 關鍵方程式與讀法

### Effectual work

\[
N_\text{performed}=N_\text{effectual}+N_\text{unexploited ineffectual}.
\]

這是 accounting identity。Sparse hardware 只有在降低第二項，且剩餘 operation 的 overhead 合理時，才改善效能。

### 理想獨立 two-sided work

\[
N_\text{effectual}\approx d_A d_B N_\text{dense}.
\]

這是教學近似。若 operand A 非零機率是 \(d_A\)，operand B 非零機率是 \(d_B\)，兩者同時非零機率是 \(d_A d_B\)。它說明 dual-sided intersection 為何可能遠優於只利用單邊 sparsity。

### HSS density

\[
d_\text{HSS}=\prod_i \frac{G_i}{H_i}.
\]

每個 hierarchy level 保留 \(G_i/H_i\) 的 entries；一個 value 必須通過所有層，所以 kept fractions 相乘。

## 硬體意涵（Hardware Implications）

- **Gating：** 需要 zero detection 與 enable signals；保留 dense timing，控制簡單但 latency 不變。
- **Skipping：** 需要 metadata 提早產生 next coordinates；會改變 cycle count，也可能造成 pipeline bubbles。
- **Dual-sided intersection：** 需要 coordinate comparison/search logic；performance 取決於 sparsity pattern 與 metadata format。
- **Compression：** 只有 metadata 小於省下的 zero payloads 時，才省 capacity 與 bandwidth。
- **Structured sparsity：** 降低 decoder complexity 與 metadata width，但限制 model。
- **HSS：** 擴大 sparsity-degree 選單，而不需為每個 degree 做完全獨立 decoder。
- **Sparse tiling：** buffer sizing 必須考慮 occupancy distributions，而不是只有 tensor shape。
- **Parallel sparse execution：** 不同 tiles nonzero 數量不同，PEs 會 load imbalance。

## 常見誤區（Common Misconceptions）

### 誤區：Sparsity 自動帶來 speedup。

Sparsity 只提供機會。Speedup 需要 skipping cycles。Gating 可以省 energy，但不降低 latency。

### 誤區：最會壓縮的 format 一定最好。

Compression 很好的 format 可能 access efficiency 很差。若硬體每個有用 value 都得掃描、累加或 random probe metadata，省下的 payload bits 不一定變成 throughput。

### 誤區：Single-sided skipping 等於 dual-sided skipping。

Single-sided skipping 只避開 leader 的 zeros。若 follower 在 leader nonzero coordinate 為零，operation 仍然 ineffectual。

### 誤區：Structured sparsity 只是為了簡化硬體而犧牲 model quality。

這個 tradeoff 的確存在，但 structured sparsity 也讓 savings 更可預測。HSS 嘗試透過組合簡單結構追回一些 flexibility。

### 誤區：Sparse tiles 應該依 worst case sizing。

Worst-case sizing 在 occupancy 高度變動時會浪費大多 buffer capacity。Overbooking 的價值就在於 average occupancy 可能遠低於 maximum occupancy。

## 與前後講次的連結（Connections）

- **L05-L06 mapping/dataflows：** sparse formats 只有在 loop order 對齊 storage order 時才有效。這仍是 mapping 問題，只是 metadata 也進入 loop。
- **L07 sparsity/pruning：** pruning 產生 weight sparsity；L08 問硬體要如何利用它。
- **L09 sparse architectures part 2：** fibertrees、CSR/CSC、projection 與 intersection 會形式化本講概念。
- **L10 sparse architectures part 3：** TeAAL 與 sparse accelerator specifications 會提供描述這些設計選擇的語言。
- **Lab 4/SparseLoop：** 本講的 SAF 詞彙會成為分析 sparse accelerator tradeoffs 的 modeling vocabulary。

## Paper Bridge: TeAAL

### Bibliographic identity

- **Title:** TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators
- **Authors:** N. Nayak et al.
- **Year / venue:** MICRO 2023
- **Used in lecture(s):** L01, L08, L09, L10
- **Local PDF:** `papers/TeAAL.pdf`

### Problem addressed

TeAAL 處理 sparse tensor accelerators 難以規格化與比較的問題。Sparse accelerators 不只 PE array 不同，loop order、tensor formats、partitioning、rank transformations、sparse orchestration 都不同；若只用非正式描述，很難比較。

### Core idea

論文用 cascades of mapped Einsums 加上 fibertrees 的 content-preserving transformations 表示 sparse accelerators。它把 computation 與 mapping/format 分開，對應本講把 skipping、format、dataflow 分開討論。

### Relevance to this lecture

Lecture 08 用 TeAAL concern stack 說明 **Format** 與 **Mapping** 分離但耦合。Compressed format 只有在 mapping 能 concordantly traverse 時才有用。Sparse tiling 也是 mapping/format 問題，因為 shape-based partitioning 與 occupancy-based partitioning 會暴露不同取捨。

### Key claims used in this chapter

- Sparse tensors 可自然表示為帶有 missing coordinate/payload pairs 的 fibertrees；見 TeAAL Section 2.1。
- Einsums 指定 computation 但不指定 iteration order；mapping 選 loop order 並影響 locality 與 load balance；見 Sections 2.2 與 2.3。
- Sparse tensors 通常被 compression 移除 zero elements，但 sparse execution 會引入 memory footprint variation、transfer imbalance 與 compute load imbalance；見 Section 2.3。
- Rank flattening、rank partitioning、rank swizzling 捕捉常見 sparse data orchestration；見 Section 3.2。

### What students should remember

1. Sparse architecture 不只是 PE microarchitecture 問題。
2. Format、mapping、binding、architecture 必須一起規格化。
3. Fibertree transformations 是討論 sparse layout 與 tiling 的精確語言。
4. Load imbalance 是 sparse design 的一等問題。

### Limitations and assumptions

TeAAL 是 modeling/specification framework，不是單一 accelerator。本章用它支撐概念，不把它當成某個 SAF 最佳的證明。

### Suggested insertion points

在解釋 format choice 為何耦合 traversal order、sparse tiling 為何需要 occupancy-aware partitioning、後續為何引入 fibertrees 時使用 TeAAL。

## Paper Bridge: SCNN

### Bibliographic identity

- **Title:** SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks
- **Authors:** A. Parashar et al.
- **Year / venue:** ISCA 2017
- **Local PDF:** `papers/L17_SCNN_Parashar_ISCA2017.pdf`

### Problem addressed

SCNN 問：如何在 convolution layers 中同時利用 pruned weights 與 ReLU-induced activation sparsity，並讓 activations/weights 在大部分 computation 中保持 compressed？

### Core idea

SCNN 使用 planar-tiled input-stationary Cartesian-product sparse dataflow。它把 nonzero weights 與 nonzero activations 的向量送入 multiplier array，做 Cartesian product，並把 products scatter 到 output accumulators。

### Relevance to this lecture

Lecture 08 把 SCNN 當成 single-sided skipping 的例子，也用它說明 sparsity benefits 並非免費。SCNN 的 scatter network、compressed buffers 與 metadata handling 正是本講警告的 overheads。

### Key claims used in this chapter

- Abstract 說 SCNN 利用 pruning 造成的 zero-valued weights 與 ReLU 造成的 zero-valued activations，並用 compressed encoding 降低 transfers 與 storage。
- Section II 報告，在論文量測的 density products 下，typical layers 可把 work 降低約 4 倍，最高可到約 10 倍。
- Section III 介紹 PT-IS-CP-sparse dataflow，說明 input-stationary Cartesian-product computation 為何匹配 sparse weights/activations。
- Section IV 描述含 compressed storage、all-to-all multiplication 與 scatter accumulation 的 PE。
- Conclusion 說 SCNN 同時利用 weight 與 activation sparsity，且當 weights 與 activations 各自低於約 85% density 時，比 dense architectures 更有效率。

### What students should remember

1. Dual-sparse work reduction 需要 values 與 coordinates。
2. Cartesian products 增加 useful multiply opportunities，但 output addresses 會 scattered。
3. Compression 只有在 decoders 與 routers 跟得上時才省 bandwidth。

### Limitations and assumptions

SCNN 針對 CNN inference，依賴 compressed sparse blocks 與 PE 中足夠 nonzero work；它不是通用 sparse tensor accelerator。

### Suggested insertion points

討論避免 zero work 為何需要額外 routing/metadata machinery 時使用 SCNN，也可作為 Lecture 09 sparse convolution dataflows 的預告。

## Paper Bridge: Eyeriss v2

### Bibliographic identity

- **Title:** Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices
- **Authors:** Y.-H. Chen et al.
- **Year / venue:** JETCAS 2019
- **Local PDF:** `papers/L17_EyerissV2_Chen_JETCAS2019.pdf`

### Problem addressed

Eyeriss v2 處理 compact and sparse DNNs 中 layer shapes 與 sparsity patterns 變化大的問題。目標是在 dense reuse assumptions 不再成立時，仍維持 throughput 與 energy efficiency。

### Core idea

它結合 hierarchical mesh NoC 與 sparse PE support。Sparse PE 以類 CSC compressed format 儲存 activations 與 weights，在 compressed domain 直接 skipping zeros，並用 SIMD support 恢復 utilization。

### Relevance to this lecture

Eyeriss v2 展示 gating 與 skipping 的差異。Original Eyeriss 對 zero activations 使用 gating；Eyeriss v2 讓資料在 on-chip 保持 compressed 並 skipping zeros 來提升 throughput。

### Key claims used in this chapter

- Section IV 說 original Eyeriss 透過 gating logic/data accesses 利用 input-activation zeros，而 Eyeriss v2 進一步 skipping zeros 以改善 throughput 與 energy。
- Section IV 描述 activations/weights 的 CSC encoding，並指出 compressed-domain processing 可以不花額外 cycles 地跳過 zeros。
- Section V 報告在該 paper 評估設定下，sparse AlexNet 與 sparse MobileNet 有大幅改善，同時也指出 workload imbalance 與 layer-shape limitations。

### What students should remember

1. 從 gating 到 skipping 會改變 PE pipeline 與 storage format。
2. Sparse support 增加 control 與 storage overhead。
3. 即使設計細緻，workload imbalance 仍存在。

### Limitations and assumptions

量化結果綁定 Eyeriss v2 的 65 nm implementation、benchmark models、batch size 與 baselines。本章把它當成 design tradeoffs 的證據，不把其 speedup 當成普世常數。

### Suggested insertion points

用於 gating/skipping 區分，以及 compressed formats 如何影響 throughput 的討論。

## Paper Bridge: The State of Sparsity in Deep Neural Networks

### Bibliographic identity

- **Title:** The State of Sparsity in Deep Neural Networks
- **Authors:** D. Blalock et al.
- **Year / venue:** MLSys 2020
- **Local PDF:** `papers/L16_StateOfPruning_Blalock_MLSys2020.pdf`

### Problem addressed

該 paper 調查 pruning methods，指出 pruning research 常有 comparison 與 metrics 不一致的問題。

### Core idea

它區分 unstructured pruning 與 structured pruning，並主張 pruning 應作為 efficiency/quality tradeoff curve 評估，而不是單一 compression number。

### Relevance to this lecture

Lecture 08 的 structured sparsity 段落依賴一個 model-side 事實：sparsity pattern 是設計選擇。Hardware-friendly pattern 可簡化 metadata 與 traversal，但可能改變模型 accuracy/efficiency frontier。

### Key claims used in this chapter

- Section 2 將 pruning 定義為產生 masked 或 removed parameters 的 model，並區分 unstructured/structured pruning。
- Section 2 強調 model efficiency 與 quality 的 tradeoff。
- 文獻回顧警告 reported compression 與 speedup metrics 不能互換。

### What students should remember

1. Sparsity 由 model decisions 與 hardware decisions 共同塑造。
2. Structured sparsity 只有在 model 能承受限制時才有價值。
3. Theoretical speedup 與 realized hardware speedup 是不同 metrics。

### Limitations and assumptions

該 paper 是 pruning evaluation，不是 sparse accelerator design。本章用它說明 hardware-friendly sparsity patterns 為何仍需 accuracy tradeoff 評估。

## 獨立學習指南（Standalone Study Guide）

建議順序：

1. 不看答案重做 dot-product accounting table。
2. 解釋為何 skipping 需要 format metadata，而 gating 不一定需要。
3. 對每個 sparse format 問：「我要如何找下一個 nonzero？」
4. 用 kept fractions 相乘計算一個 HSS density。
5. 把 overbooking 解釋成 average-case buffer-utilization strategy。

## 自我檢核問題

1. 為何 gating 省 energy 但不省 cycles？
2. Dot-product 例子中，為何 \(A\)-leader skipping 做三次 computes，但 effectual 只有兩次？
3. 為何 bitmask compression 在 100% density 時可能比 uncompressed storage 更差？
4. RLE 需要什麼 coordinate state，而 coordinate payload 不需要？
5. 固定 2:4 accelerator 為何無法任意利用 80% sparsity？
6. HSS 為何能用少量硬體模式產生更多 sparsity degrees？
7. Sparse tiling 為何讓「最大可放入 tile」變困難？
8. Overbooking 中 bumped data 是什麼？

## 練習

1. **Format calculation：** 長度 32 vector 有六個 nonzeros、8-bit values、5-bit coordinates。計算 uncompressed、bitmask、coordinate-payload 與 5-bit RLE 的 storage size。
2. **Intersection trace：** 用 merge-style algorithm 交集 \(A=\{1,4,9,10\}\) 與 \(B=\{0,4,5,10,11\}\)。計算 metadata comparisons。
3. **Leader choice：** 若 \(A\) density 0.2、\(B\) density 0.8，single-sided skipping 應選哪個 operand 當 leader？為什麼？
4. **HSS design：** 選兩層 \(G:H\) 產生 75% sparsity。說明硬體與 model-flexibility tradeoff。
5. **Tiling reasoning：** 描述一個 uniform-shape tiling 會浪費 buffer capacity 的 sparse tensor distribution，再說明 overbooking 如何改變 average tile size。
6. **Paper bridge：** 用 SCNN 解釋為何 sparse accelerator 即使少做乘法，也可能需要 scatter network。

## 關鍵詞彙（Key Terms）

| Term | Definition |
|---|---|
| **Activation sparsity（激活稀疏性）** | Activations 中的零值，常與 input 有關；硬體需 runtime detect 或 encode。 |
| **Weight sparsity（權重稀疏性）** | Trained weights 中的零值，常由 pruning 產生；inference 前通常已知。 |
| **Effectual operation** | 會改變 output 的 operation，例如兩個 nonzero operands 相乘。 |
| **Ineffectual operation** | 含 zero operand 或 zero addend、無法影響最終值的 operation。 |
| **Gating（閘控）** | 偵測零值後在該 cycle suppress reads 或 compute；省 energy 不省 time。 |
| **Skipping（跳過）** | 直接前進到有用 coordinates；省 time 與 energy，但需要 metadata/traversal logic。 |
| **Format（格式）** | Values 與 coordinates 在記憶體中的表示法。 |
| **Metadata（後設資料）** | bitmasks、coordinates、run lengths、segment pointers、offsets 等非 payload 資訊。 |
| **Single-sided intersection** | 一個 operand 的 nonzeros 驅動 traversal；另一個 operand 作 follower 被檢查或讀取。 |
| **Dual-sided intersection** | 兩個 operands 的 coordinate streams 被求交，只處理 matching nonzero coordinates。 |
| **Bitmask** | 每個 coordinate 一個 bit，表示 payload 是否 nonzero。 |
| **Coordinate payload** | 每個 nonzero value 與其 coordinate 一起儲存。 |
| **Run-length encoding（RLE）** | 儲存每個 nonzero 前方有幾個 zeros。 |
| **Structured sparsity** | 限制 sparsity 形成可預測 pattern，以降低 metadata/decoder cost。 |
| **\(G:H\) sparsity** | 每 \(H\) 個 values 中恰好有 \(G\) 個 nonzeros。 |
| **HSS** | Hierarchical Structured Sparsity；巢狀 \(G:H\) patterns，其 densities 相乘。 |
| **Tile occupancy** | Sparse tile 中 nonzeros 的數量。 |
| **Overbooking** | 選擇比 worst-case buffer capacity 更大的 nominal tile，因為大多 sparse tiles 平均可放入。 |
| **Bumped data** | Overbooked buffer 放不下而必須 streaming、不能在 buffer 中重用的 nonzeros。 |
| **Workload imbalance** | 不同 PEs 因 nonzero count 不同而工作量不均。 |

## 重點回顧（Takeaways）

- Sparsity 只有在硬體能減少 unexploited ineffectual work 時才有價值。
- Gating 省 energy；skipping 省 energy 與 time，但需要 format metadata。
- Compression efficiency 與 access efficiency 必須一起看。
- Metadata 是硬體成本，不只是註解。
- Structured sparsity 用 model flexibility 換 decoder simplicity；HSS 嘗試兼顧更多 sparsity degrees。
- Sparse tiling 的核心難題是 occupancy variation；overbooking 用 average case 提高 buffer utilization。

## 附錄 — 投影片對照表（Slide-to-Section Map）

| Slide range | Chapter section | Notes |
|---|---|---|
| L08-1 | Title | 行政資訊 |
| L08-2 to L08-8 | 本講問題、TL;DR、SAF overview | 擴充定義與 irregularity |
| L08-9 to L08-22 | Dot-product；gating vs. skipping | 重寫為 worked examples 與 accounting table |
| L08-23 to L08-48 | Representation formats | 擴充 bit-count calculations 與 access-efficiency |
| L08-49 to L08-66 | Structured sparsity and HSS | 擴充 \(G:H\) 與 HSS density equations |
| L08-67 to L08-93 | Sparse tiling and overbooking | 重寫為 buffer-utilization 敘事 |
| L08-94 to L08-96 | Dataflow interplay, summary, reading | 整合進 hardware implications、connections、source notes |

## Source Notes

- Lecture ordering 與 SAF definitions 依 Lecture 08 slides 2-7。
- Dot-product counts 依 Lecture 08 slides 9-22。
- Format bit-count examples 依 Lecture 08 slides 27-36 與 37-47。
- Structured sparsity 與 HSS 依 Lecture 08 slides 50-66。
- Sparse tiling、Tailors、Swiftiles 依 Lecture 08 slides 68-93。Worker B input 未提供 Tailors/Swiftiles 與 HighLight 的 local PDFs，因此該部分 paper-specific claims 保持 slide-anchored。
- TeAAL discussion 使用 `papers/TeAAL.pdf`，尤其 Sections 2.1、2.2、2.3、3.2。
- SCNN discussion 使用 `papers/L17_SCNN_Parashar_ISCA2017.pdf`，尤其 Sections II-IV 與 VIII。
- Eyeriss v2 discussion 使用 `papers/L17_EyerissV2_Chen_JETCAS2019.pdf`，尤其 Sections IV-V。
- Pruning context 使用 `papers/L16_StateOfPruning_Blalock_MLSys2020.pdf`，尤其 Sections 2 與 3。

## Uncertainty Notes

- 本章根據 slides 與 papers 重建可能的口頭講解；live lecture 的例子重點可能不同。
- 本章避免 embedded slide images。`assets/L08/` 下既有檔案可能仍有 copyright sensitivity，但它們不在 Worker B 要求的 write scope。
- Quantitative claims 若未由 local PDFs 獨立確認，均標為 slide-derived。
