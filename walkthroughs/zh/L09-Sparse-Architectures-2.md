# L09 — 稀疏架構，第二部分（Sparse Architectures, Part 2）

> **課程：** 6.5930/1 — Hardware Architectures for Deep Learning
> **授課教師：** Joel Emer and Vivienne Sze
> **日期：** 2026-03-04
> **主要來源：** [`Lecture/L09-Sparse_Architectures-2.pdf`](../../Lecture/L09-Sparse_Architectures-2.pdf)

本章根據公開投影片與本地 paper PDF 重建講解。原投影片大量使用動畫式 build slides；本 walkthrough 將重複 build steps 摺疊成自含敘事。

---

## TL;DR

Lecture 08 建立 sparse hardware 詞彙：gating、skipping、format、metadata、structured sparsity。Lecture 09 把這套詞彙變成設計 sparse convolution accelerator 的工作方法。

核心概念是**纖維樹（fibertree）抽象**。Tensor 被表示為 ranks、fibers、coordinates、payloads。一旦用這個角度看 sparse tensors，sparse acceleration 就變成：哪個 fiber operation 便宜？

- `getNext()` 在**協調遍歷（concordant traversal）**時便宜，也就是 loop order 與 storage order 對齊。
- `getPayload(coordinate)` 在 compressed formats 中可能昂貴，因為它是 random access。
- **投影（projection）**在 tensors 之間轉換 coordinate，例如 convolution 中的 \(w=q+s\)。
- **求交（intersection）**只保留兩個 sparse operands 都存在的 coordinates。

接著本講把這些 primitives 用到 sparse convolution：sparse weights、sparse inputs，以及兩者同時稀疏。最後以 SCNN 與 ISOSceles 展示：相同數學 sparsity 可以導向非常不同的硬體資料流。

## 本講解決什麼問題

Lecture 08 說 skipping 需要知道 nonzeros 在哪裡。Lecture 09 接著問：tensor 壓縮之後，硬體要如何遍歷它，並仍然正確計算 DNN operation？

問題不只是「儲存 sparse tensors」。真正問題是：

1. 用能暴露 coordinates 的方式表示 sparse tensors。
2. 讓 loop order 與 representation order 對齊。
3. 用 coordinate projection 連接 convolution operands。
4. 當多個 operands 都 sparse 時使用 intersection。
5. 選擇能平衡 reuse、skipping、routing、utilization 的 dataflow。

這是從 sparse format design 走向 sparse accelerator microarchitecture 的橋。

## 為什麼本講重要

在 dense convolution 中，改變 loop order 主要改變 reuse。在 sparse convolution 中，改變 loop order 可能決定 inner loop 是便宜的 iterator，還是昂貴的 random metadata lookup。這個限制更尖銳。

例如 coordinate/payload list 很適合「拜訪下一個 stored nonzero」。但若 loop 問「coordinate 137 是否存在？」它就沒那麼方便。這就是 sparse architecture design 需要 formal traversal vocabulary 的原因。

硬體意涵包括：

- **Bandwidth：** 只有 concordant traversal 時，compressed formats 才能少讀 values。
- **Latency：** 只有能快速產生下一個 useful coordinate，skipping 才成立。
- **Area：** coordinate generators、position generators、intersection units、scatter networks 變成一等硬體 blocks。
- **Utilization：** sparse work 不規則；position split 可平衡工作，但可能破壞幾何規則性。
- **Correctness：** \(w=q+s\)、\(q=w-s\) 等 projection 必須把 products 路由到正確 output coordinate。

## 先備知識與心智模型

你應記得：

- Lecture 08 的 gating、skipping、format 區分。
- 1-D convolution：\(O[q] = \sum_s I[q+s]F[s]\)。
- 早先 Einsum 講次中的 tensor ranks 與 coordinates。
- Dataflow：哪個 loop stationary、哪個 loop streaming、哪個 loop parallelized。

心智模型：

> Sparse convolution 是 dense convolution 加上顯式 coordinate bookkeeping。

MAC 本身仍然簡單。困難在於產生正確的 nonzero payload pairs，並把每個 product 送到正確 partial sum。

## 學習目標（Learning Objectives）

讀完本講後，你應該能夠：

- 定義 rank、coordinate、point、fiber、payload、fibertree。
- 解釋 coordinate 與 position 的差異。
- 比較 uncompressed arrays、coordinate/payload lists、bitmasks、RLE、hash tables、CSR、CSC、COO。
- 解釋 `getPayload()` 與 `getNext()`，以及它們的成本為何依 representation 而變。
- 區分 concordant、partially discordant、discordant traversal。
- 在 1-D convolution 中使用 \(w=q+s\)、\(q=w-s\)、\(s=w-q\) 作 coordinate projections。
- 寫出 sparse weights、sparse inputs、two-sparse convolution 的 loop nests。
- 解釋 Cnvlutin 如何利用 activation sparsity。
- 解釋 SCNN 如何使用 input-stationary Cartesian-product sparse multiplication，以及為何需要 scatter。
- 解釋 ISOSceles 用 IS-OS pipeline 與 rank swizzling 想解決什麼。

## 教科書式主敘事

### 1. Sparse tensors 需要與表示法無關的抽象

本講從基本 tensor 詞彙開始：

- **Rank** 是 tensor dimension。
- **Coordinate** 是某 rank 上的 index。
- **Point** 是一組 coordinates，每個 rank 一個。
- **Payload** 可以是 scalar value，也可以是指向下一層 fiber 的 pointer/reference。
- **Fiber** 是某 rank 上一組有序 coordinate/payload pairs。
- **Fibertree** 是一棵 tree，levels 是 ranks，edges 帶 fibers。

Dense \(3 \times 3\) matrix 中，每個 row/column coordinate 都存在。Sparse matrix 中，全零 subtrees 可被省略。像 \((2,1)\) 這種 point，會先在 H-rank fiber 找 coordinate 2，再到其 W-rank child fiber 找 coordinate 1。

這個抽象重要，因為它把 semantics 與 layout 分離。Tensor 有數學 coordinates；implementation 決定如何儲存 coordinates 與 payloads。

### 2. Coordinate 不是 position

**Coordinate** 是數學 index。**Position** 是 storage 中的 physical offset。

Uncompressed vector 中 coordinate 等於 position。Compressed coordinate/payload list 中兩者不同：

```text
Dense vector coordinates:   0  1  2  3  4  5
Dense values:               0  0  c  d  0  f

Compressed positions:       0  1  2
Stored coordinates:         2  3  5
Stored payloads:            c  d  f
```

Coordinate 5 存在 position 2。當 loop variable \(s\) 是數學 coordinate，而 hardware pointer 是 storage position 時，這個差異非常重要。

### 3. Fiber operations：`getPayload` 與 `getNext`

本講使用兩個 operations：

- `getPayload(coordinate)`：回傳指定 coordinate 的 payload，若存在。
- `getNext()`：依 traversal order 回傳下一組 coordinate/payload pair。

成本依 representation 而變：

| Representation | `getPayload(c)` | Concordant `getNext()` | 硬體直覺 |
|---|---|---|---|
| Uncompressed array | \(O(1)\) | \(O(1)\) | Address = coordinate |
| Coordinate/payload list | \(O(\log n)\) binary search，或 scan | \(O(1)\) | pointer increment 便宜；random lookup 不便宜 |
| RLE | \(O(n)\) scan/accumulate | \(O(1)\) | streaming 好，random access 差 |
| Hash table | 平均 \(O(1)\) | Ordered traversal locality 差 | 需要 hashing 與額外 references |
| Bitmask | 測 bit；payload lookup 可能需 popcount | 常很便宜 | 適合 presence checks |

這張表解釋了本講反覆提醒的一點：compressed storage 不自動等於 efficient sparse computation。它只對支援的 traversal patterns 有效率。

### 4. Concordant 與 discordant traversal

若 loop 拜訪 coordinates 的順序與 representation 儲存順序一致，就是 **concordant traversal**。對 CSR 做 row-major scan 是 concordant。對 CSR 做 column-major scan 是 discordant，因為每個 column lookup 會跨 row fibers 跳動。

CSR 與 CSC 有相同 nonzeros 和類似 storage cost，但自然 traversal orders 相反：

- CSR，記作 `Tensor<U,C>(H,W)`，H rank uncompressed、W fibers compressed；自然 row-major。
- CSC，記作 `Tensor<U,C>(W,H)`，rank order 交換；自然 column-major。

硬體意義：

- Concordant traversal 可用 counters、pointers、sequential SRAM reads 實作。
- Discordant traversal 可能需要 binary search、hash lookup、decompression 或 format conversion。

這是 Lecture 09 回到 mapping 的第一個大橋：loop order 必須與 format 一起選。

### 5. CSR 作為具體 fibertree implementation

假設 sparse matrix nonzeros：

```text
row 0: (col 0 -> a), (col 2 -> c)
row 1: empty
row 2: (col 0 -> g), (col 1 -> h)
```

CSR 儲存：

```text
segment array:    [0, 2, 2, 4]
coordinate array: [0, 2, 0, 1]
value array:      [a, c, g, h]
```

Segment array 告訴你每個 row 的 W fiber 起訖。Row 1 的 start/end 都是 2，所以是 empty。這就是 fibertree 的具體 implementation：H rank uncompressed，每個 W-rank fiber compressed。

### 6. Rank transformations：merge、split、swizzle

Sparse accelerators 常做 rank transformations：

- **Merge/flatten：** 合併 ranks，例如把 \((H,W)\) 變成單一 coordinate tuple。COO 是這個想法的 coordinate-list 版本。
- **Coordinate-space split：** 依固定 coordinate ranges 切 rank。保留幾何意義，但 nonzero counts 可能不均。
- **Position-space split：** 依 stored nonzero count 切 rank。平衡工作，但 coordinate ranges 變不規則。
- **Swizzle：** 重新排列 ranks，例如把 tensor rank order 從 \([H,R,Q]\) 改成 \([Q,R,H]\)。

這些 transformations 不改變 DNN operation 的數學意義。它們是 layout 與 scheduling choices，讓 sparse dataflow 更容易實作。

### 7. Convolution 作為 coordinate projection

本講的 1-D convolution 是：

\[
O[q] = \sum_s I[q+s]F[s].
\]

關係式 \(w=q+s\) 是**座標投影（coordinate projection）**：給定 output coordinate \(q\) 與 filter coordinate \(s\)，它告訴硬體需要哪個 input coordinate \(w\)。

Projection 有三種常見形式：

- Output-stationary view：\(w=q+s\)。
- Weight-stationary input view：\(q=w-s\)。
- Output-stationary sparse-input view：\(s=w-q\)。

算術很小，但不能省。沒有 coordinate generator，硬體無法把 product 路由到正確 partial sum。

### 8. 利用 sparse weights

若 weights sparse，而 inputs dense 或可 cheap random access，自然 schedule 是 concordantly iterate compressed filter：

```text
for q in [0, Q):
    for (s, f_val) in f:
        w = q + s
        o[q] += i[w] * f_val
```

這是 output-stationary：\(o[q]\) 是 accumulation target，filter nonzeros streaming。Filter traversal 用 `getNext()`，所以便宜。Input \(i[w]\) 最好 uncompressed，或至少 cheap to index。

Weight-stationary 版本反轉 outer loops：

```text
for (s, f_val) in f:
    for q in [0, Q):
        w = q + s
        o[q] += i[w] * f_val
```

兩者都利用 weight sparsity。差異是 reuse：一個讓 output stationary，另一個讓 weight stationary。

硬體 blocks：

- compressed filter fiber 的 position generator。
- 計算 \(w=q+s\) 的 coordinate generator。
- random-access input buffer。
- 以 \(q\) index 的 partial-sum buffer。

Cambricon-X 在投影片中作為 weight-sparsity style accelerator：weights metadata 引導 activation access。

### 9. 利用 sparse inputs

若 inputs sparse，而 weights dense 或可 cheap random access，則 iterate nonzero inputs。Weight-stationary sparse inputs：

```text
for s in [0, S):
    for (w, i_val) in i if s <= w < Q+s:
        q = w - s
        o[q] += i_val * f[s]
```

條件 \(s \le w < Q+s\) 是 sparse sliding window。每個 weight \(s\) 只與能產生合法 output coordinate \(q\) 的 input coordinates \(w\) 互動。

Output-stationary sparse inputs：

```text
for q in [0, Q):
    for (w, i_val) in i if q <= w < q+S:
        s = w - q
        o[q] += i_val * f[s]
```

這只拜訪 active window 中的 nonzero inputs，然後 lookup 對應 weight。Sparse input traversal 可以便宜；weight lookup 也必須便宜，因此這個 view 中 weights 常保持 uncompressed。

Cnvlutin 是投影片引用且由 local PDF 確認的 activation-sparsity design。核心想法是從 input stream 移除 zero-valued neurons，並以 offsets 編碼 nonzero neurons，讓 lanes 跳過 zeros 時仍能 index 正確 weights。

### 10. 同時利用 sparse weights 與 sparse inputs

當兩個 operands 都 sparse，projection alone 不夠。你必須先把一個 operand 的 coordinates 投影到另一個 coordinate space，再求交。

Output-stationary two-sparse convolution 可寫成：

```text
for q in [0, Q):
    for (s, (f_val, i_val)) in f.project(+q) & i:
        o[q] += i_val * f_val
```

仔細讀：

1. `f.project(+q)` 把每個 filter coordinate \(s\) 轉成 input coordinate \(w=s+q\)。
2. `& i` 將 projected filter fiber 與 input fiber 求交。
3. 只有兩個 operands 都存在的 coordinates 才產生 multiply。

例子：

```text
filter nonzeros s:        0, 2, 5, 6
for q = 2, projected w:   2, 4, 7, 8
input nonzeros w:         1, 2, 5, 8
intersection:             2, 8
```

只有 projected coordinates 2 與 8 有 matching input activations。硬體輸出兩個 products，而不是四個。

### 11. SCNN：Cartesian-product sparse convolution

SCNN 使用 input-stationary Cartesian-product dataflow。簡化 1-D view：

```text
for (w, i_val) in i:
    for (s, f_val) in f if w-Q <= s < w:
        q = w - s
        o[q] += i_val * f_val
```

SCNN 依 position 將 input activations 與 weights 分塊，並讓 nonzero groups 彼此相乘。若一個 tile 提供 \(I\) 個 nonzero activations 與 \(F\) 個 nonzero weights，PE 可形成最多 \(I \times F\) 個 products。

麻煩在 output routing。Product 的 output coordinate 由 \(q=w-s\) 算出，所以同一個 Cartesian product tile 產生的 products 會 scatter 到不同 output locations。SCNN 因此需要 scatter network 與 dense accumulation backend。

這是漂亮的 sparse-design tradeoff：

- Compressed frontend 讓 zeros 不進 multipliers。
- All-to-all multiplier array 暴露許多 useful products。
- Scatter/accumulator backend 支付 irregular output coordinates 的代價。

### 12. ISOSceles 與 IS-OS pipeline

投影片最後介紹 ISOSceles，一種 IS-OS dataflow。概念是透過 intermediate tensor \(T\) 拆開 convolution：

\[
T[\cdot] = I[\cdot]F[\cdot],
\qquad
O[\cdot] = \text{reduce}(T[\cdot]).
\]

投影片的精確 notation 使用 \(h=p+r\)、\(q=w-s\) 等 substitutions。概念重點是：

1. Input-stationary frontend 處理 sparse input wavefronts，產生 partial results 到 \(T\)。
2. Output-stationary backend 讀 \(T\)，累加 final outputs。
3. Intermediate \(T\) 以一種 rank order 寫入、以另一種 rank order 讀出，因此 ISOSceles 使用 rank swizzling 讓 backend traversal 變 concordant。

這與 CSR vs. CSC 是同一課題，只是現在發生在 pipeline 內：如果 tensor 以一種順序產生、另一種順序消費，rank transformation 可能比重複 discordant lookups 更便宜。

本章中的 ISOSceles quantitative results 皆為 slide-derived，因為 Worker B input 未提供 local ISOSceles PDF。

## Worked Examples

### 範例 1：CSR lookup vs. traversal

使用：

```text
segment array:    [0, 2, 2, 4]
coordinate array: [0, 2, 0, 1]
value array:      [a, c, g, h]
```

Concordant row traversal：

- Row 0 使用 positions 0 到 1：\((0,a),(2,c)\)。
- Row 1 使用 positions 2 到 1：empty。
- Row 2 使用 positions 2 到 3：\((0,g),(1,h)\)。

這很便宜，因為每個 W fiber sequential read。若問「每個 row 的 column 1 是什麼？」在 CSR 中就不便宜，因為每個 row 的 compressed coordinate list 可能都要 search。

### 範例 2：Sparse-weight convolution

令 \(f\) 有 nonzeros \((s=0,f_0=8)\) 與 \((s=2,f_2=6)\)，且 \(Q=3\)。Output-stationary sparse-weight traversal：

- \(q=0\)：使用 \(w=0\) 與 \(w=2\)。
- \(q=1\)：使用 \(w=1\) 與 \(w=3\)。
- \(q=2\)：使用 \(w=2\) 與 \(w=4\)。

若 dense traversal 且 \(S=3\)，會有 9 個 filter positions。Sparse traversal 只有 6 個，因為三個 filter coordinates 中只有兩個存在。

### 範例 3：Projection plus intersection

令 \(q=1\)，filter nonzeros \(s=\{0,3,4\}\)，input nonzeros \(w=\{1,2,5\}\)。

Filter 加上 \(+q\) 後投影為 \(w=\{1,4,5\}\)。與 input 求交得到 \(\{1,5\}\)。只有 \(s=0\) 與 \(s=4\) 產生 products。\(s=3\) 是真實 nonzero weight，但對此 output coordinate 它需要 \(w=4\)，而 input 在那裡是 zero。

## 關鍵方程式與讀法

### 1-D convolution

\[
O[q] = \sum_s I[q+s]F[s].
\]

此式表示 output coordinate \(q\) 會累加 filter coordinate \(s\) 與 input coordinate \(w=q+s\) 的 products。

### Projection equations

\[
w=q+s,\qquad q=w-s,\qquad s=w-q.
\]

這三個式子是同一個 convolution relation 解不同 coordinate。硬體中出現哪一個，取決於 dataflow。

### 理想 two-sparse work

\[
N_\text{work}\approx d_I d_F N_\text{dense}.
\]

此近似假設 sparse positions 獨立。它解釋同時利用 input 與 weight sparsity 為何可呈乘法式收益。這是 teaching model，不是保證 workload statistic。

## 硬體意涵（Hardware Implications）

- **Fibertree hardware：** 每個 rank 可用 position generator 與 coordinate/payload extractor 實作。
- **CSR/CSC：** rank order 決定哪種 traversal 便宜。若 consumer 需要不同順序，可能要 format conversion 或 rank swizzling。
- **Sparse weights：** compressed filter traversal 便宜，但 input access 變成 projected random access。
- **Sparse inputs：** input traversal 便宜，但 weight lookup 與 sliding-window restriction 變重要。
- **Two-sparse：** intersection hardware 決定 accelerator 是否真的實現 product-of-densities work reduction。
- **SCNN：** all-to-all multiplication 增加 useful work exposure，但 scatter routing 與 dense accumulators 消耗 area/energy。
- **ISOSceles：** intermediate tensors 可結合 IS 與 OS 優點，但 rank swizzling 與 buffering 成為顯式成本。

## 常見誤區（Common Misconceptions）

### 誤區：Compressed tensor 一定遍歷更快。

只有 concordant traversal 自然快。對 compressed metadata 做 random lookup 可能比 dense access 更慢。

### 誤區：Coordinate 與 position 可以互換。

只有 uncompressed ranks 中兩者相等。Compressed ranks 中，coordinate 是數學 index，position 是 storage offset。

### 誤區：Projection 等於 intersection。

Projection 改變 coordinate space。Intersection 移除兩個 operands 不共同存在的 coordinates。Two-sparse convolution 需要兩者。

### 誤區：SCNN 的 Cartesian product 表示它做 dense multiplication。

SCNN 是對 compressed nonzero groups 做 Cartesian products，不是對 dense tensors。它避開 zero operands，但仍要把 products scatter 到 irregular output coordinates。

### 誤區：Rank swizzling 只是 software transpose。

在 sparse hardware 中，rank swizzling 是設計選擇：用一次 reordering/buffering 換取後續重複 traversal 更便宜。

## 與前後講次的連結（Connections）

- **L04 Einsums：** convolution equations 是帶 index arithmetic 的 Einsums。
- **L05-L06 mapping/dataflow：** stationarity 與 loop order 決定 tensor 是 streamed、looked up 還是 accumulated。
- **L08 sparse architectures：** gating、skipping、metadata、format、intersection 是本講使用的詞彙。
- **L10 sparse architectures part 3：** TeAAL 把這些選擇形式化為 mapped Einsum cascades、formats、bindings、architecture specifications。
- **Lab 4/SparseLoop：** 只有 traversal 與 format 一起指定時，才可 modeling gating/skipping/sparse formats 的成本。

## Paper Bridge: TeAAL

### Bibliographic identity

- **Title:** TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators
- **Authors:** N. Nayak et al.
- **Year / venue:** MICRO 2023
- **Local PDF:** `papers/TeAAL.pdf`

### Problem addressed

TeAAL 提供精確方法描述 sparse tensor accelerators；這些 accelerator 的行為取決於 mapped Einsums、fibertree formats、rank transformations 與真實 sparse inputs。

### Core idea

該 paper 使用 fibertrees 作為 tensor abstraction，mapped Einsums 作為 computation/mapping abstraction，並用 flattening、partitioning、swizzling 等 transformations 表達 sparse orchestration。

### Relevance to this lecture

Lecture 09 的 fibertree 詞彙、rank transformations、concordant traversal、IS-OS rank swizzling，正是 TeAAL 形式化的概念。

### Key claims used in this chapter

- TeAAL Section 2.1 定義 ranks、coordinates、points、fibers、payloads、fibertrees，並指出 sparse fibertrees 省略 empty payloads。
- Section 2.2 說 Einsums 指定 computation，但不指定 iteration order。
- Section 2.3 解釋 loop order、partitioning、work scheduling 等 mapping attributes，並把 sparse compression 連到 load imbalance 與 memory footprint variation。
- Section 3.2 描述 rank flattening、partitioning、sorting/merging、rank swizzling 等 content-preserving transformations。

### What students should remember

1. Fibertrees 不只是圖，而是 sparse traversal 的 interface。
2. Mapping 決定 `getNext()` 或 `getPayload()` 哪個會主導成本。
3. Rank transformations 是讓 sparse traversal 可實作的架構工具。

### Limitations and assumptions

TeAAL 是 modeling accelerator 的框架；它本身不決定哪個 accelerator 最好。本章用它固定術語。

## Paper Bridge: Cnvlutin

### Bibliographic identity

- **Title:** Cnvlutin: Ineffectual-Neuron-Free Deep Neural Network Computing
- **Authors:** J. Albericio et al.
- **Year / venue:** ISCA 2016
- **Local PDF:** `papers/L18_Cnvlutin_Albericio_ISCA2016.pdf`

### Problem addressed

Cnvlutin 針對 convolutional layers 中的 zero-valued input neurons。Baseline wide-lane DNN accelerators 讓 neurons lockstep processing，所以 zero neuron 會浪費 multiplier slots 與 cycles。

### Core idea

Cnvlutin 用 Zero-Free Neuron Array format 儲存 nonzero input neurons 與 offsets。Offsets 讓每個 nonzero neuron 能找到正確 synapse/weight location，同時讓 lanes independently proceed，跳過 zero neurons 而不改變 DNN result。

### Relevance to this lecture

Cnvlutin 是 sparse-input traversal 的具體 paper bridge。它說明 skipping activation zeros 需要 format 與 dispatch mechanism，而不只是 if-statement。

### Key claims used in this chapter

- Abstract 把 zero-operand multiplications 定義為 intrinsically ineffectual，並提出 Cnvlutin 作為移除它們的 value-based acceleration。
- Section III 描述 decoupling neuron lanes，以及使用 encoded input format 把 zero-valued neurons 從 critical path 移除。
- Section IV-B 描述 Zero-Free Neuron Array format，包含以 bricks 分組的 nonzero value/offset pairs。
- Section V 報告相對 paper baseline 的 speedup 與 energy improvements；本章不重用精確數值，只作 paper context。

### What students should remember

1. Activation sparsity 是 runtime data-dependent。
2. Offsets 是 coordinate metadata，讓 compressed activations 仍能 address 正確 weights。
3. Lane decoupling 是面對 irregular nonzero counts 的 utilization solution。

### Limitations and assumptions

Cnvlutin 聚焦 convolutional layers 的 activation sparsity，並基於 DaDianNao-like baseline。它不像 SCNN 一樣同時利用 pruned weight sparsity。

## Paper Bridge: SCNN

### Bibliographic identity

- **Title:** SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks
- **Authors:** A. Parashar et al.
- **Year / venue:** ISCA 2017
- **Local PDF:** `papers/L17_SCNN_Parashar_ISCA2017.pdf`

### Problem addressed

SCNN 針對 CNN inference 中 pruned weights 與 ReLU activations 的 combined sparsity，並嘗試讓 weights/activations 在 computation 中維持 compressed。

### Core idea

PT-IS-CP-sparse dataflow 是 planar-tiled、input-stationary、Cartesian-product based。它把 compressed nonzero weight/activation groups 送入 all-to-all multiplier array，由 metadata 計算 output coordinates，並把 products scatter 到 dense accumulators。

### Relevance to this lecture

SCNN 是 two-sparse convolution 的主要具體例子。它展示 projection 與 multiplication 可以高效，但 output accumulation 會變 irregular。

### Key claims used in this chapter

- Abstract 說 SCNN 利用 pruning 造成的 zero-valued weights 與 ReLU 造成的 zero-valued activations。
- Section II 以 weight/activation densities 的乘積說明 multiplicative work reduction。
- Section III 定義 PT-IS-CP-sparse dataflow 與其 input-stationary Cartesian-product structure。
- Section IV 描述含 compressed buffers、\(F \times I\) multipliers、coordinate handling、scatter accumulation 的 PE architecture。
- Section VIII 總結 SCNN 讓 weights/activations 保持 compressed，且只把 nonzero operands 送到 multipliers。

### What students should remember

1. SCNN 的 work reduction 來自兩個 operands 都 sparse。
2. Cartesian-product multiplication 是用 nonzero groups 保持 multipliers 忙碌的方法。
3. Sparse output routing 是這種自由度的代價。

### Limitations and assumptions

該設計專門針對 CNN-style convolution，依賴 paper 的 compressed block organization 與 accumulator banking。若沒有檢查 workload/baseline，不應泛化其量化結果。

## Paper Bridge: Eyeriss v2

### Bibliographic identity

- **Title:** Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices
- **Authors:** Y.-H. Chen et al.
- **Year / venue:** JETCAS 2019
- **Local PDF:** `papers/L17_EyerissV2_Chen_JETCAS2019.pdf`

### Problem addressed

Eyeriss v2 處理 compact and sparse DNNs；這些模型的 shapes 會讓 dense reuse 與 PE utilization 變困難。

### Core idea

它使用 hierarchical mesh NoC 與 sparse PE architecture。在 sparse PE mode 中，activations/weights 使用類 CSC compressed streams，count/address metadata 允許硬體在 compressed domain 跳過 zeros。

### Relevance to this lecture

Eyeriss v2 是 Lecture 08 到 Lecture 09 轉折的具體實作：original Eyeriss gated zero activations；Eyeriss v2 使用 compressed formats 與 skipping 來提升 throughput。

### Key claims used in this chapter

- Section IV 明確區分 original Eyeriss gating 與 Eyeriss v2 會 skipping zeros 以改善 throughput 的 sparse processing。
- Section IV 描述 CSC compressed data、count/address metadata 與 sparse PE pipeline support。
- Section V 報告 implementation results，並討論 workload imbalance，以及 sparse compression 不一定創造 skippable cycles 的情況。

### What students should remember

1. Sparse support 改變 PE pipeline，不只是 memory format。
2. Flexible NoC 很重要，因為 sparse/compact models 對 bandwidth/reuse 的壓力不同。
3. Workload imbalance 仍是限制因素。

### Limitations and assumptions

Paper 報告的 improvements 依賴 implementation、workloads、baselines。本章用 Eyeriss v2 說明 architectural mechanisms，不把數字當 universal constants。

## 獨立學習指南（Standalone Study Guide）

建議分五輪讀：

1. 為一個小 sparse matrix 畫 fibertree。
2. 對每個 stored value 標出 coordinate 與 position。
3. 對每個 representation 問：`getPayload` 還是 `getNext` 便宜？
4. 把 \(O[q]=\sum_s I[q+s]F[s]\) 重寫成 output-stationary、weight-stationary、input-stationary 形式。
5. 把 SCNN 與 ISOSceles 解釋為「sparse products 在哪裡 accumulation？」的兩種答案。

## 自我檢核問題

1. 為何 CSR 讓 row traversal 便宜，但 column traversal 昂貴？
2. Compressed fiber 中 coordinate 與 position 有何差異？
3. 為何 coordinate/payload list 的 `getNext()` 便宜，但 `getPayload()` 可能昂貴？
4. Sparse-weight convolution 中，為何 input 常應保持 uncompressed？
5. Sparse-input convolution 中，sliding-window condition 做什麼？
6. Two-sparse convolution 為何需要 projection 和 intersection？
7. SCNN 為何需要 scatter network？
8. IS-OS pipeline 中 rank swizzling 解決什麼問題？

## 練習

1. **Fibertree：** 為 \(4 \times 4\) matrix 中 nonzeros \((0,1)\)、\((2,0)\)、\((2,3)\) 畫 fibertree 與 CSR arrays。
2. **Traversal：** 同一 matrix 若用 CSR，列出 row traversal 與 column traversal 需要的 operations。
3. **Projection：** 令 \(Q=4\)、\(S=3\)、filter nonzeros \(s=\{0,2\}\)。列出每個 \(q\) 的 projected \(w=q+s\)。
4. **Intersection：** 對 \(q=2\)、filter nonzeros \(s=\{0,1,4\}\)、input nonzeros \(w=\{2,3,5,7\}\)，計算 projection plus intersection。
5. **Design tradeoff：** 比較 Cnvlutin 與 SCNN。各自利用哪種 sparsity？各需什麼額外硬體？
6. **Paper reading：** 在 SCNN Section III 中找出為何 input stationarity 有吸引力，以及 accumulation 為何變難。

## 關鍵詞彙（Key Terms）

| Term | Definition |
|---|---|
| **Rank（秩）** | Tensor dimension，在 fibertree 中是一層。 |
| **Coordinate（座標）** | 某 rank 內的數學 index。 |
| **Point（點）** | 標示 tensor element 的 coordinate tuple。 |
| **Position（位置）** | Storage array 中的 physical offset；只在 uncompressed ranks 中等於 coordinate。 |
| **Payload（承載值）** | Scalar value 或指向 lower-rank fiber 的 pointer/reference。 |
| **Fiber（纖維）** | 某 rank 上的有序 coordinate/payload pairs。 |
| **Fibertree（纖維樹）** | Tensor 的 tree representation，levels 是 ranks，fibers 連接 levels。 |
| **`getPayload(c)`** | 查詢 coordinate \(c\) 的 payload。 |
| **`getNext()`** | 依 traversal order 回傳下一個 coordinate/payload pair 的 iterator。 |
| **CSR** | Compressed Sparse Row；`Tensor<U,C>(H,W)`，適合 row-major traversal。 |
| **CSC** | Compressed Sparse Column；`Tensor<U,C>(W,H)`，適合 column-major traversal。 |
| **COO** | Coordinate list format，儲存 merged coordinate tuples。 |
| **Concordant traversal** | Traversal order 與 storage/rank order 對齊，因此 sequential reads 便宜。 |
| **Discordant traversal** | Traversal order 與 storage/rank order 衝突，常需 random lookup。 |
| **Projection（投影）** | 在 tensor coordinate spaces 之間做 arithmetic mapping，例如 \(w=q+s\)。 |
| **Intersection（求交）** | 只輸出兩個 sparse operands 都存在的 coordinates。 |
| **Position-space split** | 依 stored nonzero count 切 fiber 以平衡工作。 |
| **Coordinate-space split** | 依 coordinate ranges 切 fiber 以保留幾何意義。 |
| **Rank swizzle** | 重排 tensor ranks，讓後續 traversal 變 concordant。 |
| **Cnvlutin** | 使用 zero-free neuron encoding 與 lane decoupling 的 activation-sparsity accelerator。 |
| **SCNN** | 使用 input-stationary Cartesian-product sparse multiplication 的 sparse CNN accelerator。 |
| **ISOSceles** | 投影片引用的 IS-OS sparse convolution dataflow，使用 intermediate tensor 與 rank swizzling。 |

## 重點回顧（Takeaways）

- Fibertrees 提供 dense/sparse tensor layouts 的共同語言。
- 只要 rank 被壓縮，就必須分清 coordinate 與 position。
- Sparse formats 依賴 traversal：concordant `getNext()` 可便宜，discordant `getPayload()` 可能昂貴。
- Convolution sparsity 需要 coordinate projection；two-sparse convolution 還需要 intersection。
- Cnvlutin、SCNN、Eyeriss v2、ISOSceles 分別代表不同設計點：activation skipping、two-sparse Cartesian products、compressed-domain sparse PE、IS-OS pipelining。
- Sparse dataflow design 是 reuse、skipping、output routing 三者之間的協商。

## 連結（Connections）

- **L08：** 介紹 SAFs 與 metadata tradeoffs；L09 說明 metadata 如何被 traversal。
- **L10：** 進一步走向 formal sparse accelerator specification 與 TeAAL-style modeling。
- **Mapping lectures：** stationarity 決定 tensor 是 streamed、looked up、還是 accumulated。
- **Einsum lectures：** projection 與 reduction 直接來自 convolution index expressions。
- **Paper bridges：** TeAAL 提供 abstraction，Cnvlutin 展示 activation skipping，SCNN 展示 two-sparse Cartesian products，Eyeriss v2 展示 compressed-domain sparse PE design。

## 附錄 — 投影片對照表（Slide-to-Section Map）

| Slide range | Chapter section | Notes |
|---|---|---|
| L09-1 to L09-5 | Motivation and SAF recap | 擴成 TL;DR、problem、why it matters |
| L09-6 to L09-18 | Tensor terminology and fibertree | 補 coordinate/position explanation |
| L09-19 to L09-35 | Fiber representations, CSR/CSC | 重寫為 representation/storage examples |
| L09-36 to L09-51 | Traversal efficiency | 擴充 `getPayload`/`getNext` cost model |
| L09-52 to L09-60 | Merge, split, sparsity specs, HSS | 整合進 rank transformations |
| L09-61 to L09-71 | Einsum review and convolution projection | 重寫為 coordinate projection |
| L09-72 to L09-96 | Sparse weights in convolution | 補 loop nests 與 hardware blocks |
| L09-97 to L09-111 | Sparse inputs and Cnvlutin | 補 sparse sliding window |
| L09-112 to L09-128 | Sparse inputs and weights, SCNN, intersection | 補 projection-plus-intersection example |
| L09-129 to L09-138 | ISOSceles IS-OS dataflow | slide-derived；Worker B input 無 local ISOSceles PDF |

## Source Notes

- Fibertree terminology 與 traversal ordering 依 Lecture 09 slides 6-51 以及 TeAAL Sections 2.1-2.3。
- Rank merge/split/swizzle discussion 依 Lecture 09 slides 51-60 與 TeAAL Section 3.2。
- Convolution projection 與 loop nests 依 Lecture 09 slides 72-128。
- Cnvlutin discussion 使用 `papers/L18_Cnvlutin_Albericio_ISCA2016.pdf`，尤其 Sections III、IV-B、V。
- SCNN discussion 使用 `papers/L17_SCNN_Parashar_ISCA2017.pdf`，尤其 Sections II-IV 與 VIII。
- Eyeriss v2 discussion 使用 `papers/L17_EyerissV2_Chen_JETCAS2019.pdf`，尤其 Sections IV-V。
- Cambricon-X 與 ISOSceles 僅依 Lecture 09 slide anchors 討論；Worker B input 未包含 local PDFs。
- `papers/L08_FastAlgorithmsWinograd_Lavin_2015.pdf` 有檢視，但未實質使用，因為 Winograd convolution 不是 Lecture 09 sparse traversal 主線。

## Uncertainty Notes

- 本章根據 slides 與 papers 重建可能講解；live lecture 可能重點不同。
- ISOSceles quantitative claims 只來自投影片。
- 既有 `assets/L09/` images 可能有 copyright sensitivity，但 asset cleanup 不在 Worker B write scope。
