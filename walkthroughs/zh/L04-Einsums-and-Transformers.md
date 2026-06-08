# L04 - Einsum 與 Transformer（Einsums and Transformers）

> **課程：** 6.5930/1 - 深度學習硬體架構（Hardware Architectures for Deep Learning）
> **講師：** Joel Emer 與 Vivienne Sze（MIT EECS）
> **講授日期：** 2026 年 2 月 13 日。**投影片：** 198 頁。**來源：** [`Lecture/L04-Einsums+Transformers.pdf`](../../Lecture/L04-Einsums+Transformers.pdf)
>
> 本章依據公開投影片重建缺少影片時的講解脈絡。它不是投影片摘要。L04 有大量動畫頁，特別是 FC/CONV lowering 的部分，因此本章用投影片頁碼作為 source anchor，並用自學教材的方式重新組織。

---

## TL;DR

L04 深化本課程的正式運算語言：**Einsum**。Einsum 用運算元、秩（rank）與算術關係描述張量運算，但刻意不指定 loop order、dataflow、tiling 或 memory placement。用硬體架構的語言說，Einsum 是「必須算什麼」的契約；後續 mapping 講次才決定「怎麼算」。

本講有三個連在一起的重點。第一，rank variable 告訴我們哪些張量維度會保留下來，哪些會被歸約（reduction）。只出現在右側而不出現在左側的 rank 是 **contracted rank**，它代表加總。第二，常見 DNN kernel 可以透過 **flattening** 或 **partitioning** 改寫成矩陣乘法。Fully connected layer 可以自然 flatten；convolution 則要先暴露 sliding window 的 Toeplitz-like 結構，才能 lower 成矩陣乘法。第三，Transformer self-attention 也可以寫成一串 Einsum：projection 產生 $Q$、$K$、$V$；$QK^T$ 產生 attention score；$\mathrm{softmax}$ 做正規化；$AV$ 合成 value vector；最後再做 output projection。

硬體重點不是「所有東西都只是 GEMM」。更精確的說法是：一旦把 computation 表達成 ranks 和 contractions，架構師就可以系統化地推理 reuse、data movement、parallelism、intermediate storage 與 reductions。L05、L06、L09 與 L13 都依賴這個 rank-level view。

---

## 這一講解決什麼問題

L02 把 DNN layer 介紹成張量運算，L03 說明 memory/metric 使「只看計算量」不夠。L04 要解決的是兩者中間的表示問題：

> 我們要如何用同一套 notation 寫出 FC、CONV 與 self-attention，使它精確到能支撐 hardware mapping，又抽象到不會太早選定實作方式？

一般 layer 名稱隱藏太多資訊。「Convolution」告訴我們 filter 會滑過 input，但沒有明確說出有哪些 loop ranks、哪個 rank 被 reduced。「Attention」告訴我們 tokens 會動態互動，但沒有明確說出哪幾個 matrix products 產生 $M \times M$ score tensor。硬體架構師需要 rank structure，因為 ranks 會變成 loops，loops 會變成 schedules，而 schedules 會決定資料如何穿過 memory hierarchy。

解法是用 Einsum 作為 compute-level interface。同一套 notation 可以表達 $Z_{m,n} = \sum_k A_{m,k}B_{k,n}$、$O_q = \sum_s I_{q+s}F_s$、以及 $AV_{p,f} = \sum_m A_{m,p}V_{m,f}$。這些在 neural network layer 名稱上很不同，但對 mapping 而言是同一類物件：named output ranks、named reduction ranks，以及 tensor accesses。

---

## 為什麼這一講重要

對沒有影片的讀者來說，最重要的轉折是：L04 不是為了線性代數本身而教線性代數。它是在建立後續課程能嚴謹提出硬體問題的 notation。

同一個數學 Einsum 可以用很多順序執行。對 $Z_{m,n} = \sum_k A_{m,k}B_{k,n}$，CPU loop nest 可能先跑 $m$、再跑 $n$、最後跑 $k$；systolic array 可能讓 $A$ 與 $B$ 往不同方向流動；tensor compiler 可能 tile 三個 ranks。只要它們走訪同一個 iteration space 並正確累加 contracted rank，結果都是合法的。數值結果固定；資料搬移不固定。

這個差異會影響：

- **Energy：** rank order 會控制 operands 是從 RF、SRAM 還是 DRAM reuse。
- **Bandwidth：** tensor lowering 可以產生規則 GEMM traffic，但也可能複製資料。
- **Latency：** reductions 與 softmax 造成 dependencies，不能總是完全平行化。
- **Area：** 支援像 $A \in \mathbb{R}^{M \times M}$ 這種大型 attention intermediate，需要 storage 或 fusion。
- **Utilization：** flatten 與 partition 後的 ranks 會暴露不同 parallel loops 給 PE array。
- **Correctness：** contracted ranks 必須對每個合法 coordinate 正確累加一次，即使 loop order 改變。

---

## 先備知識與心智模型

你應該帶著前面講次的三個觀念進入本章。

第一，L02 說明 tensor 有具名維度。2-D convolution 有 output channels、input channels、output spatial ranks 與 filter spatial ranks。這些不只是 shape；它們是機器最後要執行的 loops。

第二，L03 說明 memory movement 很貴，評估不能只看 MAC count。L04 還不選 mapping，但會暴露後續 mapping 要 reorder 與 tile 的 ranks。

第三，L03 第一次引入 Einsum 時已經說明：expression 指定數學運算，不指定執行順序。

本章的心智模型是：

```text
Einsum expression
    -> ranks and tensor accesses
    -> legal iteration space
    -> many possible loop nests
    -> many possible data-movement costs
```

看到一個 Einsum 時，先問四個問題：

1. 哪些 ranks 出現在 output？
2. 哪些 ranks 只出現在右側，因此會被 reduced？
3. 是否有 tensor access 把 ranks 組合在一起，例如 $q+s$ 或 $U \times p+r$？
4. 哪些 ranks 之後可以 flatten、partition、tile 或 parallelize？

---

## 學習目標

讀完本章後，你應該能夠：

- 說出 Einsum 的操作定義（Operational Definition for Einsums, ODE），並用它評估小型 expression。
- 區分 rank variables、rank names 與 rank shapes。
- 在 matrix multiplication、convolution 與 attention 中辨認 contracted ranks 與 uncontracted ranks。
- 解釋為什麼更換 rank-variable 名稱不會改變 Einsum。
- 將 partitioning 與 flattening 解釋為互逆的 rank transformations。
- 透過 flatten ranks，將 fully connected layer 轉成 matrix-vector 或 matrix-matrix multiplication。
- 解釋 convolution lowering 如何產生 Toeplitz/im2col tensor，以及為什麼這個 tensor 可能複製 input values。
- 追蹤 self-attention cascade：embedding、$Q/K/V$ projections、$QK$、$\mathrm{softmax}$、$AV$、output projection。
- 解釋 standard attention 為什麼隨 sequence length $M$ 呈二次方成長。
- 將 L04 的 rank notation 連到 L05 dataflows、L06 partitioning、sparse traversal 與 L13 data-movement analysis。

---

## 1. Einsum 是 compute contract

**Source anchor：** 投影片 L04-4 到 L04-9。

本講從一個例子開始：$Z_{m,n} = A_{m,k} \times B_{n,k}$。投影片定義 **Einsum 的操作定義（Operational Definition for Einsums, ODE）**：

> 走訪 iteration space 中所有合法 rank-variable values。在每個點，計算右側在這些 rank values 下的值。把結果 assign 到左側 coordinate；如果該 coordinate 已有值，就 reduce 進去。

對乘法型 contraction 而言，「reduce」通常就是加總。因此，$Z_{m,n} = A_{m,k} \times B_{n,k}$ 應讀成 $Z_{m,n} = \sum_k A_{m,k}B_{n,k}$。

### 直覺

Einsum 像是一份移除料理順序的精確食譜。它列出所有材料與目的地，但不說要一次做一個 output、一次做一個 reduction rank，還是一次做一個 tile。這就是它對硬體有用的原因：同一個 computation 之後可以 map 到 CPU loop nest、GPU kernel、systolic array 或 custom accelerator。

### 精確意義

**Rank variable** 是 expression 中的 index symbol，例如 $m$、$n$ 或 $k$。**Rank name** 是語義上的維度名稱，例如 $M$ 或 $K$。**Rank shape** 是該維度的大小。在 $A^{K,M}_{k,m}$ 中，rank variables 是 $k,m$，rank names 是 $K,M$，rank shapes 是這些維度的 extents。

如果一個 rank 出現在左側，它是 **uncontracted**，用來辨識 output coordinates。如果一個 rank 只出現在右側而不出現在左側，它是 **contracted**，它的 values 會被 reduce 掉。

### Worked example：讀一個小 Einsum

令 $Z_{m,n} = \sum_k A_{m,k}B_{n,k}$，且 $M=2$、$N=2$、$K=3$。要計算 $Z_{1,0}$，先把 output ranks 固定在 $m=1,n=0$，然後走訪 contracted rank $k$：

$Z_{1,0} = A_{1,0}B_{0,0} + A_{1,1}B_{0,1} + A_{1,2}B_{0,2}$。

output shape 是 $M \times N$，因為 $m,n$ 是 uncontracted。每個 output 的工作量是 $K$ 次 multiply-accumulate，因為 $k$ 是 contracted。

### 等價形式

投影片 L04-8 到 L04-9 顯示，$Z_{m,n} = A_{m,k}B_{k,n}$、$Z_{n,m} = A_{k,m}B_{n,k}$、以及 $Z_{p,q} = A_{p,r}B_{q,r}$ 都是 matrix multiplication patterns。字母不是重點；shared、output 與 reduction ranks 的模式才是重點。

### 硬體意義

ODE 建立了 correctness boundary。mapper 可以 reorder loops、tile ranks 或 parallelize work，但必須保留同一組 rank tuples 與同一組 reductions。這就是 L05 可以比較 output-stationary、weight-stationary 與 input-stationary loop nests，卻不改變 layer 數學結果的形式理由。

### 常見誤解

**誤解：** Einsum 只是 matrix equation 的簡寫。

**修正：** 在本課程中，Einsum 是 mapping 與 analysis tools 會消費的 compute specification。它明確暴露 loop ranks 與 reductions，而這正是 hardware schedule 需要操作的東西。

---

## 2. Rank transformations：flattening 與 partitioning

**Source anchor：** 投影片 L04-10 到 L04-14。

L04 接著研究一組 transformation：它們改變 ranks 的命名方式，但不改變底層 computation。

### Flattening

如果兩個 coordinates 總是一起移動，它們可以被視為一個 compound coordinate。對 2-D elementwise multiply，$Z_{i,j} = A_{i,j}B_{i,j}$ 可以用 $ij = i \times J + j$ flatten 成 $Z_{ij} = A_{ij}B_{ij}$。

Flattening 會移除 ranks。這就是形狀 $C \times H \times W$ 的 tensor 可以變成長度 $CHW$ vector 的代數原因。

### Partitioning

Partitioning 做相反的事情。單一 rank $i$ 可以用 $i = i_1 \times I_0 + i_0$ 拆成 $i_1$ 與 $i_0$。於是 $Z_i = A_iB_i$ 變成 $Z_{i_1,i_0} = A_{i_1,i_0}B_{i_1,i_0}$。

Partitioning 會新增 ranks。它是 tiling 的代數基礎：$i_1$ 命名 tile，$i_0$ 命名 tile 內的位置。

### Worked example：拆分並還原 index

假設 vector 長度為 8，選 $I_0=4$。則 $i = i_1 \times 4 + i_0$。

| 原始 $i$ | $i_1$ | $i_0$ |
|---:|---:|---:|
| 0 | 0 | 0 |
| 3 | 0 | 3 |
| 4 | 1 | 0 |
| 7 | 1 | 3 |

資料沒有改變，只是 coordinate system 改變。但在硬體上，這個 coordinate change 很有用：$i_1$ 可以變成外層 memory tile loop，$i_0$ 可以放進 local buffer。

### 硬體意義

Flattening 常讓 computation 看起來像 GEMM，這對已有高效率 matrix engines 的硬體很方便。Partitioning 常讓 computation 能符合 memory hierarchy 或 PE array。兩者在物理上都不是免費的。Flattening 改變 address generation，也可能遮蔽原本的 locality；partitioning 增加 loop structure，也可能需要處理邊界。

### 常見誤解

**誤解：** 如果兩個 computation flatten 後都是同一種 matrix multiplication，它們就有相同硬體成本。

**修正：** Flatten 後的代數形式可能相同，但 data layout、reuse pattern，以及形成 flattened tensor 的成本可能不同。Convolution lowering 是最重要的例子。

---

## 3. Convolution 作為 Einsum

**Source anchor：** 投影片 L04-15 到 L04-18，以及 L04-42 到 L04-59。

投影片中最簡單的 convolution 是一維：$O_q = \sum_s I_{q+s}F_s$。output rank $q$ 被保留；filter rank $s$ 被 contracted。input access $q+s$ 是關鍵，它表示 output position $q$ 會看到 input 的 shifted window。

### Worked example：1-D convolution

令 $I=[2,1,3,0]$、$F=[4,5]$，採 valid convolution。output 長度為 $Q=W-S+1=3$。

$O_0 = I_0F_0 + I_1F_1 = 2 \times 4 + 1 \times 5 = 13$。

$O_1 = I_1F_0 + I_2F_1 = 1 \times 4 + 3 \times 5 = 19$。

$O_2 = I_2F_0 + I_3F_1 = 3 \times 4 + 0 \times 5 = 12$。

Filter values 會跨 output positions reuse；overlapping windows 也會讓 input values 被 reuse。L05 會把這個觀察轉成 dataflow choices。

### 2-D convolution

對 batch $N$、output channel $M$、input channel $C$、output spatial ranks $P,Q$、filter ranks $R,S$ 與 stride $U$，dense 2-D convolution 是：

$$O_{n,m,p,q} = B_m + \sum_{c,r,s} I_{n,c,U \times p+r,U \times q+s}F_{m,c,r,s}.$$

contracted ranks 是 $c,r,s$。output ranks 是 $n,m,p,q$。combined input indices $U \times p+r$ 與 $U \times q+s$ 讓 convolution 成為 sliding-window operation，而不是普通 dense layer。

### 硬體意義

這個 expression 直接暴露 reuse。weight $F_{m,c,r,s}$ 可以被許多 output positions $p,q$ 與許多 batch elements $n$ 使用。input activation $I_{n,c,h,w}$ 可能貢獻給數個鄰近 outputs，因為多組 $(p,r)$ 與 $(q,s)$ 可能對應到同一個 $h,w$。partial sum $O_{n,m,p,q}$ 會在 $c,r,s$ 上收到多次 updates。這三類 operands 正是 L05 會嘗試 stationary 的對象。

### 常見誤解

**誤解：** Convolution 與 matrix multiplication 本質完全不同，所以 matrix-multiply hardware 無關。

**修正：** Convolution 有特殊 indexing pattern，但可以 lower 成 matrix multiplication。真正重要的問題是：lowering 是被 materialize 在 memory 裡，還是由 address generation 隱式表示。

---

## 4. FC 與 CONV lowering 成矩陣乘法

**Source anchor：** 投影片 L04-20 到 L04-172。本節綜合一段很長的動畫序列；許多頁是 visual build states，因此抽出的文字較少。

### Fully connected layer

對 input activation tensor $I_{c,h,w}$ 與 $M$ 個 output neurons，fully connected layer 計算：

$$O_m = \sum_{c,h,w} I_{c,h,w}F_{m,c,h,w}.$$

這裡 $m$ 是 uncontracted，$c,h,w$ 是 contracted。把 input coordinates flatten 成 $chw$ 後，expression 變成 $O_m = \sum_{chw} I_{chw}F_{m,chw}$，也就是 matrix-vector multiply。

加入 batch rank $n$ 後，同一層變成：

$$O_{n,m} = \sum_{chw} I_{n,chw}F_{m,chw}.$$

這就是 matrix-matrix multiplication。batch rank 把多個獨立 matrix-vector products 變成一個較大的 matrix computation。

### Loop nest view

FC Einsum 可對應到像這樣的 loop nest：

```text
for m in [0, M):
  O[m] = 0
  for c in [0, C):
    for h in [0, H):
      for w in [0, W):
        O[m] += I[c,h,w] * F[m,c,h,w]
```

這個 code block 不是 Einsum 的定義，而是一種合法 traversal order。其他 loop orders 會算出相同 output，但造成不同 reuse。

### Convolution lowering

對 convolution 而言，只有 flatten 不夠，因為 input 使用 $p+r$ 與 $q+s$。標準 lowering 想法是建立 patch tensor：

$$T_{n,p,q,c,r,s} = I_{n,c,U \times p+r,U \times q+s}.$$

接著 convolution 變成：

$$O_{n,m,p,q} = \sum_{c,r,s} T_{n,p,q,c,r,s}F_{m,c,r,s}.$$

把 $(c,r,s)$ flatten 成 $crs$，把 $(n,p,q)$ flatten 成 $npq$：

$$O_{m,npq} = \sum_{crs} F_{m,crs}T_{npq,crs}.$$

這就是 matrix multiplication：形狀 $M \times CRS$ 的 filter matrix，乘上形狀 $NPQ \times CRS$ 的 patch matrix；實際左右乘與 transpose 取決於 convention。

### Worked example：2-D im2col

取 single-channel $3 \times 3$ input 與 $2 \times 2$ filter，stride 1。output 有 $P=2,Q=2$，所以有四個 output positions。patch matrix 有四列，每列對應一個 output position；有四欄，每欄對應一個 filter coordinate：

```text
input coordinates in each row of T[pq,rs]

pq=00: (0,0) (0,1) (1,0) (1,1)
pq=01: (0,1) (0,2) (1,1) (1,2)
pq=10: (1,0) (1,1) (2,0) (2,1)
pq=11: (1,1) (1,2) (2,1) (2,2)
```

注意 duplication。input coordinate $(1,1)$ 出現在四列中。Materialized im2col 讓 GEMM 容易，但可能增加 memory traffic 與 storage。Convolution accelerator 也可以即時計算同樣的 addresses，避免真的把 $T$ 存下來。

### 硬體意義

Lowering 解釋了為什麼 GEMM engines 很通用，但也暴露 tradeoff。Lowered matrix 可以讓 compute 規則化並改善 PE utilization；但 materializing patch matrix 可能放大 memory traffic。Custom convolution dataflow 可以利用同一個數學關係，而不物理複製 input values。這正是 L05 要問的問題：對同一個 Einsum，哪種 traversal order 與 storage placement 能最小化 movement？

---

## 5. 從 convolution 到 attention

**Source anchor：** 投影片 L04-174 到 L04-178。

接著本講換到另一個 workload family。Convolution 使用固定 local receptive field：output position 會看到 filter shape $R \times S$ 決定的區域。Attention 使用動態 global receptive field：一個 token 可以對任何其他 token 給高權重，不受距離限制。

投影片 L04-175 到 L04-177 用三個重點說明動機：

- Convolution 很自然地 model spatial-neighbor dependencies，但 filter window 是固定的。
- Attention 用來 model long-range dependencies；投影片引用 Vaswani et al. 2017，說 attention 可以不受距離限制地 model global dependencies。
- Input 會被切成 tokens：文字中的 words、vision 中的 image patches，以及 audio 中的 spectrogram patches。

### Teaching interpretation

Attention 不是「連得更多的 convolution」。它改變的是 dependency rule。Convolution 的 dependency 由幾何決定：output $(p,q)$ 讀 fixed window。Attention 的 dependency 由資料決定：query vector 會與所有 key vectors 比較，然後 $\mathrm{softmax}$ 把 scores 轉成 weights。

### 硬體意義

這種 dynamic global dependency 會為 sequence length $M$ 建立 $M \times M$ attention tensor。對長序列而言，這個 intermediate 可能主導 storage 與 traffic。因此硬體問題不只是快速矩陣乘法，還包括是否 materialize、tile、stream、fuse、sparsify 或 approximate score 與 attention tensors。

---

## 6. Self-attention 作為 Einsums

**Source anchor：** 投影片 L04-179 到 L04-198。

L04 的 self-attention 部分把每個 stage 都寫成 tensor expression。投影片註明省略某些 constant scaling steps，因此本章依投影片範圍說明，並在 source notes 標明這個限制。

### Rank names

| Rank | 意義 |
|---|---|
| $M$ | self-attention 中 query、key、value 的 sequence length |
| $P$ | $QK$ 中 query/output 端 sequence length 的 alias |
| $C$ | dictionary 或 vocabulary size |
| $D$ | input/global embedding dimension，常記為 $d_{\text{model}}$ |
| $E$ | query/key local embedding dimension，常記為 $d_k$ |
| $F$ | value local embedding dimension，常記為 $d_v$ |
| $G$ | output embedding dimension |
| $B$ | batch size |
| $H$ | attention heads 數量 |

### Single-head computation

第一層會用 $I_{m,d} = \sum_c IR_{m,c}WI_{c,d}$ 將 one-hot 或 raw token input $IR_{m,c}$ embedded 成 dense token。後續層通常已經接收 dense $I$。

Query 與 key 是投影到 $E$ 維空間：

$$Q_{m,e} = \sum_d I_{m,d}WQ_{d,e}, \qquad K_{m,e} = \sum_d I_{m,d}WK_{d,e}.$$

pre-softmax score tensor 會比較每個 query position $p$ 與每個 key position $m$：

$$QK_{m,p} = \sum_e Q_{p,e}K_{m,e}.$$

投影片 convention 是對每個固定 $p$，沿 $m$ 正規化：

$$SN_{m,p} = \exp(QK_{m,p}), \qquad SD_p = \sum_m SN_{m,p}, \qquad A_{m,p} = SN_{m,p}/SD_p.$$

Value projection 為 $V_{m,f} = \sum_d I_{m,d}WV_{d,f}$。attention output 是 value 的 weighted sum：

$$AV_{p,f} = \sum_m A_{m,p}V_{m,f}.$$

最後，$Z_{p,g} = \sum_f AV_{p,f}WZ_{f,g}$ 投影回 output embedding space。

### Worked example：兩個 token 的 attention scores

令 $M=2$、$E=2$。假設 $Q_0=[1,0]$、$Q_1=[0,1]$、$K_0=[1,1]$、$K_1=[2,0]$。對 query position $p=0$，scores 是：

$QK_{0,0}=Q_0 \cdot K_0=1$，且 $QK_{1,0}=Q_0 \cdot K_1=2$。

這個 query 的 normalized attention weights 是：

$A_{0,0}=e^1/(e^1+e^2)$，且 $A_{1,0}=e^2/(e^1+e^2)$。

接著 $AV_{0,f}=A_{0,0}V_{0,f}+A_{1,0}V_{1,f}$。這就是 attention 的核心意義：position 0 的 output 變成來自所有 token positions 的 value vectors 的資料相依加權混合。

### Computation properties

投影片 L04-190 到 L04-191 給出重要 scheduling properties：

- $Q$、$K$、$V$ projections 彼此獨立，可以平行計算。
- 在 attention 內部，$QK$ 必須先於 $\mathrm{softmax}$，而 $\mathrm{softmax}$ 必須先於 $AV$。
- 推論時 $WQ$、$WK$、$WV$、$WZ$ 是 static；$Q$、$K$、$V$、$QK$、$A$、$AV$、$Z$ 是 dynamic。
- MAC count 會隨 token count 二次方成長，因為 $QK$ 與 $M^2E$ 成正比，$AV$ 與 $M^2F$ 成正比。

### Batched attention

加入 batch 很機械化。對 dynamic tensors 加上 batch rank $b$：$QK_{b,m,p} = \sum_e Q_{b,p,e}K_{b,m,e}$、$A_{b,m,p}=SN_{b,m,p}/SD_{b,p}$、以及 $AV_{b,p,f}=\sum_m A_{b,m,p}V_{b,m,f}$。Weight matrices 在 batch elements 之間共享。

### Multi-head attention

Multi-head attention 加入 head rank $h$，讓每個 head 有獨立 projections：

$$Q_{b,h,m,e} = \sum_d I_{b,m,d}WQ_{d,h,e}, \quad K_{b,h,m,e} = \sum_d I_{b,m,d}WK_{d,h,e}, \quad V_{b,h,m,f} = \sum_d I_{b,m,d}WV_{d,h,f}.$$

每個 head 各自計算 $QK$、$\mathrm{softmax}$ 與 $AV$。Per-head outputs 會沿 head/value dimensions concatenate，例如 $C_{b,p,h \times F+f}=AV_{b,h,p,f}$，再由 $Z_{b,p,d}=\sum_g C_{b,p,g}WZ_{g,d}$ 投影；其中 $g$ 代表 flattened head-value dimension。

### 硬體意義

Attention 同時有有利與困難的硬體特性。Projection matrices 是 static 且可 reuse，適合 weight reuse。$QK$ 與 $AV$ 是 dense matrix multiplications，適合 array compute。但 score 與 attention tensors 是 dynamic、依 sequence length 而變，而且常很大。Softmax 引入 exponentiation、reduction、division 與 ordering constraints。Attention accelerators 與 fused attention kernels 之所以存在，是因為天真 materialize $QK$ 與 $A$ 會浪費 bandwidth。

### 常見誤解

**誤解：** Multi-head attention 只是比較大的 single-head attention。

**修正：** 它新增明確的 head rank $H$。這個 rank 暴露 parallelism 與獨立 projections，但也改變 storage layout、concatenation 與 final projection。Mapper 必須決定 heads 是平行處理、循序處理，還是 tile 處理。

---

## 7. 論文與來源橋接（Paper and Source Bridge）

**Source anchor：** 投影片 L04-175 到 L04-178，以及 L04-193 到 L04-194。

### Paper Bridge: Attention Is All You Need

**Bibliographic identity：** Ashish Vaswani et al.，*Attention Is All You Need*，NeurIPS 2017。Local PDF：`papers/Transformer (Attention).pdf`。

**Problem addressed：** Sequence transduction models 傳統上依賴 recurrence 或 convolution 在 positions 之間傳遞資訊。Paper 問的是：能否只用 attention mechanisms 連接 positions 並建立 sequence representations？

**Core idea：** Transformer 用 stacked self-attention 與 position-wise feed-forward layers 取代 recurrence。Scaled dot-product attention 計算 $\mathrm{softmax}(QK^T/\sqrt{d_k})V$；multi-head attention 則把多個 projected attention heads parallel 執行後再合併。

**Relevance to L04：** L04 在 convolution 之後，用 Transformer attention 作為第二個主要 workload，練習 Einsum thinking。Paper 提供 $Q$、$K$、$V$、scaled dot-product attention、multi-head attention 的語義，以及 token-token interaction 為何造成不同於 convolution 的硬體問題。

**Key claims used in this chapter：**

- Attention 將 query 與 key-value pairs 映射到 output。來源：Section 3.2。
- Scaled dot-product attention 定義為 $\mathrm{softmax}(QK^T/\sqrt{d_k})V$。來源：Section 3.2.1, Equation 1。
- Multi-head attention 將 queries、keys、values project 到多個 subspaces，並 parallel 執行 attention。來源：Section 3.2.2。
- Self-attention 以 constant sequential path length 連接所有 positions；對 sequence length $n$ 與 representation dimension $d$，per-layer complexity 是 $O(n^2d)$。來源：Section 4 與 Table 1。
- Paper 使用 positional encodings，因為 model 沒有 recurrence 或 convolution 預設編碼 position。來源：Section 3.5。

**What students should remember：** Attention 不只是新的 neural-network layer。它改變 workload 的 tensor shape：intermediate attention matrix 隨 token-token pairs 成長，這就是後續硬體討論 materialization、tiling、fusion、long-sequence memory pressure 的原因。

**Limitations and assumptions：** 本章使用 paper 支撐 attention 的 mathematical structure，不重現 translation benchmark claims。L04 遵循投影片 rank-name convention，可能相對於 paper matrix notation 轉置 indices。

**Suggested insertion points：** 如果 $Q$、$K$、$V$ 或 multi-head attention 仍像未解釋名詞，先讀這段 bridge 再回到第 6 節。

### Paper Bridge: Squeeze-and-Excitation Networks

**Bibliographic identity：** Jie Hu、Li Shen、Samuel Albanie、Gang Sun、Enhua Wu，*Squeeze-and-Excitation Networks*，CVPR 2018。Local PDF：`papers/L03_SENet_Hu_CVPR2018.pdf`。

**Problem addressed：** Standard convolution 會在 local receptive field 中混合 spatial 與 channel information，但不會明確 model global channel interdependencies。

**Core idea：** SE block 先 **squeeze** 每個 channel 成 global descriptor，再用 learned、input-dependent weights **excite** channels。這個 block 透過 channel-wise gates 重新校準 feature maps。

**Relevance to L04：** L04 對比 convolution 的 fixed local receptive field 與 attention-like mechanisms 的 dynamic emphasis。SENet 不是 Transformer self-attention，但它提供一個 CNN 中 data-dependent weighting 的具體例子：network 會依 current input 改變 channel importance。

**Key claims used in this chapter：**

- Paper 將 SE block 描述為 explicit modeling channel interdependencies，並 adaptively recalibrating channel-wise feature responses。來源：Abstract。
- Squeeze stage 使用 global information 形成 channel descriptor。來源：Section 3.1。
- Excitation stage 將 descriptor 映射到 channel weights，並做 channel-wise scaling。來源：Section 3.2。
- Paper 報告 SE blocks 可加入 existing architectures，且只增加 slight computational cost。來源：Abstract 與 Section 4。

**What students should remember：** Attention 不只是一組 Transformer equations。更廣義的架構想法是 data-dependent emphasis。SENet 強調 channels；Transformer attention 強調 token-token interactions。

**Limitations and assumptions：** SENet 不能支撐 L04 的 $QK^T$、沿 token positions 的 softmax、或 multi-head attention equations。它只用來釐清投影片提到的 local-vs-dynamic-emphasis 概念。

### 其他投影片列出的例子

投影片 L04-176 與 L04-178 引用 GPT-3（Brown et al., NeurIPS 2020）、AST（Gong et al., Interspeech 2021）、ViT（Dosovitskiy et al., ICLR 2021）、Jalammar 的 illustrated Transformer，以及 Dive into Deep Learning 作為例子或 figure sources。本章不重製它們的圖，也不使用它們的 paper-specific quantitative claims。

---

## 硬體意義總結

L04 的主要硬體意義如下：

- **Einsum 將 compute 與 schedule 分離。** 這讓 mapper 可以搜尋 loop orders，而不改變 layer 數學式。
- **Contracted ranks 是 reduction work。** 它們影響 accumulator lifetime、reduction trees 與 partial-sum storage。
- **Uncontracted ranks 是 output space。** 它們通常暴露 parallel work，但最佳 parallel rank 取決於 hardware shape 與 data reuse。
- **Flattening 讓 compute 規則化。** 它可以暴露 GEMM structure，但可能改變 address generation 並隱藏原本 locality。
- **Partitioning 是 tiling 的代數。** 它建立 tile ranks，後續可 map 到 memory levels 或 PE array dimensions。
- **CONV lowering 在規則性與可能 duplication 間取捨。** Materialized im2col 可能擴大 traffic；implicit lowering 保留 storage，但需要更特殊的 address generation。
- **Attention 引入大型 dynamic intermediates。** $M \times M$ score 與 attention tensors 對長序列造成 storage 與 bandwidth pressure。
- **Softmax 是 scheduling boundary。** 它包含 reductions 與 normalization，因此限制 fusion 與 parallel execution。

---

## 常見誤解

### 誤解：左側 output 就能告訴我們全部成本。

Output shape 只是一部分。Contracted ranks 決定每個 output 需要多少 products，而 tensor access functions 決定 reuse。$O_{n,m,p,q}$ 本身沒有告訴你 summing over $c,r,s$ 或搬移 $I$、$F$ 的成本。

### 誤解：Convolution lower 成 GEMM 表示 im2col 一定要 materialize。

代數 lowering 定義 patch matrix $T$。硬體可以 materialize $T$，也可以 on demand 產生它的 elements，或使用完全不在 storage 中命名 $T$ 的 direct convolution dataflow。

### 誤解：Softmax 只是 matrix multiplication 後的小細節。

Softmax 會改變 scheduling problem。為了對每個 $p$ 沿 $m$ 正規化，機器需要相關 score values、它們的 exponentials 或數值穩定等價形式、denominator，然後才能得到 normalized weights 供 $AV$ 使用。

### 誤解：Attention 的二次方成本來自 embedding projection。

Projections 的成本像 $MDE$、$MDF$ 或 $MFG$。二次項來自 token-token interactions：$QK$ 隨 $M^2E$ 成長，$AV$ 隨 $M^2F$ 成長。

---

## 連結

- **L02：** 提供 DNN layer vocabulary：FC、CONV、channels、batches、filters 與 tokens。
- **L03：** 引入 memory hierarchy 與第一批 Einsum/attention 例子。L04 讓 rank manipulation 更明確。
- **L05：** 使用同樣 Einsums，詢問哪種 loop order 形成 output-stationary、weight-stationary 或 input-stationary reuse。
- **L06：** 將 partitioning 轉成 temporal 與 spatial tiling，包含 distributed matrix multiplication 與 attention。
- **L07-L10：** Sparse tensors 仍然是 Einsum 中的 tensors；差別在 formats 與 traversal 必須跳過或表示 missing coordinates。
- **L12：** Precision choices 作用在 Einsum 的 operands 與 accumulators 上；reductions 通常需要更寬的 accumulators。
- **L13：** 把 ranks、tensor accesses 與 schedules 轉成 formal spaces/maps，用於精確 data-movement calculation。

---

## 獨立學習指南

### 進入下一講前必須掌握

- 給定一個 Einsum，能標出 output ranks 與 contracted ranks。
- 用自己的話解釋 ODE，而且不依賴任何特定 loop order。
- Flatten 與 unflatten 小 coordinate，例如 $(i,j)$。
- 將 $O_m = \sum_{c,h,w} I_{c,h,w}F_{m,c,h,w}$ 轉成 matrix-vector form。
- 解釋為什麼 $O_q = \sum_s I_{q+s}F_s$ 有 sliding-window reuse。
- 描述 materialized im2col 與 implicit convolution lowering 的差異。
- 追蹤 attention chain 從 $I$ 到 $Z$，並辨識哪些 tensors 是 static 或 dynamic。

### 自我檢核問題

1. 在 $Z_{m,n} = \sum_k A_{m,k}B_{n,k}$ 中，哪些 ranks 決定 output shape，哪些 ranks 決定每個 output 的工作量？
2. 為什麼更改 rank-variable names 不會改變 Einsum？
3. 若 $i=i_1 \times I_0+i_0$，當 $i=11,I_0=4$ 時，$i_1$ 與 $i_0$ 是多少？
4. 在 $O_q = \sum_s I_{q+s}F_s$ 中，哪些 values 會跨鄰近 $q$ reuse？
5. 當 $3 \times 3$ input 為了 $2 \times 2$ filter 被 lowered 時，可能出現什麼 data duplication？
6. 為什麼 $QK_{m,p} = \sum_e Q_{p,e}K_{m,e}$ 會產生 $M \times M$ tensor？
7. 哪些 attention operations 可以在 core attention 之前平行化？哪些必須有順序依賴？
8. 為什麼在此脈絡下 softmax 不只是 elementwise operation？

### 練習

1. 為 $Z_{m,n} = \sum_k A_{m,k}B_{k,n}$ 寫一個 loop nest。再寫第二個不同 loop order，並說明為什麼兩者都合法。
2. 對 $C=2,H=3,W=4$，將 $(c,h,w)$ flatten 成一個 rank $chw$。用 $chw=c \times H \times W+h \times W+w$ 計算 $(c,h,w)=(1,2,3)$ 的 flat index。
3. 對 $I=[1,2,0,3,4]$ 與 $F=[2,-1,1]$，計算 valid 1-D convolution outputs。
4. 為 single-channel $4 \times 4$ input 與 $2 \times 2$ filter、stride 1 建立 im2col coordinate table。patch matrix 有幾列幾欄？
5. 對 $M=4,E=8,F=8$，計算 $QK$ 與 $AV$ 的 MACs。再對 $M=8$ 重算。改變了什麼？
6. 設計題：如果 accelerator 無法儲存完整 $M \times M$ attention matrix，有哪兩種 implementation strategies 可以避免 materialize 它？

---

## 關鍵詞彙

| 詞彙 | 意義 |
|---|---|
| **Einsum** | 一種 tensor expression，指定 operands、ranks 與 reductions，但不選定 execution order。 |
| **Operational Definition for Einsums (ODE)** | 走訪每個合法 rank tuple、計算右側、並 assign 或 reduce 到左側的規則。 |
| **Rank variable** | Expression 中使用的 index symbol，例如 $m$ 或 $k$。 |
| **Rank name** | 維度的語義名稱，例如 $M$，依脈絡可表示 output channels 或 sequence length。 |
| **Rank shape** | 某個 rank 的 extent。 |
| **Contracted rank** | 只出現在右側、不出現在左側的 rank，因此會被 reduced。 |
| **Uncontracted rank** | 出現在 output 上、用來辨識 output coordinates 的 rank。 |
| **Iteration space** | ODE 走訪的所有合法 rank-variable tuples。 |
| **Flattening** | 將多個 ranks 合併成一個 compound rank，例如 $(c,h,w) \rightarrow chw$。 |
| **Partitioning** | 將一個 rank 拆成多個 ranks，例如 $i \rightarrow (i_1,i_0)$。 |
| **Toeplitz/im2col lowering** | 透過形成 input patch tensor 或等價 address pattern，將 convolution 改寫為 matrix multiplication。 |
| **Static tensor** | 推論時不隨 input examples 改變的 tensor，例如 trained weight matrix。 |
| **Dynamic tensor** | 每次 input 都會重新計算的 activation 或 intermediate tensor。 |
| **Self-attention** | $Q$、$K$、$V$ 都來自同一 input sequence 的 attention。 |
| **Attention tensor** | Softmax-normalized 的 $M \times M$ tensor $A$，用來為每個 query position 加權 value vectors。 |
| **Multi-head attention** | 加入 head rank $H$ 的 attention，使每個 head 有不同 projections 與 attention patterns。 |
| **Quadratic scaling** | Standard attention 中 token-token products 隨 sequence length $M$ 呈 $M^2$ 成長。 |

---

## 重點回顧

- Einsum 是本課程描述 compute 的精確語言：它指定 ranks、tensor accesses 與 reductions，同時保持 schedule 開放。
- Contracted ranks 是讀懂工作量與 reductions 的關鍵；uncontracted ranks 是讀懂 output shape 的關鍵。
- Flattening 與 partitioning 不是表面改寫。它們是 GEMM lowering、tiling 與 spatial mapping 背後的代數工具。
- FC layers 可以直接 flatten 成 matrix-vector 或 matrix-matrix multiplication。
- CONV 透過 patch tensor lower 成 matrix multiplication，但 materialize 這個 tensor 可能複製 input data。
- Attention 是 matrix-like Einsums 加上 $\mathrm{softmax}$ 的 cascade；token-token products 造成 $M^2$ work 與 intermediates。
- 硬體推理時要區分 static weights 與 dynamic intermediates，因為它們有不同 reuse opportunities。
- L04 準備了後續 mapping、partitioning、sparse traversal、precision choices 與 exact data-movement analysis 所需的 formal rank language。

---

## 附錄 - 投影片對照表

| 投影片範圍 | 本章章節 | 備註 |
|---|---|---|
| L04-1 到 L04-3 | Header 與 source context | 標題與 acknowledgements |
| L04-4 到 L04-9 | 第 1 節、關鍵詞彙 | ODE、tensor references、matrix/einsum patterns |
| L04-10 到 L04-14 | 第 2 節 | Rank tuples、partitioning、flattening、matrix-multiply variants |
| L04-15 到 L04-18 | 第 3 節 | 1-D convolution 與 rank-shape examples |
| L04-20 到 L04-41 | 第 4 節 | FC animation sequence 綜合為 FC lowering explanation |
| L04-42 到 L04-159 | 第 3 與第 4 節 | CONV visual build sequence 綜合說明；不嵌入 slide images |
| L04-160 到 L04-172 | 第 4 節 | Toeplitz/im2col 與 CONV-to-matmul summary |
| L04-174 到 L04-178 | 第 5 與第 7 節 | Convolution vs. attention、Transformers、tokens、source examples |
| L04-179 到 L04-189 | 第 6 節 | Attention mechanism as Einsums、full cascade |
| L04-190 到 L04-192 | 第 6 節與硬體意義 | Computation properties、static/dynamic tensors、batched attention |
| L04-193 到 L04-194 | 第 6 節 | Multi-head attention |
| L04-195 到 L04-198 | 第 6 節與關鍵詞彙 | Rank names 與 tensor glossary |

## 來源註記（Source Notes）

- 本章的講次順序、公式與 attention rank names 依據 `Lecture/L04-Einsums+Transformers.pdf`。
- ODE 的文字是根據投影片 L04-4 改寫。
- Convolution 與 CONV-to-matmul 解釋根據投影片 L04-15 到 L04-18，以及 L04-42 到 L04-172 重建；其中許多頁是動畫 frames，抽出的文字很少。
- Attention 公式依據投影片 L04-183 到 L04-194。投影片註明某些 constant scaling steps 沒有畫出；本章遵循投影片範圍，沒有額外加入被省略的 scaling factor。
- 小 vector、小 im2col coordinate table、two-token attention examples 是本章原創教學例子。
- Vaswani et al. 2017 與 Hu et al. 2018 的 paper bridges 使用本地 PDFs：`papers/Transformer (Attention).pdf` 與 `papers/L03_SENet_Hu_CVPR2018.pdf`；Brown et al. 2020、Gong et al. 2021、Dosovitskiy et al. 2021、Jalammar、D2L 仍是投影片明列的 examples，而不是本次獨立重讀的 paper sources。

## 不確定性註記（Uncertainty Notes）

- 現場講課可能對某些 animation frames 有不同強調；本章是根據投影片 sequence 重建可能的講解。
- 部分 CONV lowering 細節屬於 teaching interpretation，因為對應投影片多為 visual builds。
- Attention notation 遵循投影片 convention：$A_{m,p}$ 對每個 query/output position $p$ 沿 key/source position $m$ 正規化。其他教材常使用轉置過的 convention。
