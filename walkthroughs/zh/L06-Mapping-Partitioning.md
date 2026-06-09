# L06 - Mapping: Partitioning（映射：分割）

> **課程：** 6.5930/1 - Hardware Architectures for Deep Learning
> **授課者：** Joel Emer & Vivienne Sze（MIT EECS）
> **日期：** 2026-02-18 · **投影片：** 76 頁 · **來源：** [`Lecture/L06-Mapping-Partitioning.pdf`](../../Lecture/L06-Mapping-Partitioning.pdf)
>
> 本章重建投影片之間缺少的教學敘事。投影片提供順序與符號；paper bridge 提供外部技術脈絡；直覺、例子、硬體意義與常見誤解則是為了自學者重新撰寫。

---

## TL;DR

Partitioning（分割）是 mapping（映射）裡把 tensor index 切成「外層 tile index」與「內層 in-tile index」的決策。如果原本的 index 是 $i$、範圍是 $I$，切成 $(i_1,i_0)$ 之後，會有 $I=I_1I_0$ 與 $i=i_1I_0+i_0$。這個看似單純的代數變形，會產生兩個硬體後果。第一，它讓 working set（工作集合）變小，資料比較可能留在快速 buffer，而不是反覆從遠端記憶體讀回來。第二，它暴露出可以分配給不同 processing elements（PEs）的獨立 partitions。

真正困難的地方不是符號，而是要決定：哪些 ranks 要切？哪些外層 split 是 temporal（時間上依序執行）？哪些是 spatial（空間上平行分配到 PEs）？哪些 split 會讓 reduction 或 communication 變成必要？

L06 接續 L05 的 dataflow。L05 問：「loop 用什麼順序跑？weights、activations、partial sums 放在哪裡？」L06 進一步問：「每個 loop tile 多大？哪些 tile loops 要平行跑？」loop order、tile shape、spatial assignment 合在一起，才是一個完整 mapping。

---

## 這堂課要解決什麼問題

天真的 mapping 想法是：把 loop nest 寫出來，然後讓硬體跑。但硬體通常做不到這麼理想。DNN layer 的 tensor 可能遠大於 PE 附近的 buffer，而 accelerator 又可能有很多 PEs 必須被餵飽。如果 loop nest 沒有 partitioning，會出現兩個問題。

第一，reuse 來得太晚。假設某個 $B_k$ 會被很多輸出 $Z_m$ 使用。如果 computation 要走過很大的矩陣之後才回來重用它，那 $B_k$ 可能早就被 local buffer 淘汰。數學答案仍然正確，但 energy 被浪費在重讀資料上。

第二，parallelism 仍然只是隱含存在。tensor expression 可能有上千個互相獨立的 output elements，但硬體需要明確規則：哪個 PE 負責哪一部分？Partitioning 透過把一個 index 變成多層 index，讓其中某些外層 index 可以成為 spatial loops。

Source note：L06 slide 3 與 slide 8 直接列出兩個 partitioning 目標：降低 reuse distance，以及辨識可平行計算的資料集合。

---

## 為什麼這堂課重要

Partitioning 是抽象 tensor algebra 變成 machine schedule 的地方。同一個 matrix multiplication，可以映射成小型 local tile、大型 streaming tile，也可以分散到多個 devices 上執行。這些選擇會影響：

- **Energy（能量）：**資料來自 RF、SRAM、global buffer，還是 DRAM。
- **Bandwidth（頻寬）：**記憶體系統是否餵得動 PEs。
- **Latency（延遲）：**工作是依序跑，還是分散到多個 PEs 跑。
- **Utilization（利用率）：**PEs 是否拿到均衡且足夠的工作。
- **Communication（通訊）：**split 的 rank 是否需要跨 partitions reduction。
- **Programmability（可程式化性）：**mapping 能否被 Timeloop 或 TeAAL 這類工具系統化描述。

一個有用的心智模型是：partitioning 會把 computation 在 Roofline plot 上移動。好的 tiling 可以減少每個 operation 需要搬動的 bytes，因此提高 operational intensity；更多 spatial partitioning 可以暴露 compute throughput，但前提是 memory 和 communication system 撐得住。

---

## 先備知識與心智模型

讀本章前，最好熟悉前面幾堂課的四個觀念。

**Einsum notation：**重複出現的 indices 會被加總；留在左側的 indices 是 output free indices。例如 $Z_m=\sum_k A_{k,m}B_k$ 是 matrix-vector multiply。

**Loop nests：**einsum 可以實作成 loops。free indices 與 reduction indices 的 loop order 會控制 locality 與 accumulation。

**Memory hierarchy：**越靠近 PE 的資料越便宜；DRAM 通常最貴。精確能耗比取決於製程與設計，但「local reuse 很重要」這個質性結論穩定成立。

**Dataflow：**dataflow 是 weights、activations、partial sums 的儲存與排程策略。Partitioning 是創造 dataflow 的主要旋鈕之一。

Teaching interpretation：可以把 partitioning 想成把方格紙蓋到 tensor 上。每一格是一個 tile。Temporal tile 表示「同一份硬體先處理這格，再處理下一格」；spatial tile 表示「不同 PEs 同時處理不同格」。

---

## 學習目標

讀完本章後，你應該能夠：

- 解釋為什麼 partitioning 一個 index 會增加一個 rank。
- 把簡單 tensor expression 從未分割形式改寫成 partitioned notation。
- 區分 temporal partitioning 與 spatial partitioning。
- 解釋 partition size 如何改變 reuse distance 與 working-set size。
- 判斷 split reduction rank 什麼時候會產生跨 partition reduction。
- 追蹤 distributed matrix multiply 例子裡，partitioned tensors 如何導向 delayed reduction。
- 比較 attention 的 tensor parallel、head parallel、data/model-dimension parallel。
- 用 Roofline reasoning 解釋為什麼更好的 tiling 可能提高 attainable performance。

---

## 主要教材式敘事

### 1. Partitioning 改變結構，但不改變數學

L06 最重要的規則是：

$$
i \rightarrow (i_1,i_0), \qquad i = i_1 I_0 + i_0, \qquad I = I_1 I_0.
$$

tensor 裡的值沒有變，變的是 address 方式。原本的 vector $A_i$ 變成 $A_{i_1,i_0}$。如果 $I=16$、$I_0=4$，則 $I_1=4$。原本的 $A_{11}$ 會變成 $A_{2,3}$，因為 $11=2\cdot4+3$。

這就是投影片說「partitioning always adds a rank」的意思。原本一個 index $i$，現在用兩個 indices 表示。mapper 因此有兩個 loops 可以安排、重排，或分配到不同 PEs。

常見誤解：partitioning 不是把 tensor 切得比較好看而已。它改變硬體看到的 loop structure；而 loop structure 會決定 locality、communication、PE assignment。

Source note：vector 與 matrix 的 index identities 直接根據 L06 slides 5-6。

### 2. Temporal partitioning 降低 reuse distance

考慮投影片中的 matrix-vector multiply：

$$
Z_m = \sum_k A_{k,m}B_k.
$$

未分割 loop nest 是：

```python
for m in range(M):
    for k in range(K):
        Z[m] += A[k, m] * B[k]
```

對每個 output $m$，computation 都走過所有 $k$。如果 $K$ 很大，active row/column working set 可能放不進 PE 附近的 buffer。原本稍後要 reuse 的值，可能在 reuse 前就被逐出。

現在把 $m$ 與 $k$ 都切開：

$$
m = m_1 M_0 + m_0, \qquad k = k_1 K_0 + k_0.
$$

同一個 computation 變成：

$$
Z_{m_1,m_0} = \sum_{k_1,k_0} A_{k_1,k_0,m_1,m_0} B_{k_1,k_0}.
$$

一種可能 loop nest 是：

```python
for m1 in range(M1):
    for k1 in range(K1):
        for m0 in range(M0):
            for k0 in range(K0):
                Z[m1, m0] += A[k1, k0, m1, m0] * B[k1, k0]
```

當 $m_1$ 與 $k_1$ 固定時，內層 loops 只碰到形狀為 $K_0\times M_0$ 的 $A$ tile、$K_0$ 個 $B$ values，以及 $M_0$ 個 $Z$ partial sums。如果這個 working set 放得進 local buffer，硬體可以在移到下一個 tile 前重用這些值。

小例子：令 $M=8$、$K=8$、$M_0=2$、$K_0=4$。固定 $(m_1,k_1)$ 時，內層工作使用 $2\times4=8$ 個 $A$ values、$4$ 個 $B$ values、$2$ 個 $Z$ partial sums。partitioned version 給 mapper 一個明確 local working set：$8+4+2=14$ 個 tensor values。如果 PE-local buffer 放得下，tile 就能用較少高層記憶體流量完成。

硬體意義：temporal partitioning 主要是 locality 工具。它不會自動增加 PEs，而是讓 reuse 及早發生，進而降低 energy 與 bandwidth pressure。

### 3. Spatial partitioning 把 tile loops 變成平行工作

同一個 split 也可以用於 spatial execution。對 elementwise multiply：

$$
Z_i = A_iB_i,
$$

把 $i$ 切成 $(i_1,i_0)$：

$$
Z_{i_1,i_0}=A_{i_1,i_0}B_{i_1,i_0}.
$$

如果有 $I_1$ 個 PEs，外層 loop 可以是 spatial：

```python
spatial_for i1 in range(I1):
    for i0 in range(I0):
        Z[i1, i0] = A[i1, i0] * B[i1, i0]
```

這個例子容易平行化，是因為 $i_1$ 是 output free partition。不同 $i_1$ 寫入不同 output elements，不需要彼此通訊。

因此可以得到一條實用規則：

- split **free rank** 通常會產生獨立 output partitions。
- split **reduction rank** 會產生 partial sums，最後必須合併。

這條規則是後半堂課的核心轉折。

### 4. Distributed matrix multiply：partitioning 會創造 communication

matrix multiply case study 從下式開始：

$$
Z_{m,n}=\sum_k A_{k,m}B_{k,n}.
$$

投影片說此 implementation based on ThunderKittens distributed matrix multiply algorithm，但本 worker pass 在 local repository 沒有找到該 paper PDF。因此本章把 ThunderKittens 細節視為 slide-derived anchors；一般 partitioning 解釋則是 teaching interpretation。

partitioned expression 是：

$$
Z_{m_1,m_0,n}=\sum_{k_1,k_0}A_{k_1,k_0,m_1,m_0}B_{k_1,k_0,n}.
$$

flattening 可確認它仍然是同一個 matrix multiply：

$$
Z_{m_1M_0+m_0,n}
=\sum_{k_1,k_0}A_{k_1K_0+k_0,m_1M_0+m_0}B_{k_1K_0+k_0,n}.
$$

如果 $k_1$ 被分散到 $G$ 個 PEs，每個 PE 只擁有 reduction dimension 的一部分。PE $g$ 可以計算自己 local $k_1=g$ slice 的貢獻，但無法單獨產生 final $Z$，因為最後答案需要跨所有 $k_1$ partitions 加總。

這就是 communication lesson：split reduction rank 可以買到 parallel work，但結果是 partial；parallelism 的代價是 reduction。

### 5. Delayed reduction 把 dependency 明確化

投影片引入 delayed-reduction variant。不要立刻對 $k_1$ reduction，而是先把 $k_1$ 放到左側：

$$
ZT_{k_1,m_1,m_0,n}
=\sum_{k_0}A_{k_1,k_0,m_1,m_0}B_{k_1,k_0,n}.
$$

再另外 reduction：

$$
Z_{m_1,m_0,n}=\sum_{k_1}ZT_{k_1,m_1,m_0,n}.
$$

這不是改變答案的數學技巧，而是改變 communication 發生時間的 scheduling trick。每個 PE 可以先用 local data 算出自己的 $ZT$ slice，再由獨立 communication/reduction stage 合併。

硬體意義：delayed reduction 可以讓 PEs 在同步前做更久的 local matrix multiply，改善 utilization。它也讓 compiler 或 runtime 能明確看見 communication boundary，選擇 all-reduce、reduce-scatter 或其他 collective pattern。代價是 $ZT$ 比 $Z$ 多帶一個 $k_1$ rank，在 reduction 前需要額外暫存。

### 6. Distributed tensor shapes 是 mapping contracts

Slides 13-17 介紹 $AD_{g,k_0,m_1,m_0}$、$BD_{g,k_0,n}$、$ZD_{g,m_0,n}$。這些不是單純重新命名的 arrays，而是 mapping 與 machine 之間的 contracts。

- $AD_{g,k_0,m_1,m_0}$ 表示 PE/group $g$ 擁有 $A$ 的 $k_1=g$ slice。
- $BD_{g,k_0,n}$ 表示同一 PE/group 擁有 matching $k_1=g$ 的 $B$ slice。
- $ZD_{g,m_0,n}$ 表示某個 output partition 在 communication 前後由 local side 持有。

讀 distributed tensor name 時，可以問三個問題：

1. 哪個 rank 決定 PE ownership？
2. 哪些 ranks 是 PE 內部 local loops？
3. 哪些缺失的 ranks 代表 replication、reduction 或後續 assembly？

### 7. Attention partitioning：三種切 Transformer work 的方式

attention section 把同樣的 rank-splitting logic 用在 transformer equations。投影片列出三種策略。

**Tensor parallel：**把 batch dimension $b\rightarrow(b_1,b_0)$，並平行執行 $b_1$。forward pass 中不同 batch elements 彼此獨立，所以 attention core 的 communication 較少。代價是 weight replication：每個 PE group 都需要 projection weights。

**Head parallel：**把 attention head dimension $h\rightarrow(h_1,h_0)$，並平行執行 $h_1$。不同 heads 在 concatenation 與 output projection 前大多可獨立計算。這很自然，因為 transformer attention 本來就把 computation 分解到 heads。

**Data/model-dimension parallel：**把 $h$ 與 $d$ 都切開，並平行執行 $h_1$ 與 $d_1$。這暴露更多 parallelism，但 split $d$ 會碰到 projection reductions，因此 partial results 可能需要 communication。

Slides 20-22 的 tensor names 很密，但設計問題很簡單：你切的是 free rank、自然獨立的 structural rank，還是 contraction rank？

小例子：假設 transformer layer 有 $H=8$ heads 與 $4$ 個 PE groups。head-parallel split 可以選 $H_1=4$、$H_0=2$。每個 PE group 計算兩個 heads，這些 heads 的 attention score 與 value contraction 可以 local 執行。之後 outputs 必須 concatenated 到 model dimension，再交給 output projection。如果改成 split model dimension $d$，Q/K/V projections 會產生 $d$ 上的 partial sums，所以 reduction 或 collective communication 會出現。

連結 L04：L04 把 attention 介紹為 query、key、value、sequence、head dimensions 上的 einsums；L06 問的是這些 ranks 哪些該為硬體被 split。

---

## Worked Examples

### Example 1：還原原始座標

令 $I=12$、$I_0=3$、$I_1=4$。原始 index 是 $i=i_1I_0+i_0$。

- $(i_1,i_0)=(0,2)$ 對應 $i=2$。
- $(i_1,i_0)=(2,1)$ 對應 $i=7$。
- $(i_1,i_0)=(3,2)$ 對應 $i=11$。

硬體意義：$i_1$ 可以命名 tile 或 PE assignment；$i_0$ 命名 tile 內的位置。

### Example 2：Free-rank split vs. reduction-rank split

對 $Z_m=\sum_kA_{k,m}B_k$：

- split $m$ 會產生不同 output tiles。不同 PEs 可以擁有不同 $m_1$，因為它們寫不同 $Z$ elements。
- split $k$ 會產生 partial sums。不同 PEs 可以擁有不同 $k_1$，但它們都貢獻到同一批 final $Z_m$，所以需要 reduction。

硬體意義：兩種 split 都暴露 parallelism，但只有 free-rank split 通常不需要 communication。

### Example 3：用 Roofline 讀 tile

Operational intensity 是：

$$
\text{operational intensity}=\frac{\text{operations}}{\text{bytes moved from the chosen memory level}}.
$$

假設一個小 tile 做 $128$ 個 MACs。如果 implementation 從 DRAM 搬 $512$ bytes，operational intensity 是 $128/512=0.25$ MAC/byte。若更好的 temporal partitioning 讓同樣 $128$ MACs 只需從 DRAM 搬 $128$ bytes，operational intensity 變成 $1$ MAC/byte。MAC 數沒有變，變的是 mapping 導致的 memory traffic。

Source note：這是原創教學例子。definition 與 performance-bound interpretation 依據 Williams、Waterman、Patterson 的 Roofline paper，CACM 2009，Roofline Model section 與 Figure 1 討論。

---

## Key Equations and How to Read Them

### Partition identity

$$
i=i_1I_0+i_0,\qquad I=I_1I_0.
$$

$i_1$ 選 tile；$i_0$ 選 tile 內座標。硬體設計者在意的是：$i_1$ 可以成為 temporal tile loop，也可以成為 spatial PE-assignment loop。

### Partitioned matrix-vector multiply

$$
Z_{m_1,m_0}
=\sum_{k_1,k_0}A_{k_1,k_0,m_1,m_0}B_{k_1,k_0}.
$$

free ranks $(m_1,m_0)$ 識別 output elements；repeated ranks $(k_1,k_0)$ 識別 reduction。如果 $k_1$ 是 spatial，final output correctness 需要跨 $k_1$ partitions 加總。

### Delayed reduction

$$
ZT_{k_1,m_1,m_0,n}
=\sum_{k_0}A_{k_1,k_0,m_1,m_0}B_{k_1,k_0,n},
\qquad
Z_{m_1,m_0,n}=\sum_{k_1}ZT_{k_1,m_1,m_0,n}.
$$

temporary tensor $ZT$ 保留 partitioned reduction rank，讓 communication stage 清楚浮現。

### Roofline bound

$$
\text{attainable performance}
=\min(\text{peak compute},\ \text{peak bandwidth}\times\text{operational intensity}).
$$

這條式子說，一個 mapping 可能 compute-bound，也可能 bandwidth-bound。當 partitioning 降低同樣 operations 所需 bytes moved 時，就可能提高 operational intensity。

---

## Hardware Implications

**Buffer sizing：**tile dimensions 必須讓 active weights、activations、partial sums 放進目標 buffer level。

**Bandwidth：**temporal tile 太大會 spill 到高層 memory；太小則可能 reuse 不足、loop overhead 過高。

**PE utilization：**spatial partitioning 暴露 parallel work，但每個 partition 必須有足夠且相近的工作量。

**Reduction cost：**split reduction rank 會創造 communication；mapping 必須考慮 collective bandwidth 與 synchronization。

**Area：**更多 local storage 可支援較大 tile，但會消耗 area，也可能影響 clock 或 PE count。

**Correctness：**partitioning 必須保留原本 flattened coordinate mapping，且不能漏掉 reductions。

**Programmability：**明確 rank splits 讓 mapping 更容易被 formal tools 描述，因為 tile shape 與 spatial assignment 變成一等公民。

---

## Common Misconceptions

### 誤解：partitioning 和 parallelism 是同一件事。

Partitioning 只是在 tensor/loop 結構中加入 rank levels。只有當某個 level 被 spatial assignment 到 PEs，parallelism 才真正出現。

### 誤解：split 任何 rank 都不需要 communication。

split free output rank 通常容易；split reduction rank 會產生 partial sums，最後必須合併。

### 誤解：tile 越小越好。

小 tile 可能放進 local memory，但也可能降低 reuse、增加 loop overhead、讓 PEs 吃不飽。tile size 是 tradeoff，不是單調越小越好。

### 誤解：$AD$、$BD$ 這種 tensor name 只是 bookkeeping。

Distributed tensor names 編碼了 ownership。它們告訴我們哪個 PE 持有哪些資料，以及哪個 communication step 會出現。

---

## Connections to Previous and Later Lectures

**L01-L03：**memory hierarchy 與 operational intensity 解釋 partitioning 為何重要。數學 computation 一樣，但 energy 與 bandwidth 可能差很多。

**L04：**attention 被介紹成 tensor algebra。L06 使用同樣 ranks 推理 transformer execution 的 parallelism。

**L05：**loop order 與 dataflow 描述資料如何穿越 PE array。L06 加上 tile shape 與 spatial assignment。

**L07-L10：**sparse architectures 仍需要 partitioning，但 sparsity 讓 tile work 不均、metadata 出現，load balance 更難。

**L13：**calculating motion 會把這些 qualitative mapping choices 轉成 explicit reads、writes、transfers 計數。

---

## Paper Bridge: Roofline

### Bibliographic identity

- **Title:** "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
- **Authors:** Samuel Williams, Andrew Waterman, and David Patterson
- **Year / venue:** Communications of the ACM, 2009
- **Used in lecture(s):** 支撐 L01/L02 的 roofline material，也支撐 L06 的 partitioning-as-locality 討論。

### Problem addressed

這篇 paper 要解決的是：在 compute capability 與 memory bandwidth 都差異很大的 multicore systems 上，如何用簡單模型理解 performance bottleneck。它不是精準預測 runtime，而是提供 bound-and-bottleneck model，幫助判斷 kernel 是 compute throughput limited 還是 memory bandwidth limited。

### Core idea

Roofline 把 attainable performance 對 operational intensity 作圖。Operational intensity 是 operations per byte of DRAM traffic，而且 traffic 是經過 cache hierarchy 過濾後到 DRAM 的 bytes。performance 被 peak compute 與 peak memory bandwidth times operational intensity 的較小者限制。

### Relevance to this lecture

Partitioning 會改變同樣 arithmetic operations 需要搬動多少 bytes。因此，partitioning 可以透過提高 operational intensity，把 kernel 在 Roofline plot 上往右移。它也能提醒我們：如果 bandwidth 已經是限制，單純加更多 spatial PEs 不一定會更快。

### Key claims used in this chapter

- Operational intensity 定義為 operations per byte of DRAM traffic，且 memory traffic 是 cache-filtered traffic。Source：Roofline paper，Roofline Model section，CACM 2009，pp. 66-67。
- Roofline bound 是 $\min(\text{peak compute},\text{peak bandwidth}\times\text{operational intensity})$。Source：Roofline paper，p. 67 formula。
- Ridge point 表示達到 peak compute 所需的最小 operational intensity。Source：Roofline paper，Figure 1 discussion，p. 67。

### What students should remember

- Roofline 不會替你選 partition，但會解釋 partition 為何有意義。
- 能 local reuse data 的 mapping 可以提高 operational intensity。
- 沒有足夠 bandwidth 的 spatial parallelism，只是在同一條 bandwidth roof 上更用力。

### Limitations and assumptions

Roofline 是 bound model，不是精準 simulator。它抽象掉 control overhead、synchronization、bank conflicts、irregular sparsity 等細節。在 DNN accelerators 中，它適合作為進入 detailed mapping/energy tools 前的 intuition。

### Suggested insertion points

在說明 temporal partitioning 如何降低 bandwidth pressure，以及評估 spatial partition 是否可能 compute-bound 或 bandwidth-bound 時引用。

---

## Paper Bridge: TeAAL

### Bibliographic identity

- **Title:** "TeAAL: A Declarative Framework for Modeling Sparse Tensor Accelerators"
- **Authors:** Nayak et al.
- **Year / venue:** MICRO 2023
- **Used in lecture(s):** L01 pyramid context、L06 mapping formalism，以及後續 sparse accelerator lectures。

### Problem addressed

Sparse tensor accelerators 很難比較，因為每個設計都混合 algorithm、tensor formats、mappings、architecture resources 與 bindings。TeAAL 提供 declarative 方法描述這些 concerns，並生成 accelerator models。

### Core idea

TeAAL 把 tensor computation 與 mapping 分開。Extended einsums 描述要算什麼；mapping specifications 描述 ranks 如何排序、partition、schedule。這正對應 L06：partitioning 改變 mapping，但不改變 mathematical einsum。

### Relevance to this lecture

L06 可以視為 TeAAL mapping layer 的手算版。當我們把 $i$ 切成 $(i_1,i_0)$，再決定 $i_1$ 是 temporal 或 spatial，我們就在做 TeAAL 想明確表示的 mapping choice。

### Key claims used in this chapter

- TeAAL 使用 einsums 指定 computation，並用 mapping specifications 描述 rank ordering、partitioning、scheduling。Source：TeAAL Sections 2.2 and 2.3。
- TeAAL 包含 mapping、tensor format、architecture、binding 等 separate specifications。Source：TeAAL Sections 3-4。
- 該 framework 針對 sparse tensor accelerators，其中 mapping 與 format choices 強烈互動。Source：TeAAL abstract and Section 2。

### What students should remember

- Einsum 說要算什麼；mapping 說如何執行。
- Partitioning 是 mapping operation，不是改變數學結果。
- 明確 rank splits 有用，因為工具可以分析 locality、parallelism、communication。

### Limitations and assumptions

TeAAL 是 modeling framework，不是會自動找出最佳 mapping 的萬能 optimizer。本章用 TeAAL 釐清 abstraction，不主張 L06 例子有特定 speedup。

### Suggested insertion points

在 partition identity 之後，以及說明 distributed tensors 是 mapping contracts 時引用。

---

## 獨立學習指南

### 如何讀這堂課

1. 練習把一個 index $i$ 轉成 $(i_1,i_0)$，直到 coordinate mapping 變直覺。
2. 對每個 split 問：這是 free rank 還是 reduction rank？
3. 對每個 outer split rank 標記 temporal 或 spatial。
4. 估算 inner tile 的 active working set。
5. 找出是否需要 communication 或 reduction。

### Self-check questions

1. 為什麼把 $i$ partition 成 $(i_1,i_0)$ 會增加 tensor rank？
2. 在 $Z_m=\sum_kA_{k,m}B_k$ 中，split $m$ 和 split $k$ 差在哪裡？
3. partitioned matrix-vector tile 需要哪些資料放進 local buffer？
4. 為什麼 delayed reduction 會引入 $ZT_{k_1,m_1,m_0,n}$？
5. 在 attention 中，為什麼 head parallelism 通常比 split contraction dimension 容易？
6. partitioning 如何提高 operational intensity？

### Exercises

1. **Conceptual：**各用一句話說明 tile loop 與 spatial loop 的差別。
2. **Small calculation：**令 $I=24$、$I_0=6$。$I_1$ 是多少？$i=17$ 的 partitioned coordinate 是什麼？
3. **Loop-nest rewrite：**用 $M_0=2$、$K_0=4$ partition $Z_m=\sum_kA_{k,m}B_k$，並寫出 loop nest。
4. **Design tradeoff：**你有四個 PEs。matrix-vector multiply 會先 split $m$ 還是 $k$？說明 communication tradeoff。
5. **Paper bridge：**用 Roofline 說明，為什麼減少 DRAM traffic 的 tile 即使 MAC count 不變，也可能改善 attainable performance。
6. **Open-ended architecture reasoning：**對 $H=16$ heads 與 $8$ PE groups 的 attention，提出 head-parallel split，並列出哪些 tensors 被 sharded 或 replicated。

---

## 關鍵詞彙

### Partitioning（分割）

把 index range 切成多個 rank levels，例如 $i\rightarrow(i_1,i_0)$。硬體上重要，因為新的 rank levels 可以成為 tile loops 或 PE-assignment loops。

### Tile（分塊）

由固定 outer partition indices 選出的 tensor 或 iteration-space 子集合。好的 tile 要大到能 reuse，也要小到能放進目標 buffer。

### Reuse distance（重用距離）

同一個 value 兩次使用之間隔了多少 accesses。Temporal partitioning 試圖縮短 reuse distance，讓 value 留在快速儲存中。

### Temporal partitioning（時間式分割）

outer tile loop 依序執行的 partition。主要用途是 locality 與 buffer management。

### Spatial partitioning（空間式分割）

outer tile loop 被分散到多個 PEs 的 partition。主要用途是 parallelism。

### Free rank（自由 rank）

出現在 output 的 index。split free rank 通常會產生獨立 output partitions。

### Reduction rank（歸約 rank）

只出現在 einsum 右側、最後被加總掉的 index。split reduction rank 會產生需要合併的 partial sums。

### Working set（工作集合）

tile inner loops 執行期間必須保持 live 的資料，包括 input tiles 與 partial sums。

### Delayed reduction（延遲歸約）

先把 partitioned reduction rank 留在 temporary output，再於後續 stage reduction。它讓 communication stage 明確化。

### Operational intensity（操作強度）

每搬動一 byte 可執行的 operations，classic Roofline 通常以 DRAM traffic 為基準。operational intensity 越高，越可能發揮 compute throughput。

### Ridge point（屋脊點）

Roofline 中 bandwidth roof 與 compute roof 相交的位置，表示要變成 compute-bound 所需的 operational intensity。

### Distributed tensor（分散式 tensor）

indices 中包含 PE/device ownership rank 的 tensor，例如 $AD_{g,k_0,m_1,m_0}$ 裡的 $g$。

---

## 重點回顧

- Partitioning 改變 tensor rank structure，但不改變數學結果。
- Temporal partitioning 是 locality；spatial partitioning 是 parallel work。
- Free-rank splits 通常比 reduction-rank splits 更容易平行化。
- Delayed reduction 讓 communication 明確，給 mapper 安排 collectives 的位置。
- Attention partitioning 是同一套 rank-splitting idea 套用到 transformer dimensions。
- Roofline 解釋 partitioning 為何能透過減少 bytes moved per operation 改善 performance。

---

## 連結

本章往回連到 L05，因為 dataflow 需要 tile shapes 與 spatial loops 才能形成完整 mapping。它也連到 L03/L04，因為 einsum notation 讓 rank splitting 變得精確。往後它連到 L07-L10，因為 sparse tensors 會讓 partitioning 遇到 irregular densities 與 load balance 問題。它也直接連到 L13，因為 calculating data motion 必須知道精確的 partitioned loop nest。

---

## 附錄 - Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L06-1 | Title and metadata | Lecture identity |
| L06-2 | Main narrative | Partitioning topic setup |
| L06-3 | 這堂課要解決什麼問題 | objectives 直接來自投影片 |
| L06-4 | Temporal partitioning | unpartitioned matrix-vector example |
| L06-5 | Partitioning 改變結構 | vector rank split |
| L06-6 | Partitioning 改變結構 | matrix rank split |
| L06-7 | Temporal partitioning | partitioned matrix-vector loop nest |
| L06-8 | Spatial partitioning | `spatial_for` example |
| L06-9-L06-18 | Distributed matrix multiply | 擴充 reduction/communication 解釋 |
| L06-19-L06-22 | Attention partitioning | 擴充 transformer-rank interpretation |
| L06-23-L06-76 | Source notes | extracted text 多數為空白或 animation frames，未視為新增概念內容 |

---

## Source Notes

- 本章順序與主要 examples 依據 `Lecture/L06-Mapping-Partitioning.pdf`。
- partitioning objectives 直接來自 L06 slide 3。
- vector 與 matrix partition identities 直接來自 L06 slides 5-7。
- distributed matrix multiply notation 與 delayed reduction 依據 L06 slides 10-18。ThunderKittens 由 slide deck 引用，但本 worker pass 沒有 local paper PDF，因此 ThunderKittens-specific claims 僅視為 slide-stated。
- attention partitioning strategies 依據 L06 slides 20-22，並使用 L04 的 standard transformer background。
- Roofline bridge 使用 `papers/Roofline Model.pdf`，尤其 Roofline Model section 與 Figure 1 discussion。
- TeAAL bridge 使用 `papers/TeAAL.pdf`，尤其 Sections 2.2、2.3、3、4。
- worked examples 除非另有標註，都是原創教學例子。

## Uncertainty Notes

- live lecture 可能對後段 animation frames 有不同強調；L06-22 之後的 extracted text 多數是空白頁面。
- 若之後加入 ThunderKittens 原 paper，distributed matrix multiply 的 implementation details 應再核對。
- 本章沒有刪除或審核 `assets/L06` 既有 slide-derived assets；只是避免新增 copied figures。
