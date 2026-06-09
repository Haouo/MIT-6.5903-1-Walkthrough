# L13 — 計算資料搬移（Calculating Data Motion）

> **課程：** 6.5930/1 — 深度學習硬體架構（Hardware Architectures for Deep Learning）
> **講師：** Joel Emer 與 Vivienne Sze（MIT EECS）
> **講授日期：** 2026 年 3 月 18 日 · **投影片：** 66 頁 · **來源：** [`Lecture/L13-Calculating_Motion.pdf`](../../Lecture/L13-Calculating_Motion.pdf)
>
> *鳴謝：Angshuman Parashar / Michael Gilbert。*
>
> *本文是以「概念」為單位重建講課脈絡的導讀（walkthrough），依主題而非逐頁編排。每一節都標註其對應的投影片範圍，方便你對照原始投影片閱讀。*

---

## 一句話總結（TL;DR）

知道一個 DNN 加速器採用哪種資料流（dataflow）還不夠——你需要精確算出每一個資料元素在記憶體階層的每一層之間被搬移了幾次。本講發展出一套嚴謹的代數方法來完成這件事。以 **ISL（Integer Set Library，整數集合函式庫）** 為數學基礎，它依序建構：(1) 一個列舉所有乘加（MAC）運算的「迭代空間（iteration space）」；(2) 把迭代空間中的點對應到它們所存取資料元素的「投影映射（projection map）」；(3) 在迭代空間上施加排程（資料流）的「時間戳映射（timestamp map）」；以及 (4) 用來精確計算每個時步（time step）需要讀入多少新資料的「差量計算（delta calculation）」。同一套機制接著被提升到 **L1 分塊（tile）** 層級，以計算緩衝區層級的資料搬移量，並自然延伸至「縮退（shrink）」計算（資料何時離開緩衝區）。本講最後藉由將排程從輸出駐留（output-stationary）改為權重駐留（weight-stationary）來對比兩者的資料搬移剖析（profile），展示這套形式化工具可以直接比較不同資料流的代價。

---

## 本講要解決的問題

前面的 mapping lectures 教了定性的 dataflow labels：output-stationary 讓 partial sums 靠近 PE，weight-stationary 讓 weights 靠近 PE，row-stationary 嘗試平衡多種 reuse。這些名稱很有用，但還不足以設計硬體。硬體架構師最後需要的是數字：

- 有多少 input values 進入 L1？
- 有多少 weights 可以跨 tiles 保留？
- Output partial sum 什麼時候可以被 evict？
- Buffer 的 minimum live set 多大？
- 換一個 loop order 後，bandwidth 需求如何改變？

天真的作法是模擬 loop nest 並計數 accesses。這對小例子可行，但不適合 design-space exploration。L13 教的是 symbolic method：把 computation、data access、schedule、fill 與 eviction 都表示成 integer sets 與 relations。這樣 memory traffic 就變成 set algebra。

## 為什麼本講重要

Data movement 是整門課的主成本主題，但「資料搬移很貴」容易說、難以精準計算。這一講補上 quantitative layer。只要能寫出 iteration space、projection maps 與 timestamp map，你就可以不用只靠 stationarity 直覺，而是直接問「哪些資料真的跨過這個 memory boundary？」

這很重要，因為 buffer capacity 與 bandwidth 都是 architecture contracts。如果 L1 buffer 必須同時容納 \(S\) 個 weights、\(S\) 個 sliding-window inputs 與一個 output accumulator，那 PE design、SRAM banking、NoC pressure 與 compiler schedule 都必須尊重這個 live set。錯估 data motion 會讓設計在紙上看起來很有效率，實作時卻碰到 timing、bandwidth 或 energy 目標。

## 先備知識與心智模型

你需要前幾講的三個觀念：

1. **Einsum / loop-nest thinking：** 每個 DNN layer 都能描述成一組 loop indices 與 tensor accesses。
2. **Dataflow thinking：** Mapping 決定哪些 loop dimensions 是 temporal、哪些是 spatial，以及每個 level reuse 什麼 data。
3. **Memory-hierarchy thinking：** RF、L1、global buffer、DRAM 的 access cost 不同。

這一講的心智模型是一張行事曆。Iteration space 說明有哪些工作項目；projection maps 說明每個工作需要哪些資料；timestamp map 把工作排進時間；delta 與 shrink 則比較相鄰時間格，判斷什麼資料必須進房間、什麼資料可以離開。

---

## 學習目標（Learning Objectives）

讀完本講後，你應該能夠：

- 以 ISL 集合定義**迭代空間**，並寫出從迭代空間點到資料元素（輸入、權重、輸出）的投影映射。
- 建構編碼資料流（迴圈順序）的**時間戳映射**，並推導其反映射與合成。
- 用**差量計算**（當前集合減去前一集合）在最細粒度下計算新資料讀取次數。
- 透過分塊時間戳維度，將單一時步的差量提升到 **L1 分塊層級**，並計算「填充（fill）」與「縮退（shrink）」。
- 說明如何透過改變時間戳映射中的迴圈順序來改變資料搬移剖析，並將其連結到前幾講中資料流比較的討論。

---

## 第一章 — 貫穿全講的範例：一維卷積（1-D Convolution）

> *投影片：L13-1 … L13-3*

本講將所有抽象概念紮根於一個具體的運算：**一維卷積**，這是 DNN 核心原語中最簡單的一種。

運算定義為：

```
O[q] += I[q + s] * F[s]
```

維度為：W（輸入寬度）、S（濾波器大小）、Q = W − ⌈S/2⌉（輸出寬度）。以 Python 迴圈表示：

```python
for q in [0, Q):
    for s in [0, S):
        o[q] += i[q + s] * f[s]
```

本講來源將這個 traversal 標記為**輸出駐留（Output Stationary, OS）**：內層迴圈遍歷 `s`，外層迴圈走 `q`，使得每個輸出在排程移往下一個 `q` 之前就累積完所有的部份和。這是後續所有分析的*參考映射*。

> **為什麼重要：** 一維卷積簡單到可以用手追蹤，卻包含了完整 DNN 核心的所有結構——輸入、權重、輸出、歸約維度（reduction dimension）以及資料的重複使用。這裡推導出的每一個公式都直接適用於二維卷積和其他 Einsum 形式。

---

## 第二章 — ISL 入門：集合、映射與迭代空間

> *投影片：L13-4 … L13-24*

在計算資料搬移之前，本講先用 ISL 建立代數詞彙——ISL 是一個針對整數集合進行多面體算術（polyhedral arithmetic）的函式庫。

### 集合（Sets）

**ISL 集合**是由「空間名稱（space name）」和「約束（constraints）」所定義的整數坐標點的集合。空間名稱至關重要：`SetXY[x, y]` 與 `SetWX[x, y]` 屬於不同的空間，即使它們的坐標範圍相同，它們也**不相等**（投影片 L13-4 … L13-7）。同樣地，坐標順序也很重要：`SetXY[x, y]` 與 `SetXY[y, x]` 是不同的集合（L13-8 … L13-9）。

### 迭代空間（Iteration Space）

一維卷積的**迭代空間**是 Q × S 的笛卡爾積——每一個對應到一次乘加運算的 `(q, s)` 對：

```python
ispace = isl.Set('{ IterationSpace[q, s] : 0 <= q < Q and 0 <= s < S }')
```

迭代空間純粹是一個組合物件，不帶有時間或記憶體位置的概念。這些概念由「投影映射」和「時間戳映射」引入。

### 投影映射（Projection Maps）

ISL **映射（map）** 是形如 `Domain[...] -> Range[...]` 的關係，由仿射等式（affine equalities）約束。三個投影映射將迭代空間中的點連接到資料元素：

- **權重投影**（`is2weight`）：`IterationSpace[q, s] -> Weight[s]` ——同一列 `s` 中的每個點都映射到同一個權重元素。

  這個多對一的關係正是**權重駐留性（weight stationarity）** 所利用的：權重 `F[s]` 在所有 `q` 位置上被*重複使用*。

- **輸出投影**（`is2output`）：`IterationSpace[q, s] -> Output[q]` ——同一列 `q` 中的每個點都映射到同一個輸出累加器。
- **輸入投影**（`is2input`）：`IterationSpace[q, s] -> Input[q + s]` ——較為複雜的映射，反映了卷積的滑動視窗（sliding-window）特性：每個輸入元素可能被多個 `(q, s)` 對共享。

映射可以通過 `apply_range` 合成，支援 Timestamp → IterationSpace → Data 這樣的鏈式連接。

> **為什麼重要：** 投影映射是「資料重用（data reuse）」的代數形式化。它們所揭示的多對一模式決定了哪個張量在哪種資料流下是駐留的——而量化這種重用正是計算記憶體流量的第一步。

---

## 第三章 — 加入時間：排程與時間戳

> *投影片：L13-25 … L13-34*

迭代空間沒有順序——它只是一個集合。**排程**（資料流）通過將每個迭代空間點映射到一個*時間戳*（邏輯時步）來施加全序。

### 時間戳映射

對於輸出駐留（output-stationary）的 traversal（外層迴圈 `q`，內層迴圈 `s`），時間戳為：

```python
df_is2ts = isl.Map('{ IterationSpace[q, s] -> Timestamp[t1, t0] : t1 = q and t0 = s }')
```

時間戳是一個雙分量的 tuple `(t1, t0)`，`t1` 是較慢（外層）的維度。其反映射給出在任意邏輯時刻所執行的迭代空間點：

```python
df_ts2is = df_is2ts.reverse()
```

### 合成映射以連接資料

通過 `apply_range` 合成映射，時間戳可以直接與任何資料元素相連接：

```python
df_ts2weight = df_ts2is.apply_range(is2weight)   # Timestamp -> Weight
df_ts2input  = df_ts2is.apply_range(is2input)    # Timestamp -> Input
df_ts2output = df_ts2is.apply_range(is2output)   # Timestamp -> Output
```

這些合成映射回答了：*「在時步 (t1, t0) 時，存取的是哪個權重／輸入／輸出？」*——這是計算連續步驟之間資料變化次數的前提。

> **為什麼重要：** 將排程（時間戳映射）與演算法（投影映射）分離，意味著你只需替換時間戳映射就能切換資料流——所有後續計算（差量、分塊資料搬移）都會自動更新。這正是該方法具有**組合性（compositionality）** 的原因。

---

## 第四章 — 計算資料搬移：差量計算

> *投影片：L13-35 … L13-43*

在將每個時步映射到資料元素之後，本講介紹**計算資料搬移**的核心步驟。

方案如下：

1. 建立從每個時間戳到其**前一個**時間戳的映射。
2. 確定當前時步所存取的資料元素。
3. 確定前一個時步所存取的資料元素。
4. 取**集合差**（當前減前一）——這些就是必須被讀取的*新*資料元素。
5. 對每個資料張量（輸入、權重、輸出）重複以上步驟。

### 前一時間戳映射

```python
timestamp_previous = isl.Map(f'{{ Timestamp[t1p, t0p] -> Timestamp[t1, t0] :
    t1p = t1 and t0p = t0 + 1
    or
    t1p = t1 + 1 and t0p = 0 and t0 = T0_MAX - 1 }}')
```

這捕捉了 `t0` 在達到最大值時的進位邏輯，此時 `t1` 遞增。

### 權重差量

**權重差量（weight delta）** 為：

```python
df_ts2weight_previous = timestamp_previous.apply_range(df_ts2weight_current)
df_ts2weight_delta = df_ts2weight_current.subtract(df_ts2weight_previous)
```

在輸出駐留 traversal 下，每個時步都使用一個新的權重（因為 `s` 是快速索引），產生 Q × S 個差量條目——每個權重在每次輸出計算中都被讀取一次。

### 輸入與輸出差量

- **輸入差量（input delta）**：每個時步也讀取一個新的輸入（滑動視窗的存取模式意味著 `s` 每前進一步就到達新的 `I[q + s]`）。
- **輸出差量（output delta）**：只有 Q 個條目——在每個 `q`-分塊的起始（`s = 0` 時）開啟一個新的輸出累加器。後續的 `s` 步驟則累積到同一個 `O[q]` 中。

這些差量代表了最細粒度的資料搬移——即在每個時鐘週期內必須跨越最底層暫存器檔邊界的資料量。

> **為什麼重要：** 差量計算將一個抽象的排程轉換為資料傳輸的具體計數。將所有時步的差量集合求和，就能得到總資料搬移量，可直接與硬體設計的記憶體頻寬預算相比較。

---

## 第五章 — L1 緩衝：分塊時間戳

> *投影片：L13-44 … L13-56*

在真實的加速器中，資料是在 **L1（片上）緩衝區**中暫存的，而非每次 MAC 傳輸一個元素。本講展示如何利用一個直接的時間戳分塊映射，將逐時步的分析提升到**分塊（tile）層級**。

### 分塊時間戳

通過投影出快速維度，將細粒度時間戳分組到 L1 分塊：

```python
timestamp2L1timestamp = isl.Map('{ Timestamp[t1, t0] -> L1Timestamp[t1] }')
L1timestamp2timestamp = timestamp2L1timestamp.reverse()  # 一對多
```

每個 L1 分塊 `L1Timestamp[t1]` 對應到共享相同 `t1` 的所有細粒度時步——即一次完整的內層迴圈對 `s` 的掃描。

### L1 差量計算

同樣的差量方案在分塊層級同樣適用：

```python
df_L1ts2weight_current = L1timestamp2timestamp.apply_range(df_ts2weight_current)
L1timestamp_previous = isl.Map("{ L1Timestamp[tp] -> L1Timestamp[t] : tp = t + 1 }")
df_L1ts2weight_previous = L1timestamp_previous.apply_range(df_L1ts2weight_current)
df_L1ts2weight_delta = df_L1ts2weight_current.subtract(df_L1ts2weight_previous)
```

在輸出駐留下，每個 L1 分塊都需要所有 S 個權重。分塊 0 是初始讀入（3 個權重），但後續分塊發現相同的權重已存在於緩衝區中，故差量為零。**權重在分塊間完全駐留（perfectly stationary）。**

對於**輸入**：分析揭示了一個**滑動視窗（sliding window）** 模式：

```
分塊 0 讀入：Input[0], Input[1], Input[2]   （3 個新的）
分塊 1 讀入：Input[3]                        （1 個新的）
分塊 2 讀入：Input[4]                        （1 個新的）
...
```

每個後續分塊只需要一個新的輸入元素，反映了一維卷積的單位步幅（unit stride）。這是一個經典結論：滑動視窗卷積的輸入重用模式正好是每個輸出位置一個新元素。

對於**輸出**：每個分塊恰好寫入一個新的輸出累加器（即當前 `q` 的那個），排除完整部份和後共產生 Q 個差量條目。

> **為什麼重要：** 分塊層級的差量正是硬體設計師實際用來決定 L1 緩衝區大小、以及估計 L1 緩衝區與全域緩衝區（或 DRAM）之間所需頻寬的工具。識別出輸入的滑動視窗模式，立即就能提示出適合採用**行緩衝（line buffer）** 微架構。

---

## 第六章 — 縮退計算與資料流比較

> *投影片：L13-57 … L13-66*

### 縮退：資料何時離開緩衝區？

「填充」差量（fill delta，即前面計算的差量）告訴我們資料何時進入緩衝區。本講還介紹了**縮退（shrink）計算**——資料何時可以*被驅逐*：

```python
L1timestamp_next = L1timestamp_previous.reverse()  # 分塊 t -> 分塊 t+1
df_L1ts2weight_shrink = df_L1ts2weight_current.subtract(df_L1ts2weight_next)
```

一個資料元素在分塊 `t` 縮退，當且僅當它在分塊 `t` 的緩衝區中存在，但在分塊 `t+1` 時不再被需要。在輸出駐留下，所有 S 個權重在每個分塊都被需要，所以縮退為零，直到最後一個分塊——所有三個權重只有在計算結束時才能被驅逐。對於輸入：滑動視窗每步前移，最舊的輸入 `I[q]` 在其最後一次使用後的下一個分塊就不再需要了，因此恰好在最後一次使用後一個分塊被驅逐。

「填充」與「縮退」共同確定了 L1 緩衝區在任何給定分塊必須持有的**最小活躍集（live set）**，從而決定最小所需緩衝區容量。

### 比較輸出駐留與權重駐留

本講最後將排程切換為**權重駐留（Weight Stationary, WS）**：

```python
ws_schedule = isl.Map('{ IterationSpace[q, s] -> Timestamp[t1, t0] : t1 = s and t0 = q }')
```

外層迴圈現在走 `s`（慢速），內層迴圈走 `q`（快速）。通過新的時間戳映射運行相同的差量流水線，產生了完全不同的資料搬移剖析：在 WS 下，權重是駐留的（分塊 0 之後差量為零），而輸入在每個 `s`-分塊都必須重新讀入，輸出則在所有分塊中累積部份和，產生 Q × (S − 1) 次部份和回讀事件。

這個結尾比較使整講的回報變得具體：**形式化的 ISL 框架讓你可以機械地計算並比較任意兩種資料流的資料搬移代價**，無需建構硬體——這正是驅動 TeAAL 關注點金字塔 Mapping 層（映射層）設計決策的那種分析。

> **為什麼重要：** 資料流選擇是加速器設計中槓桿最高的決策之一。ISL 方法將這個決策從直覺變成算術：你寫下時間戳映射，運行差量流水線，讀出記憶體流量數字。這是自動化設計空間探索工具的基礎。

---

## Worked Examples（worked examples）

### Example 1：列舉 OS schedule

令 \(Q = 3\)、\(S = 2\)。Iteration space 有六個 MACs：

\[
(q,s) \in \{(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)\}.
\]

對 output-stationary schedule，\(t_1=q\)、\(t_0=s\)。Timestamp order 因此是 \((0,0)\)、\((0,1)\)、\((1,0)\)、\((1,1)\)、\((2,0)\)、\((2,1)\)。Output projection 會把前兩個點映射到 \(O[0]\)，中間兩個映射到 \(O[1]\)，最後兩個映射到 \(O[2]\)。Output fill delta 只在 \((0,0)\)、\((1,0)\)、\((2,0)\) 非零，因為每個 output 的第二個 MAC reuse 同一個 accumulator。

硬體意義：OS 在最低層級降低 partial-sum traffic，因為 accumulator 在 reduction loop 期間保持 live。

### Example 2：L1 input sliding window

令 \(Q = 5\)、\(S = 3\)，也就是投影片的小卷積。OS tiling 下，一個 L1 tile 對應固定 \(q\) 的所有 \(s\) values。Tile 0 需要 inputs \(\{I[0], I[1], I[2]\}\)。Tile 1 需要 \(\{I[1], I[2], I[3]\}\)。Delta 只有 \(\{I[3]\}\)，因為 \(I[1]\) 與 \(I[2]\) 已經 live。之後同理：初始載入 \(S\) 個 inputs 後，stride 1 每個 tile 只需要一個新 input。

硬體意義：這正是 convolution line buffer 有效的原因。Symbolic delta calculation 重新推導出 sliding-window microarchitecture。

### Example 3：Fill 與 shrink 決定 capacity

對 \(Q = 5, S = 3\)，OS L1 input live set 在 warmup 後維持大小 3：\(\{I[q], I[q+1], I[q+2]\}\)。Fill 在 window 前進時加入 \(I[q+2]\)；shrink 在舊值最後一次使用後移除 \(I[q-1]\)。因此 maximum live input set 是 3 個元素，而不是整個 input tensor 的 \(Q+S-1 = 7\) 個元素。

硬體意義：bandwidth 與 capacity 是不同問題。Delta 計算新進資料；shrink 加上 carry-over 決定 live capacity。

---

## 重要公式與讀法（Key Equations）

- Iteration space：\(\mathcal{I} = \{(q,s) : 0 \le q < Q,\ 0 \le s < S\}\)。這是 MACs 的集合，不是 data elements 的集合。
- Projections：\(P_W(q,s)=s\)、\(P_O(q,s)=q\)、\(P_I(q,s)=q+s\)。這些 maps 定義 reuse：多個 iteration points 可能指向同一個 data element。
- Output-stationary schedule：\(T_{\mathrm{OS}}(q,s)=(q,s)\)。Output index 慢變；reduction index 快變。
- Weight-stationary schedule：\(T_{\mathrm{WS}}(q,s)=(s,q)\)。Weight index 慢變；output index 快變。
- Fill delta at time \(t\)：\(D_{\mathrm{fill}}(t)=A(t)-A(t-1)\)，其中 \(A(t)\) 是 time \(t\) 存取的 data elements set。在 ISL 中，這會實作成 current access 與 previous access 的 set/map difference。
- Shrink at time \(t\)：\(D_{\mathrm{shrink}}(t)=A(t)-A(t+1)\)。Fill 問「什麼要進來」；shrink 問「什麼可以離開」。

這些公式是投影片 ISL relations 的教學式記法。真的實作時 ISL syntax 很重要，但數學概念就是 relation composition 加 set difference。

---

## 硬體意涵（Hardware Implications）

- **Bandwidth：** Fill deltas 計算跨過某個 boundary 的新 values。把它們加總就是該 memory level 的 bandwidth demand。
- **Capacity：** 由 fill/shrink 推出的 live sets 決定 minimum buffer size。Capacity 可能小但 total traffic 大，也可能反過來。
- **Latency hiding：** 如果下一個 tile 的 fill set 能 symbolically 算出，compiler 或 DMA engine 就能 prefetch。
- **SRAM banking：** Schedule 決定 accesses 是 sequential、strided，還是 repeated。這影響 bank conflicts 與 port requirements。
- **NoC traffic：** Projection maps 揭示 multicast opportunities。如果很多 iterations 共用同一個 weight 或 input，interconnect 可以 broadcast，而不是發出獨立 reads。
- **Correctness：** Shrink 不只是 optimization，也是 correctness 問題。太早 evict value 會改變 computation，或造成昂貴 reload。
- **Design-space exploration：** 只改 timestamp map，就能比較 OS、WS 與其他 schedules，而保持同一個 mathematical layer。

---

## 常見誤解（Common Misconceptions）

### 誤解：Iteration space 已經告訴你 data movement。

Iteration space 只告訴你有多少 MACs。Data movement 取決於 projection maps、schedule order 與 buffer tiling。兩個 schedules 可以有相同 MAC count，但 traffic 完全不同。

### 誤解：Stationary dataflow 表示該 tensor 完全不動。

Stationary 表示 mapping 嘗試在某個 level 保留該 tensor。它仍可能需要 initial fill、final eviction、跨 tile reloads，或在其他 hierarchy levels 移動。

### 誤解：Delta 等於 total access count。

Delta 計算的是相對於前一個 time / tile 的**新資料**。一個 value live 期間可能被存取很多次，但不一定跨過被建模的 boundary 重新讀入。

### 誤解：Shrink 只是可有可無的 bookkeeping。

沒有 shrink，你只能算 fills，不能 sizing buffer。Shrink 告訴你 value 何時死亡，因此 storage 何時能被 reuse。

---

## Paper Bridge：TETRIS

### Bibliographic identity

- Title: *TETRIS: Scalable and Efficient Neural Network Acceleration with 3D Memory*
- Authors: Mingyu Gao, Jing Pu, Xuan Yang, Mark Horowitz, Christos Kozyrakis
- Year / venue: ASPLOS 2017
- Used in this lecture: 作為 analytical scheduling 與 data-motion accounting 為何重要的 system-level example。

### Problem addressed

TETRIS 處理的是 NN accelerator 擴大 PE arrays 與 network sizes 時出現的 memory-system bottleneck。Compute 增加沒有用，除非 buffers、DRAM bandwidth 與 interconnect traffic 能有效餵飽它。

### Core idea

這篇 paper 使用 3D memory 重新平衡 compute 與 buffers 的面積配置，把部分 accumulation 移到 memory 附近，並以 analytical method 推導 dataflow schedules，而不是只靠 exhaustive search。它與 L13 的連結是：scheduling 與 data movement 必須被建模成可計算的量，而不只是 architecture intuition。

### Relevance to this lecture

L13 教的是這類 analytical data-motion reasoning 的機械步驟。TETRIS 顯示這些步驟在完整 accelerator 中為什麼重要：memory hierarchy、bypassing、accumulation location 與 partitioning 都取決於知道哪些 data 在哪裡移動。

### Key claims used in this chapter

- Abstract 與 Section 1 說明 scaling NN accelerators 會加劇 on-chip SRAM 與 off-chip DRAM overhead，使 memory system 成為 bottleneck。
- Sections 3.2 與 3.3 說明 3D memory 改變 PE/buffer balance，並可用 in-memory accumulation 降低 memory traffic。
- Section 4 發展 software scheduling 與 partitioning techniques，包含 analytical scheduling choices，而不只是 exhaustive search。
- Section 6 報告在 paper 假設下，TETRIS 相對 conventional 2D DRAM accelerator baselines 改善 performance 與 energy。

### What students should remember

- Data-motion accounting 不是課堂練習；它決定 area allocation、buffer bypass 與 partitioning。
- 不同 memory technology 會改變 cost model，因此 optimal schedule 可能改變。
- Analytical scheduling 很有價值，因為所有 mapping / partitioning choices 的 exhaustive search 很快就變昂貴。

### Limitations and assumptions

TETRIS 研究的是 3D-memory accelerator design space，不是 L13 投影片中完全相同的 ISL recipe。這裡應把它當作 analytical scheduling 與 memory traffic 的 system-level bridge，而不是「所有 accelerator 都應使用 3D memory」的證據。

### Suggested insertion points

讀完第五章的 L1 buffering recipe 與第六章 OS-vs-WS comparison 後，再讀這篇 bridge，可以看到手算方法如何走向 automated mapping tools。

---

## 獨立學習指南（Standalone Study Guide）

### 進入下一講前必須掌握

- 將迴圈巢表示成具有具名 dimensions 與 constraints 的 ISL iteration space。
- 建立 projection maps，從 compute iterations 對應到它們碰到的 data elements。
- 用 timestamp map 將 traversal order 轉成 schedule。
- 把 fill、read、update、shrink events 視為 set/map operations 來計算。
- 在保持 computation 不變的情況下，透過改變 timestamp map 比較兩種 dataflows。

### 自我檢核問題

1. Iteration-space point 與 data-space point 有什麼差異？
2. 問「某個時間碰到哪些資料」時，為什麼 inverse timestamp map 有用？
3. 比較 OS 與 WS schedules 時，改變的是 Einsum、projection maps，還是 timestamp map？

### 練習

1. 為一個小型 1-D convolution 寫出 iteration-space constraints，並列舉所有 valid points。
2. 建立 input projection map `w = q + s`，並套用到每個 iteration-space point。
3. 對一個很小的 tiled timestamp range，手動列出哪些 weights 與 inputs 必須 fill 到 L1，哪些可以 retained。

### 常見誤區

- 把 ISL 當成只是語法。核心概念是對 iteration、time、data spaces 做 set algebra。
- 混淆 data movement counts 與 MAC counts。它們透過 projection maps 相關，但不是同一件事。
- 比較 mappings 時不小心改變 computation。Computation 應保持固定；改變的是 schedule。

---

## 關鍵詞彙（Key Terms）

| 詞彙 | 說明 |
|---|---|
| **ISL（Integer Set Library，整數集合函式庫）** | 一個用於整數集合和關係的多面體算術的 C/Python 函式庫；本講的計算基礎。 |
| **迭代空間（Iteration Space）** | 對應到計算中每個 MAC 運算的所有「迴圈變數」元組的集合。 |
| **投影映射（Projection Map）** | 從迭代空間點到它們所存取的資料元素（輸入、權重或輸出）的 ISL 映射。 |
| **時間戳映射（Timestamp Map）** | 為每個迭代空間點賦予字典序時間坐標的 ISL 映射，編碼了排程／資料流。 |
| **輸出駐留（Output Stationary, OS）** | 外層迴圈遍歷輸出位置 `q`、內層迴圈遍歷歸約位置 `s` 的資料流；輸出在移往下一個位置之前先完成本地累積。 |
| **權重駐留（Weight Stationary, WS）** | 外層迴圈遍歷濾波器位置 `s` 的資料流；每個權重固定不動，同時所有 `q` 個輸出各累積一次部份和。 |
| **差量／填充（Delta / Fill）** | 在時步 `t` 存取但在時步 `t-1` 未存取的資料元素集合；計算讀入緩衝區的新資料量。 |
| **縮退（Shrink / Eviction）** | 在分塊 `t` 中存在但在分塊 `t+1` 不再需要的資料元素集合；標識資料何時可以離開緩衝區。 |
| **L1 時間戳／分塊（L1 Timestamp / Tile）** | 通過將細粒度時間戳分組到分塊而獲得的較粗粒度時間，對應到一次內層迴圈掃描。 |
| **滑動視窗（Sliding Window）** | 卷積的輸入存取模式：輸出位置每推進一格就引入一個新的輸入元素，在任意時刻的總活躍集大小為 Q + S − 1。 |
| **活躍集（Live Set）** | 必須同時駐留於某個緩衝區的資料元素集合；由填充和前一分塊的遺留資料共同決定。 |
| **多對一重用（Many-to-one Reuse）** | 投影映射的一個屬性，即多個迭代空間點映射到同一個資料元素——資料駐留性（data stationarity）的形式化基礎。 |

---

## 重點回顧（Takeaways）

- 每個 DNN 映射都可以分解為「迭代空間 + 投影映射 + 時間戳映射」，而這個分解完全決定了資料搬移。
- **差量計算**——當前存取集合與前一集合的集合差——在任何記憶體階層上都能機械地計算新資料讀取次數，無需模擬。
- 在 **L1 分塊層級**，同樣的差量方案揭示了結構：在輸出駐留下，權重完全駐留（讀入一次，永不重載），而輸入遵循**滑動視窗**模式（初始讀入後每個分塊僅一個新元素）。
- **縮退計算**是填充的對偶，兩者合起來決定給定映射所需的最小緩衝區容量。
- 從輸出駐留切換到權重駐留的時間戳映射，立即產生不同的（且定量精確的）資料搬移剖析——**資料流比較從直覺變成了算術**。
- 本講提供了 **TeAAL** 等自動化映射工具的底層數學引擎：給定硬體描述和映射，工具就能以封閉形式計算資料流量。

---

## 與後續講次的連結（Connections）

這是 6.5930/1 的最後一講。它完成了 L01 所開啟的整條弧線：

- **L01** 確立了「資料搬移主導能耗」（DRAM ≈ ALU 的 200 倍）並引入 **TeAAL 關注點金字塔**。
- **L02–L04** 將 DNN 運算形式化為 Einsum，並為真實網路中的迭代空間和資料張量建立目錄。
- **L05–L06** 定性地引入了資料流（輸出駐留、權重駐留、列駐留）並展示它們如何影響資料重用。
- **L07–L10** 將框架延伸到稀疏資料（sparse data），此時迭代空間本身變得不規則。
- **L11–L12** 涵蓋進階實作技術（降低精度、記憶體內運算），改變了每個記憶體層級的能耗代價。
- **L13（本講）** 閉合了迴路：它提供了精確計算在任何映射、任何記憶體階層，對任何以 Einsum 表示的運算所產生的資料搬移量的*定量*方法。ISL 框架直接使能了 TeAAL 和 Accelergy 等工具內部執行的自動化分析，也是本課程中所有設計空間探索和映射最佳化最終所立足的嚴謹基礎。

---

## 附錄 — 投影片對照表（Slide-to-Section Map）

| 投影片 | 章節 |
|---|---|
| L13-1 | 標題 |
| L13-2 … L13-3 | 第一章 — 一維卷積貫穿範例、輸出駐留 traversal |
| L13-4 … L13-12 | 第二章 — ISL 集合：空間名稱、坐標順序、約束 |
| L13-13 … L13-16 | 第二章 — 定義迭代空間 |
| L13-17 … L13-18 | 第二章 — ISL 映射、定義域／值域、有界映射 |
| L13-19 … L13-24 | 第二章 — 投影映射：權重、輸出、輸入、運算 |
| L13-25 … L13-29 | 第三章 — 時間戳映射、traversal 視覺化、反映射 |
| L13-30 | 第三章 — 用 apply_range 合成映射 |
| L13-31 … L13-34 | 第三章 — 時間戳到資料的合成、探測排程 |
| L13-35 … L13-43 | 第四章 — 差量計算（填充）——權重、輸入、輸出 |
| L13-44 … L13-56 | 第五章 — L1 緩衝：分塊時間戳、L1 填充差量 |
| L13-57 … L13-64 | 第六章 — 縮退計算——權重、輸入、輸出 |
| L13-65 | 第六章 — 比較 OS vs. WS：切換時間戳映射 |
| L13-66 | 結語 |

## 來源註記（Source Notes）

- 本章順序、1-D convolution example、ISL set/map syntax、timestamp construction、delta recipe、L1 tiling recipe、shrink recipe 與 OS-vs-WS comparison 依據 `Lecture/L13-Calculating_Motion.pdf`。
- OS loop nest 與 projections 主要來自 L13-2 與 L13-19 到 L13-24。
- Delta / fill 解釋依據 L13-35 到 L13-43；L1 tile-level explanation 依據 L13-44 到 L13-56；shrink 依據 L13-57 到 L13-64。
- TETRIS paper bridge 依據 Gao et al., *TETRIS: Scalable and Efficient Neural Network Acceleration with 3D Memory*, ASPLOS 2017，尤其 Abstract、Sections 1、3、4、6。
- OS enumeration、sliding-window live-set、capacity examples 是本 walkthrough 根據投影片 1-D convolution setup 建立的 teaching examples。

## 不確定性註記（Uncertainty Notes）

- Local file `papers/L13_Buffets_Pellauer_ASPLOS2019.pdf` 抽出的內容看起來是 Douglas-Rachford / ADMM optimization paper，而不是檔名所指的 Buffets paper。因此本章沒有把它當作來源引用。
- 本章以教學方式呈現 ISL operations。實際 implementation details 可能依 ISL Python binding 版本與 local helper functions 而不同。
- Live lecture 可能強調一些投影片 PDF 中看不到的 tool implementation details；本 walkthrough 是根據投影片與可用 local papers 重建 likely narration。
