# L07 - Co-Design of DNN Models and Hardware: Sparsity（DNN 模型與硬體共同設計：稀疏性）

> **課程：** 6.5930/1 - Hardware Architectures for Deep Learning
> **授課者：** Joel Emer & Vivienne Sze（MIT EECS）
> **日期：** 2026-02-25 · **投影片：** 53 頁 · **來源：** [`Lecture/L07 - Sparsity.pdf`](../../Lecture/L07%20-%20Sparsity.pdf)
>
> 本章根據投影片與 local papers 重建缺少的 lecture narration。它是自學教材，不是投影片逐頁摘要。

---

## TL;DR

Sparsity（稀疏性）表示 tensor 中有許多重複值，DNN 裡最常見的是重複的 zeros。對 DNN hardware 來說，zero 帶來兩種機會：不要儲存或搬動這個 zero；不要執行結果不會改變 output 的 multiply-add。L07 解釋 zeros 從哪裡來，也解釋為什麼 zeros 不會自動變成好處。Activation sparsity 可以自然來自 ReLU、input data 的相關性，或 graph structure。Weight sparsity 通常由 pruning（剪枝）創造；pruning 是一個 model-hardware co-design process，需要決定移除哪些 weights 或 groups、如何恢復 accuracy，以及要最佳化哪個硬體指標。

核心警告是：sparsity 是 opportunity，不是 guarantee。要跳過 zeros，需要 metadata、compression formats、zero detection、scheduling 與 load balancing。dense accelerator 可能仍然執行所有 ineffectual work；sparse accelerator 雖然能跳過工作，卻要付 overhead。最佳設計取決於 sparsity level、granularity、dataflow、memory hierarchy 與實際 deployment platform。

---

## 這堂課要解決什麼問題

前幾堂課已經說明，DNN accelerator efficiency 往往受 memory movement 與 PE utilization 支配，而不只是 MAC count。L07 問的是：如果 DNN 裡很多值是 zero，系統能不能避免搬動與計算它們？

天真的答案是「可以，因為 anything times zero is zero」。硬體答案更謹慎：系統必須夠早知道哪些值是 zero，才能避免 fetch、route 或 multiply。它也必須在移除工作後仍讓 PEs 忙碌。如果 nonzeros 分布不規則，一個 PE 可能拿到很多 nonzeros，另一個 PE 卻沒事做。sparse representation 還需要 indices、run lengths、masks 或其他 metadata；metadata 本身會消耗 storage、bandwidth 與 control logic。

因此，這堂課把 sparsity 視為 co-design。model 決定 zeros 從哪裡來；hardware 決定那些 zeros 是否真的變成 savings。

Source note：L07 slides 2-6 直接定義本堂課 goals、sparsity definition，以及 effectual/ineffectual operation framework。

---

## 為什麼這堂課重要

Sparsity 是降低 DNN inference cost 的主要槓桿之一，但它橫跨 accelerator stack 的多層。

- **Algorithm/model：**pruning 會改變 trained network，也可能改變 accuracy。
- **Format：**compressed representations 決定 nonzeros 與 metadata 如何儲存。
- **Mapping：**dataflow 決定 sparse values 在哪裡 reuse，以及 zero-skipping 發生在哪裡。
- **Architecture：**PEs 需要 gating、skipping、intersection、buffering 或 load-balancing support。
- **Evaluation：**正確 metric 可能是 energy、latency、storage、throughput、accuracy，或完整 tradeoff curve。

這堂課是 L08-L10 的前置。L07 說明 sparsity 從哪裡來、引入哪些設計決策；後續 lectures 才會說 sparse accelerator mechanisms。

---

## 先備知識與心智模型

你應該知道：

- ReLU：$\operatorname{ReLU}(x)=\max(0,x)$。
- Dense MAC：例如 $z\leftarrow z+a\cdot w$ 的 multiply-accumulate。
- Memory hierarchy：資料從越遠的 memory 搬動，通常 energy 越高。
- Dataflow and partitioning：mapping 決定 values 住在哪裡、何時 reuse。

心智模型：把 dense accelerator 想成一條會處理每個箱子的產線，即使箱子是空的也照樣搬、照樣檢查。Sparsity 問的是：空箱子能不能在消耗 conveyor bandwidth、storage、worker time 前被辨識出來？答案取決於 label system、routing system，以及剩下非空箱子分布是否均勻。

---

## 學習目標

讀完本章後，你應該能夠：

- 定義 sparsity、density、effectual operation、ineffectual operation。
- 區分 activation sparsity 與 weight sparsity。
- 解釋 ReLU 為什麼創造自然 activation zeros。
- 解釋 graph adjacency matrix 為什麼帶來 structural sparsity。
- 描述 pruning loop：scoring、grouping、ranking、fine-tuning、scheduling。
- 比較 magnitude-based、feature-based、energy-aware、platform-aware pruning objectives。
- 解釋 unstructured sparsity 與 structured sparsity 的硬體 tradeoffs。
- 解釋 MAC count 與 weight count 為什麼可能是 latency/energy 的差代理指標。
- 說明 L08-L10 必須加入什麼機制，才能把 sparse workloads 變成硬體 speedups。

---

## 主要教材式敘事

### 1. Sparsity 產生 effectual 與 ineffectual work

在這堂課中，sparsity 通常指 zeros。若一個 tensor 有 75% zeros，density 就是 25%，表示只有四分之一 entries 是 nonzero。

硬體機會來自兩個 identity：

$$
a\times0=0,\qquad a+0=a.
$$

如果一個 MAC 使用 zero activation 或 zero weight，multiply 可能產生 zero，add 也可能不改變 accumulator。投影片把有用的 operations 稱為 **effectual operations**，把 zero-related 的無效 operations 稱為 **ineffectual operations**：

$$
\text{total operations}=\text{effectual operations}+\text{ineffectual operations}.
$$

但硬體實際執行通常是：

$$
\text{actual operations}=\text{effectual operations}+\text{unexploited ineffectual operations}.
$$

ineffectual operations 與 unexploited ineffectual operations 之間的差距，就是 sparse accelerator design space。理想 sparse machine 會跳過所有 ineffectual work；實際 sparse machine 只能跳過一部分，並為此付出 overhead。

硬體意義：zero-skipping 只有在節省的 arithmetic 與 movement 大於 zero detection、metadata、data steering、load balancing 成本時才值得。

Source note：effectual/ineffectual operation framework 直接來自 L07 slides 5-8。

### 2. Compression 省的不只是容量，也是 movement

Compression 只儲存 useful values，加上足以重建位置的 metadata。對 sparse tensors，這可能是 coordinate lists、compressed sparse rows、run-length encoding、masks 或 structured patterns。

compression format 必須 uniquely decodable。對 DNN inference，通常也要 lossless，因為改變 tensor values 可能影響 accuracy，除非近似本身就是 model design 的一部分。

compression 的硬體意義不只是 capacity。如果 compressed activation tile 放得進 local buffer，但 dense tile 放不進去，accelerator 就可能避免 DRAM reads。storage win 會變成 data-movement win。

常見誤解：compression 本身保證 speedup。Compression 可以減少 bytes moved，但如果 compute engine 必須 decompress 成 dense stream 並照跑 dense MACs，operation count 不會下降。要 speedup，compute path 也必須 exploit sparse representation。

Source note：compression requirements 與 benefits 直接來自 L07 slide 11。

### 3. Activation sparsity：資料與 nonlinearities 自然產生 zeros

Activation sparsity 指 intermediate feature maps 裡的 zeros。L07 指出三個來源：ReLU、input correlations、sparse input structure。

**ReLU：**如果 pre-activation 是負的，ReLU 輸出 zero。對小矩陣：

$$
\begin{bmatrix}
9 & -1 & -3\\
1 & -5 & 5\\
-2 & 6 & -1
\end{bmatrix},
$$

ReLU 會產生：

$$
\begin{bmatrix}
9 & 0 & 0\\
1 & 0 & 5\\
0 & 6 & 0
\end{bmatrix}.
$$

九個值中有五個是 zero。在 convolution layer 中，如果硬體能跳過，單一 zero activation 可能消除許多 weight multiplications。

投影片報告 AlexNet convolutional output feature maps 經 ReLU 後，在顯示的 convolution layers 中約有 75% zeros。這是 L07 slide 10 的 slide-stated quantitative claim。

**Spatial correlation：**image feature map 中相鄰 activations 往往相似。系統可以處理 full values 的 differences/deltas；若鄰近值很接近，delta representation 可能 sparse。這依賴 application 與 representation。

**Temporal correlation：**連續 video frames 很相似。系統可以 reuse 前一 frame 的 computation 或處理 motion/delta information，但需要額外 storage 與找出 redundancy 的方法。這通常只適合相同 operation 在時間上反覆套用的情境。

**Graph structure：**graph neural networks 使用 adjacency matrices。真實 graphs 通常不是 complete graph，所以 adjacency matrices 通常 sparse。layer 可寫為：

$$
X^{(\ell+1)}=\sigma(\hat{A}X^{(\ell)}W^{(\ell)}),
$$

其中 $\hat{A}$ 是 normalized sparse adjacency matrix，$X^{(\ell)}$ 是 dense node-feature matrix，$W^{(\ell)}$ 是 dense weight matrix。multiplication order 很重要：$(\hat{A}X)W$ 與 $\hat{A}(XW)$ 在結合律下數學結果相同，但 intermediate density 與 memory traffic 可以不同。

硬體意義：activation sparsity 是 dynamic 的。zero pattern 會隨 input 改變。硬體需要 runtime detection，或使用攜帶 nonzero positions 的 format。

Source note：ReLU、Cnvlutin、SnaPEA、PredictiveNet/Song、Diffy、temporal-correlation examples 與 GNN setup 依據 L07 slides 10-23。本 worker pass 中，architecture result numbers 只使用 slide deck 作為來源。

### 4. Weight sparsity：由 model design 創造 zeros

Weight sparsity 指 learned parameters 中的 zeros。不同於 activation sparsity，weight sparsity 可以在 training 後固定，因此較容易 offline 壓縮 weights，並依此規劃 mappings。

投影片先指出，即使 pruning 之前，weights 也可能有 redundancy。若 filter 是 $[A,B,A]$、input 是 $[1,2,3]$，dense processing 會算：

$$
A\cdot1+B\cdot2+A\cdot3.
$$

因為 $A$ 出現兩次，可以改寫成：

$$
A(1+3)+B\cdot2.
$$

這把三次 multiplications 與三次 weight reads 降成兩次 multiplications 與兩次 weight reads，代價是多一次 addition，且進入 multiply 的值可能需要更寬 bitwidth。這個例子直接根據 L07 slide 26。

這還不是 pruning，而是利用 repeated weights。Pruning 進一步把選定 weights 或 groups 設成 zero。

### 5. Pruning pipeline

Pruning pipeline 可拆成五個概念。

**Scoring：**對 weight 或 group 指派分數。Magnitude-based pruning 使用 $|w|$。Feature-based pruning 估計對 output features 的影響。更進階方法可能使用 gradients、saliency、energy 或 measured latency。

**Grouping：**決定 granularity。可以移除 individual weights、rows、channels、filters、blocks 或 patterns。

**Ranking：**比較 scores 並選擇要移除的項目。

**Fine-tuning：**更新 surviving weights 以恢復 accuracy。

**Scheduling：**決定每次 iteration pruning 多激進。

簡化 loop 是：

```text
train dense model
repeat:
    score weights or groups
    remove the lowest-ranked weights or groups
    fine-tune the remaining weights
until target accuracy/energy/latency/storage budget is reached
```

早期經典方法 Optimal Brain Damage 使用 second-derivative saliency 估計刪除 weight 對 training error 的影響。投影片引用 LeCun, NeurIPS 1989 作為此歷史脈絡。

Source note：pipeline 依據 L07 slides 27-30 與 46-48。Blalock et al., "What is the State of Neural Network Pruning?" Sections 2.1-2.4 支撐 pruning masks、高階 prune/fine-tune algorithm、structure/scoring/scheduling/fine-tuning categories，以及 evaluation metrics。

### 6. Energy-aware 與 platform-aware pruning

Magnitude-based pruning 問：「哪些 weights 看起來對 accuracy 最不重要？」Energy-aware pruning 問的是另一件事：「對這個 hardware 與 dataflow，移除哪些 weights 最能降低 energy？」

這個差別重要，因為硬體上不是每個 weight 的成本都一樣。從 DRAM fetch 的 weight 遠比已在 register file 的 weight 昂貴。高 reuse layer 裡的 weight，也可能比低 reuse layer 裡的 weight 有不同 energy impact。

L07 slide 32 給出 Yang, CVPR 2017 的 memory hierarchy energy table：ALU/RF 作為 $1\times$、NoC $2\times$、global buffer $6\times$、DRAM $200\times$。這些比例應視為 course example 的 slide-stated values，不是 universal constants。

投影片報告 Energy-Aware Pruning 在 cited work 中讓 AlexNet energy 降低 $3.7\times$，並比 magnitude-based pruning 多 $1.7\times$ improvement。Source：L07 slide 35，citing Yang, CVPR 2017。

Platform-aware pruning 更進一步。NetAdapt 直接在 target platform 上量測 latency 或 energy，而不是只依賴 MAC count。本講來源用這個例子說明，最佳化 MACs 不一定最佳化 real latency。Source：L07 slides 37-41，citing Yang, ECCV 2018。

硬體意義：pruning objective 必須對齊 deployment objective。如果目標是 phone latency，就量 phone latency；如果目標是 accelerator energy，就用實際 mapping 與 memory hierarchy 去 model 或 measure accelerator energy。

### 7. Structured vs. unstructured sparsity

Grouping 是 model accuracy 與 hardware regularity 之間的橋。

**Unstructured sparsity：**移除 individual weights。這給 model 最大彈性，在同 sparsity level 下通常較能保 accuracy，但 zero pattern 不規則。硬體需要 indices、intersection 或 sparse scheduling。

**Structured sparsity：**移除 rows、channels、filters、blocks 或 fixed patterns。這對 dense SIMD、systolic、vector hardware 比較友善，因為剩下的 computation 形狀規則。代價是整組移除是較粗的 model change，accuracy 可能較早下降。

granularity spectrum 可這樣讀：

| Granularity | What disappears | Hardware effect | Model risk |
|---|---|---|---|
| Weight | individual scalar | irregularity 最高、彈性最大 | 較低 |
| Row/block/pattern | 小型 regular group | 有部分 vector regularity | 中等 |
| Channel | 完整 input channel | 移除重複 dense work | 較高 |
| Filter | 完整 output channel/filter | 縮小 dense layer shape | 較高 |

投影片引用 Scalpel 作為 matching SIMD organization 的 pruning，pattern-based pruning 作為 middle ground，mixture-of-experts 作為 dynamic coarse-grained sparsity。這些是 L07 slides 43-45 的 slide-stated examples。

常見誤解：90% unstructured sparsity 的 model 一定比 50% structured sparsity 的 model 快。前者可能 compression 更好，但 hardware utilization 更差；後者可能能映射到 dense kernels，因此更快。

### 8. Pruning 與 layer type、starting architecture 互動

Pruning 對不同 layers 的影響不一樣。以 AlexNet 為例，投影片報告 fully connected layers 的 weight reduction 大於 convolutional layers，且 MAC reduction 小於 weight reduction。Source：L07 slide 49，citing Han, NeurIPS 2015。

現代 efficient models 也更難 prune。MobileNet、EfficientNet 這類 model 原本就透過 architecture design 移除許多 redundancy，因此 pruning 可能更快造成 accuracy drop。投影片明確指出，modern efficient DNN models accuracy drops more quickly，且 unpruned efficient model 可以 outperform pruned inefficient model。Source：L07 slides 50-51，citing Hoefler JMLR 2021 and Blalock MLSys 2020。

硬體意義：pruning 不是 architecture design 的替代品，而是更廣泛 model-hardware co-design loop 中的一個工具。

### 9. 為什麼需要 specialized hardware

L07 結尾指出，practical DNN sparsity 常在 30-70% 量級，而 conventional sparse software libraries 多為更高 sparsity levels 設計。Source：L07 slide 52。

這解釋為什麼需要後續 lectures。在 30-70% sparsity 下，zeros 多到值得在意，但還沒有多到 generic sparse libraries 必然勝出。硬體必須用低 overhead exploit moderate sparsity。

Specialized sparse hardware 可能需要：

- compressed storage formats，
- fetch 前或 MAC 前 zero detection，
- 避免無用 switching 的 gating，
- metadata decode 與 address generation，
- sparse intersection logic，
- variable-rate nonzero streams 的 buffering，
- PEs 之間 load balancing，
- account for density 的 mapping strategies。

---

## Worked Examples

### Example 1：effectual fraction

假設 dot product 有 $8$ 個 weights 與 $8$ 個 activations。activation vector 在 positions 1、4、7 是 zero；weight vector 在 positions 2、7 是 zero。

只要任一 operand 是 zero，該 MAC 就是 ineffectual。受 zero 影響的位置是 $\{1,2,4,7\}$，所以 $8$ 個 MACs 中有 $4$ 個 ineffectual。effectual fraction 是 $4/8=50\%$。

硬體意義：dense PE 會跑完全部 $8$ 個 MACs。sparse PE 可能只跑 $4$ 個，但前提是能有效辨識與排程 nonzero pairs。

### Example 2：metadata 可能壓過 tiny values

假設 length $16$ 的 sparse vector 有四個 nonzeros。每個 value 是 8 bits，每個 index 是 4 bits，compressed representation 需要 $4(8+4)=48$ bits。dense representation 需要 $16\cdot8=128$ bits。compression 有幫助。

如果 vector 有十二個 nonzeros，compressed storage 是 $12(8+4)=144$ bits，反而比 dense storage 差。這就是 sparse format choice 取決於 density 與 metadata cost 的原因。

### Example 3：pruning objective 會改變答案

假設兩個 candidate weights 被 prune 後 accuracy loss 相同。$w_1$ 只從 local buffer fetch 一次；$w_2$ 因 mapping 缺少 reuse，會從 DRAM fetch 多次。若 $|w_1|\approx|w_2|$，magnitude pruning 可能把它們視為相近。energy-aware pruning 應優先移除 $w_2$，因為它節省更多 movement。

---

## Key Equations and How to Read Them

### Sparsity and density

$$
\text{sparsity}=\frac{\#\text{zeros}}{\#\text{total values}},\qquad
\text{density}=1-\text{sparsity}.
$$

sparsity 告訴你 opportunity size；density 告訴你還剩多少 useful work。硬體成本還取決於 metadata 與 load balance。

### Effectual operation accounting

$$
\text{actual operations}
=\text{effectual operations}+\text{unexploited ineffectual operations}.
$$

sparse hardware 的目標是降低第二項，同時不要讓每個剩餘 operation 的 overhead 太高。

### GNN layer

$$
X^{(\ell+1)}=\sigma(\hat{A}X^{(\ell)}W^{(\ell)}).
$$

$\hat{A}$ 是 sparse graph structure；$X^{(\ell)}$ 與 $W^{(\ell)}$ 通常 dense。mapping 必須決定早點還是晚點乘上 $\hat{A}$，因為 intermediate density 會影響 memory traffic。

### Pruned model mask

$$
f(x;M\odot W)
$$

$M$ 是 binary mask，$W$ 是 weight tensor。若 $M_i=0$，weight $W_i$ 被 pruned。此 notation 依據 Blalock et al., Section 2.1。

---

## Hardware Implications

**Energy：**跳過 zero 可以省 compute energy 與 data movement energy，但 metadata 與 control 也會耗能。

**Bandwidth：**compressed tensors 可以降低 bandwidth，但 irregular nonzero streams 可能造成 bursty 或難以 coalesce 的 accesses。

**Latency：**skipping work 只有在 scheduler 避免 bubbles 與 load imbalance 時才會降低 latency。

**Area：**sparse support 需要額外硬體：decoders、comparators、masks、queues、arbiters、intersection units。

**Utilization：**random sparsity 可能讓 PEs 閒置，除非 work 被動態平衡。

**Correctness：**pruning 可能改變 model outputs；activation skipping 必須保留 exact dense semantics，除非本來就設計為 approximate inference。

**Programmability：**sparse mappings 較難指定，因為 format、mapping、architecture 互相影響。

---

## Common Misconceptions

### 誤解：只要值是 zero，data movement 就會自動省下來。

只有當 zeros 沒被 fetch，或 compressed storage 避免搬動 zeros，movement 才會省下來。如果 zeros 已用 dense format 從 DRAM fetch 到 PE 才丟掉，DRAM bandwidth 已經花掉了。

### 誤解：sparsity 越高，execution 一定越快。

高 sparsity 可能表示較少工作，但也可能帶來更多 irregularity、metadata overhead 與 load imbalance。granularity 很重要。

### 誤解：weight count 是好的 energy metric。

不同 weights 可能因 layer shape、reuse、dataflow 而造成不同 memory traffic。Energy-aware pruning 的存在就是因為 weight count alone 不夠。

### 誤解：prune 一個 inefficient model 等同設計一個 efficient model。

Pruning 可以幫忙，但從更好的 architecture 開始可能更重要。L07 明確指出 unpruned efficient model 可以 outperform pruned inefficient model。

---

## Connections to Previous and Later Lectures

**L01-L03：**energy 與 memory hierarchy 解釋為什麼 skipping movement 可能比 skipping MACs 更重要。

**L05-L06：**pruning energy 取決於 mapping；同一個 zero 在不同 dataflows 與 partitions 下，可能節省不同 memory levels。

**L08-L10：**這些 lectures 會提供 exploit sparse weights 與 activations 的 accelerator mechanisms。

**L12：**precision 與 sparsity 是互補 compression axes：一個降低 bits per value，另一個降低 represented values 的數量。

**L13：**sparse workloads 的 data motion counting 必須計入 nonzeros、metadata，以及每個 memory level 的 movement。

---

## Paper Bridge: Computing's Energy Problem

### Bibliographic identity

- **Title:** "Computing's Energy Problem (and what we can do about it)"
- **Author:** Mark Horowitz
- **Year / venue:** ISSCC 2014
- **Used in lecture(s):** 支撐 L01/L03 的 memory-energy themes，也支撐 L07 避免 data movement 的動機。

### Problem addressed

這篇 paper 解釋為什麼 energy，而不只是 transistor count 或 peak operations，成為 computing 的核心限制。它強調 memory movement 可能主導 energy。

### Core idea

energy-efficient systems 需要 locality 與 specialization。DRAM accesses 遠比 internal cache accesses 或 functional operations 昂貴，因此 algorithms 與 hardware 應最大化 reuse 並避免不必要 movement。

### Relevance to this lecture

Sparsity 吸引人的原因之一，是 zeros 不必被搬動。Horowitz 提供 energy context，解釋為什麼避免 DRAM fetch zero 可能比避免單一 arithmetic operation 更重要。

### Key claims used in this chapter

- DRAM access energy 約為 1-2 nJ，而 internal cache access 或 functional operation 約為 10 pJ。Source：Horowitz ISSCC 2014，Section 5 "Don't Forget the Memory Energy."
- energy-efficient computation 需要強 locality 與 many operations per memory fetch。Source：Horowitz ISSCC 2014，Sections 5-6。

### What students should remember

- Sparse acceleration 不只是 arithmetic，也同樣關於 memory movement。
- 從 DRAM fetch 的 zero 已經花掉 energy。
- locality 與 format choices 決定 sparsity 是否成為真實硬體收益。

### Limitations and assumptions

數值 energy values 依 technology 與 system 而變。請把它們當成 scale intuition，不是 universal constants。

### Suggested insertion points

在 compression、energy-aware pruning，以及「太晚在 PE 才丟掉 zero 不如一開始就不要搬」的段落引用。

---

## Paper Bridge: What is the State of Neural Network Pruning?

### Bibliographic identity

- **Title:** "What is the State of Neural Network Pruning?"
- **Authors:** Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, John Guttag
- **Year / venue:** MLSys 2020
- **Used in lecture(s):** L07 pruning methodology 與 evaluation discipline。

### Problem addressed

這篇 paper 問：pruning literature 是否有可靠、可比較的證據說明哪些方法最好？它指出 dataset、architecture、metrics、baselines 不一致，使比較非常困難。

### Core idea

paper 把 pruning 表示為對 model 套用 binary mask，並從 structure、scoring、scheduling、fine-tuning、evaluation metrics 比較 pruning methods。它主張 standardized benchmarking，並提出 ShrinkBench。

### Relevance to this lecture

L07 教 pruning design space；Blalock et al. 說明為什麼這個 design space 必須嚴謹評估：parameter count、FLOPs、theoretical speedup、latency、accuracy 不能互相替換。

### Key claims used in this chapter

- pruned model 可表示為 $f(x;M\odot W')$，其中 $M$ 是 binary mask。Source：Blalock et al., Section 2.1。
- 許多 pruning strategies 遵循 train、score/prune、fine-tune、iterate。Source：Algorithm 1 and Section 2.2。
- methods 主要差異在 structure、scoring、scheduling、fine-tuning。Source：Section 2.3。
- evaluation goals 不同；parameter count 與 FLOPs 是 latency、throughput、memory usage、power 的 loose proxies。Source：Section 2.4。
- paper 指出 inconsistent metrics 與 benchmarking 是 pruning research 的重大問題。Source：abstract and Section 5.2。

### What students should remember

- Pruning 不是單一方法，而是一組 choices。
- 永遠要問 pruning result 最佳化的是什麼 metric。
- sparse model 應該以 accuracy-efficiency curve 評估，而不是 cherry-picked single point。
- hardware-facing pruning 應盡可能報告 direct metrics。

### Limitations and assumptions

這篇 paper 是 pruning survey 與 benchmarking study，不是 sparse accelerator architecture paper。它支撐 pruning methodology 與 evaluation discipline，不支撐特定硬體機制。

### Suggested insertion points

在 pruning pipeline、direct-metrics discussion，以及 efficient architectures vs. pruned inefficient models 的警告中引用。

---

## 獨立學習指南

### 如何讀這堂課

1. 先掌握 effectual vs. ineffectual operations。
2. 把 activation sparsity 與 weight sparsity 分開。
3. 對每個 sparsity source 問：zero pattern 是否在 runtime 前已知？
4. 對每個 pruning strategy 問：它最佳化哪個 metric？
5. 對每個 sparse representation 估計 metadata 與 load-balance cost。

### Self-check questions

1. sparsity 與 density 差在哪裡？
2. ReLU 為什麼創造 activation sparsity？
3. 為什麼在 PE 才 skip zero，比一開始就不要 fetch zero 省得少？
4. 為什麼只依 weight magnitude pruning 可能錯過 energy opportunities？
5. unstructured sparsity 與 structured sparsity 差在哪裡？
6. MAC count 與 latency 為什麼不能互換？
7. 為什麼 L08-L10 需要 specialized hardware mechanisms？

### Exercises

1. **Conceptual：**用兩元素 dot product 解釋 effectual 與 ineffectual operations。
2. **Small calculation：**一個 vector 有 64 entries、20 nonzeros、8-bit values、6-bit indices。比較 dense 與 coordinate-list storage。
3. **Design tradeoff：**選一個 layer，判斷 weight pruning 或 activation skipping 哪個比較容易 exploit，並說明 assumptions。
4. **Pruning pipeline：**替小型 MLP 設計 pruning loop，指定 scoring、grouping、ranking、fine-tuning、stopping criterion。
5. **Paper-reading bridge：**根據 Blalock et al.，解釋為什麼只報 parameter reduction 會誤導。
6. **Architecture reasoning：**設計一個 PE-level zero-skipping mechanism，列出一個 benefit 與一個 overhead。

---

## 關鍵詞彙

### Sparsity（稀疏性）

tensor values 中為 zero 或可被利用的 repeated values 的比例。本章通常指 zeros。

### Density（密度）

nonzero values 的比例。70% sparse tensor 有 30% density。

### Effectual operation（有效操作）

會改變 output 的 computation。對 MAC 來說，通常表示兩個 operands 都有意義。

### Ineffectual operation（無效操作）

涉及 zero 且不改變結果的 computation。sparse hardware 試圖跳過這些操作。

### Activation sparsity（activation 稀疏性）

intermediate activations 裡的 zeros。通常是 dynamic 且 input-dependent，尤其在 ReLU 後。

### Weight sparsity（weight 稀疏性）

model parameters 裡的 zeros。通常由 pruning 創造，且 inference 前通常已知。

### Compression format（壓縮格式）

儲存 nonzeros 與 metadata 的表示法。metadata 可能在 moderate density 下吃掉 savings。

### Pruning（剪枝）

移除 weights 或 groups of weights，通常透過 binary mask 並 fine-tune remaining model。

### Scoring（評分）

pruning 前對 weights 或 groups 指派重要性分數。

### Grouping（分組）

選擇 pruning granularity：individual weights、blocks、rows、channels、filters 或 patterns。

### Ranking（排序）

排序 scored weights/groups 以決定移除對象。

### Fine-tuning（微調）

pruning 後 retrain remaining weights 以恢復 accuracy。

### Scheduling（排程）

決定每次 iteration prune 多少 weights 或 groups。

### Unstructured sparsity（非結構化稀疏）

任意位置的 individual zeros。對 model 彈性高，對硬體困難。

### Structured sparsity（結構化稀疏）

zeros 以 regular groups 排列。對硬體較容易，對 accuracy 通常較限制。

### Energy-aware pruning（能量感知剪枝）

不只根據 weight magnitude 或 accuracy，也把移除項目的 energy impact 納入 scoring。

### Platform-aware pruning（平台感知剪枝）

使用 target platform 上的 latency 或 energy 等 direct measurements 來 pruning/adapt network。

### Metadata（中繼資料）

定位 nonzeros 所需的額外資訊，例如 indices、masks、run lengths。

---

## 重點回顧

- Sparsity 提供節省 storage、data movement、computation 的可能。
- Effectual/ineffectual accounting 區分數學機會與硬體實際跳過的工作。
- Activation sparsity 通常 dynamic；weight sparsity 通常可在 training 後固定。
- Pruning 是 scoring、grouping、ranking、fine-tuning、scheduling 的 pipeline。
- 正確 pruning metric 取決於 deployment objective。
- Unstructured sparsity 偏向 model flexibility；structured sparsity 偏向 hardware regularity。
- moderate DNN sparsity 需要 specialized hardware support，才會可靠轉成 speedup 或 energy savings。

---

## 連結

L07 透過 memory energy 連到 L01-L03，透過 mapping-dependent reuse 連到 L05-L06，透過 sparse accelerator mechanisms 連到 L08-L10，透過 precision/compression tradeoffs 連到 L12，並透過 explicit data-movement accounting 連到 L13。統一問題是：當一個 value 是 zero，stack 的哪一層能夠夠早知道，並避免成本？

---

## 附錄 - Slide-to-Section Map

| Slide range | Chapter section | Notes |
|---|---|---|
| L07-1 | Title and metadata | Lecture identity |
| L07-2-L07-8 | Sparsity 產生 effectual/ineffectual work | 擴充硬體 overhead 解釋 |
| L07-9-L07-17 | Activation sparsity | ReLU、compression、activation skipping、correlation |
| L07-18-L07-23 | Graph sparsity | GNN notation 與 operation-order discussion |
| L07-24-L07-26 | Weight redundancy | Gauss/UCNN-style repeated-weight intuition |
| L07-27-L07-31 | Pruning pipeline | Scoring 與 classic pruning |
| L07-32-L07-41 | Energy/platform-aware pruning | 擴充 direct-metric reasoning |
| L07-42-L07-45 | Structured vs. unstructured sparsity | 擴充 granularity tradeoff |
| L07-46-L07-48 | Ranking, fine-tuning, scheduling | 整合進 pruning pipeline |
| L07-49-L07-51 | Layer type 與 model architecture 互動 | 擴充 Blalock connection |
| L07-52-L07-53 | Summary and readings | 用於 takeaways 與 paper bridge |

---

## Source Notes

- 本章順序依據 `Lecture/L07 - Sparsity.pdf`。
- effectual/ineffectual operation definitions、sparsity sources、ReLU/AlexNet activation sparsity、activation-sparsity architecture examples、GNN setup、pruning pipeline、energy-aware pruning examples、NetAdapt examples、structured-pruning examples、summary sparsity range 依據 L07 slides 2-53。
- 約 75% AlexNet feature-map zeros、Cnvlutin speedup/area overhead、Minerva speed/power results、Energy-Aware Pruning energy improvements、NetAdapt latency improvements、Han pruning reductions、30-70% practical sparsity 等 quantitative claims 都是 L07 slide-stated claims；若要在本 companion 外重用，應回到原 papers 查證。
- memory-energy motivation 使用 `papers/L07_ComputingsEnergyProblem_Horowitz_ISSCC2014.pdf`，尤其 Section 5。
- pruning methodology 與 evaluation discipline 使用 `papers/L16_StateOfPruning_Blalock_MLSys2020.pdf`，尤其 Sections 2.1-2.4 與 5.2。
- worked examples 是原創教學例子。

## Uncertainty Notes

- slides 引用的多個 architecture examples，包括 Cnvlutin、Minerva、SnaPEA、Diffy、UCNN、EAP、NetAdapt、Scalpel、PCONV、PatDNN，本 worker pass 未以 local PDFs 獨立核對。
- 本章沒有刪除 `assets/L07` 既有 slide-derived assets；只是避免新增 copied figures。
- 精確 sparse-hardware mechanisms 刻意留到 L08-L10。
