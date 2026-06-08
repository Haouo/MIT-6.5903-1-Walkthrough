# MIT 6.5930/1 — Hardware Architectures for Deep Learning · Bilingual Lecture Walkthroughs

In-depth, **concept-organized walkthroughs** of the MIT 6.5930/1 lecture slides
(Joel Emer & Vivienne Sze, Spring 2026), written in **English** and **Traditional Chinese (繁體中文)**.

There are no recorded videos for this course online, so these walkthroughs reconstruct the teaching
narrative from the slide decks: each lecture is rewritten as a short textbook-style chapter set —
explaining *what* each idea is, *why* it matters, and *how* the pieces connect — with the key slide
figures extracted and embedded inline.

> 本專案是 MIT 6.5930/1「深度學習硬體架構」課程投影片的**概念式導讀**，提供**英文**與**繁體中文**兩種完整版本。由於網路上沒有課程錄影，這些導讀以投影片為依據重建講課脈絡，並擷取關鍵投影片圖片內嵌於文中。

---

## How to read

- Each lecture has two standalone files — pick your language:
  - English → [`walkthroughs/en/`](walkthroughs/en/)
  - 繁體中文 → [`walkthroughs/zh/`](walkthroughs/zh/)
- Both language versions share the **same structure and the same figures**, so you can cross-reference.
- Every section cites its source slide range (e.g. `Slides: L01-26 … L01-35`) so you can follow along
  with the original PDF in [`Lecture/`](Lecture/).

Each walkthrough follows a fixed 8-part template: **TL;DR → Learning Objectives → conceptual chapters
(with embedded figures and a "Why it matters" note each) → Key Terms glossary → Takeaways →
Connections to later lectures → Slide-to-Section map.**

---

## Lecture Index

| # | Lecture | English | 繁體中文 | Status |
|---|---|---|---|---|
| L01 | Introduction and Applications | [EN](walkthroughs/en/L01-Intro-and-Applications.md) | [ZH](walkthroughs/zh/L01-Intro-and-Applications.md) | ✅ Done |
| L02 | Overview on DNN Components | [EN](walkthroughs/en/L02-Overview-on-DNN-Components.md) | [ZH](walkthroughs/zh/L02-Overview-on-DNN-Components.md) | ✅ Done |
| L03 | Memory, Metrics, Einsums & Transformers | [EN](walkthroughs/en/L03-Memory-Metrics-Einsums-Transformers.md) | [ZH](walkthroughs/zh/L03-Memory-Metrics-Einsums-Transformers.md) | ✅ Done |
| L04 | Einsums & Transformers | [EN](walkthroughs/en/L04-Einsums-and-Transformers.md) | [ZH](walkthroughs/zh/L04-Einsums-and-Transformers.md) | ✅ Done |
| L05 | Mapping — Dataflows | [EN](walkthroughs/en/L05-Mapping-Dataflows.md) | [ZH](walkthroughs/zh/L05-Mapping-Dataflows.md) | ✅ Done |
| L06 | Mapping — Partitioning | [EN](walkthroughs/en/L06-Mapping-Partitioning.md) | [ZH](walkthroughs/zh/L06-Mapping-Partitioning.md) | ✅ Done |
| L07 | Sparsity | [EN](walkthroughs/en/L07-Sparsity.md) | [ZH](walkthroughs/zh/L07-Sparsity.md) | ✅ Done |
| L08 | Sparse Architectures | [EN](walkthroughs/en/L08-Sparse-Architectures.md) | [ZH](walkthroughs/zh/L08-Sparse-Architectures.md) | ✅ Done |
| L09 | Sparse Architectures 2 | [EN](walkthroughs/en/L09-Sparse-Architectures-2.md) | [ZH](walkthroughs/zh/L09-Sparse-Architectures-2.md) | ✅ Done |
| L10 | Sparse Architectures 3 | [EN](walkthroughs/en/L10-Sparse-Architectures-3.md) | [ZH](walkthroughs/zh/L10-Sparse-Architectures-3.md) | ✅ Done |
| L11 | Advanced Technologies | [EN](walkthroughs/en/L11-Advanced-Technologies.md) | [ZH](walkthroughs/zh/L11-Advanced-Technologies.md) | ✅ Done |
| L12 | Precision | [EN](walkthroughs/en/L12-Precision.md) | [ZH](walkthroughs/zh/L12-Precision.md) | ✅ Done |
| L13 | Calculating (Data) Motion | [EN](walkthroughs/en/L13-Calculating-Motion.md) | [ZH](walkthroughs/zh/L13-Calculating-Motion.md) | ✅ Done |

> All 13 lectures are complete in both languages. L01 was the pilot that established the style; L02–L13 follow the same pipeline and 8-section template.
>
> **Note on L13:** the deck is titled "Calculating Data Motion" — a formal lecture on computing exact data-movement counts for a mapping using the Integer Set Library (ISL), with 1-D convolution as the running example. It is *not* about optical-flow/motion estimation.

---

## Repository Structure

```
MIT-6.5930-1/
├── README.md                      # this file — overview, index, conventions
├── Lecture/                       # source slide PDFs (read-only)
│   ├── L01-Intro_and_Applications.pdf
│   └── … L02 … L13
├── walkthroughs/
│   ├── en/                        # English walkthroughs
│   │   └── L01-Intro-and-Applications.md
│   └── zh/                        # 繁體中文 walkthroughs
│       └── L01-Intro-and-Applications.md
├── assets/
│   └── L01/                       # extracted slide figures (PNG), shared by EN + ZH
│       ├── L01-p02-ai-ingredients.png
│       ├── L01-p28-teaal-pyramid.png
│       └── …
└── scripts/
    └── extract_slides.sh          # pdftoppm wrapper — render slide pages to PNG
```

### Conventions

- **Walkthrough file:** `L<NN>-<Kebab-Title>.md`, with the **same stem** in `en/` and `zh/`.
- **Figure file:** `L<NN>-p<page>-<kebab-slug>.png` — the page number is kept for slide traceability.
- Figures live under `assets/L<NN>/` and are referenced from both language files via the relative
  path `../../assets/L<NN>/…`.

---

## Regenerating / Adding Figures

Slide figures are rendered from the source PDFs with [`scripts/extract_slides.sh`](scripts/extract_slides.sh),
a small wrapper around `pdftoppm` (from `poppler-utils`). It takes a PDF, a lecture id, and a list of
`page:slug` pairs:

```bash
# Render the key figures for L01 into assets/L01/
scripts/extract_slides.sh "Lecture/L01-Intro_and_Applications.pdf" L01 \
  2:ai-ingredients \
  9:compute-demand-growth \
  22:moore-dennard-slowdown \
  26:every-accelerator-unique \
  28:teaal-pyramid \
  29:fusemax-enhancements \
  43:accelerator-energy-cost \
  45:roofline-model \
  37:course-outline
```

Output defaults to 150 DPI; override with the `DPI` env var (e.g. `DPI=200 scripts/extract_slides.sh …`).

---

## Workflow for adding a lecture (L02–L13)

1. **Skim the deck:** `pdftotext "Lecture/L0N-….pdf" -` to read slide text and pick the conceptual chapters.
2. **Extract figures:** run `scripts/extract_slides.sh` with the figure-bearing slides.
3. **Write `walkthroughs/en/L0N-….md`** following the 8-part template.
4. **Write `walkthroughs/zh/L0N-….md`** — a faithful adaptation (natural academic Chinese; English
   technical terms in parentheses on first use), same structure and figures.
5. **Update the index table** above (mark EN/ZH done).
6. **Fidelity pass:** confirm every stat/claim in the prose traces to a real slide.

---

## Source & Credits

- Slides © Joel Emer, Vivienne Sze, and the 6.5930/1 course staff (MIT EECS). PDFs in `Lecture/` are the
  source of truth; figures in `assets/` are rendered directly from them.
- Textbook reference: V. Sze, Y.-H. Chen, T.-J. Yang, J. Emer, *Efficient Processing of Deep Neural
  Networks*.
- Course website: <http://csg.csail.mit.edu/6.5930/>

These walkthroughs are study notes derived from the public lecture slides for educational use.
