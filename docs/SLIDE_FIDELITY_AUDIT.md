# Slide Fidelity Audit

This audit records how completely the walkthroughs cover the source slide PDFs in
`Lecture/`. It applies to both English and Traditional Chinese walkthroughs because
the two language trees have matching structure, figures, and slide maps.

The goal is not to make the prose slide-by-slide. The goal is stronger: every slide
should be either explained in the conceptual narrative, summarized in an appendix map,
or explicitly treated as animation/logistics/background material.

---

## Audit Method

1. Treat the PDFs in `Lecture/` as the source of truth.
2. Treat each walkthrough's `Slide-to-Section Map` as the traceability contract.
3. Count physical PDF pages covered by that map.
4. Manually inspect non-obvious gaps:
   - animation-heavy ranges,
   - background/logistics pages,
   - decks with internal slide labels that differ from repository lecture IDs,
   - ranges whose appendix entry existed but whose prose was too thin.

Coverage status terms:

| Status | Meaning |
|---|---|
| Covered in prose | The main walkthrough explains the idea directly. |
| Synthesized | Multiple animation or build slides are collapsed into one conceptual explanation. |
| Background/logistics | The slide is not central technical content, but is named in the map. |
| Traceability note | A source-label mismatch or other source artifact is explicitly documented. |

---

## Coverage Summary

All 1,243 PDF pages across L01-L13 are mapped by the English slide-to-section maps.
The Chinese walkthroughs mirror the same lecture structure and slide mapping.

| Lecture | PDF pages | Mapped pages | Coverage | Notes |
|---|---:|---:|---:|---|
| L01 | 53 | 53 | 100% | Includes course logistics and outline pages in the map. |
| L02 | 102 | 102 | 100% | Covers methodology, CNN components, CONV, FC, and batch matmul. |
| L03 | 127 | 127 | 100% | Composite deck; map uses physical pages plus internal slide labels. |
| L04 | 198 | 198 | 100% | Animation-heavy CONV and attention build slides are synthesized. |
| L05 | 111 | 111 | 100% | Dataflow animations and LoopTree section are mapped. |
| L06 | 76 | 76 | 100% | Slides 23-76 are loop-nest visualization/animation frames. |
| L07 | 53 | 53 | 100% | Activation and weight sparsity arc fully mapped. |
| L08 | 96 | 96 | 100% | Tailors/Swiftiles evaluation range L08-86-L08-93 is now included. |
| L09 | 138 | 138 | 100% | Rank merge/split and sparsity-specification range has added prose. |
| L10 | 66 | 66 | 100% | PDF internal labels are `L06-*`; walkthrough includes traceability note. |
| L11 | 81 | 81 | 100% | Photonics and final summary are covered in the advanced-tech arc. |
| L12 | 76 | 76 | 100% | Bit-width determination slides L12-8-L12-9 are now included. |
| L13 | 66 | 66 | 100% | ISL set/map derivation and OS/WS comparison are mapped. |

---

## Non-Obvious Source Artifacts

### L03: Composite deck

`Lecture/L03-Memory+Metrics+Einsums+Transformers.pdf` combines multiple internal
slide-label sequences. Its walkthrough map uses physical pages for traceability and
names the internal sections:

- Memory and metrics pages.
- Efficient CNN model background.
- Extended Einsums.
- Kernel computation.
- Attention Einsums.

This is intentional: physical page numbers are the stable reference for this PDF.

### L06: Animation-heavy tail

L06 pages 23-76 are mapped as loop-nest visualization and execution-trace animation
frames. The walkthrough synthesizes these frames rather than reproducing each
intermediate build state.

### L08: Sparse tiling evaluation

The source deck's overbooking discussion extends through Tailors and Swiftiles
evaluation slides. The walkthrough prose already described the mechanism and results;
the slide map now explicitly covers L08-67-L08-93.

### L09: Sparse specification range

L09 pages 52-71 are a compact but important bridge from fibertree representation to
rank transformations and sparsity specifications. The walkthrough now includes a
dedicated explanation of:

- rank merging,
- coordinate-space vs. position-space splitting,
- channel-based sparsity,
- sub-kernel sparsity,
- fully unstructured `CRS` sparsity,
- 2:4 sparsity,
- hierarchical structured sparsity.

### L10: Internal slide-label mismatch

`Lecture/L10-Sparse_Architectures-3.pdf` is the repository's Lecture 10 deck, but
the PDF's internal labels read `L06-*`. The walkthrough keeps the PDF labels in
slide references for source matching and adds an explicit traceability note so readers
do not confuse it with `Lecture/L06-Mapping-Partitioning.pdf`.

### L12: Bit-width determination slides

L12 pages 8-9 contain the practical question of determining input, weight, and
partial-sum bit-widths. The walkthrough prose covers accumulator headroom via RSC,
and the slide map now includes L12-2-L12-9 as the first chapter range.

---

## Per-Lecture Fidelity Notes

| Lecture | Fidelity note |
|---|---|
| L01 | Conceptual narrative covers motivation, specialization, design pyramid, roofline, course structure, and logistics. |
| L02 | Technical coverage follows the deck from methodology through CNN/CONV/FC components; no intentionally skipped technical range. |
| L03 | Uses physical-page traceability because internal slide labels are mixed; efficient CNN pages are treated as background for workload context. |
| L04 | Dense animation frames are synthesized into stable explanations of rank patterns, convolution lowering, and attention. |
| L05 | OS/WS/IS dataflow animations are summarized by loop nests, data placement, and energy implications. |
| L06 | Early conceptual slides are explained in prose; later visualization frames are mapped as animation frames. |
| L07 | Sparsity sources, pruning pipeline, scoring, structured/unstructured trade-offs, and summary are covered directly. |
| L08 | SAFs, formats, structured sparsity, HSS, sparse tiling, overbooking, Tailors, Swiftiles, and summary are covered. |
| L09 | Fibertrees, traversal, rank specifications, sparse CONV loop nests, SCNN, and ISOSceles are covered. |
| L10 | Single-sparse and joint-sparse accelerator cases are covered; source-label mismatch documented. |
| L11 | CiM, crossbars, ADC bottleneck, RAELLA, substrate comparison, CiMLoop, photonics, and summary are covered. |
| L12 | Energy/area case, quantization, formats, accuracy, mixed precision, hardware support, binary/ternary nets, and summary are covered. |
| L13 | ISL sets/maps, timestamps, delta/fill/shrink calculations, L1 tiling, and OS/WS comparison are covered. |

---

## Maintenance Checklist

When a walkthrough is edited or a slide PDF changes:

1. Re-run `scripts/check_walkthroughs.sh`.
2. Confirm every embedded image still resolves.
3. Confirm every lecture still has a `Standalone Study Guide`.
4. Recompute slide-map coverage from the appendix.
5. Inspect any missing physical pages:
   - add them to the slide map if they contain content,
   - mark them as animation/background/logistics if they are not central prose,
   - add a traceability note if the PDF has unusual internal labels.
6. Update this audit when a non-obvious traceability decision changes.

This keeps the walkthroughs usable as self-guided material while preserving the PDFs
as the source of truth.
