# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A bilingual (English + Traditional Chinese) textbook-style study companion for **MIT 6.5930/1 — Hardware Architectures for Deep Learning** (Emer & Sze). There are no public lecture videos, so each lecture is reconstructed as a self-contained chapter that explains the narration *between* the slides. This is a **content repository**, not a software project — the "source" is Markdown prose and the "build artifacts" are rendered slide figures.

`AGENTS.md` is the authoritative pedagogical contract (chapter structure, source/citation discipline, copyright rules, EN/ZH parity policy, reusable authoring prompts). **Read it before writing or revising any chapter** — the rules below are operational; AGENTS.md is the why and how.

## Commands

```bash
# Validate everything CI checks (run before every commit)
scripts/check_walkthroughs.sh

# Lint shell scripts (CI also runs this)
shellcheck scripts/*.sh

# Read a lecture's slide text to plan a chapter
pdftotext "Lecture/L0N-….pdf" -

# Render specific slide pages to PNG figures (needs poppler-utils)
scripts/extract_slides.sh "Lecture/L0N-….pdf" L0N <page:slug> [<page:slug> ...]
# e.g. scripts/extract_slides.sh "Lecture/L01-Intro_and_Applications.pdf" L01 2:ai-ingredients 28:teaal-pyramid
# DPI defaults to 150; override with DPI=200 scripts/extract_slides.sh …
```

There is no compiler, package manager, or test framework. `check_walkthroughs.sh` is the closest thing to a test suite, and it must pass for CI (`.github/workflows/ci.yml`) to go green.

## What `check_walkthroughs.sh` enforces

The validator (no PDF tooling, runs anywhere) fails on any of:

1. **EN/ZH filename parity** — every file in `walkthroughs/en/` must have an identically-named twin in `walkthroughs/zh/`.
2. **Image links resolve** — every embedded `![…](../../assets/…)` path must point to a real file.
3. **Required sections present** — each file must contain all template section markers, matched as substrings:
   - EN: `TL;DR`, `Learning Objectives`, `Standalone Study Guide`, `Key Terms`, `Takeaways`, `Connections`, `Appendix`
   - ZH: `TL;DR`, `學習目標`, `獨立學習指南`, `關鍵詞彙`, `重點回顧`, `連結`, `附錄`

If you rename a section heading, update `EN_SECTIONS`/`ZH_SECTIONS` in the script to match, or the check breaks.

## Structure & conventions

- `Lecture/` — source slide PDFs (read-only source of truth). Do **not** redistribute or alter.
- `walkthroughs/en/` and `walkthroughs/zh/` — one Markdown chapter per lecture. **English is canonical; Chinese is a full pedagogical rewrite with the same conceptual depth, not a summary.**
- `assets/L<NN>/` — slide figures (PNG), shared by both language versions.
- `docs/CROSS_LECTURE_INDEX.md` — concept threads across L01–L13; update when major concepts are added/renamed.
- `docs/SLIDE_FIDELITY_AUDIT.md` — slide-coverage/traceability notes.
- `scripts/` — `check_walkthroughs.sh` (validation) and `extract_slides.sh` (figure rendering).

Naming (enforced by parity + traceability):
- Walkthrough: `L<NN>-<Kebab-Title>.md`, **same stem** in `en/` and `zh/`.
- Figure: `L<NN>-p<page>-<kebab-slug>.png` — the slide page number is part of the name for traceability.
- Figures are referenced from both language files via the relative path `../../assets/L<NN>/…`.

## Workflow for adding or revising a lecture

Follow AGENTS.md's `inspect → diagnose → propose plan → edit → self-review → report` discipline. In practice:

1. `pdftotext` the deck; pick the conceptual chapters.
2. Extract figure-bearing slides with `extract_slides.sh`.
3. Write/revise `walkthroughs/en/L0N-….md` (canonical), then the `zh/` twin with matching examples, misconceptions, and paper bridges.
4. Cite a slide range (e.g. `Slides: L0N-26 … L0N-35`) or paper section for every non-background claim — never invent quantitative numbers.
5. Update the index table in `README.md` and `docs/CROSS_LECTURE_INDEX.md` if concepts changed.
6. Run `scripts/check_walkthroughs.sh` locally before committing.

## Copyright

Treat slide PDFs and paper figures as copyright-sensitive. Reference slide page numbers and redraw diagrams (ASCII/Mermaid/original) rather than copying. Do not delete existing copyright-sensitive files without explicit instruction — report risk and recommended action instead (see AGENTS.md "Copyright Discipline").
