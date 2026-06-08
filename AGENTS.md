# AGENTS.md

## Project Mission

This repository is a self-contained, bilingual, textbook-style learning companion for MIT 6.5930/1 Hardware Architectures for Deep Learning.

The target reader is a motivated self-learner who has access to the public lecture slides and this repository, but does not have access to lecture videos.

Therefore, each chapter must reconstruct the missing lecture narration and explain the concepts deeply enough for independent study.

This project is not:

* a slide summary,
* a compressed translation,
* a transcript,
* a collection of bullet-point notes,
* or a paper dump.

It is a self-contained pedagogical companion.

The most important rule:

> A good chapter should explain what the lecturer would have said between the slides.

If a section would not help a student who has no lecture video, it is not good enough.

---

## Core Principle

Slides provide the syllabus and ordering backbone.

Papers provide technical depth, historical context, original claims, rigorous definitions, and examples.

The chapter must provide the missing teaching layer:

* motivation,
* intuition,
* definitions,
* worked examples,
* reasoning steps,
* hardware implications,
* common misconceptions,
* and cross-lecture connections.

Do not merely describe what is on the slides. Explain why the topic exists, what problem it solves, and how a hardware architect should think about it.

---

## Workflow Discipline

For non-trivial tasks, follow this workflow:

```text
inspect → diagnose → propose plan → edit → self-review → report uncertainty
```

Do not directly modify files before understanding the current state.

Before editing a lecture chapter:

1. Inspect the lecture slides.
2. Inspect the existing English chapter.
3. Inspect the existing Traditional Chinese chapter.
4. Inspect relevant papers if provided.
5. Diagnose what is missing.
6. Propose a concrete revision plan.
7. Only then edit files.
8. After editing, self-review against this AGENTS.md.
9. Report changed files, sources used, remaining uncertainties, and human-review items.

Avoid vague edits such as:

* “make this better,”
* “add more details,”
* “summarize this lecture,”
* “translate this into Chinese.”

Prefer concrete actions such as:

* identify missing prerequisites,
* explain missing reasoning steps,
* add a worked example,
* define the term precisely,
* explain hardware implications,
* add common misconceptions,
* connect the concept to previous and later lectures,
* cite slide pages or paper sections,
* rewrite the Chinese section with the same conceptual depth.

---

## Source Discipline

Always distinguish among four types of content:

1. **Directly stated in lecture slides**
2. **Derived from assigned or relevant papers**
3. **Standard background explanation**
4. **Authorial teaching interpretation**

Do not blur these categories.

Any number, energy ratio, speedup, benchmark result, architecture comparison, historical claim, or paper-specific claim must cite a source.

Acceptable source anchors include:

* lecture number and slide page,
* paper section,
* paper page,
* paper figure,
* paper table,
* official course page,
* official project documentation,
* textbook section.

Do not invent quantitative claims.

If a claim is plausible but not directly verified, mark it as teaching interpretation or uncertainty.

Use wording such as:

```text
Teaching interpretation:
...

Source note:
This explanation is based on Lecture XX slides YY-ZZ and PaperName Section A.B.

Uncertainty:
This chapter explains the concept in this way for pedagogy, but the exact emphasis may differ from the live lecture.
```

---

## Copyright Discipline

This repository should be safe as a public self-study companion.

Do not redistribute original lecture PDFs unless explicit permission exists.

Do not extract and commit slide figures unless explicit permission exists.

Do not copy paper figures verbatim.

Prefer:

* references to official slide page numbers,
* official course links,
* original diagrams redrawn from scratch,
* ASCII diagrams,
* Mermaid diagrams,
* textual descriptions.

If a diagram is inspired by a slide or paper figure, label it clearly:

```text
Redrawn based on Lecture XX slide YY.
```

or:

```text
Redrawn based on PaperName Figure Z.
```

If copyright-sensitive files already exist in the repository, do not delete them without explicit instruction. Instead, report:

* risky file path,
* risk level,
* reason,
* recommended action.

Recommended actions may include:

* remove,
* replace with official link,
* redraw,
* cite more clearly,
* or keep with explanation.

---

## Pedagogical Style

Write patiently and concretely.

For every major concept, include:

1. **Intuition**
   What is the mental model?

2. **Precise meaning**
   What does the term technically mean?

3. **Motivation**
   Why does this concept exist?

4. **Small example**
   Give a concrete example, preferably with numbers or a tiny tensor/matrix.

5. **Hardware implication**
   How does it affect energy, bandwidth, latency, area, utilization, programmability, or correctness?

6. **Common misconception**
   What does a beginner often misunderstand?

7. **Connection**
   How does this concept connect to earlier and later lectures?

Avoid shallow statements such as:

```text
Data movement is expensive.
```

Instead explain:

* which data moves,
* between which memory levels,
* why that movement costs energy,
* how mapping/dataflow changes that movement,
* and what tradeoff appears.

Example of preferred style:

```text
A DNN accelerator is not merely a large array of MAC units. The MAC units perform arithmetic, but the energy and performance of the system are often dominated by how weights, activations, and partial sums move through the memory hierarchy. A good dataflow tries to keep frequently reused data close to the processing elements so that one expensive memory access can support many arithmetic operations.
```

---

## Chapter Requirements

Each lecture chapter should contain the following sections unless there is a strong reason not to:

1. TL;DR
2. What problem this lecture solves
3. Why this lecture matters
4. Prerequisites and mental model
5. Learning objectives
6. Main textbook-style narrative
7. Worked examples
8. Key equations and how to read them
9. Hardware implications
10. Common misconceptions
11. Connections to previous and later lectures
12. Paper bridge
13. Study guide
14. Self-check questions
15. Exercises
16. Glossary
17. Slide-to-section map
18. Source notes and uncertainty notes

Not every section needs to be equally long, but every chapter should be self-contained enough for a reader without lecture videos.

---

## No-Video Requirement

Assume the reader cannot watch the lecture.

Do not write:

```text
As discussed in lecture...
```

```text
The slide shows...
```

```text
This is intuitive...
```

Instead, reconstruct the missing explanation.

For every transition between topics, explain why the lecture moves from one idea to the next.

Each chapter should answer:

* What problem are we trying to solve?
* Why is the naive solution insufficient?
* What abstraction is introduced?
* What does the abstraction hide?
* What tradeoffs does it expose?
* How would a hardware architect use this idea?

Good chapters should include “between-slide narration”: the explanation that would normally be spoken by the lecturer but is not fully present in the slides.

---

## English and Traditional Chinese Policy

English is the canonical version.

Traditional Chinese is not a compressed summary.

The Traditional Chinese version must preserve the same conceptual substance as the English version.

The Traditional Chinese version may be rewritten pedagogically rather than translated sentence-by-sentence.

Important English technical terms should be kept in parentheses.

Example:

```text
資料流（dataflow）不是單純指資料移動方向，而是指 weight、activation、partial sum 如何在 PE array 與記憶體階層中停留、流動與重用。
```

Do not rely only on translated jargon such as:

* 映射
* 資料流
* 重用
* 稀疏性
* 量化
* 階層式記憶體
* 利用率
* 資料搬移
* 部分和

Always explain the hardware meaning behind the term.

For the Traditional Chinese version:

* If the English version has an example, Chinese must include the same example.
* If the English version has a misconception, Chinese must include the same misconception.
* If the English version has a paper bridge, Chinese must include the same paper bridge.
* If the English version explains a term deeply, Chinese must also explain it deeply.
* Do not reduce the Chinese version into a short summary.

A good Chinese chapter should help a reader learn the topic, not merely check the meaning of English terms.

---

## Paper Usage Policy

Do not summarize entire papers unless explicitly asked.

Use papers as bridges that support lecture concepts.

For each paper used in a chapter, explain:

1. What problem the paper addresses.
2. What core idea, abstraction, architecture, or model it contributes.
3. Which lecture concept it supports.
4. What result or insight matters for this chapter.
5. What limitation or assumption the reader should know.

Avoid paper dumping.

A paper bridge should answer:

```text
Why does this paper matter for this lecture?
What concept does it clarify?
What should the student remember from it?
```

When using a paper, prefer compact pedagogical integration over long paper summaries.

---

## Paper Bridge Template

When asked to create or revise a paper bridge, use this structure:

```markdown
## Paper Bridge: [Paper Title]

### Bibliographic identity

- Title:
- Authors:
- Year / venue:
- Used in lecture(s):

### Problem addressed

Explain what problem the paper tries to solve.

### Core idea

Explain the main abstraction, architecture, model, or method.

### Relevance to this lecture

Explain which lecture concepts this paper supports.

### Key claims used in this chapter

List only claims that should appear in the chapter.

Each claim should include a source anchor such as section, page, figure, or table.

### What students should remember

Give 3-5 takeaways.

### Limitations and assumptions

Explain what the paper does not solve or what assumptions it makes.

### Suggested insertion points

Explain where this paper should be referenced in the lecture chapter.
```

---

## Worked Example Discipline

Whenever possible, include small examples.

Prefer tiny, concrete examples over large realistic ones.

Examples may include:

* a 2×2 matrix multiplication,
* a tiny convolution,
* a small loop nest,
* a few processing elements,
* a small memory hierarchy,
* a tiny sparse vector,
* a small quantization example,
* a toy dataflow mapping.

A worked example should show reasoning, not only final results.

For example, when explaining data reuse:

```text
Suppose one weight is used by 16 activations. If the accelerator reads this weight from SRAM every time, it performs 16 SRAM reads. If the dataflow keeps the weight in a local register near the PE, the accelerator may read it once and reuse it 16 times. The arithmetic is unchanged, but the memory traffic is reduced.
```

For equations:

* explain what each symbol means,
* explain what the equation measures,
* show a small numerical example,
* explain the hardware meaning of the result.

---

## Common Misconception Discipline

Each chapter should include useful misconceptions.

Good misconceptions are not trivial. They should target realistic beginner errors.

Examples:

```markdown
### Misconception: A DNN accelerator is just a large MAC array.

A MAC array is important, but accelerator performance and energy are often dominated by memory hierarchy, data movement, mapping, utilization, and programmability.
```

```markdown
### Misconception: Dataflow only means the direction data moves.

In DNN accelerators, dataflow usually refers to a broader scheduling and storage policy: which data stays stationary, which data moves, where partial sums accumulate, and how reuse is exploited.
```

```markdown
### Misconception: Sparsity always saves energy.

Sparsity can reduce arithmetic and memory traffic, but it also introduces metadata, indexing, irregular access, and load-balancing overhead.
```

---

## Hardware Implication Discipline

Every major idea should eventually connect back to hardware.

Important hardware implications include:

* energy,
* bandwidth,
* latency,
* area,
* utilization,
* memory capacity,
* memory hierarchy,
* interconnect traffic,
* programmability,
* scheduling complexity,
* mapping search space,
* correctness,
* scalability.

Avoid leaving concepts purely mathematical.

For example, after explaining convolution lowering or tensor algebra, also explain how the representation affects:

* PE array mapping,
* memory access patterns,
* reuse opportunities,
* buffer sizing,
* and data movement.

---

## Cross-Lecture Connection Discipline

Each chapter should explain how it connects to previous and later lectures.

Use phrasing such as:

```text
This lecture builds on Lecture XX by...
```

```text
This idea will reappear in Lecture YY when...
```

```text
The concept introduced here becomes important later because...
```

Cross-lecture connections are especially important for:

* data movement,
* memory hierarchy,
* mapping,
* dataflow,
* sparsity,
* precision,
* tensor algebra,
* cost modeling,
* accelerator design space,
* programmability.

Update the cross-lecture index when major concepts are added or renamed.

---

## Glossary Discipline

Glossary entries should not be mere translations.

Each glossary entry should include:

* English term,
* Traditional Chinese term if applicable,
* concise definition,
* intuition,
* why it matters in hardware,
* common confusion if relevant.

Example:

```markdown
### Dataflow（資料流／資料配置策略）

In DNN accelerator design, dataflow describes how weights, activations, and partial sums are scheduled, stored, moved, and reused across the PE array and memory hierarchy.

It is not merely the geometric direction of movement. It is a combined compute, storage, and communication policy.
```

---

## Self-Check and Exercise Discipline

Self-check questions should be answerable from the chapter.

They should test understanding, not memorization only.

Good self-check questions ask:

* why a design choice matters,
* what tradeoff appears,
* what would happen under a different mapping,
* how a small example behaves,
* why a misconception is wrong.

Exercises should vary in difficulty:

1. Conceptual
2. Small calculation
3. Design tradeoff
4. Paper-reading bridge
5. Open-ended architecture reasoning

Each chapter should include at least a few questions that force the reader to reason about hardware cost.

---

## Slide-to-Section Map Discipline

Each chapter should preserve slide-to-section traceability.

The slide-to-section map should help the reader map the original course slides to the rewritten textbook chapter.

It should include:

* lecture number,
* slide range,
* corresponding chapter section,
* notes if the chapter reorders or expands material.

Example:

```markdown
| Slide range | Chapter section | Notes |
|---|---|---|
| 1-5 | Motivation | Rewritten as problem setup and learning objectives |
| 6-12 | Data movement cost | Expanded with memory hierarchy explanation |
| 13-20 | Dataflow examples | Expanded with worked examples |
```

If a section adds background not directly present in slides, label it as background or teaching interpretation.

---

## Source Notes and Uncertainty Notes

Every chapter should end with source notes and uncertainty notes.

Example:

```markdown
## Source Notes

- The lecture ordering follows Lecture XX slides.
- The explanation of row-stationary dataflow is based on [paper/source].
- The worked example is original and created for pedagogy.

## Uncertainty Notes

- The live lecture may have emphasized some examples differently.
- This chapter reconstructs the likely lecture narration from slides and papers.
```

Use uncertainty notes honestly. Do not pretend that reconstructed lecture narration is official.

---

## Validation Checklist

Before finishing any chapter, verify:

* The chapter can be read without lecture video.
* The chapter is not merely a slide summary.
* Every major term is defined.
* Every important equation has an explanation.
* Important equations have worked examples when useful.
* Every major hardware concept has intuition.
* Every important claim is sourced or clearly marked as interpretation.
* Quantitative claims have citations.
* Paper-derived claims are attributed.
* The Traditional Chinese version is not substantially thinner than the English version.
* The Chinese version preserves examples, misconceptions, and paper bridges.
* The slide-to-section map is updated.
* Glossary entries are consistent.
* Self-check questions are answerable from the chapter.
* Exercises are aligned with the chapter.
* Copyright-sensitive figures are not copied directly.
* Remaining uncertainties are reported.

---

# Standard Codex Prompts

The following prompts are reusable. Use them when working on this repository.

---

## Prompt 1: Self-Containedness Audit

Use this before editing a lecture.

```text
Do not edit files yet.

Please inspect Lecture [XX] and produce a self-containedness audit.

Inputs:
- Slides: [path]
- English chapter: [path]
- Traditional Chinese chapter: [path]
- Relevant papers: [paths]
- Project instructions: AGENTS.md

Goal:
A motivated reader should be able to learn this lecture without watching the lecture video.

Evaluate:

1. Missing prerequisites
   - What background does the chapter assume but not explain?

2. Missing lecture narration
   - Where do the slides likely require oral explanation?
   - Which transitions between ideas are too abrupt?

3. Shallow concepts
   - Which terms are named but not deeply explained?
   - Which terms need intuition, examples, or hardware implications?

4. Missing worked examples
   - Which equations, mappings, dataflows, or architectural ideas need concrete examples?

5. Citation gaps
   - Which claims need slide or paper citations?
   - Which claims look potentially unsupported?

6. Chinese-version gaps
   - Where is the Chinese version thinner than the English version?
   - Where does it rely on translated jargon without explanation?

7. Copyright issues
   - Are there copied figures or slide-derived assets that should be removed, redrawn, or cited differently?

8. Proposed revision plan
   - List concrete sections to add, remove, or expand.

Do not make changes yet.
```

---

## Prompt 2: Implement Lecture Revision

Use this after the audit.

```text
Now implement the revision plan for Lecture [XX] according to AGENTS.md.

Goal:
The reader has no access to lecture videos. They should be able to learn this lecture independently from this chapter, the public slides, and the cited papers.

Editing requirements:

1. English chapter
   - Rewrite as the canonical self-contained chapter.
   - Expand explanations that currently assume lecture video.
   - Add missing intuition, definitions, worked examples, and hardware implications.
   - Add paper bridges where relevant.
   - Keep slide-to-section traceability.

2. Traditional Chinese chapter
   - Rewrite after the English version.
   - Do not summarize.
   - Preserve all major explanations, examples, misconceptions, and paper bridges.
   - Keep important English technical terms in parentheses.
   - Explain translated technical terms instead of relying on jargon.

3. Source discipline
   - Mark slide-derived claims.
   - Mark paper-derived claims.
   - Mark teaching interpretations.
   - Do not invent numerical claims.

4. Copyright discipline
   - Do not copy slide or paper figures directly.
   - If a figure is necessary, write a text description for an original diagram or use Mermaid/ASCII.

5. End-of-file quality
   - Update glossary.
   - Update self-check questions.
   - Update exercises.
   - Update slide-to-section map.
   - Add source notes and uncertainty notes.

After editing:
- Run available validation scripts.
- Report changed files.
- Report sources used.
- Report remaining uncertainties.
- Report places needing human review.
```

---

## Prompt 3: Strict Technical Review

Use this after a chapter has been revised.

```text
Act as a strict technical reviewer and course instructor.

Do not edit files yet.

Review Lecture [XX] after the latest revision.

Project goal:
The chapter must be self-contained for a motivated reader without access to lecture videos.

Evaluate:

1. Self-containedness
   - Can a reader understand the lecture without video?
   - What still requires outside explanation?

2. Technical depth
   - Are any concepts still too shallow?
   - Are any causal links missing?

3. Worked examples
   - Are there enough concrete examples?
   - Are equations explained with examples where needed?

4. Hardware intuition
   - Does the chapter explain implications for energy, bandwidth, latency, utilization, area, programmability, or correctness?

5. Source discipline
   - Are quantitative claims cited?
   - Are paper-derived claims clearly attributed?
   - Are any claims likely hallucinated?

6. Chinese parity
   - Is the Traditional Chinese version conceptually as complete as the English version?
   - Does it preserve examples, misconceptions, and paper bridges?
   - Does it explain terms rather than only translating them?

7. Copyright safety
   - Are any copied figures or slide assets present?
   - Should any diagrams be redrawn or replaced with descriptions?

8. Pedagogy
   - Are common misconceptions useful?
   - Are self-check questions answerable?
   - Are exercises aligned with the chapter?

Output:
- Critical issues
- Medium-priority issues
- Minor issues
- Concrete patch plan
- Pass/fail recommendation

Do not edit files unless explicitly asked.
```

---

## Prompt 4: Paper Bridge

Use this when adding a paper PDF.

```text
Create a paper bridge for [Paper Title] in the context of Lecture [XX].

Inputs:
- Paper PDF: [path]
- Lecture slides: [path]
- Current chapter: [path]
- Project instructions: AGENTS.md

Do not summarize the entire paper.

Instead, extract only what is pedagogically necessary for this lecture.

Output:

1. Bibliographic identity
   - Title
   - Authors
   - Year / venue if available

2. Problem addressed
   - What problem does the paper solve?

3. Core idea
   - What abstraction, architecture, model, or method does it introduce?

4. Relevance to Lecture [XX]
   - Which lecture sections need this paper?
   - Which concepts does it clarify?

5. Key claims used by this chapter
   - List only claims that should appear in the chapter.
   - Include page, section, figure, or table references.
   - Mark uncertain claims.

6. Concepts to explain
   - What terms from the paper need explanation for students?

7. Worked example opportunity
   - Is there a small example that can be created from the paper's idea?

8. Limitations
   - What assumptions or limitations should students know?

9. Suggested insertion points
   - Where should this paper be referenced in the lecture chapter?

Do not edit the chapter yet.
```

---

## Prompt 5: Chinese Parity Review

Use this when the Chinese version feels too thin.

```text
Review the Traditional Chinese version of Lecture [XX] against the English version.

Goal:
The Chinese version must be a pedagogical rewrite with the same conceptual substance, not a compressed summary.

Check:

1. Missing concepts
   - Which English concepts are missing in Chinese?

2. Missing examples
   - Which worked examples appear in English but not Chinese?

3. Missing misconceptions
   - Which common misconceptions are missing or simplified?

4. Missing paper bridges
   - Which paper-related explanations are thinner in Chinese?

5. Jargon problems
   - Which Chinese terms are translated but not explained?
   - Identify terms such as 映射, 資料流, 重用, 稀疏性, 量化, 階層式記憶體 if they lack explanation.

6. Tone
   - Does the Chinese version sound like a compressed translation?
   - Where should it be rewritten more naturally?

Then patch the Chinese version so that:
- It preserves the English version's conceptual depth.
- Important English technical terms remain in parentheses.
- Every major concept has intuition, definition, example, hardware implication, and common misconception.
```

---

## Prompt 6: Copyright Audit

Use this before making the repository more public.

```text
Audit the repository for copyright-sensitive course materials.

Goal:
Make this repository safer as a public self-study companion.

Check for:

1. Original lecture PDFs committed into the repo.
2. Extracted slide images.
3. Verbatim copied slide text beyond short references.
4. Paper figures copied directly.
5. Missing attribution.
6. Missing license/disclaimer.
7. Places where original diagrams should be redrawn.

Produce:
- A list of risky files.
- A risk level for each file: high / medium / low.
- Recommended action:
  - remove,
  - replace with official link,
  - redraw,
  - cite more clearly,
  - or keep.
- A proposed SOURCE_POLICY.md.
- A proposed COPYRIGHT_POLICY.md.
- A proposed README disclaimer.

Do not delete files yet.
```

---

## Prompt 7: Gold-Standard Chapter

Use this before scaling to all lectures. Prefer a central lecture such as Mapping/Dataflow.

```text
We will create Lecture [XX] as the gold-standard chapter for the whole repository.

Do not touch other lectures.

Goal:
This chapter should define the target quality bar for all future chapters.

Process:
1. Audit current L[XX].
2. Read relevant slides and papers.
3. Propose an ideal chapter outline.
4. Rewrite English version.
5. Rewrite Traditional Chinese version.
6. Add paper bridge.
7. Add worked examples.
8. Add glossary.
9. Add self-check and exercises.
10. Add slide-to-section map.
11. Add source notes.
12. Self-review against AGENTS.md.
13. Update AUTHORING_GUIDE.md with lessons learned from this chapter.

Quality bar:
A reader should be able to understand the lecture without video.
The chapter should feel like a compact textbook chapter, not a slide walkthrough.
```

---

## Prompt 8: Authoring Guide from Gold Standard

Use this after the gold-standard chapter is good.

```text
Based on the completed gold-standard Lecture [XX], create or update AUTHORING_GUIDE.md.

The guide should explain how future chapters should be written.

Include:

1. Target reader
2. Desired depth
3. Chapter structure
4. How to use slides
5. How to use papers
6. How to write English canonical chapters
7. How to write Traditional Chinese pedagogical rewrites
8. How to handle technical terms
9. How to create worked examples
10. How to write common misconceptions
11. How to cite sources
12. How to avoid copyright issues
13. Review checklist
14. Examples of good and bad prose

Make the guide concrete enough that Codex can follow it in later tasks.
```

---

## Minimal One-Shot Prompt

Use this when you want a shorter command.

```text
Revise [Lecture XX] according to AGENTS.md.

Do not merely summarize the slides.

The chapter must be self-contained for a reader without lecture video.

Before editing:
1. Inspect the slides, current EN/ZH chapters, and relevant papers.
2. Diagnose missing prerequisites, missing reasoning steps, shallow concepts, missing examples, citation gaps, Chinese parity problems, and copyright risks.
3. Propose a revision plan.

Then edit:
1. Rewrite English as the canonical textbook-style chapter.
2. Rewrite Traditional Chinese as a full pedagogical rewrite, not a summary.
3. Add intuition, definitions, worked examples, hardware implications, common misconceptions, paper bridges, glossary entries, study questions, exercises, slide-to-section map, and source notes.
4. Cite slide pages or paper sections for all important claims.
5. Do not copy figures directly.

After editing:
1. Run validation checks.
2. Self-review against AGENTS.md.
3. Report changed files, sources used, uncertainties, and human-review items.
```

---

## Suggested Repository Files

If asked to improve the repository structure, consider adding:

```text
AGENTS.md
AUTHORING_GUIDE.md
SOURCE_POLICY.md
COPYRIGHT_POLICY.md
PAPER_INDEX.md
TERM_GLOSSARY_EN_ZH.md
CHAPTER_REVIEW_CHECKLIST.md
docs/learning_paths/
docs/paper_bridges/
```

Do not create all files automatically unless explicitly asked. Prefer creating `AGENTS.md` first, then one gold-standard chapter, then derive the rest.

---

## Recommended Development Strategy

Do not rewrite all lectures at once.

Preferred strategy:

1. Pick one central lecture as the gold-standard chapter.
2. Recommended candidate: Mapping/Dataflow.
3. Expand that chapter until it is truly self-contained.
4. Use it to refine `AGENTS.md` and create `AUTHORING_GUIDE.md`.
5. Apply the same standard to other lectures one by one.
6. Run strict technical review after every chapter.
7. Run Chinese parity review after every Chinese chapter.
8. Run copyright audit before public release.

The goal is consistency and trustworthiness, not merely longer chapters.

---

## Quality Bar

A chapter passes only if:

1. A motivated reader can understand it without lecture video.
2. It explains the missing reasoning between slides.
3. It gives intuition before formalism.
4. It defines all major terms.
5. It includes concrete examples.
6. It connects concepts to hardware cost.
7. It distinguishes slides, papers, background, and interpretation.
8. It does not invent unsupported claims.
9. The Chinese version is not a compressed summary.
10. Copyright-sensitive materials are handled carefully.

If these conditions are not met, the chapter needs revision.

