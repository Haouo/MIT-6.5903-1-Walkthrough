# Cross-Lecture Index

This index turns the 13 standalone walkthroughs into a coherent self-guided course.
It applies to both `walkthroughs/en/` and `walkthroughs/zh/`, which share the same
lecture structure, figures, and slide-to-section mapping.

Use this file when you want to follow a concept across lectures instead of reading
the course strictly in chronological order.

---

## Course Spine

| Arc | Lectures | What the arc teaches |
|---|---:|---|
| Motivation and workload model | L01-L02 | Why DNN acceleration is data-movement dominated, and what DNN layers look like as tensor computations. |
| Formal compute language | L03-L04 | Memory metrics, Einsum notation, FC/CONV/attention as tensor programs. |
| Mapping | L05-L06 | How loop order, dataflow, partitioning, and parallelism determine reuse and communication. |
| Sparsity co-design | L07-L10 | Where zeros come from, how they are represented, and how hardware skips them. |
| Technology and numerical co-design | L11-L12 | How new substrates and reduced precision change energy, area, and accuracy trade-offs. |
| Exact data-movement accounting | L13 | How to compute data movement for a mapping using ISL sets and maps. |

---

## Concept Threads

### Data Movement

Start with L01's energy-cost argument, then follow the thread into increasingly
formal treatments.

| Step | Lecture | Role |
|---:|---|---|
| 1 | L01 | Establishes that moving data, especially from DRAM, can dominate arithmetic. |
| 2 | L03 | Defines memory hierarchy and metrics for comparing designs. |
| 3 | L05 | Shows how dataflow changes memory traffic while leaving the computation unchanged. |
| 4 | L06 | Adds partitioning so working sets fit in local storage and parallel machines can distribute work. |
| 5 | L08-L10 | Shows how sparse formats and skipping change both traffic and scheduling regularity. |
| 6 | L11 | Moves compute toward memory and exposes peripheral-circuit costs. |
| 7 | L13 | Converts the intuition into exact set/map calculations. |

Key terms to connect: memory hierarchy, reuse distance, stationarity, tile,
working set, arithmetic intensity, Roofline, delta, fill, shrink.

### Einsum and Tensor Ranks

Einsum is the course's common language. It starts as notation and becomes the
bridge between algorithms, mappings, formats, and exact analysis.

| Step | Lecture | Role |
|---:|---|---|
| 1 | L02 | Introduces tensors, ranks, FC, CONV, and batch dimensions. |
| 2 | L03 | Uses Einsums for memory/metrics examples and attention. |
| 3 | L04 | Deepens rank patterns, flattening, partitioning, FC/CONV lowering, and attention cascades. |
| 4 | L05 | Maps the same Einsum through different loop orders. |
| 5 | L06 | Splits ranks for temporal reuse and spatial parallelism. |
| 6 | L09-L10 | Extends the rank language to sparse fibers, projection, and intersection. |
| 7 | L13 | Turns ranks and loops into ISL spaces and maps. |

Key terms to connect: rank, free rank, reduction rank, flattening, partitioning,
projection, fiber, coordinate, position, iteration space.

### Mapping and Dataflow

Mapping is the control layer between the mathematical computation and the hardware.

| Step | Lecture | Role |
|---:|---|---|
| 1 | L05 | Defines output-stationary, weight-stationary, and input-stationary dataflows. |
| 2 | L06 | Adds temporal/spatial partitioning and distributed execution. |
| 3 | L08 | Shows that sparse formats must match traversal order. |
| 4 | L09-L10 | Shows how sparse dataflow choices determine projection, intersection, scatter, and utilization. |
| 5 | L13 | Compares mappings by changing timestamp maps while keeping the computation fixed. |

Key terms to connect: loop nest, dataflow, stationarity, LoopTree, temporal loop,
spatial-for, delayed reduction, concordant traversal, discordant traversal.

### Sparsity

The sparsity arc is intentionally multi-stage: model zeros, storage formats,
traversal, then real architectures.

| Step | Lecture | Role |
|---:|---|---|
| 1 | L07 | Explains activation sparsity, weight pruning, granularity, and accuracy trade-offs. |
| 2 | L08 | Introduces gating, skipping, compressed formats, structured sparsity, HSS, and sparse tiling. |
| 3 | L09 | Formalizes sparse tensors as fibertrees and introduces projection/intersection. |
| 4 | L10 | Studies concrete joint-sparse accelerators: Eyeriss, Cambricon-X, Cnvlutin, SCNN, ISOSceles. |
| 5 | L12 | Connects binary/ternary precision to induced sparsity and low-cost arithmetic. |

Key terms to connect: activation sparsity, pruning, gating, skipping, bitmask,
coordinate-payload, RLE, fibertree, fiber intersection, Cartesian product,
scatter network, rank swizzle.

### Precision

Precision is the second major co-design axis after sparsity.

| Step | Lecture | Role |
|---:|---|---|
| 1 | L01 | Frames precision as a data-attribute optimization in the design pyramid. |
| 2 | L03 | Introduces accuracy as a metric that can constrain efficiency claims. |
| 3 | L11 | Shows why analog and near-memory technologies must account for limited precision and conversion overhead. |
| 4 | L12 | Gives the full precision taxonomy, quantization methods, mixed precision, and scalable MACs. |
| 5 | L13 | Provides a method that can be reused to count movement for different operand widths. |

Key terms to connect: quantization, accumulator width, fixed-point, floating-point,
bfloat16, MX, log-domain quantization, codebook, QAT, PTQ, precision-scalable MAC.

---

## Lecture Dependency Map

| Lecture | Depends most on | Prepares you for |
|---|---|---|
| L01 | None | All later lectures; especially L03, L05, L07, L12. |
| L02 | L01 | L03-L06; gives the layer and tensor vocabulary. |
| L03 | L01-L02 | L04, L05, L13; establishes metrics and formal notation. |
| L04 | L02-L03 | L05-L06, L09-L10, L13; deepens rank manipulation. |
| L05 | L03-L04 | L06, L08-L10, L13; teaches dataflow reasoning. |
| L06 | L04-L05 | L09-L10 and distributed attention/matmul reasoning. |
| L07 | L01-L05 | L08-L10; introduces sparse model-side choices. |
| L08 | L07 plus L05 | L09-L10; introduces sparse formats and sparse tiling. |
| L09 | L04-L08 | L10; formal sparse tensor traversal and intersection. |
| L10 | L08-L09 | L11-L12 context; closes sparse architecture case studies. |
| L11 | L01, L03, L05 | L12; device/circuit constraints for co-design. |
| L12 | L01, L03, L11 | Later precision-aware architecture work. |
| L13 | L04-L06 | Independent mapping analysis and final-project modeling. |

---

## Suggested Self-Guided Paths

### First full pass

Read L01-L13 in order. Use each lecture's Standalone Study Guide as a stop point:
do not move on until the "What to master before moving on" bullets feel answerable.

### If your goal is mapping and data movement

Read L01, L03, L04, L05, L06, then L13. Add L08 only after you understand dense
mapping, because sparse traversal adds irregularity to the same mapping questions.

### If your goal is sparse accelerators

Read L01, L04, L05, then L07-L10. Return to L06 when a sparse design uses splitting,
position-space partitioning, or delayed reduction.

### If your goal is architecture/technology co-design

Read L01, L03, L05, L11, L12. Then use L07-L10 as examples of data-attribute
co-design and L13 as the method for quantifying movement.

### If your goal is Transformers

Read L02, L03, L04, L06, then revisit L05 for mapping and L12 for precision.
Use L06's attention partitioning section as the bridge from Einsum notation to
distributed implementation.

---

## Common Cross-Lecture Confusions

| Confusion | Clarification | Where to review |
|---|---|---|
| Einsum vs. loop nest | The Einsum defines the computation; the loop nest defines one execution order. | L03-L05 |
| Tiling vs. spatial parallelism | Both start from rank splitting, but temporal loops improve reuse while spatial loops assign work to PEs. | L06 |
| Coordinate vs. position | Coordinate is the mathematical index; position is the physical storage slot in a representation. | L09 |
| Gating vs. skipping | Gating saves energy in an existing cycle; skipping removes the cycle. | L08-L10 |
| Structured vs. unstructured sparsity | Structured sparsity reduces metadata/control cost but may constrain model accuracy. | L07-L08 |
| Precision vs. accuracy | Reduced precision changes representation and can affect accuracy, so it is a co-design choice. | L12 |
| Compute-in-memory vs. free compute | CiM reduces movement but introduces peripheral and precision costs. | L11 |
| MAC count vs. data movement | MACs are counted from the computation; data movement depends on the mapping. | L05, L13 |

