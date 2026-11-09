# CAD Intelligence — Technical Plan: Reverse Engineering POC

**Version:** 0.1  
**Date:** 2026-11-09  
**Status:** Proposed execution plan; implementation choices remain unapproved  
**Owner:** Яков / FrogFlow Studio

## 1. Decision and objective

The first technical experiment is **reverse engineering from STL/OBJ**. Do not divert the first sprint into a separate text-to-CAD product. The agent/tooling comparison for FreeCAD, CadQuery and OpenCascade runs in parallel as a bounded architecture study, not as a reason to delay the first pipeline test.

Goal: demonstrate one honest, repeatable end-to-end path from a real mesh to a reconstructed CAD result for one narrowly chosen geometry class, with quantitative comparison against the source mesh.

The POC is not required to reconstruct arbitrary objects, recover hidden geometry, infer unknown tolerances, or generate a complete parametric design history.

## 2. Critical first gate: inspect the input

Before selecting algorithms, obtain an STL/OBJ file and permission to process it. Preserve the original unchanged.

Record:
- file format, file size and triangle/vertex count;
- units and scale, if known; otherwise mark them unknown;
- bounding box and connected components;
- watertightness, degenerate faces, duplicate vertices/faces and normal consistency;
- scan quality, holes, occlusion, noise and possible outliers;
- available reference dimensions or known design intent.

**Gate A:** choose the test object only after this audit. Prefer a single part with visible, reasonably clean planar and/or cylindrical features. If the client scan is too complex or unavailable, use a clearly labelled test asset, not a claim of success on the client’s part.

## 3. POC scope

### Required pipeline

1. **Ingest:** load STL/OBJ, preserve source and record metadata.
2. **Validate:** diagnose mesh integrity, scale, units and connected components.
3. **Prepare:** apply only documented, reversible cleanup; keep raw and processed copies.
4. **Recognize:** attempt a limited set of features that the input supports, initially planes and cylinders/holes where appropriate.
5. **Reconstruct:** fit candidate analytic surfaces and construct a CAD B-rep for the selected class; use a fallback or report failure if reconstruction is not reliable.
6. **Export:** write STEP only when a valid B-rep result exists. A mesh export must not be labelled CAD reconstruction.
7. **Compare:** align reconstructed geometry and source mesh using a declared method; calculate source-to-result and result-to-source deviations.
8. **Report:** provide preview, recognized features, output type, metrics, units, method, known limitations and failed stages.
9. **Repeat:** run from a clean state using documented commands.

### Explicitly out of scope

- universal reconstruction of arbitrary freeform surfaces;
- guaranteed recovery of original design intent or parametric feature history;
- inference of invisible/internal surfaces from an exterior-only scan;
- automatic assignment of missing dimensions or tolerances;
- production approval, certification or engineering safety claims;
- polished SaaS UI, accounts, billing or cloud deployment.

## 4. Architecture hypothesis

Use a modular Python pipeline. Keep the LLM out of low-level numeric geometry calculations.

- **Mesh adapter:** load, inspect and optionally clean mesh data.
- **Feature analysis:** deterministic geometry fitting/segmentation and confidence estimates.
- **Reconstruction adapter:** create exact geometry and B-rep through the selected CAD kernel/tool.
- **Validation module:** shape validity, units, dimensions, topology and deviation metrics.
- **Orchestrator:** initially a deterministic workflow; add an LLM only where it can make bounded choices, explain diagnostics, select among approved strategies or recover from documented errors.
- **Report/preview:** generate machine-readable JSON plus a simple human-readable report and visual comparison.

Do not begin with a multi-agent architecture. First prove one deterministic end-to-end pipeline. Then add a single LLM orchestrator with restricted tools and explicit stopping conditions.

## 5. Candidate stack and decision rule

Candidates to evaluate:
- Open3D and/or Trimesh for mesh inspection and processing;
- OpenCascade (OCCT) for geometric primitives, B-rep and STEP operations;
- FreeCAD for an accessible CAD automation and inspection environment;
- CadQuery for code-oriented parametric construction where suitable.

These are candidates, not a locked stack. FreeCAD and CadQuery are not substitutes for the geometric kernel: CadQuery is built on OpenCascade bindings; FreeCAD exposes its own Python automation APIs and also uses OCCT.

Choose based on a small reproducible experiment, not popularity:
- can it load or interoperate with the required input/output formats?
- can it represent the required fitted surfaces as valid B-rep?
- can it export and re-import STEP reliably?
- can it run headlessly/reproducibly in our environment?
- is its API inspectable and controllable by an LLM tool layer?
- what are installation, licensing, platform and packaging constraints?
- can tests check results without relying on screenshots alone?

## 6. Validation requirements

Do not use a single scalar score as proof of engineering correctness.

Report as applicable:
- source and reconstructed bounding boxes and units;
- validity of the CAD shape and number of solids/shells;
- surface-distance distribution, mean/median and high percentile; maximum distance only with a robust explanation of outlier sensitivity;
- fraction of sampled source points within a stated threshold, if a justified threshold exists;
- bidirectional comparison, since one-way source-to-result distance can hide missing or excess geometry;
- alignment method and whether rigid registration changes the comparison;
- known reference dimensions, when available;
- visual deviation map as supplementary evidence, not the only evidence.

If units or scale are unknown, do not claim an absolute tolerance. If no agreed tolerance/reference dimensions exist, report measured deviations without declaring the part manufacturing-ready.

## 7. Execution phases and deliverables

### Phase 0 — Input audit
Deliverable: `INPUT_AUDIT.md` with file facts, defects, known units, candidate geometry class and a proposed success criterion.

### Phase 1 — Minimal mesh pipeline
Deliverable: runnable script that loads STL/OBJ, validates it and emits diagnostics/preview without modifying the original.

### Phase 2 — One feature family
Deliverable: documented feature-fitting experiment on the selected geometry, including failures and confidence/fit residuals.

### Phase 3 — CAD reconstruction and STEP round-trip
Deliverable: reconstructed CAD result if supported; STEP export and re-import test; validity report. If the chosen feature family cannot produce a valid result, document why and stop or revise scope.

### Phase 4 — Quantitative comparison
Deliverable: comparison report and deviation visualization using a declared alignment and sampling method.

### Phase 5 — Optional bounded LLM control
Only after Phases 1–4 are reproducible: expose a small tool set such as `inspect_mesh`, `fit_features`, `build_candidate`, `validate_shape`, `compare_surfaces`, and `export_step`. Require the agent to inspect results, choose from allowed strategies, keep a trace of actions, and stop when acceptance criteria fail or retry limits are reached.

### Phase 6 — Demo
Deliverable: reproducible live run, recorded fallback clearly labelled as a recording, sample input, output files, metrics and limitations.

## 8. Acceptance criteria for the first POC

A POC counts as a success only if:
- the selected input is processed reproducibly from a clean run;
- the chosen feature family is explicitly stated;
- a reconstructed B-rep/STEP is produced and successfully re-opened if STEP success is claimed;
- quantitative source-vs-result comparison is reported with units/method;
- no mesh is misrepresented as a parametric CAD model;
- failure states are visible and documented;
- an engineer can inspect the source, output and report.

The result may be a successful limited reconstruction or a well-documented technical negative result. A polished interface without a validated geometry result does not count.

## 9. Immediate next actions

1. Ask the studio contact for a suitable STL/OBJ and permission to use it for a private technical demo.
2. Ask for known dimensions, units and what the reconstructed model is expected to be used for.
3. In parallel, ask Claude and Север to compare FreeCAD, CadQuery and OpenCascade specifically for an autonomous reverse-engineering workflow, using the attached evaluation brief.
4. Run a minimal input-audit script before choosing the final reconstruction method.
5. Do not lock the stack until the input audit and tool comparison are available.

## 10. Decision boundaries

The owner decides the product scope and final stack. The AI advisors should recommend, cite primary documentation, state uncertainty and propose small tests. No model should claim it ran a test unless it actually ran it and records environment, command and result.


---

## 11. Pilot studio integration and reconstruction semantics

### 11.1 Observed customer workflow

Founder-reported facts about the prospective pilot studio:
- daily damaged-part reconstruction;
- simple and medium-complexity parts;
- a separate 3D scanner is available, exact model unknown;
- the described equipment appears to include a tripod-mounted camera/sensor unit, a handheld scanning instrument and a smaller rotating-platform setup; this description is insufficient to identify the manufacturer, model or scanning technology;
- SolidWorks is the studio's CAD system;
- manual geometry construction is the main bottleneck;
- current manual effort is approximately 1–3 hours per part;
- target process is scan → automated reconstruction → engineer review → 3D print.

Treat these as reported workflow facts pending observation and time study. Do not assume every daily repair is scanned or that all scanner outputs are STL/OBJ.

### 11.2 First integration boundary

For the POC, use **file-based integration**, not direct scanner control:
- identify scanner make/model and capture software;
- determine available export formats, units, scale and coordinate conventions;
- export a sample scan to STL/OBJ or another format only after explicitly adding support;
- preserve the original source file;
- produce STEP only if a valid B-rep is generated;
- test the STEP file in the studio's actual SolidWorks version;
- record import warnings, units, body count, geometry integrity and engineer edits.

Direct scanner SDK integration, live capture, SolidWorks add-ins and automated printer submission are future options, not first-POC dependencies.

### 11.3 Reconstruction task contract

Every run must identify the requested result mode:

- **As-built:** represent observed physical geometry. Do not silently idealize wear.
- **Nominal:** reconstruct intended/idealized geometry using known dimensions, symmetry, repeated features or engineer-approved constraints.
- **Parametric:** deliver an editable construction recipe/feature history and verify that changing selected parameters rebuilds the model.

Record geometry provenance by region or feature: scan-observed, separately measured, inferred by symmetry/repetition, constraint-derived, or assumed and awaiting approval.

For missing/broken regions, the system must not infer an original shape solely from an exterior scan. Ask for measurements, a matching part, assembly context or engineer input. Present multiple candidates when ambiguity is material.

### 11.4 Geometry reconstruction stages

Do not treat feature fitting as complete reconstruction. Keep these stages separate:
1. mesh and scan quality audit;
2. registration / coordinate and scale determination;
3. segmentation and primitive/surface hypotheses;
4. feature relationships and adjacency;
5. design constraints and known dimensions;
6. candidate topology / solid construction;
7. B-rep validity and task-specific checks;
8. STEP round-trip and target-CAD inspection;
9. source comparison with observed-region and inferred-region reporting;
10. engineer correction and acceptance.

LLM may choose among approved strategies, ask questions, interpret diagnostics and orchestrate tools. It must not invent dimensions, tolerances, topology or missing design intent.

### 11.5 Comparison and acceptance method

For every run, preserve:
- raw scan and processed copy;
- scale/units source and registration transform;
- alignment method (best-fit versus engineering datum-based);
- sampling method and observable scan coverage;
- source-to-CAD and CAD-to-source distance distributions;
- mean/median and robust high percentiles;
- known critical dimensions and independent measurements;
- regions omitted from comparison and why;
- geometry provenance and uncertainty reasons.

A partial scan makes whole-model bidirectional distance potentially misleading. Report observed and inferred regions separately. Do not use a low fitting residual as proof that the nominal part is correct. Confidence scores must not be presented as calibrated probabilities unless calibration data exists.

### 11.6 Initial geometry family

Do not lock the first family before auditing real files. Candidate families for a first experiment may include prismatic parts, brackets, spacers, bushings, simple housings, plates with holes, and other parts dominated by planes/cylinders. These are candidates only, not promised supported classes.

### 11.7 Foundational operational safeguards

Before processing third-party files, provide:
- explicit authorization and purpose limitation;
- file-size/complexity limits and validation;
- isolated worker with CPU/RAM/time/process limits;
- restricted filesystem and no arbitrary network access;
- no unrestricted execution of model-generated code;
- durable task status, cancellation and bounded retry;
- versioned inputs, parameters, tools and outputs;
- resource budget, retention and deletion controls.

A polished multi-tenant platform is not needed for the POC, but safe handling is required even for a small pilot.
