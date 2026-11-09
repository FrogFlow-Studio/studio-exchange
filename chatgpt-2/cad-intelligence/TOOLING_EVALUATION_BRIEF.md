# CAD Intelligence — Tooling Evaluation Brief for Claude and Север

**Date:** 2026-11-09  
**Purpose:** Independent comparison to support the first STL/OBJ reverse-engineering POC.  
**Question:** Which role should FreeCAD, CadQuery and OpenCascade play in an autonomous agent workflow, and what is the smallest test that will let us choose responsibly?

## Required context

The owner has chosen **STL/OBJ reverse engineering as the first experiment**. Do not redirect the project to text-to-CAD as the main first milestone. We want an agent that can inspect a scan, identify a limited class of geometric features, reconstruct supported geometry, validate it, compare it with the scan and report uncertainty.

The POC must be narrow, reproducible and honest. A STEP file does not by itself prove that a model is parametric. A mesh must not be presented as a reconstructed B-rep solid.

## Please evaluate these candidates

1. **FreeCAD Python API**
2. **CadQuery**
3. **OpenCascade Technology (OCCT), including Python bindings such as OCP where relevant**

Also comment briefly on whether Open3D or Trimesh should be used for mesh diagnostics and feature-fitting support, but keep the main comparison focused on the three CAD candidates.

## Evaluation dimensions

For each candidate, provide:
- its actual role: application/automation host, high-level modeling API, or geometric kernel;
- STL/OBJ ingestion and mesh interoperability limits;
- ability to fit planes/cylinders/cones and create exact B-rep geometry from fitted parameters;
- STEP export and re-import/shape validation options;
- suitability for a persistent, iterative agent session;
- API discoverability, error messages, introspection and rollback/transaction support;
- headless execution and reproducible deployment considerations;
- platform, packaging and dependency friction;
- licence and distribution considerations, with primary-source links;
- strengths, weaknesses and the failure modes most likely to matter in our POC.

Use current official documentation and repositories wherever possible. Distinguish verified facts from assumptions. Do not assert that a library automatically converts arbitrary STL meshes into editable parametric CAD unless primary documentation and an actual tested workflow support that claim.

## Deliverables

1. A concise comparison table.
2. A recommended architecture for the first POC, including which tool performs each operation.
3. One preferred stack and one fallback stack, with rationale.
4. A minimal 1–2 hour smoke test that could invalidate your recommendation quickly.
5. Exact acceptance criteria for that test.
6. The biggest unresolved technical risk in reverse engineering from STL/OBJ.
7. Links to the official docs/source for important claims.

## Suggested smoke test

Use a simple mesh with known geometry, for example a block-like part with one cylindrical hole, or another object appropriate to the installed tooling. Inspect the mesh; fit/identify a plane and cylinder; construct a candidate B-rep; export STEP; re-import the STEP; validate the shape; compare the reconstructed surface to the source mesh. Record all deviations and any manual intervention.

Do not claim the test passed unless it was actually run. If you cannot run code, give commands and a test protocol, and label them unexecuted.

## Independence rule

Please make your recommendation independently before reading the other advisor's conclusion. Save your answer in your own folder in the shared repository, then state:
- what is supported by documentation;
- what you personally tested versus only inferred;
- what would change your recommendation.
