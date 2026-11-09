# CAD Intelligence
## Master Product & Technology Strategy v1.0

**Project:** FrogFlow Studio  
**Direction:** AI CAD / Engineering Automation  
**Product model:** Hybrid AI CAD Platform  
**Development strategy:** Speed-first, platform-first architecture  
**Initial team:** Founder + AI  
**First technical module:** Reverse Engineering  
**Status:** Strategic proposal, pending founder approval  
**Date:** 9 November 2026

---

## 1. Executive vision

CAD Intelligence is envisioned as a hybrid AI engineering platform combining three modes:

1. **AI CAD Copilot:** assists an engineer with CAD tasks and operations.
2. **Autonomous CAD Agent:** plans and executes bounded engineering workflows with minimal human intervention.
3. **CAD Intelligence Platform:** provides a unified workspace for projects, files, models, tasks, validation, reports, and collaboration.

The intended user experience: a user describes an engineering task in natural language and supplies the required source materials. CAD Intelligence interprets the task, plans the work, invokes specialized tools, creates or modifies geometry, validates the result, and explains what was done.

The platform is not expected to automate every engineering task at launch. Capabilities expand through validated workflows.

**Long-term goal:** build a commercial AI platform that automates a meaningful share of repetitive CAD and engineering work while preserving human control over ambiguous, safety-critical, and consequential decisions.

## 2. Product pillars

### 2.1 CAD Copilot

An interactive assistant for engineers.

Potential capabilities:
- Understand natural-language instructions.
- Inspect uploaded CAD files and explain model structure.
- Modify supported features and parameters.
- Assist with dimensions and geometric constraints.
- Generate scripts for supported CAD operations.
- Diagnose construction errors.
- Prepare engineering documentation and reports.

Example: “Increase this hole diameter to 12 mm, preserve its position relative to the central axis, and check the remaining dimensions.”

The Copilot should execute supported operations, validate the result, and show the changes rather than merely return code.

### 2.2 Autonomous CAD Agent

An agent that executes bounded workflows with minimal intervention.

Expected workflow:
1. Receive a task.
2. Identify missing information and constraints.
3. Produce an execution plan.
4. Select approved CAD tools.
5. Execute operations.
6. Inspect intermediate state.
7. Diagnose and attempt bounded recovery from errors.
8. Ask a human when ambiguity or risk requires it.
9. Validate and report the final result.

Autonomy is measured by the share of tasks completed correctly without manual intervention, not by the number of actions performed without a person.

### 2.3 CAD Intelligence Platform

The commercial and operational environment:
- User workspace and authentication.
- Projects and file management.
- Source and output file storage.
- Operation history and model versioning.
- 3D preview and comparison.
- Task and agent management.
- Validation and reports.
- Export and API.
- Team collaboration.
- Usage tracking and billing.

All three pillars should use shared platform and geometry infrastructure.

## 3. Product modules and priorities

| Module | Purpose | Priority |
|---|---|---|
| Reverse Engineering | Reconstruct supported CAD geometry from scans | P0 |
| Geometry Analysis | Inspect and recognize supported geometry | P0 |
| Project Workspace | Projects, files, task runs, and history | P0 |
| Engineering Reports | Validation results and limitations | P1 |
| CAD Copilot | Natural-language assistance for CAD operations | P1 |
| CAD Agent | Bounded autonomous task execution | P1 |
| CAD Modification | Modify existing models | P1 |
| CAD Generation | Create models from specifications | P2 |
| Integrations | External CAD systems and API | P2 |
| Team Workspace | Collaboration and team controls | P2 |

P0 means required for the first useful product, not that every P0 module must be completed simultaneously. P1 and P2 are expansion priorities, subject to validation and demand.

## 4. Core technology architecture

**Core principle: the LLM coordinates the process; specialized geometry tools perform engineering computation.**

The LLM must not guess geometry. It should invoke inspectable operations and reason over their outputs.

### 4.1 Platform layer

Responsibilities:
- Authentication and authorization.
- User workspace and projects.
- File management.
- Job orchestration and status.
- Version history.
- Usage tracking.
- API and billing foundations.

### 4.2 AI orchestration layer

Responsibilities:
- Task understanding.
- Planning and decomposition.
- Tool selection.
- Execution control.
- Error analysis and bounded recovery.
- Human-in-the-loop decisions.
- Result interpretation.

Use a provider/model abstraction so Claude, GPT, and future models can be evaluated and replaced. Model selection should consider task quality, cost, latency, and tool-use reliability. Do not hard-code the product to one model provider.

### 4.3 CAD core layer

Candidate technologies, not yet approved:
- **OpenCascade / OCP:** geometric kernel and B-rep operations.
- **FreeCAD:** CAD automation and access to geometry objects.
- **CadQuery:** programmatic parametric construction.
- **Open3D:** scan and point-cloud processing.
- **Trimesh:** mesh inspection and operations.

The final stack must follow independent comparison and practical smoke tests. Not every candidate necessarily belongs in production.

### 4.4 Validation layer

An independent validation layer should check:
- Geometric validity.
- Topological integrity.
- Units and scale assumptions.
- Dimensions and known constraints.
- Number of solids and shells where relevant.
- Deviations from source data.
- Export and re-import integrity.
- Compliance with user-specified constraints.

A valid CAD object is not automatically an engineering-correct part.

### 4.5 File and geometry layer

Initial formats:
- STL.
- OBJ.
- STEP.

Additional formats should be added when justified by real workflows.

Keep these concepts distinct:
- **Mesh:** polygonal surface representation.
- **B-rep:** boundary representation of solids and surfaces.
- **Parametric CAD:** model with editable parameters and construction history.

Exporting STEP does not, by itself, create a parametric feature history.

## 5. Development strategy: platform-first, validation always

Do not build a disposable one-off prototype with an assumed full rewrite later. Establish reusable platform boundaries from the beginning, but implement only what the next validated milestone needs.

The first workflow is **Reverse Engineering from STL/OBJ**.

### Phase 0. Foundation and technical selection

**Goal:** select a defensible technical foundation.

Work:
- Compare FreeCAD Python API, CadQuery, and OpenCascade/OCP.
- Evaluate Open3D and Trimesh for mesh diagnostics.
- Run a practical smoke test.
- Define component responsibilities and interfaces.
- Specify reproducible execution and testing requirements.

Deliverable: documented architecture decision and selected initial stack, with unresolved risks recorded.

### Phase 1. First working engine

**Goal:** produce a real, measurable engineering result.

Initial workflow:

`STL/OBJ → Mesh Analysis → Feature Recognition → Reconstruction → B-rep → STEP → Validation → Report`

Work:
- Ingest and inspect source mesh.
- Establish units and scale assumptions.
- Diagnose mesh integrity and defects.
- Recognize a deliberately limited set of geometric features.
- Reconstruct only supported geometry.
- Export STEP only when a valid B-rep exists.
- Compare reconstructed geometry with the source using a documented alignment and deviation method.
- Produce a technical report.

Deliverable: repeatable Reverse Engineering pipeline for a limited, explicitly stated class of parts.

### Phase 2. Platform MVP

**Goal:** make the engine usable as a product.

Work:
- Web workspace and authentication.
- Project creation.
- File upload and storage.
- Task execution and status.
- 3D preview.
- Operation history.
- Result download.
- Technical reports.

Deliverable: a user can create a project, upload a supported model, run processing, inspect the result, and download available outputs.

### Phase 3. AI Copilot

**Goal:** add natural-language interaction.

Work:
- Natural-language task interpretation.
- Clarifying questions for missing constraints.
- Operation planning.
- Approved CAD tool selection.
- Project context.
- Result explanations and error summaries.

Deliverable: users can direct supported CAD workflows through conversation.

### Phase 4. Autonomous Agent

**Goal:** increase reliable task autonomy.

Work:
- Multi-step execution.
- Intermediate state tracking.
- Automated error diagnosis.
- Bounded recovery strategies and retries.
- Validation gates.
- Human approval for ambiguous or consequential operations.

Deliverable: agent completes selected, tested workflows with measurable reliability.

### Phase 5. Multi-module platform

**Goal:** expand beyond Reverse Engineering.

Potential directions:
- CAD Modification.
- CAD Generation.
- Assembly Analysis.
- Drawing Assistance.
- Engineering Documentation.
- Design Optimization.
- Integrations with existing CAD systems.

Each module requires separate technical feasibility and demand validation.

### Phase 6. Commercial scale

**Goal:** prepare for broader adoption.

Work:
- Team workspaces and permissions.
- Public or customer API.
- Subscription and usage-based billing.
- Compute accounting.
- Job queues and worker scaling.
- Monitoring and observability.
- Secure file handling and retention controls.
- Backup and recovery.
- Operational support.

## 6. Solo-founder operating model

Initial operating model: **Founder + AI**.

Founder responsibilities:
- Product direction and final decisions.
- Architecture approval.
- Acceptance testing.
- Customer discovery and feedback.
- Prioritization and release control.

AI responsibilities:
- Technology research.
- Implementation of bounded components.
- Test and code generation.
- Debugging assistance.
- Documentation.
- Independent architecture and code review.

Bring in specialists selectively when needed, especially for CAD engineering, geometry-kernel expertise, or infrastructure.

### Multi-AI workflow

Multiple AI systems may take complementary roles:
- Implementation.
- Independent review.
- Architecture and risk assessment.

Avoid simultaneous uncontrolled edits to the same code. Use a shared Git workflow, isolated ownership, tests, and founder approval.

### Development rules

1. Never claim a capability is complete without evidence.
2. Every module has tests and acceptance criteria.
3. No unreviewed production changes.
4. No secrets or client personal data in shared research materials.
5. Autonomous tools operate with least privilege and explicit boundaries.
6. Record failed attempts and limitations.
7. Keep AI providers and CAD tools replaceable behind interfaces.

## 7. Product experience

The interface should expose a clear engineering workflow, not the complexity of internal infrastructure.

### Dashboard
- Recent projects.
- Active tasks.
- Usage and status.
- Available tools.

### Projects
- Source files.
- CAD models.
- Versions.
- Results.
- Operation history.

### AI Workspace
- Copilot conversation.
- Agent task submission.
- Execution plan.
- Approval checkpoints.
- Errors and results.

### 3D Viewer
- Source model.
- Reconstructed geometry.
- Side-by-side or overlay comparison.
- Deviation visualization.
- Version switching.

### Reports
- Technical results.
- Geometry metrics.
- Known assumptions and limitations.
- Exportable report.

The first interface should be focused and polished around the initial workflow, not crowded with speculative features.

## 8. Engineering reliability

Engineering credibility is a product requirement.

For every result, distinguish:
- What was measured.
- What was reconstructed.
- What was inferred or assumed.
- What remains unknown.
- Which validations passed or failed.
- Which limitations apply.

A scan does not reveal all hidden surfaces, material properties, wear history, original design intent, or unknown tolerances. Ask for known dimensions and constraints when they materially affect correctness.

Do not label outputs manufacturing-ready without appropriate evidence, checks, and engineering review.

For scan comparison, document units, scale, alignment method, sampling, and deviation statistics. If scale or units are unknown, do not claim absolute dimensional tolerances. Do not rely on a single metric as proof of engineering correctness.

## 9. Initial validation environment

A small 3D studio known to the founder is a potential first validation environment.

Potential advantages:
- Real engineering tasks and files.
- Practical testing.
- Professional feedback.
- A possible pilot customer.
- Opportunity to measure time saved.

Questions to resolve:
- Which CAD applications are used?
- What part types are common?
- How often are scans used?
- Which file formats are exchanged?
- Which tasks consume the most time?
- What accuracy and tolerance requirements apply?
- What does a successful deliverable look like?

This is a potential test environment and acquisition channel, not a confirmed customer or contract.

## 10. Speed-first execution principles

1. **Build only what the next milestone needs.** Do not build for millions of users before validating the first workflows.
2. **Keep architecture extensible.** New modules should use shared interfaces without breaking existing workflows.
3. **Prefer working software over excess documentation.** Documentation supports decisions; validated capabilities demonstrate progress.
4. **Reuse before rebuilding.** Use proven CAD kernels and libraries instead of creating a geometry engine from scratch.
5. **Validate before expanding.** Add modules based on demonstrated feasibility and customer problems.
6. **Automate the development loop.** Invest early in repeatable tests, reproducible runs, and clear failure reports.

## 11. Success metrics

### Technical
- Task completion rate.
- Geometric validity rate.
- Deviation distributions.
- Processing time.
- Manual correction rate.
- Repeatability across runs.
- Export and re-import success.

### Product
- Active users.
- Projects created.
- Models processed.
- Repeat usage.
- Share of tasks completed without manual intervention.
- Time from upload to usable result.

### Commercial
- Pilot customers.
- Pilot-to-paid conversion.
- Average revenue per customer.
- Compute cost per task.
- Gross margin.
- Retention.

Early progress should be measured by reliability and customer value, not the number of agents or advertised supported formats.

## 12. Initial business model

Potential monetization:
1. Subscription.
2. Usage-based processing and compute.
3. Team plans.
4. Enterprise plans with custom integrations and controls.
5. API access.

A sensible initial hypothesis is a subscription with an included usage allowance and separate billing for resource-intensive tasks. Pricing must follow measured compute costs and feedback from early customers.

## 13. Initial roadmap

Indicative sequence, not a delivery guarantee:

| Period | Target |
|---|---|
| First 7 days | Technical selection and first smoke test |
| Weeks 2–3 | First Reverse Engineering pipeline |
| Weeks 4–5 | Workspace and project workflows |
| Weeks 6–7 | Copilot integration |
| Weeks 8–10 | Bounded Autonomous Agent |
| After week 10 | Pilot validation, module expansion, commercialization |

Actual timing depends on input geometry, kernel/tool limitations, and reconstruction quality. Re-plan after each milestone based on measured results.

## 14. Immediate priorities

1. Obtain a real STL/OBJ file with permission to process it.
2. Learn which CAD applications and task types the studio uses.
3. Receive independent comparison of FreeCAD, CadQuery, and OpenCascade/OCP.
4. Run the technical smoke test.
5. Approve the initial CAD Core architecture.
6. Build the minimum reusable Platform Layer.
7. Implement the first Reverse Engineering pipeline.
8. Validate it on a real part.
9. Prepare a demonstration with measurable results.
10. Obtain feedback from a potential pilot user.

## 15. Strategic risks

| Risk | Response |
|---|---|
| Scope becomes too broad | One primary workflow at a time |
| Reconstruction is difficult | Limit supported part classes and state boundaries |
| LLM makes mistakes | Tool constraints and independent validation |
| Dependence on one CAD kernel | Adapter architecture |
| Compute costs grow | Usage tracking and task-based model selection |
| Limited CAD expertise | Practical tests and external engineering review |
| Founder overload | Automate tests, documentation, and review |
| Premature scaling | Validate demand before major infrastructure investment |

## 16. Final strategic position

CAD Intelligence is intended to become a hybrid AI platform for engineering work.

It is not merely:
- An STL-to-STEP converter.
- A chat interface for CAD.
- A 3D model generator.
- A collection of FreeCAD scripts.

It is a unified system combining engineering-task understanding, planning, geometric computation, automation, validation, and human collaboration.

**Start with Reverse Engineering. Design the architecture for a broader platform.**

The first success is not a long feature list. It is proof that the platform can complete a concrete engineering task, produce a valid and measurable result, and explain why that result can be trusted.

**Strategic principle:**  
*Understand the task. Build the geometry. Validate the result. Empower the engineer.*


---

# Strategy update v1.1 — customer evidence and commercial validation

**Update date:** 2026-11-09  
**Status:** evidence-based refinement; strategic proposal still pending founder approval.  
**Scope:** incorporates the founder's direct description of the pilot studio and the independent reviews by Север and Компас. It does not imply that technical feasibility, willingness to pay, or a commercial agreement has been proven.

## A. Updated initial customer workflow

The founder has direct prior work experience in the prospective pilot studio. The studio is small, works in 3D, and receives damaged parts. For small or relatively uncomplicated parts, staff currently measure the part with calipers, manually reconstruct its CAD geometry, and 3D-print a replacement.

Additional founder-provided facts:
- A separate 3D scanner is available at the studio.
- The scanner is used as part of a workflow that the founder describes as a large camera/sensor unit on a tripod, a handheld scanning instrument, and a smaller setup with a rotating plate. Exact manufacturer, model, capture technology, software and export formats are unknown.
- Damaged-part reconstruction occurs daily.
- SolidWorks is the studio's CAD system.
- The main time-consuming activity is manual geometry construction.
- Manual work currently takes approximately 1–3 hours per part.
- Desired workflow: scan → automated reconstruction → engineer review → 3D printing.
- The studio handles simple and medium-complexity parts.

These are founder-reported observations, not independently audited operational data. The frequency and time range should be measured across a sample of real jobs.

### Initial product wedge

**AI-Assisted Part Reconstruction for small 3D studios and repair-oriented workshops.**

The initial job is not generic "AI CAD". It is to reduce repetitive manual reconstruction work for supported classes of damaged mechanical parts, using an existing scan as geometric evidence and retaining engineer approval before production.

Proposed workflow:

1. Scan the part using the studio's existing equipment.
2. Export a supported file, initially STL/OBJ; direct scanner integration is not required for the first POC.
3. Inspect the scan, units, scale, mesh quality, coverage and defects.
4. Separate observed intact geometry from worn, broken, occluded or missing regions.
5. Reconstruct supported geometry using deterministic fitting, constraints and explicit assumptions.
6. Generate a CAD result of a clearly declared type.
7. Validate geometry, known dimensions and deviations.
8. Open the result in SolidWorks for engineer review and correction.
9. Export/hand off the approved model for the studio's existing 3D-printing workflow.

The first deliverable should fit into the studio's current SolidWorks workflow. Native SolidWorks integration is not a first-POC requirement. A STEP handoff is a candidate interface and must be tested in the studio's actual SolidWorks version and workflow.

### What the scan can and cannot establish

A scan provides evidence about visible, captured surfaces. It does not by itself reveal the original geometry of a broken-away feature, hidden/internal surfaces, design intent, material properties or required tolerances.

For each reconstructed region, preserve provenance:
- observed from scan;
- measured separately (for example, caliper measurement);
- inferred from symmetry or repeated geometry;
- derived from an agreed engineering constraint;
- assumed by the system and awaiting engineer confirmation.

Never silently turn an assumption into a measured fact. If the original shape is ambiguous, show alternatives or request an engineer decision.

## B. Output contract: three reconstruction modes

A task must declare which result is wanted. These modes are related but not interchangeable:

| Mode | Meaning | Acceptance focus |
|---|---|---|
| As-built | Representation of the currently observed physical surface | Agreement with captured, observable geometry |
| Nominal reconstruction | Idealized intended geometry, correcting wear/damage where justified | Known dimensions, constraints, symmetry and engineer-approved assumptions |
| Parametric reconstruction | Editable construction recipe with parameters/features | Rebuildability, editable dimensions and stable feature dependencies, in addition to geometric checks |

A STEP solid may be a valid B-rep without a parametric feature history. Do not call it a parametric model unless the delivered artifact actually contains an editable construction recipe/tree and passes a rebuild/edit test.

The engineer or customer selects the intended mode and supplies known dimensions, critical interfaces, tolerances and manufacturing purpose where relevant.

## C. Revised interpretation of advisor reviews

The reviews by Север and Компас correctly identify unresolved issues:
- reconstruction from noisy or incomplete scans is technically difficult;
- fit surfaces alone do not establish a valid part topology or design intent;
- real files and known dimensions are required before choosing algorithms;
- competitors and existing workflows must be benchmarked;
- customer willingness to pay, service economics and stop criteria are missing;
- building a full SaaS platform before proving the geometry engine is premature;
- file isolation, resource limits, job recovery and permissions are foundational requirements;
- independent engineering review is necessary.

The founder's new evidence changes one important premise in the reviews: the prospective studio does not merely have occasional access to scans. The founder reports daily damaged-part reconstruction, an available scanner, SolidWorks, manual geometry construction as the bottleneck, and 1–3 hours of manual work per part. This supports keeping reverse engineering as the first product experiment rather than switching the first milestone to text-to-CAD.

It does **not** prove that every daily repair job is scan-based, that all parts are automatable, that the scanner produces suitable mesh files, or that the studio will pay. Those questions remain validation work.

The reviews' warnings about scope, technical risk, competition and commercial validation remain active.

## D. Commercial hypothesis and validation before scaling

Initial buyer hypothesis:
- small 3D-printing / reverse-engineering studios;
- repair workshops that recreate unavailable or damaged replacement parts;
- small manufacturers with recurring low-volume spare-part reconstruction.

Initial value hypothesis: reduce engineer time spent on repetitive geometry construction while preserving control over uncertain design decisions.

Do not publish a price or claim savings until the baseline and assisted workflow are measured. Record for each pilot job:
- part category and complexity;
- current manual time, including measurement and CAD cleanup;
- scan preparation and processing time;
- engineer review/correction time;
- percentage of geometry automated;
- output type and SolidWorks import result;
- critical dimensions and agreed tolerances;
- whether the result was accepted for printing;
- compute, software and human service cost.

Compare three workflows on the same parts where possible:
1. studio's current manual process;
2. an available commercial scan-to-CAD tool/workflow;
3. CAD Intelligence assisted workflow.

Relevant alternatives to investigate include Geomagic Design X / Geomagic, QUICKSURFACE, Mesh2Surface, Creaform VXmodel, and native or add-in workflows in SolidWorks, Fusion and FreeCAD. This is a research list, not a claim that each is a direct substitute for every target part. Verify current features, licensing, file support and hands-on performance using the same source files.

A plausible initial go-to-market path is a **paid reconstruction service / assisted pilot**, where the founder operates the pipeline and an engineer approves the output. A self-service platform follows only when repeatable demand and delivery economics justify it. This is a proposed path, not a committed business model.

## E. Technical and commercial gates

Do not adopt a universal accuracy threshold such as ±0.2 mm for all parts. Acceptance tolerances depend on part function, size, scan quality, material, printer, fit and the engineer's requirements.

Use staged gates:

**Gate 0 — Customer data and workflow**
- Obtain permission to process 3–5 representative scan files.
- Obtain known dimensions or caliper measurements for each where possible.
- Record the intended output and use for each part.
- Identify scanner make/model, capture software, export formats and units.
- Confirm the SolidWorks version and preferred handoff format.
- Measure current manual time on each selected job.

**Gate 1 — Technical feasibility**
- Select one narrow geometry family based on the actual files.
- Produce a reproducible pipeline and report.
- Demonstrate valid CAD geometry and a successful STEP round-trip if STEP is claimed.
- Compare against scan and independent known dimensions.
- Explicitly report failures and manual interventions.

**Gate 2 — Engineering usefulness**
- Engineer can inspect and correct the result in SolidWorks.
- Engineer records acceptance/rejection and reasons.
- Measure total assisted time, not only machine processing time.
- Confirm whether the result is suitable for the agreed print/use case.

**Gate 3 — Commercial signal**
- Studio confirms the workflow solves a recurring problem.
- Ask for a specific paid pilot or paid next batch; do not infer willingness to pay from compliments.
- Calculate full delivery cost, including compute, retries, founder/engineer time and support.
- Compare the cost and time with the current process and available alternatives.

**Stop / pivot conditions**
- No representative files or permission can be obtained.
- The selected class cannot be reconstructed reliably enough for its stated use.
- Human correction removes the time advantage.
- Existing tools provide equivalent results at a lower total cost and acceptable workflow friction, with no clear underserved segment.
- The studio does not confirm a recurring problem or a plausible paid use case after a fair pilot.
- The effort materially displaces the founder's existing revenue work without a defined time/budget limit.

A failed gate should lead to narrowing the part class, changing the service workflow, testing a different buyer problem, or stopping. It must not be hidden by expanding the feature list.

## F. Revised execution order

1. Customer workflow interview and scanner/software identification.
2. Collect 3–5 permitted representative files and known measurements.
3. Audit the inputs and classify part families.
4. Benchmark the current manual process and at least one relevant existing tool where practical.
5. Select one tractable geometry family.
6. Run a deterministic mesh-to-CAD pipeline and validate it.
7. Test STEP handoff in the studio's SolidWorks workflow.
8. Run an engineer-reviewed pilot and measure total time saved.
9. Seek a paid pilot/service order.
10. Build only the minimal reusable interface and job infrastructure needed by repeated work.
11. Expand toward Copilot, Agent and the wider platform only after the initial workflow demonstrates repeatability and demand.

This refines the earlier "platform-first" wording: keep platform boundaries and reusable interfaces, but **engine-first and customer-workflow-first in execution**. Do not build authentication, billing, collaboration or a broad dashboard ahead of the first validated job.

## G. Foundational safety and operations

Even a small pilot that handles third-party engineering files needs:
- explicit permission and purpose limitation;
- access control and least privilege;
- isolated processing workers with CPU, RAM, time, file-size and process limits;
- no arbitrary network access or unrestricted execution of agent-generated code;
- durable job status, cancellation, bounded retries and recovery;
- versioned source files, parameters, tool versions and output artifacts;
- clear retention and deletion rules;
- resource accounting and a per-job budget;
- protection against one malformed or oversized file affecting other jobs.

Advanced team management and autoscaling can wait. Basic file and worker safety cannot.

## H. Founder decisions still required

The following remain unapproved until the founder explicitly decides:
- final product name and public positioning;
- first supported part family;
- target tolerances and acceptance rules per application;
- selected CAD/mesh stack;
- whether first revenue is service, pilot, subscription or a combination;
- pricing and available development budget;
- whether to pursue direct scanner integration;
- whether the first output needs a parametric feature tree or a validated B-rep is sufficient.

**Updated strategic position:** retain Reverse Engineering as the first experiment because the founder has identified a daily, concrete studio workflow and a real manual bottleneck. Prove a narrow, engineer-approved reconstruction service before investing in the full CAD Intelligence platform.
