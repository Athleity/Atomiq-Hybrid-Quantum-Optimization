# Presentation audit refresh

## Why the publication was revised

The first GitHub release had a dense report layout, and the README was text-only. This made the evidence hard to scan. This refresh reorganizes the README around the three challenge stages, embeds six saved result figures directly in the page, and adds short indexes for documentation, reports, and machine-readable results.

## PDF redesign

The final report is an A4 PDF built from saved, audited artifacts. The local environment has no LaTeX engine, so the already-installed ReportLab runtime was used to improve layout without changing the project environment. The report now uses a restrained navy/teal palette, larger type, consistent margins and page furniture, readable result tables, intentional page breaks, concise captions, and a clear distinction between actual-candidate simulation and synthetic QAOA hardware evidence. It is organized into eight sections across nine pages; the extra Stage 1 diagnostic page includes ROC/PR and calibration figures rather than leaving a mostly blank page.

The report includes Stage 1 model comparison, ranking and calibration; the exact Stage 2 shortlist; actual-pool QAOA simulation diagnostics; Stage 3 exact probability, individual-qubit validation, shot convergence, and IBM results; synthetic QAOA hardware comparison; the Marrakesh provenance correction; limitations; and reproduction instructions. It does not reconstruct optimizer trajectories or alter experimental values.

Final PDF: output/pdf/atomiq_final_research_report.pdf  
Pages: 9  
Size: 2,947,561 bytes  
SHA-256: cd7502169c1cc2019c0f964d1109a27f2fafc72a249ae201743187889b7c2d0e

## Visual inspection

All nine pages were rendered after generation. The cover, Stage 1 validation pages, Stage 2 result pages, Stage 3 circuit and hardware pages, synthetic QAOA hardware page, and reproduction/limitations page were reviewed at page size and in contact sheets. No text clipping or orphaned provenance section was observed. Existing result figures were reused from saved validated outputs; no experiment was rerun to create the README figures or refresh the PDF.

## Scientific integrity

The README and report distinguish:

- Training-only nested-CV validation from test metrics.
- Exact classical enumeration/MILP from QAOA diagnostics.
- Actual-candidate local statevector/Aer results from the separate synthetic IBM QAOA experiment.
- Marrakesh Stage 3 hardware data from Kingston calibration/simulator data.
- Raw and readout-mitigated estimates, including inconclusive or negative mitigation findings.

The Stage 3 source export remains immutable, and the canonical backend/layout correction remains documented. No quantum advantage is claimed.

