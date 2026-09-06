# Publication figure-quality revision

Date: 2026-09-04. This revision changes figure rendering and the two LaTeX
sources; it does not rerun or alter any particle trajectory. Previous code and
theory corrections remain unchanged.

## Population qualifier

The Methods Appendix explicitly states that, for alpha greater than pi/2, the
reported quantities are conditional statistics of the population selected by
the Q0 dead band and high-alpha candidate filter, rather than net chirality of
every near-wall particle. PRL states "selected boundary population" in both
the stability caption and Discussion. The W schematic now calls W a
near-tangential membership proxy and does not describe it as a measured force
or a strict causal wall-dependence test.

## Redrawn material

Thirteen figures were redrawn as native vector PDFs, each with a 600 dpi PNG
counterpart:

- two terminal-state particle comparisons, using the exact 28 terminal frames
  and drawing all 56,000 particles;
- the two stability/selection plots, directly from the 44-row published CSV;
- three algorithm-definition schematics at 6.5-inch publication width;
- two directional-dispersion plots, two Chern-platform plots, and the PRL
  alpha=pi/2 strip plot;
- the five-phase matched-parameter strip comparison, rearranged to two columns.

The three lattice-scale plots were already true vector PDFs. Their display
width was increased from 0.62 to 0.80 of the Appendix text width instead of
resampling them. The periodic snapshot figure was likewise already vector,
apart from its smooth color ramp, and was retained.

All 13 replacement PDFs contain zero raster-image objects. Normal plot labels
were designed at about 7-10 points at their final displayed size; the three
definition schematics use at least 8.3 points for ordinary text. The PNG
counterparts are intended for platforms that cannot ingest vector PDF, not as
the LaTeX sources.

## Numerical preservation

- Chern figures reproduce all 2,142 saved CSV records, including NA gaps.
- The five-phase strip figure reuses 108,540 stored mode samples. Its crossing
  counts, intersection locations, and exclusions agree with the previous
  machine-readable diagnostics.
- The PRL strip figure reuses 21,708 stored eigenvalues and the same localization
  classification; the two reference counts remain (2,-2).
- Redrawn one-dimensional dispersions use the same 1,000-point cuts. The maximum
  difference between the old and current bulk-matrix implementations on these
  cuts is 3.55e-15.
- Stability curves are exact CSV values; endpoint unavailability, disconnected
  critical markers, and shorter-window marker fills are retained.

The PRL projector paragraph was also corrected to say that the displayed Chern
platforms use the screened pole formula and that representative points are
independently checked by Fukui discretization. This matches the figure caption,
Methods, and implementation.

## Reproduction and QA artifacts

Redraw scripts and provenance files are stored beside the generated figures in
`particles/`, `stability/`, `schematics/`, `theory/`, and `strip_sweep/`.
`Figures/HighResolution20260903/` is the exact publication payload referenced
by the staged TeX sources. Poppler page renders and contact sheets are under
`qa/`; the `before/` directory contains the pre-revision paper sources and PDFs.

The final PDFs were compiled twice with XeLaTeX. The Appendix is 39 pages and
PRL is 7 pages. The Appendix log contains no overfull, underfull, or
unresolved-reference warning. PRL retains its pre-existing underfull
bibliography/paragraph warnings and has no overfull or unresolved-reference
warning. A structural scan found zero raster-image objects in every one of the
13 replacement PDFs; all 39 Appendix pages and all 7 PRL pages were rendered
for visual review, including a separate final inspection of Appendix pages
37--39 after pagination was tightened.

Independent particle/selection, bulk-theory, and topology/strip agents all
reported a final pass after checking the staged sources, PDFs, saved numerical
records, labels, and layout. No blocking inconsistency remains in their scoped
reviews.

## Publication verification

The reviewed sources and PDFs were published to `D:\LaTex\Boundary Flow`, and
the 13 vector PDFs plus their 13 PNG counterparts were published under
`Figures\HighResolution20260903`. A post-copy SHA-256 readback found all four
documents and all 26 figure assets byte-identical to the reviewed staging
copies.

For an independent source-level reproduction test, the published TeX sources,
bibliography output, and complete figure directory were copied into a clean QA
mirror and compiled twice with XeLaTeX. The rebuild produced the same 39-page
Appendix and 7-page PRL without overfull boxes or unresolved references. The
Appendix's extracted text is exactly identical to the published PDF; PRL is
exactly identical after normalizing the expected date change produced by its
existing `\today` command.
