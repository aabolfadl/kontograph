# Paper

`kontograph.tex` — IEEE conference format, single author, 6 pages.

## Build

```bash
pdflatex kontograph.tex
pdflatex kontograph.tex   # second pass resolves cross-references
```

No BibTeX run is needed; the bibliography is a `thebibliography` environment.
No external figures — the paper is self-contained, which is what arXiv prefers.

## arXiv submission

Upload `kontograph.tex` alone. arXiv runs pdfLaTeX itself and ships IEEEtran, so
no class file is required.

Suggested categories:

- Primary: `cs.LG` (Machine Learning)
- Cross-list: `cs.SE` (Software Engineering), `q-fin.RM` (Risk Management)

The software-engineering cross-list is worth including: the leakage-suite and
ONNX-parity results are as much SE findings as ML ones.

## Every number is traceable

| Claim in the paper | Regenerate with |
|---|---|
| Ablation ladder, confidence intervals | `kg eval report` |
| Latency budget | `kg serve smoke --events 2000` |
| ONNX parity | `kg serve export-onnx` |
| Explainer fidelity | `artifacts/kaggle/out/02_distill_explainer/metrics.json` |
| Dataset statistics | `data/<dataset>/manifest.json` |
