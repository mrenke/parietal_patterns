# prep_results — DD Paper Figures

Final figure-generation notebooks for the dyscalculia (DD) paper submission.
Reuses analyses from `nets_PFM/`, `gradients_noHalo/`, etc., but this folder is
where the paper-ready panels get assembled — kept visually consistent across
figures and across the two machines this repo is worked on from.

## House style — single source of truth

All figures share one style module: **`plotting.py`**. Any new figure
notebook must start with:

```python
from parietal_patterns.prep_results.plotting import (
    set_style, render_surf_panel, plot_group_comparison_panel,
    add_panel_letter, add_subpanel_label,
    GROUP_LABELS, GROUP_ORDER, GROUP_PALETTE,
)
set_style()
```

before any plotting happens. Don't hand-roll rcParams, colors, or fonts in a
new notebook — if a new figure needs something `plotting.py` doesn't have
yet, add a helper there so every figure keeps calling the same function
rather than each notebook re-deriving its own version.

### Style origin

The convention follows the **"scientific-figures" Claude skill** from
Gilles' skills repo:
https://github.com/Gilles86/gilles-claude-skills/tree/main/skills/scientific-figures
— vision-science publication style: seaborn-on-matplotlib, Helvetica,
despined/offset axes, no gridlines, muted categorical palette, vector-safe
export. `plotting.py`'s `set_style()` is this repo's baked-in copy of those
conventions. If that skill is installed locally it has more rationale behind
the choices than the code comments here; either way, `plotting.py` is what
the notebooks actually import, so it's the binding contract, not the skill.

### Key conventions (see `plotting.py` for the actual code)

- **Colors:** `GROUP_PALETTE = {'Control': '#4C72B0', 'Dyscalculia': '#DD8452'}`
  (blue/orange). Always import `GROUP_LABELS`/`GROUP_ORDER`/`GROUP_PALETTE`
  rather than redefining group naming, ordering, or colors per notebook.
- **Group comparison panel** (`plot_group_comparison_panel`): bar (alpha=0.5)
  + swarm overlay + a significance bracket showing only the p-value (no test
  statistic). Originates in `gradients_noHalo/rep_groupDiffs_dParams.ipynb`;
  kept as the house idiom for group comparisons rather than switching mark
  types per figure.
- **Surface panels** (`render_surf_panel`): renders one inflated-surface view
  to a whitespace-auto-cropped PIL image so panels pack into a matplotlib
  `GridSpec` without manual trimming. Works with fsaverage5 or fsLR 32k
  surfaces/maps interchangeably — Fig 1 uses fsaverage5, Fig 2 uses fsLR 32k.
- **p-values:** `format_pvalue(p, prefix='p')` — `'p < 0.001'` below the
  reporting floor, else `'p = 0.036'`. Used for frequentist p-values and for
  Bayesian posterior tail probabilities (`prefix='p_bayes'`), so a tail with
  no draws on one side never prints as `p = 0.000`.
- **Panel letters:** `add_panel_letter()` — bold, Nature-style, top-left —
  for main panels (A, B, ...); `add_subpanel_label()` — smaller italic — for
  sub-panels grouped under one letter (i, ii, iii, ...).
- **Figure width:** journal double-column width is 7.25in (see Fig 1's
  combined-figure `figsize`).
- **Vector export:** `pdf.fonttype=42` / `svg.fonttype='none'` keep exported
  text editable/non-outlined — don't override these per-figure.

## Figures in this folder

- `paperDD_behavFig1.ipynb` — **all behavioural figure code**, in three
  independent sections: Fig. 1 panel B (accuracy/RT bar+swarm + choice curves
  carrying the probit interaction p_bayes), Fig. 1 panel D (`rdm_full_cont`
  group posteriors), and the magnitude-comparison **parameter correlation
  matrix** (same |ρ| + stars idiom as Fig. 4, one triangle, applied within the
  task). Each section exports a **standalone drop-in PDF** — Fig. 1 itself is
  assembled by hand in Affinity, so this notebook deliberately does *not*
  build a combined figure. Panels B and D were formerly
  `panelB_standalone.py` / `panelD_standalone.py`, now folded in here; those
  two scripts are redundant. Runs on the Mac, not the remote — needs
  `numrisk` + `pingouin` + the magjudge traces under
  `ds-dnumrisk/derivatives/cogmodels_magjudge`.
- `paperDD_combFig1.ipynb` — **Fig. 4**, behaviour × neural cross-measure
  correlations: panel A the |ρ| matrix, panel B scatters of the pairs that
  survive it. Reworked from `thesis_ch-DD-neuro_01.ipynb`'s
  `cross_measure_partial_r2_behav-neural_group-removed.pdf`: gradient measures
  and `mean_iq` dropped, cells show **|ρ| rather than ρ²** (shading unsigned,
  printed coefficient signed), and the two controls (partialling out group,
  multiple-comparison correction) are independent switches in the config cell —
  the output filename encodes which were on. Both panels are drawn by
  `draw_heatmap(ax)` / `draw_scatter(ax, ...)` so the combined figure and the
  standalone exports share one code path. Same env as the behaviour figure
  (`behav_fit`, needs `numrisk` + `pingouin`).
  - Two things not to undo when editing: `AXIS_CLIP` crops the perceptual-noise
    axis for readability but is **view-only** — the fit and the statistics keep
    all subjects, and `apply_clip` returns/prints the off-scale count so a
    clipped panel can't silently drop a point. And string literals carrying
    mathtext need an `r` prefix — a bare `'\\rho'` is a carriage return.
- `paperDD_neuralFig1.ipynb` — nPRF R² in NPC (group-mean surfaces) + 3
  robustness-check bar panels.
- `paperDD_neuralFig2.ipynb` — PFM/DAN patch results (group-average DAN
  patches, fsLR 32k; in progress).
- `ohbm26_poster_plotsAstats_01.ipynb`, `thesis_ch-DD-*.ipynb` — earlier
  poster/thesis notebooks. Some idioms (bar+swarm) originate here but they
  predate `plotting.py` and don't import it — don't treat them as the style
  reference for new paper figures, `plotting.py` supersedes them.

## Workflow note

This folder is actively being reworked in parallel on two machines via git
push/pull. When asked to create a new figure here, read `plotting.py` (and
this file) first, and extend it rather than introducing a one-off style.
