# Pipeline Comparison: Gordon 2017 vs. Previous fsaverage5 Implementation

This document records the differences between the Gordon 2017 original pipeline
(now implemented here in fsLR 32k) and the previous fsaverage5 pipeline used
in this project. Useful for interpreting any differences in results.

---

## Denoising

| Parameter | Gordon 2017 (this pipeline) | Previous fsaverage5 pipeline |
|---|---|---|
| Motion model | Friston 1996 Volterra expansion | Ciric 2017 36-parameter model |
| Motion regressors | 24: 6(t) + 6(t−1) + 6²(t) + 6²(t−1) | 24: 6 + 6-deriv + 6² + 6-deriv² |
| Key difference | **Lagged** motion terms | **Power2 of derivatives** instead |
| Global signal | Yes (signal only) | Yes + derivative + power2 + deriv² |
| White matter | Yes (signal only) | Yes + derivative + power2 + deriv² |
| CSF | Yes (signal only) | Yes + derivative + power2 + deriv² |
| Total confound regressors | ~27 (24 motion + 3 physio) | 36 |
| Scrubbing threshold | FD > 0.2 mm | FD > 0.2 mm |
| Scrubbing method | **Interpolate → filter → excise** (Power 2014) | Censor before filtering |
| Bandpass | 0.009–0.08 Hz | 0.009–0.08 Hz |
| Pre-surface CoV masking | Yes (voxels > 0.5 SD above local mean excluded) | No |

### Notes on motion model differences
- **Friston (lagged)**: captures slow drift in scanner-related motion artefacts by
  including the previous TR's parameters. Better at modelling spin-history effects.
- **Ciric-36 (power2 of deriv)**: more aggressively captures nonlinear motion
  effects at the same TR. Benchmark paper (Ciric 2017) shows it performs similarly
  to Friston-24 on most QC metrics.
- Practically: both are reasonable. Any connectivity differences between the two
  pipelines are likely subtle but could matter for fine-grained network topology.

### Notes on scrubbing method
- Gordon's interpolate-then-excise prevents bandpass filtering from smearing
  artefact signal from high-motion frames into neighbouring clean frames.
- The simple censor-before-filter approach introduces spectral ringing near
  censored frames. This is a meaningful difference, especially for subjects with
  frequent motion spikes.

---

## Surface Space

| Parameter | Gordon 2017 (this pipeline) | Previous fsaverage5 pipeline |
|---|---|---|
| Target surface space | **fsLR 32k** (32,492 vertices/hemi) | fsaverage5 (10,242 vertices/hemi) |
| Surface format | CIFTI `.dtseries.nii` | GIFTI `.func.gii` per hemisphere |
| Subcortical included | Yes (voxel-level in CIFTI) | No |
| Cerebellum included | Yes (voxel-level in CIFTI) | No |
| Smoothing | σ = 2.55 mm geodesic (surface) + Euclidean (volume) | — |
| Resolution | ~2 mm effective (32k) | ~9 mm effective (fsaverage5) |

---

## Network Detection

| Parameter | Gordon 2017 (this pipeline) | Previous fsaverage5 pipeline |
|---|---|---|
| Unit of analysis | Vertex-wise (all brain vertices + subcortical voxels) | — |
| Correlation matrix | All-to-all, local connections (< 30 mm) zeroed out | — |
| Algorithm | Infomap at multiple density thresholds (0.3%–5%) | — |
| Network assignment | Consensus across thresholds, matched to group templates | — |
| Small network removal | Networks < 400 vertices/voxels removed | — |

---

## What differences in results to expect

If you compare fsLR 32k (this pipeline) to fsaverage5 (previous):

1. **Finer spatial detail** — 32k has ~3× more vertices than fsaverage5; small
   network pieces (< ~2 cm²) visible in 32k will be blurred/absent in fsaverage5.
2. **Subcortical connectivity** — fsaverage5 pipeline had no subcortical data;
   the CIFTI pipeline will show thalamo-cortical and striato-cortical networks.
3. **Slightly different network boundaries** — due to CoV masking and the
   interpolate-then-excise scrubbing in the Gordon pipeline.
4. **Motion sensitivity** — Gordon's scrubbing is more conservative; subjects
   with frequent motion may show more data loss but cleaner connectivity.

---

## References

- Gordon et al. 2017, Neuron 95, 791–807
- Friston et al. 1996, Magn. Reson. Med. 35, 346–355 (Volterra motion model)
- Ciric et al. 2017, NeuroImage 154, 174–187 (36-parameter benchmark)
- Power et al. 2014, NeuroImage 84, 320–341 (interpolation scrubbing)
