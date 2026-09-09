# Error audit: F827, H038, and H374

This audit examines the three first-batch cases called out for follow-up. It uses
all 16 indexed photographs, the same calibrated 50, 100, and 150 mm center crops
as the experiments, the measured curves, all six held-out predictions, and
reconstructed held-out 3-neighbor sets from the saved per-photo features.

The generated panels are [F827](../artifacts/reports/error_audit/F827.png),
[H038](../artifacts/reports/error_audit/H038.png), and
[H374](../artifacts/reports/error_audit/H374.png). Exact pooled and per-camera
errors are in [metrics.csv](../artifacts/reports/error_audit/metrics.csv), and
neighbor identities and within-experiment distances are in
[neighbors.csv](../artifacts/reports/error_audit/neighbors.csv). Generated files
remain ignored by Git.

## Observations

### F827

- The measured curve is the finest of the 24 labels at 0.063 mm: 89.56% finer,
  versus 75.66% for the next sample. It is also 99.90% finer at 0.2 mm.
- Every held-out prediction is too coarse through the fine-size range. Ridge is
  closest at 58.19 EMD; gray 50 mm scores 93.88, and the other four predictions
  score 107.40.
- H030 is the first neighbor for every k-NN variant. G190 and H038 occur in four
  of five neighbor sets. No convex average of three other measured curves can
  reach F827's training-set maximum at 0.063 mm, so this k-NN design has a known
  extrapolation limit for this held-out extreme.
- The Motorola and Samsung frames differ visibly in framing, brightness, and
  color. Under RGB 100 mm, their separate predictions score 31.35 and 104.35
  EMD and disagree by 73.00 EMD. Ridge reduces their disagreement to 22.15 EMD.
  The pooled feature average gives 3/5 weight to the three Samsung frames.

### H038

- The measured curve is consistently coarser than all six predictions from
  0.02 through 20 mm. Normalized gray 100 mm is best at 93.38 EMD, followed by
  gray 50 and 150 mm at 97.59. Ridge is worst at 125.80.
- F827 and H038 are mutual or near neighbors in RGB and gray 100 mm even though
  their measured curves are 191.47 EMD apart. The plotted crops show substantial
  within-sample variation in the visible coarse pieces, especially across the
  eight center locations.
- The Motorola and Samsung views again differ visibly in framing and color.
  Camera consistency does not imply accuracy here: gray 100 mm camera predictions
  disagree by only 14.95 EMD, while their errors are 160.13 and 150.95 EMD.
  The pooled feature average gives 5/8 weight to the five Samsung frames.

### H374

- H374 is the only audited sample photographed with the Motorola Edge 60 Fusion
  and has no paired-camera diagnostic. Its three frames visibly contain many
  coarse particles, with large pieces moving into and out of the center crop.
- RGB 100 mm predicts a much finer curve and has the largest first-batch error,
  126.53 EMD. Gray 150 mm (24.16), ridge (26.72), and normalized gray 100 mm
  (28.41) are much closer; gray 50 and 100 mm remain above 71 EMD.
- Crop or normalization changes replace all three RGB neighbors. The nearest
  RGB squared standardized distance is 69.11, versus 3.88 for gray 150 mm. These
  distances are meaningful only within an experiment because the feature sets
  have different dimensions and scaling.

## Interpretation and next checks

The photographs establish scene differences, not their causes. Camera model,
lighting, framing, center location, photo count, and sample arrangement all vary
together. In particular, H374 cannot support a camera comparison, and none of
these observations tests performance on the iPhones used for the competition
test set.

The F827/H038 neighbor-label mismatch is evidence that the current summary
features do not reliably order these two soils by their measured curves. It is
not evidence that either label is wrong. Keep both samples and their labels in
validation unless independent provenance or laboratory evidence says otherwise.

Useful declared follow-ups are: compare equal-per-camera aggregation with the
current equal-per-photo average for paired-camera samples; retain a regression
candidate that can extrapolate for F827-like extremes; and test a fixed
multi-scale representation for H374-like heterogeneous coarse scenes. Judge each
on all 24 grouped folds and camera sensitivity, rather than choosing a rule from
these three cases.

Reproduce the audit after the first experiment batch with:

```bash
make audit
```
