# Model Final State

This file reflects the repository after the cleanup on 2026-03-22.

Large losing checkpoints and duplicate exports were deleted. The small summaries, histories, plots, and leaderboard JSON files were intentionally kept so the experiment record still exists even though most heavy model weights are gone.

## Cleanup Result

- Repository size dropped from about `24G` to about `3.9G`.
- Local `.git` dropped from about `4.2G` to about `110M` after removing temporary pack garbage and running git cleanup.
- `runpod_results` dropped from about `17G` to about `581M`.
- Remaining model artifacts in the repo: `10` files total.
- Of those `10`, `8` are retained trained model artifacts and `2` are MediaPipe `.task` utility models used for feature extraction.

## Hugging Face Weights Repo

The retained final weights were also uploaded to a dedicated Hugging Face model repo:

- Repo: [sanjuhs/nn-training-final-weights](https://huggingface.co/sanjuhs/nn-training-final-weights)
- Current visibility: `public`
- Uploaded payload: about `579MB` across the retained `8` trained model artifacts

This is the right place for the heavier model files. The GitHub repo can now stay lightweight while the final weights live in Hugging Face.

## GitHub Push Readiness

The repo is now much safer to commit and push normally:

- `runpod_results/` is now ignored in `.gitignore`, so the large local experiment folder should not get staged accidentally.
- Local model artifact directories like `V2A-over-training-old-nn/.../models/*.pth` and `*.onnx` are also ignored for future work.
- Because your branch was still `0` ahead and `0` behind `origin/main`, there was no need for a force-push or history rewrite just to make the next push small.

Important nuance:

- The current public GitHub history may still contain older tracked model artifacts from earlier commits.
- That does **not** make the next normal push huge, but it does mean fully erasing all historical weight blobs from GitHub would require a separate history-rewrite pass later.

## Models Currently Kept

These are the trained model artifacts that still exist in the repository.

| Model | Family | Files kept | Best val loss | Overall MAE | Mouth MAE | Jaw/open corr | Smile corr | Why it was kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber` | transformer | `best.pth` + `.onnx` | `0.406168` | `0.051743` | `0.03158` | `0.252572` | `0.355528` | Best overnight full run by recorded sweep score. |
| `overnight_20260321_205258__conv_d320_l12_b16` | transformer | `best.pth` + `.onnx` | `0.549753` | `0.049641` | `0.029998` | `0.229706` | `0.414111` | Strongest smile / expressiveness checkpoint. |
| `baseline_d192_l6` | transformer | `best.pth` + `.onnx` | `0.06394` | `0.062378` | `0.030038` | `0.214903` | `0.317995` | Best full-pipeline combined-dataset baseline retained as the clean reference model. |
| `tcn_model` | TCN | `best_tcn_model.pth` + `tcn_model.onnx` | `0.008304` | `0.06489` | `0.076518` | `0.209255` | `0.241591` | Best legacy TCN reference kept for historical comparison. |

## Exact Artifact Paths Still Present

### Retained trained model artifacts

- `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber_best.pth` (`195.1MB`)
- `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber.onnx` (`63.7MB`)
- `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d320_l12_b16/overnight_20260321_205258__conv_d320_l12_b16_best.pth` (`195.0MB`)
- `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d320_l12_b16/overnight_20260321_205258__conv_d320_l12_b16.onnx` (`63.7MB`)
- `runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6_best.pth` (`32.9MB`)
- `runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6.onnx` (`11.4MB`)
- `V2A-over-training-old-nn/2_architecture_training/models/best_tcn_model.pth` (`8.95MB`)
- `V2A-over-training-old-nn/2_architecture_training/models/tcn_model.onnx` (`1.59MB`)

### Utility model artifacts still present

- `V2A-over-training-old-nn/models/face_landmarker_v2_with_blendshapes.task` (`3.76MB`)
- `V2A-over-training-old-nn/1_data_cleaning/models/face_landmarker_v2_with_blendshapes.task` (`3.76MB`)

These two `.task` files are not part of the TCN-vs-transformer comparison. They are MediaPipe utility models used for extraction / preprocessing, so I left them alone.

## Transformer vs TCN Verdict

Short answer: yes, the transformer family won for the current project direction.

The careful version:

- The raw `val_loss` values are not apples-to-apples across the old TCN setup and the newer transformer runs, because the datasets, normalization, and loss definitions changed.
- So the fair comparison is not just the headline loss. The better comparison is the visually important metrics like mouth MAE, jaw correlation, and smile correlation.

On those metrics, the retained transformers beat the legacy TCN:

| Model | Mouth MAE | Jaw/open corr | Smile corr | Overall MAE |
| --- | --- | --- | --- | --- |
| `tcn_model` | `0.076518` | `0.209255` | `0.241591` | `0.06489` |
| `baseline_d192_l6` | `0.030038` | `0.214903` | `0.317995` | `0.062378` |
| `conv_d320_l12_b16` | `0.029998` | `0.229706` | `0.414111` | `0.049641` |
| `conv_d320_l12_b20_nadam_huber` | `0.03158` | `0.252572` | `0.355528` | `0.051743` |

Interpretation:

- Even the smaller retained transformer baseline is already better than the legacy TCN on mouth MAE and smile correlation.
- The `conv_d320_l12_b16` transformer is the strongest retained checkpoint for facial expressiveness.
- The `conv_d320_l12_b20_nadam_huber` transformer is the strongest retained checkpoint for the newer overnight composite objective.
- The legacy TCN still matters as a historical baseline, but it is no longer the best model family for the current work.

## Did The Old Overfits Beat The Transformers?

Based on the files that were actually in the repo, there is no recorded evidence that the old overfit/debug checkpoints beat the retained transformers.

Why:

- The old overfit-style artifacts such as `best_tcn_model_train_50`, `best_tcn_model_train_15`, `best_tcn_model_test_35`, and the old tiny-transformer checkpoints did not have a complete, fair held-out evaluation record in the repo that showed them outperforming the current winners.
- Some of those files were clearly debug / overfit / duplicate export artifacts rather than strong production candidates.
- Since they were not the strongest recorded models and they were not the current best path, I removed them.

So the practical verdict is:

- The retained transformer checkpoints are the best production-facing models left in the repo.
- The retained TCN is now only a historical reference, not the winner.
- The old overfit artifacts were not strong enough, or not well-enough proven, to justify keeping them.

## What Was Removed

The cleanup removed the large losing model artifacts from these families:

- larger conv transformer variants such as `d512` and `d640`
- conv-gated transformer variants
- conformer variants
- multiscale variants
- smoke / corrvar / older baseline artifact copies
- old tiny-transformer artifacts
- old TCN duplicate exports and overfit/debug checkpoints

Important:

- I kept the small experiment records like `summary.json`, `history.json`, `curves.json`, leaderboard files, and plots wherever possible.
- So the training history is still available, but the large losing model weights are gone.

## Current Recommendation

If you want to keep the repo lean and still useful, this is a good final state:

- `conv_d320_l12_b20_nadam_huber` as the main best checkpoint
- `conv_d320_l12_b16` as the expressive alternate checkpoint
- `baseline_d192_l6` as the clean baseline reference
- `tcn_model` as the legacy historical reference

That gives you the strongest current transformer path, a strong expressive variant, a clean baseline, and one old TCN for comparison, without carrying the huge losing model files anymore.

The same retained weights are now also backed up in Hugging Face, so you can keep GitHub focused on code, docs, and small metadata rather than heavy model binaries.
