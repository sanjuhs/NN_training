# Model Inventory

Generated from the actual metadata, summary, leaderboard, and history files currently present in this repository.

## What To Do With The Huge `.pth` Files

- Keep the markdown inventory, then keep only the shortlist of winning checkpoints plus any baseline you still need for comparison.
- Delete completed `*_last.pth` checkpoints first if you want the fastest safe cleanup. That alone would reclaim about `8.1GB` from `runpod_results`.
- For losing runs, you usually only need one of these: the best checkpoint, an ONNX export, or the metrics in this report. You almost never need both `best` and `last` once a run is finished.
- Move any archive-worthy checkpoints out of the git repo entirely. Keep them on external storage, Hugging Face, or another artifact store instead of inside the public repository.
- Current Runpod artifact footprint in this repo: `runpod_results = 16.4GB` with about `7.4GB` of unique `best` checkpoints and `8.1GB` of unique `last` checkpoints.

## Recommended Keep Shortlist

- Best overnight full run by recorded sweep score: `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber` in `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber` with score `0.382858` and best val loss `0.406168`.
- Best overnight pilot worth resuming if you want to continue exploring: `overnight_20260322_065411__multiscale_d384_l12_b12_huber` in `runpod_results/overnight_sweeps/overnight_20260322_065411/overnight_20260322_065411__multiscale_d384_l12_b12_huber` with score `0.405873`.
- Best full-pipeline combined-dataset baseline in this repo: `baseline_d192_l6` with best val loss `0.06394`.
- Best legacy local model by available val loss: `tcn_model` with best val loss `0.008304`.
- If facial expressiveness is the deciding factor, keep the strong `conv_d320_l12` family checkpoints even if you delete most larger models.

## Comparison Notes

- `score` is only available for the overnight sweep leaderboard entries.
- `best_val_loss` is not always directly comparable across older L1-only runs and newer Huber/composite sweeps, so compare both the loss family and the correlation metrics.
- When a metadata JSON was missing, the report keeps the run but marks the missing fields as `n/a` instead of guessing.

## Scored Models

| Area | Name | Phase | Variant | IO | Shape | Best val loss | Overall MAE | Mouth MAE | Smile corr | Score | Folder size | Best `.pth` | Last `.pth` | ONNX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overnight_sweeps/overnight_20260322_053753 | `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber` | `full` | `conv_transformer` | `80 -> 59` | `d320, l12, h8, ffn1280, k9` | `0.406168` | `0.051743` | `0.03158` | `0.355528` | `0.382858` | `436.9MB` | `186.1MB` | `186.1MB` | `63.7MB` |
| overnight_sweeps/overnight_20260322_053753 | `overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber` | `full` | `conv_gated_transformer` | `80 -> 59` | `d512, l16, h8, ffn2048, k9` | `0.422205` | `0.051814` | `0.031852` | `0.332957` | `0.399496` | `1.9GB` | `823.2MB` | `823.2MB` | `277.0MB` |
| overnight_sweeps/overnight_20260322_065411 | `overnight_20260322_065411__multiscale_d384_l12_b12_huber` | `pilot` | `multiscale_transformer` | `n/a` | `d384, l12, h8, ffn1536, k9` | `0.425908` | `0.052681` | `0.032171` | `0.283525` | `0.405873` | `870.8MB` | `434.9MB` | `434.9MB` | `0B` |
| overnight_sweeps/overnight_20260322_053753 | `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber` | `pilot` | `conv_transformer` | `80 -> 59` | `d320, l12, h8, ffn1280, k9` | `0.424152` | `0.053482` | `0.03234` | `0.218626` | `0.409218` | `436.9MB` | `186.1MB` | `186.1MB` | `63.7MB` |
| overnight_sweeps/overnight_20260322_053753 | `overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber` | `pilot` | `conv_gated_transformer` | `80 -> 59` | `d512, l16, h8, ffn2048, k9` | `0.435818` | `0.052837` | `0.032246` | `0.245969` | `0.418319` | `1.9GB` | `823.2MB` | `823.2MB` | `277.0MB` |
| overnight_sweeps/overnight_20260322_053753 | `overnight_20260322_053753__conv_d512_l16_b10_radam_huber` | `pilot` | `conv_transformer` | `n/a` | `d512, l16, h8, ffn2048, k9` | `0.44549` | `0.055793` | `0.035472` | `0.225034` | `0.429826` | `1.2GB` | `630.7MB` | `630.7MB` | `0B` |
| overnight_sweeps/overnight_20260322_065411 | `overnight_20260322_065411__conformer_d384_l12_b12_huber` | `pilot` | `conformer_transformer` | `n/a` | `d384, l12, h8, ffn1536, k15` | `0.445175` | `0.053851` | `0.033485` | `0.185872` | `0.432139` | `943.0MB` | `471.0MB` | `471.0MB` | `0B` |
| overnight_sweeps/overnight_20260322_065411 | `overnight_20260322_065411__conformer_d512_l16_b8_nadam_huber` | `pilot` | `conformer_transformer` | `n/a` | `d512, l16, h8, ffn2048, k15` | `0.458553` | `0.053572` | `0.033016` | `0.230206` | `0.442881` | `2.2GB` | `1.1GB` | `1.1GB` | `0B` |
| overnight_sweeps/overnight_20260321_205258 | `overnight_20260321_205258__conv_d320_l12_b16` | `full` | `conv_transformer` | `80 -> 59` | `d320, l12, h8, ffn1280, k9` | `0.549753` | `0.049641` | `0.029998` | `0.414111` | `0.523781` | `436.7MB` | `186.0MB` | `186.0MB` | `63.7MB` |
| overnight_sweeps/overnight_20260321_205258 | `overnight_20260321_205258__conv_d512_l16_b10` | `full` | `conv_transformer` | `80 -> 59` | `d512, l16, h8, ffn2048, k9` | `0.554401` | `0.05019` | `0.030381` | `0.380404` | `0.530203` | `1.4GB` | `630.7MB` | `630.7MB` | `212.9MB` |
| overnight_sweeps/overnight_20260321_205258 | `overnight_20260321_205258__conv_d320_l12_b16` | `pilot` | `conv_transformer` | `80 -> 59` | `d320, l12, h8, ffn1280, k9` | `0.573302` | `0.0514` | `0.03095` | `0.225517` | `0.557817` | `436.7MB` | `186.0MB` | `186.0MB` | `63.7MB` |
| overnight_sweeps/overnight_20260321_205258 | `overnight_20260321_205258__conv_d512_l16_b10` | `pilot` | `conv_transformer` | `80 -> 59` | `d512, l16, h8, ffn2048, k9` | `0.576515` | `0.052205` | `0.031232` | `0.216509` | `0.561138` | `1.4GB` | `630.7MB` | `630.7MB` | `212.9MB` |
| overnight_sweeps/overnight_20260321_205258 | `overnight_20260321_205258__conv_d640_l18_k15_b8` | `pilot` | `conv_transformer` | `n/a` | `d640, l18, h10, ffn2560, k15` | `0.587657` | `0.052077` | `0.031314` | `0.198964` | `0.572104` | `2.2GB` | `1.1GB` | `1.1GB` | `0B` |
| overnight_sweeps/overnight_20260321_205258 | `overnight_20260321_205258__convgated_d512_l16_b10` | `pilot` | `conv_gated_transformer` | `n/a` | `d512, l16, h8, ffn2048, k9` | `0.591581` | `0.051444` | `0.031335` | `0.208477` | `0.576587` | `1.6GB` | `823.1MB` | `823.1MB` | `0B` |
| overnight_sweeps/overnight_20260322_053753 | `overnight_20260322_053753__gated_d384_l12_b16_nadam` | `pilot` | `gated_transformer` | `n/a` | `d384, l12, h8, ffn1536, k9` | `0.60611` | `0.052721` | `0.031782` | `0.152133` | `0.59475` | `656.0MB` | `327.5MB` | `327.5MB` | `0B` |
| full_pipeline/runs/baseline_d192_l6 | `baseline_d192_l6` | `single` | `baseline` | `80 -> 59` | `d192, l6, h6, ffn768, k9` | `0.06394` | `0.062378` | `0.030038` | `0.317995` | `n/a` | `74.4MB` | `31.4MB` | `31.4MB` | `11.4MB` |
| full_pipeline/runs/conv_transformer_d224_l8 | `conv_transformer_d224_l8` | `single` | `conv_transformer` | `80 -> 59` | `d224, l8, h8, ffn896, k9` | `0.064167` | `0.062618` | `0.029966` | `0.294647` | `n/a` | `144.6MB` | `61.4MB` | `61.4MB` | `21.6MB` |
| full_pipeline/runs/gated_transformer_d224_l8 | `gated_transformer_d224_l8` | `single` | `gated_transformer` | `80 -> 59` | `d224, l8, h8, ffn896, k9` | `0.064606` | `0.063048` | `0.030202` | `0.267998` | `n/a` | `176.2MB` | `74.9MB` | `74.9MB` | `26.1MB` |
| current_baseline_existing | `baseline_existing` | `single` | `baseline` | `80 -> 59` | `d192, l6, h6, ffn768, k9` | `0.076364` | `0.074656` | `0.035716` | `0.003066` | `n/a` | `74.4MB` | `31.4MB` | `31.4MB` | `11.4MB` |
| overnight_sweeps/overnight_20260322_065411 | `overnight_20260322_065411__multiscale_d512_l16_b8_radam_huber` | `summary_only` | `n/a` | `n/a` | `n/a` | `0.459387` | `0.054704` | `0.033699` | `0.209221` | `n/a` | `1.3GB` | `277.3MB` | `1.0GB` | `0B` |
| conv_transformer_d320_l12_corrvar_std | `conv_transformer_d320_l12_corrvar_std` | `single` | `conv_transformer` | `80 -> 59` | `d320, l12, h8, ffn1280, k9` | `0.523375` | `0.050086` | `0.031807` | `0.347757` | `n/a` | `436.7MB` | `186.0MB` | `186.0MB` | `63.7MB` |
| conv_transformer_d768_l20_b20_quick_v2 | `conv_transformer_d768_l20_b20_quick_v2` | `single` | `n/a` | `n/a` | `n/a` | `0.543007` | `0.051662` | `0.031311` | `0.200154` | `n/a` | `7.0KB` | `0B` | `0B` | `0B` |
| conv_transformer_smoke | `conv_transformer_smoke` | `single` | `conv_transformer` | `80 -> 59` | `d320, l12, h8, ffn1280, k9` | `0.70455` | `0.058267` | `0.055976` | `0.061824` | `n/a` | `436.4MB` | `185.9MB` | `185.9MB` | `63.7MB` |
| legacy_local | `tcn_model` | `legacy` | `TCN_Audio_to_Blendshapes` | `80 -> 59` | `n/a` | `0.008304` | `0.06489` | `0.076518` | `0.241591` | `n/a` | `10.1MB` | `0B` | `0B` | `1.5MB` |
| legacy_local | `tiny_transformer_full10s_l1` | `legacy` | `tiny_transformer_encoder` | `80 -> 59` | `d128, l3, h4, ffn256` | `0.072877` | `n/a` | `n/a` | `n/a` | `n/a` | `7.3MB` | `5.0MB` | `0B` | `2.3MB` |
| legacy_local | `tiny_transformer_full10s_l1temp` | `legacy` | `n/a` | `n/a` | `n/a` | `0.074392` | `n/a` | `n/a` | `n/a` | `n/a` | `5.0MB` | `5.0MB` | `0B` | `0B` |
| legacy_local | `tiny_transformer_l1` | `legacy` | `tiny_transformer_encoder` | `80 -> 59` | `d128, l3, h4, ffn256` | `0.082626` | `n/a` | `n/a` | `n/a` | `n/a` | `7.4MB` | `5.0MB` | `0B` | `2.3MB` |
| legacy_local | `tiny_transformer_l1temp` | `legacy` | `tiny_transformer_encoder` | `80 -> 59` | `d128, l3, h4, ffn256` | `0.082846` | `n/a` | `n/a` | `n/a` | `n/a` | `7.4MB` | `5.0MB` | `0B` | `2.3MB` |

## Detailed Run Notes

### `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber`

- Area: `overnight_sweeps/overnight_20260322_053753`
- Phase: `full`
- Source path: `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber`
- Metadata file: `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber.json`
- Variant / architecture: `conv_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d320, l12, h8, ffn1280, k9`
- Config extras: `batch_size=20; eval_batch_size=4; optimizer=nadam; base_loss=huber; dropout=0.08; corr_weight=0.2; variance_weight=0.06; warmup_epochs=6; patience=20; output_space=natural_range; num_parameters=16223099; model_size_mb=61.886`
- Best epoch: `26`
- Best val loss: `0.406168`
- Sweep score: `0.382858`
- Artifacts: total `436.9MB`, best checkpoints `186.1MB`, last checkpoints `186.1MB`, ONNX `63.7MB`
- Recorded metrics:
  - `train_loss` = `0.408099`
  - `train_l1_loss` = `0.256907`
  - `train_temporal_loss` = `0.112112`
  - `train_corr_loss` = `0.616144`
  - `train_var_loss` = `0.372628`
  - `train_grad_norm` = `0.233352`
  - `val_loss` = `0.406168`
  - `val_l1_loss` = `0.211776`
  - `val_temporal_loss` = `0.04046`
  - `val_corr_loss` = `0.862347`
  - `val_var_loss` = `0.331646`
  - `val_overall_mae` = `0.051743`
  - `val_mouth_mae` = `0.03158`
  - `val_jaw_open_mae` = `0.088285`
  - `val_mouth_close_mae` = `0.001179`
  - `val_smile_mae` = `0.086015`
  - `val_pose_mae` = `0.060751`
  - `val_jaw_open_corr` = `0.252572`
  - `val_mouth_close_corr` = `0.035385`
  - `val_smile_corr` = `0.355528`
  - `val_mouth_jaw_corr_mean` = `0.179194`
  - `val_overall_blendshape_corr_mean` = `0.215758`
  - `learning_rate` = `0.000139`
- Config notes: `Large-batch control with NAdam and Huber.`

### `overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber`

- Area: `overnight_sweeps/overnight_20260322_053753`
- Phase: `full`
- Source path: `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber`
- Metadata file: `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber/overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber.json`
- Variant / architecture: `conv_gated_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d512, l16, h8, ffn2048, k9`
- Config extras: `batch_size=10; eval_batch_size=4; optimizer=nadam; base_loss=huber; dropout=0.08; corr_weight=0.22; variance_weight=0.06; warmup_epochs=8; patience=20; output_space=natural_range; num_parameters=71885371; model_size_mb=274.221`
- Best epoch: `24`
- Best val loss: `0.422205`
- Sweep score: `0.399496`
- Artifacts: total `1.9GB`, best checkpoints `823.2MB`, last checkpoints `823.2MB`, ONNX `277.0MB`
- Recorded metrics:
  - `train_loss` = `0.40862`
  - `train_l1_loss` = `0.249262`
  - `train_temporal_loss` = `0.106808`
  - `train_corr_loss` = `0.60001`
  - `train_var_loss` = `0.366916`
  - `train_grad_norm` = `0.320342`
  - `val_loss` = `0.422205`
  - `val_l1_loss` = `0.212107`
  - `val_temporal_loss` = `0.040526`
  - `val_corr_loss` = `0.862737`
  - `val_var_loss` = `0.3045`
  - `val_overall_mae` = `0.051814`
  - `val_mouth_mae` = `0.031852`
  - `val_jaw_open_mae` = `0.087375`
  - `val_mouth_close_mae` = `0.001243`
  - `val_smile_mae` = `0.087907`
  - `val_pose_mae` = `0.060477`
  - `val_jaw_open_corr` = `0.240859`
  - `val_mouth_close_corr` = `0.057144`
  - `val_smile_corr` = `0.332957`
  - `val_mouth_jaw_corr_mean` = `0.178743`
  - `val_overall_blendshape_corr_mean` = `0.221037`
  - `learning_rate` = `0.000109`
- Config notes: `Hybrid conv+gated model with NAdam and Huber.`

### `overnight_20260322_065411__multiscale_d384_l12_b12_huber`

- Area: `overnight_sweeps/overnight_20260322_065411`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260322_065411/overnight_20260322_065411__multiscale_d384_l12_b12_huber`
- Variant / architecture: `multiscale_transformer`
- Input -> output: `n/a`
- Architecture shape: `d384, l12, h8, ffn1536, k9`
- Config extras: `batch_size=12; eval_batch_size=4; optimizer=adamw; base_loss=huber; dropout=0.08; corr_weight=0.22; variance_weight=0.06; warmup_epochs=8; patience=24`
- Best epoch: `10`
- Best val loss: `0.425908`
- Sweep score: `0.405873`
- Artifacts: total `870.8MB`, best checkpoints `434.9MB`, last checkpoints `434.9MB`, ONNX `0B`
- Recorded metrics:
  - `train_loss` = `0.442241`
  - `train_l1_loss` = `0.268284`
  - `train_temporal_loss` = `0.114732`
  - `train_corr_loss` = `0.653202`
  - `train_var_loss` = `0.408612`
  - `train_grad_norm` = `0.236924`
  - `val_loss` = `0.425908`
  - `val_l1_loss` = `0.215057`
  - `val_temporal_loss` = `0.036325`
  - `val_corr_loss` = `0.849634`
  - `val_var_loss` = `0.368595`
  - `val_overall_mae` = `0.052681`
  - `val_mouth_mae` = `0.032171`
  - `val_jaw_open_mae` = `0.089181`
  - `val_mouth_close_mae` = `0.001214`
  - `val_smile_mae` = `0.090162`
  - `val_pose_mae` = `0.062607`
  - `val_jaw_open_corr` = `0.20432`
  - `val_mouth_close_corr` = `0.035272`
  - `val_smile_corr` = `0.283525`
  - `val_mouth_jaw_corr_mean` = `0.162376`
  - `val_overall_blendshape_corr_mean` = `0.202728`
  - `learning_rate` = `0.00015`
- Config notes: `Multi-scale local/global transformer fusion.`

### `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber`

- Area: `overnight_sweeps/overnight_20260322_053753`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber`
- Metadata file: `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber/overnight_20260322_053753__conv_d320_l12_b20_nadam_huber.json`
- Variant / architecture: `conv_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d320, l12, h8, ffn1280, k9`
- Config extras: `batch_size=20; eval_batch_size=4; optimizer=nadam; base_loss=huber; dropout=0.08; corr_weight=0.2; variance_weight=0.06; warmup_epochs=6; patience=20; output_space=natural_range; num_parameters=16223099; model_size_mb=61.886`
- Best epoch: `6`
- Best val loss: `0.424152`
- Sweep score: `0.409218`
- Artifacts: total `436.9MB`, best checkpoints `186.1MB`, last checkpoints `186.1MB`, ONNX `63.7MB`
- Recorded metrics:
  - `train_loss` = `0.46144`
  - `train_l1_loss` = `0.285582`
  - `train_temporal_loss` = `0.136459`
  - `train_corr_loss` = `0.711173`
  - `train_var_loss` = `0.446683`
  - `train_grad_norm` = `0.209629`
  - `val_loss` = `0.424152`
  - `val_l1_loss` = `0.219142`
  - `val_temporal_loss` = `0.039645`
  - `val_corr_loss` = `0.897929`
  - `val_var_loss` = `0.390696`
  - `val_overall_mae` = `0.053482`
  - `val_mouth_mae` = `0.03234`
  - `val_jaw_open_mae` = `0.088726`
  - `val_mouth_close_mae` = `0.001254`
  - `val_smile_mae` = `0.086948`
  - `val_pose_mae` = `0.065704`
  - `val_jaw_open_corr` = `0.087515`
  - `val_mouth_close_corr` = `0.026424`
  - `val_smile_corr` = `0.218626`
  - `val_mouth_jaw_corr_mean` = `0.118006`
  - `val_overall_blendshape_corr_mean` = `0.156088`
  - `learning_rate` = `0.00016`
- Config notes: `Large-batch control with NAdam and Huber.`

### `overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber`

- Area: `overnight_sweeps/overnight_20260322_053753`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber`
- Metadata file: `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber/overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber.json`
- Variant / architecture: `conv_gated_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d512, l16, h8, ffn2048, k9`
- Config extras: `batch_size=10; eval_batch_size=4; optimizer=nadam; base_loss=huber; dropout=0.08; corr_weight=0.22; variance_weight=0.06; warmup_epochs=8; patience=20; output_space=natural_range; num_parameters=71885371; model_size_mb=274.221`
- Best epoch: `6`
- Best val loss: `0.435818`
- Sweep score: `0.418319`
- Artifacts: total `1.9GB`, best checkpoints `823.2MB`, last checkpoints `823.2MB`, ONNX `277.0MB`
- Recorded metrics:
  - `train_loss` = `0.466686`
  - `train_l1_loss` = `0.280115`
  - `train_temporal_loss` = `0.128389`
  - `train_corr_loss` = `0.69906`
  - `train_var_loss` = `0.439317`
  - `train_grad_norm` = `0.306225`
  - `val_loss` = `0.435818`
  - `val_l1_loss` = `0.217162`
  - `val_temporal_loss` = `0.039745`
  - `val_corr_loss` = `0.881801`
  - `val_var_loss` = `0.377879`
  - `val_overall_mae` = `0.052837`
  - `val_mouth_mae` = `0.032246`
  - `val_jaw_open_mae` = `0.088606`
  - `val_mouth_close_mae` = `0.00155`
  - `val_smile_mae` = `0.08922`
  - `val_pose_mae` = `0.064069`
  - `val_jaw_open_corr` = `0.190291`
  - `val_mouth_close_corr` = `0.011432`
  - `val_smile_corr` = `0.245969`
  - `val_mouth_jaw_corr_mean` = `0.142665`
  - `val_overall_blendshape_corr_mean` = `0.18157`
  - `learning_rate` = `0.000093`
- Config notes: `Hybrid conv+gated model with NAdam and Huber.`

### `overnight_20260322_053753__conv_d512_l16_b10_radam_huber`

- Area: `overnight_sweeps/overnight_20260322_053753`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__conv_d512_l16_b10_radam_huber`
- Variant / architecture: `conv_transformer`
- Input -> output: `n/a`
- Architecture shape: `d512, l16, h8, ffn2048, k9`
- Config extras: `batch_size=10; eval_batch_size=4; optimizer=radam; base_loss=huber; dropout=0.08; corr_weight=0.22; variance_weight=0.06; warmup_epochs=8; patience=20`
- Best epoch: `6`
- Best val loss: `0.44549`
- Sweep score: `0.429826`
- Artifacts: total `1.2GB`, best checkpoints `630.7MB`, last checkpoints `630.7MB`, ONNX `0B`
- Recorded metrics:
  - `train_loss` = `0.47819`
  - `train_l1_loss` = `0.285575`
  - `train_temporal_loss` = `0.131349`
  - `train_corr_loss` = `0.721449`
  - `train_var_loss` = `0.455479`
  - `train_grad_norm` = `0.325584`
  - `val_loss` = `0.44549`
  - `val_l1_loss` = `0.225961`
  - `val_temporal_loss` = `0.03998`
  - `val_corr_loss` = `0.888396`
  - `val_var_loss` = `0.368061`
  - `val_overall_mae` = `0.055793`
  - `val_mouth_mae` = `0.035472`
  - `val_jaw_open_mae` = `0.087266`
  - `val_mouth_close_mae` = `0.001324`
  - `val_smile_mae` = `0.106994`
  - `val_pose_mae` = `0.06723`
  - `val_jaw_open_corr` = `0.150142`
  - `val_mouth_close_corr` = `0.014471`
  - `val_smile_corr` = `0.225034`
  - `val_mouth_jaw_corr_mean` = `0.125844`
  - `val_overall_blendshape_corr_mean` = `0.166518`
  - `learning_rate` = `0.000101`
- Config notes: `Wider conv model with RAdam and Huber.`

### `overnight_20260322_065411__conformer_d384_l12_b12_huber`

- Area: `overnight_sweeps/overnight_20260322_065411`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260322_065411/overnight_20260322_065411__conformer_d384_l12_b12_huber`
- Variant / architecture: `conformer_transformer`
- Input -> output: `n/a`
- Architecture shape: `d384, l12, h8, ffn1536, k15`
- Config extras: `batch_size=12; eval_batch_size=4; optimizer=adamw; base_loss=huber; dropout=0.08; corr_weight=0.22; variance_weight=0.06; warmup_epochs=8; patience=24`
- Best epoch: `7`
- Best val loss: `0.445175`
- Sweep score: `0.432139`
- Artifacts: total `943.0MB`, best checkpoints `471.0MB`, last checkpoints `471.0MB`, ONNX `0B`
- Recorded metrics:
  - `train_loss` = `0.464174`
  - `train_l1_loss` = `0.280283`
  - `train_temporal_loss` = `0.139624`
  - `train_corr_loss` = `0.689852`
  - `train_var_loss` = `0.419033`
  - `train_grad_norm` = `0.226233`
  - `val_loss` = `0.445175`
  - `val_l1_loss` = `0.220608`
  - `val_temporal_loss` = `0.033265`
  - `val_corr_loss` = `0.900958`
  - `val_var_loss` = `0.41154`
  - `val_overall_mae` = `0.053851`
  - `val_mouth_mae` = `0.033485`
  - `val_jaw_open_mae` = `0.090126`
  - `val_mouth_close_mae` = `0.001261`
  - `val_smile_mae` = `0.094757`
  - `val_pose_mae` = `0.067191`
  - `val_jaw_open_corr` = `0.099525`
  - `val_mouth_close_corr` = `-0.007983`
  - `val_smile_corr` = `0.185872`
  - `val_mouth_jaw_corr_mean` = `0.101255`
  - `val_overall_blendshape_corr_mean` = `0.153356`
  - `learning_rate` = `0.000133`
- Config notes: `Speech-oriented Conformer-style stack with Huber loss.`

### `overnight_20260322_065411__conformer_d512_l16_b8_nadam_huber`

- Area: `overnight_sweeps/overnight_20260322_065411`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260322_065411/overnight_20260322_065411__conformer_d512_l16_b8_nadam_huber`
- Variant / architecture: `conformer_transformer`
- Input -> output: `n/a`
- Architecture shape: `d512, l16, h8, ffn2048, k15`
- Config extras: `batch_size=8; eval_batch_size=2; optimizer=nadam; base_loss=huber; dropout=0.08; corr_weight=0.24; variance_weight=0.06; warmup_epochs=10; patience=28`
- Best epoch: `10`
- Best val loss: `0.458553`
- Sweep score: `0.442881`
- Artifacts: total `2.2GB`, best checkpoints `1.1GB`, last checkpoints `1.1GB`, ONNX `0B`
- Recorded metrics:
  - `train_loss` = `0.478142`
  - `train_l1_loss` = `0.279598`
  - `train_temporal_loss` = `0.119316`
  - `train_corr_loss` = `0.693706`
  - `train_var_loss` = `0.434813`
  - `train_grad_norm` = `0.270953`
  - `val_loss` = `0.458553`
  - `val_l1_loss` = `0.221051`
  - `val_temporal_loss` = `0.033734`
  - `val_corr_loss` = `0.886556`
  - `val_var_loss` = `0.384036`
  - `val_overall_mae` = `0.053572`
  - `val_mouth_mae` = `0.033016`
  - `val_jaw_open_mae` = `0.089046`
  - `val_mouth_close_mae` = `0.001656`
  - `val_smile_mae` = `0.091234`
  - `val_pose_mae` = `0.063307`
  - `val_jaw_open_corr` = `0.19361`
  - `val_mouth_close_corr` = `0.054225`
  - `val_smile_corr` = `0.230206`
  - `val_mouth_jaw_corr_mean` = `0.123485`
  - `val_overall_blendshape_corr_mean` = `0.162583`
  - `learning_rate` = `0.00012`
- Config notes: `Larger Conformer-style run with NAdam.`

### `overnight_20260321_205258__conv_d320_l12_b16`

- Area: `overnight_sweeps/overnight_20260321_205258`
- Phase: `full`
- Source path: `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d320_l12_b16`
- Metadata file: `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d320_l12_b16/overnight_20260321_205258__conv_d320_l12_b16.json`
- Variant / architecture: `conv_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d320, l12, h8, ffn1280, k9`
- Config extras: `batch_size=16; eval_batch_size=4; dropout=0.08; corr_weight=0.18; variance_weight=0.06; warmup_epochs=6; output_space=natural_range; num_parameters=16223099; model_size_mb=61.886`
- Best epoch: `23`
- Best val loss: `0.549753`
- Sweep score: `0.523781`
- Artifacts: total `436.7MB`, best checkpoints `186.0MB`, last checkpoints `186.0MB`, ONNX `63.7MB`
- Recorded metrics:
  - `train_loss` = `0.544512`
  - `train_l1_loss` = `0.41231`
  - `train_temporal_loss` = `0.106768`
  - `train_corr_loss` = `0.577734`
  - `train_var_loss` = `0.381186`
  - `train_grad_norm` = `0.256129`
  - `val_loss` = `0.549753`
  - `val_l1_loss` = `0.374738`
  - `val_temporal_loss` = `0.036641`
  - `val_corr_loss` = `0.860101`
  - `val_var_loss` = `0.306083`
  - `val_overall_mae` = `0.049641`
  - `val_mouth_mae` = `0.029998`
  - `val_jaw_open_mae` = `0.08744`
  - `val_mouth_close_mae` = `0.001007`
  - `val_smile_mae` = `0.07777`
  - `val_pose_mae` = `0.057645`
  - `val_jaw_open_corr` = `0.229706`
  - `val_mouth_close_corr` = `0.04921`
  - `val_smile_corr` = `0.414111`
  - `val_mouth_jaw_corr_mean` = `0.189234`
  - `val_overall_blendshape_corr_mean` = `0.234322`
  - `learning_rate` = `0.000163`
- Config notes: `Control rerun with warmup and larger batch.`

### `overnight_20260321_205258__conv_d512_l16_b10`

- Area: `overnight_sweeps/overnight_20260321_205258`
- Phase: `full`
- Source path: `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d512_l16_b10`
- Metadata file: `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d512_l16_b10/overnight_20260321_205258__conv_d512_l16_b10.json`
- Variant / architecture: `conv_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d512, l16, h8, ffn2048, k9`
- Config extras: `batch_size=10; eval_batch_size=4; dropout=0.08; corr_weight=0.18; variance_weight=0.06; warmup_epochs=8; output_space=natural_range; num_parameters=55075387; model_size_mb=210.096`
- Best epoch: `18`
- Best val loss: `0.554401`
- Sweep score: `0.530203`
- Artifacts: total `1.4GB`, best checkpoints `630.7MB`, last checkpoints `630.7MB`, ONNX `212.9MB`
- Recorded metrics:
  - `train_loss` = `0.549817`
  - `train_l1_loss` = `0.414844`
  - `train_temporal_loss` = `0.102655`
  - `train_corr_loss` = `0.589983`
  - `train_var_loss` = `0.394051`
  - `train_grad_norm` = `0.307259`
  - `val_loss` = `0.554401`
  - `val_l1_loss` = `0.37777`
  - `val_temporal_loss` = `0.036077`
  - `val_corr_loss` = `0.870029`
  - `val_var_loss` = `0.303688`
  - `val_overall_mae` = `0.05019`
  - `val_mouth_mae` = `0.030381`
  - `val_jaw_open_mae` = `0.086728`
  - `val_mouth_close_mae` = `0.001027`
  - `val_smile_mae` = `0.079602`
  - `val_pose_mae` = `0.058019`
  - `val_jaw_open_corr` = `0.220168`
  - `val_mouth_close_corr` = `0.053168`
  - `val_smile_corr` = `0.380404`
  - `val_mouth_jaw_corr_mean` = `0.178851`
  - `val_overall_blendshape_corr_mean` = `0.222539`
  - `learning_rate` = `0.000135`
- Config notes: `Wider/deeper conv transformer.`

### `overnight_20260321_205258__conv_d320_l12_b16`

- Area: `overnight_sweeps/overnight_20260321_205258`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d320_l12_b16`
- Metadata file: `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d320_l12_b16/overnight_20260321_205258__conv_d320_l12_b16.json`
- Variant / architecture: `conv_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d320, l12, h8, ffn1280, k9`
- Config extras: `batch_size=16; eval_batch_size=4; dropout=0.08; corr_weight=0.18; variance_weight=0.06; warmup_epochs=6; output_space=natural_range; num_parameters=16223099; model_size_mb=61.886`
- Best epoch: `6`
- Best val loss: `0.573302`
- Sweep score: `0.557817`
- Artifacts: total `436.7MB`, best checkpoints `186.0MB`, last checkpoints `186.0MB`, ONNX `63.7MB`
- Recorded metrics:
  - `train_loss` = `0.639565`
  - `train_l1_loss` = `0.475145`
  - `train_temporal_loss` = `0.120735`
  - `train_corr_loss` = `0.71596`
  - `train_var_loss` = `0.491842`
  - `train_grad_norm` = `0.260994`
  - `val_loss` = `0.573302`
  - `val_l1_loss` = `0.38636`
  - `val_temporal_loss` = `0.036008`
  - `val_corr_loss` = `0.891963`
  - `val_var_loss` = `0.409816`
  - `val_overall_mae` = `0.0514`
  - `val_mouth_mae` = `0.03095`
  - `val_jaw_open_mae` = `0.087674`
  - `val_mouth_close_mae` = `0.001137`
  - `val_smile_mae` = `0.081527`
  - `val_pose_mae` = `0.060187`
  - `val_jaw_open_corr` = `0.107983`
  - `val_mouth_close_corr` = `0.026732`
  - `val_smile_corr` = `0.225517`
  - `val_mouth_jaw_corr_mean` = `0.12365`
  - `val_overall_blendshape_corr_mean` = `0.157793`
  - `learning_rate` = `0.00018`
- Config notes: `Control rerun with warmup and larger batch.`

### `overnight_20260321_205258__conv_d512_l16_b10`

- Area: `overnight_sweeps/overnight_20260321_205258`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d512_l16_b10`
- Metadata file: `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d512_l16_b10/overnight_20260321_205258__conv_d512_l16_b10.json`
- Variant / architecture: `conv_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d512, l16, h8, ffn2048, k9`
- Config extras: `batch_size=10; eval_batch_size=4; dropout=0.08; corr_weight=0.18; variance_weight=0.06; warmup_epochs=8; output_space=natural_range; num_parameters=55075387; model_size_mb=210.096`
- Best epoch: `4`
- Best val loss: `0.576515`
- Sweep score: `0.561138`
- Artifacts: total `1.4GB`, best checkpoints `630.7MB`, last checkpoints `630.7MB`, ONNX `212.9MB`
- Recorded metrics:
  - `train_loss` = `0.651576`
  - `train_l1_loss` = `0.481622`
  - `train_temporal_loss` = `0.120809`
  - `train_corr_loss` = `0.739985`
  - `train_var_loss` = `0.511934`
  - `train_grad_norm` = `0.365219`
  - `val_loss` = `0.576515`
  - `val_l1_loss` = `0.391115`
  - `val_temporal_loss` = `0.036864`
  - `val_corr_loss` = `0.889041`
  - `val_var_loss` = `0.392169`
  - `val_overall_mae` = `0.052205`
  - `val_mouth_mae` = `0.031232`
  - `val_jaw_open_mae` = `0.087396`
  - `val_mouth_close_mae` = `0.001086`
  - `val_smile_mae` = `0.083486`
  - `val_pose_mae` = `0.061163`
  - `val_jaw_open_corr` = `0.165816`
  - `val_mouth_close_corr` = `0.043517`
  - `val_smile_corr` = `0.216509`
  - `val_mouth_jaw_corr_mean` = `0.126461`
  - `val_overall_blendshape_corr_mean` = `0.159162`
  - `learning_rate` = `0.000077`
- Config notes: `Wider/deeper conv transformer.`

### `overnight_20260321_205258__conv_d640_l18_k15_b8`

- Area: `overnight_sweeps/overnight_20260321_205258`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__conv_d640_l18_k15_b8`
- Variant / architecture: `conv_transformer`
- Input -> output: `n/a`
- Architecture shape: `d640, l18, h10, ffn2560, k15`
- Config extras: `batch_size=8; eval_batch_size=2; dropout=0.08; corr_weight=0.2; variance_weight=0.06; warmup_epochs=8`
- Best epoch: `6`
- Best val loss: `0.587657`
- Sweep score: `0.572104`
- Artifacts: total `2.2GB`, best checkpoints `1.1GB`, last checkpoints `1.1GB`, ONNX `0B`
- Recorded metrics:
  - `train_loss` = `0.641057`
  - `train_l1_loss` = `0.466307`
  - `train_temporal_loss` = `0.112776`
  - `train_corr_loss` = `0.700831`
  - `train_var_loss` = `0.482415`
  - `train_grad_norm` = `0.412861`
  - `val_loss` = `0.587657`
  - `val_l1_loss` = `0.389922`
  - `val_temporal_loss` = `0.035017`
  - `val_corr_loss` = `0.865373`
  - `val_var_loss` = `0.381827`
  - `val_overall_mae` = `0.052077`
  - `val_mouth_mae` = `0.031314`
  - `val_jaw_open_mae` = `0.086694`
  - `val_mouth_close_mae` = `0.001096`
  - `val_smile_mae` = `0.08605`
  - `val_pose_mae` = `0.061107`
  - `val_jaw_open_corr` = `0.208716`
  - `val_mouth_close_corr` = `0.022688`
  - `val_smile_corr` = `0.198964`
  - `val_mouth_jaw_corr_mean` = `0.134974`
  - `val_overall_blendshape_corr_mean` = `0.173069`
  - `learning_rate` = `0.000085`
- Config notes: `Deeper model with wider local kernel.`

### `overnight_20260321_205258__convgated_d512_l16_b10`

- Area: `overnight_sweeps/overnight_20260321_205258`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260321_205258/overnight_20260321_205258__convgated_d512_l16_b10`
- Variant / architecture: `conv_gated_transformer`
- Input -> output: `n/a`
- Architecture shape: `d512, l16, h8, ffn2048, k9`
- Config extras: `batch_size=10; eval_batch_size=4; dropout=0.08; corr_weight=0.2; variance_weight=0.06; warmup_epochs=8`
- Best epoch: `6`
- Best val loss: `0.591581`
- Sweep score: `0.576587`
- Artifacts: total `1.6GB`, best checkpoints `823.1MB`, last checkpoints `823.1MB`, ONNX `0B`
- Recorded metrics:
  - `train_loss` = `0.646597`
  - `train_l1_loss` = `0.470111`
  - `train_temporal_loss` = `0.116051`
  - `train_corr_loss` = `0.708149`
  - `train_var_loss` = `0.48423`
  - `train_grad_norm` = `0.374013`
  - `val_loss` = `0.591581`
  - `val_l1_loss` = `0.388366`
  - `val_temporal_loss` = `0.035613`
  - `val_corr_loss` = `0.887781`
  - `val_var_loss` = `0.397974`
  - `val_overall_mae` = `0.051444`
  - `val_mouth_mae` = `0.031335`
  - `val_jaw_open_mae` = `0.087364`
  - `val_mouth_close_mae` = `0.001225`
  - `val_smile_mae` = `0.085332`
  - `val_pose_mae` = `0.060181`
  - `val_jaw_open_corr` = `0.150373`
  - `val_mouth_close_corr` = `0.00084`
  - `val_smile_corr` = `0.208477`
  - `val_mouth_jaw_corr_mean` = `0.119853`
  - `val_overall_blendshape_corr_mean` = `0.168686`
  - `learning_rate` = `0.000101`
- Config notes: `Conv mixer plus gated FFN hybrid.`

### `overnight_20260322_053753__gated_d384_l12_b16_nadam`

- Area: `overnight_sweeps/overnight_20260322_053753`
- Phase: `pilot`
- Source path: `runpod_results/overnight_sweeps/overnight_20260322_053753/overnight_20260322_053753__gated_d384_l12_b16_nadam`
- Variant / architecture: `gated_transformer`
- Input -> output: `n/a`
- Architecture shape: `d384, l12, h8, ffn1536, k9`
- Config extras: `batch_size=16; eval_batch_size=4; optimizer=nadam; base_loss=l1; dropout=0.08; corr_weight=0.2; variance_weight=0.06; warmup_epochs=6; patience=20`
- Best epoch: `6`
- Best val loss: `0.60611`
- Sweep score: `0.59475`
- Artifacts: total `656.0MB`, best checkpoints `327.5MB`, last checkpoints `327.5MB`, ONNX `0B`
- Recorded metrics:
  - `train_loss` = `0.661947`
  - `train_l1_loss` = `0.479583`
  - `train_temporal_loss` = `0.119413`
  - `train_corr_loss` = `0.73083`
  - `train_var_loss` = `0.50379`
  - `train_grad_norm` = `0.308732`
  - `val_loss` = `0.60611`
  - `val_l1_loss` = `0.399127`
  - `val_temporal_loss` = `0.038023`
  - `val_corr_loss` = `0.904389`
  - `val_var_loss` = `0.403401`
  - `val_overall_mae` = `0.052721`
  - `val_mouth_mae` = `0.031782`
  - `val_jaw_open_mae` = `0.090739`
  - `val_mouth_close_mae` = `0.001249`
  - `val_smile_mae` = `0.082303`
  - `val_pose_mae` = `0.062294`
  - `val_jaw_open_corr` = `0.08887`
  - `val_mouth_close_corr` = `0.0132`
  - `val_smile_corr` = `0.152133`
  - `val_mouth_jaw_corr_mean` = `0.090356`
  - `val_overall_blendshape_corr_mean` = `0.145717`
  - `learning_rate` = `0.00016`
- Config notes: `Pure gated FFN transformer with NAdam.`

### `baseline_d192_l6`

- Area: `full_pipeline/runs/baseline_d192_l6`
- Phase: `single`
- Source path: `runpod_results/full_pipeline/runs/baseline_d192_l6`
- Metadata file: `runpod_results/full_pipeline/runs/baseline_d192_l6/baseline_d192_l6.json`
- Variant / architecture: `baseline`
- Input -> output: `80 -> 59`
- Architecture shape: `d192, l6, h6, ffn768, k9`
- Config extras: `dropout=0.1; num_parameters=2733947; model_size_mb=10.429`
- Best epoch: `24`
- Best val loss: `0.06394`
- Artifacts: total `74.4MB`, best checkpoints `31.4MB`, last checkpoints `31.4MB`, ONNX `11.4MB`
- Recorded metrics:
  - `train_loss` = `0.06699`
  - `train_l1_loss` = `0.066706`
  - `train_temporal_loss` = `0.014204`
  - `train_grad_norm` = `0.039925`
  - `val_loss` = `0.06394`
  - `val_l1_loss` = `0.063836`
  - `val_temporal_loss` = `0.005232`
  - `val_overall_mae` = `0.062378`
  - `val_mouth_mae` = `0.030038`
  - `val_jaw_open_mae` = `0.08626`
  - `val_mouth_close_mae` = `0.001004`
  - `val_smile_mae` = `0.079053`
  - `val_pose_mae` = `0.166057`
  - `val_jaw_open_corr` = `0.214903`
  - `val_mouth_close_corr` = `0.04932`
  - `val_smile_corr` = `0.317995`
  - `learning_rate` = `0.000019`

### `conv_transformer_d224_l8`

- Area: `full_pipeline/runs/conv_transformer_d224_l8`
- Phase: `single`
- Source path: `runpod_results/full_pipeline/runs/conv_transformer_d224_l8`
- Metadata file: `runpod_results/full_pipeline/runs/conv_transformer_d224_l8/conv_transformer_d224_l8.json`
- Variant / architecture: `conv_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d224, l8, h8, ffn896, k9`
- Config extras: `dropout=0.1; num_parameters=5347611; model_size_mb=20.4`
- Best epoch: `23`
- Best val loss: `0.064167`
- Artifacts: total `144.6MB`, best checkpoints `61.4MB`, last checkpoints `61.4MB`, ONNX `21.6MB`
- Recorded metrics:
  - `train_loss` = `0.065996`
  - `train_l1_loss` = `0.065733`
  - `train_temporal_loss` = `0.013166`
  - `train_grad_norm` = `0.040188`
  - `val_loss` = `0.064167`
  - `val_l1_loss` = `0.064077`
  - `val_temporal_loss` = `0.00452`
  - `val_overall_mae` = `0.062618`
  - `val_mouth_mae` = `0.029966`
  - `val_jaw_open_mae` = `0.086758`
  - `val_mouth_close_mae` = `0.000991`
  - `val_smile_mae` = `0.079395`
  - `val_pose_mae` = `0.168358`
  - `val_jaw_open_corr` = `0.176654`
  - `val_mouth_close_corr` = `0.059122`
  - `val_smile_corr` = `0.294647`
  - `learning_rate` = `0.000001`

### `gated_transformer_d224_l8`

- Area: `full_pipeline/runs/gated_transformer_d224_l8`
- Phase: `single`
- Source path: `runpod_results/full_pipeline/runs/gated_transformer_d224_l8`
- Metadata file: `runpod_results/full_pipeline/runs/gated_transformer_d224_l8/gated_transformer_d224_l8.json`
- Variant / architecture: `gated_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d224, l8, h8, ffn896, k9`
- Config extras: `dropout=0.1; num_parameters=6535707; model_size_mb=24.932`
- Best epoch: `19`
- Best val loss: `0.064606`
- Artifacts: total `176.2MB`, best checkpoints `74.9MB`, last checkpoints `74.9MB`, ONNX `26.1MB`
- Recorded metrics:
  - `train_loss` = `0.068787`
  - `train_l1_loss` = `0.068518`
  - `train_temporal_loss` = `0.013431`
  - `train_grad_norm` = `inf`
  - `val_loss` = `0.064606`
  - `val_l1_loss` = `0.064506`
  - `val_temporal_loss` = `0.005003`
  - `val_overall_mae` = `0.063048`
  - `val_mouth_mae` = `0.030202`
  - `val_jaw_open_mae` = `0.086433`
  - `val_mouth_close_mae` = `0.001004`
  - `val_smile_mae` = `0.080163`
  - `val_pose_mae` = `0.16938`
  - `val_jaw_open_corr` = `0.205233`
  - `val_mouth_close_corr` = `0.053632`
  - `val_smile_corr` = `0.267998`
  - `learning_rate` = `0.000021`

### `baseline_existing`

- Area: `current_baseline_existing`
- Phase: `single`
- Source path: `runpod_results/current_baseline_existing`
- Metadata file: `runpod_results/current_baseline_existing/models/baseline_existing_best.json`
- Variant / architecture: `baseline`
- Input -> output: `80 -> 59`
- Architecture shape: `d192, l6, h6, ffn768, k9`
- Config extras: `dropout=0.1; num_parameters=2733947; model_size_mb=10.429`
- Best epoch: `16`
- Best val loss: `0.076364`
- Artifacts: total `74.4MB`, best checkpoints `31.4MB`, last checkpoints `31.4MB`, ONNX `11.4MB`
- Recorded metrics:
  - `train_loss` = `0.064622`
  - `train_l1_loss` = `0.064353`
  - `train_temporal_loss` = `0.013425`
  - `train_grad_norm` = `0.047131`
  - `val_loss` = `0.076364`
  - `val_l1_loss` = `0.076245`
  - `val_temporal_loss` = `0.005926`
  - `val_overall_mae` = `0.074656`
  - `val_mouth_mae` = `0.035716`
  - `val_jaw_open_mae` = `0.106573`
  - `val_mouth_close_mae` = `0.001227`
  - `val_smile_mae` = `0.102479`
  - `val_pose_mae` = `0.178517`
  - `val_jaw_open_corr` = `-0.003674`
  - `val_mouth_close_corr` = `0`
  - `val_smile_corr` = `0.003066`
  - `learning_rate` = `0.000019`

### `overnight_20260322_065411__multiscale_d512_l16_b8_radam_huber`

- Area: `overnight_sweeps/overnight_20260322_065411`
- Phase: `summary_only`
- Source path: `runpod_results/overnight_sweeps/overnight_20260322_065411/overnight_20260322_065411__multiscale_d512_l16_b8_radam_huber`
- Variant / architecture: `n/a`
- Input -> output: `n/a`
- Architecture shape: `n/a`
- Best epoch: `5`
- Best val loss: `0.459387`
- Artifacts: total `1.3GB`, best checkpoints `277.3MB`, last checkpoints `1.0GB`, ONNX `0B`
- Recorded metrics:
  - `train_loss` = `0.498666`
  - `train_l1_loss` = `0.287988`
  - `train_temporal_loss` = `0.134488`
  - `train_corr_loss` = `0.733338`
  - `train_var_loss` = `0.465872`
  - `train_grad_norm` = `0.351003`
  - `val_loss` = `0.459387`
  - `val_l1_loss` = `0.22258`
  - `val_temporal_loss` = `0.040554`
  - `val_corr_loss` = `0.886716`
  - `val_var_loss` = `0.366129`
  - `val_overall_mae` = `0.054704`
  - `val_mouth_mae` = `0.033699`
  - `val_jaw_open_mae` = `0.087622`
  - `val_mouth_close_mae` = `0.001526`
  - `val_smile_mae` = `0.093077`
  - `val_pose_mae` = `0.064744`
  - `val_jaw_open_corr` = `0.165909`
  - `val_mouth_close_corr` = `0.001065`
  - `val_smile_corr` = `0.209221`
  - `val_mouth_jaw_corr_mean` = `0.117637`
  - `val_overall_blendshape_corr_mean` = `0.153745`
  - `learning_rate` = `0.000069`
- Notes:
  - Present on disk but not listed in the sweep leaderboard.

### `conv_transformer_d320_l12_corrvar_std`

- Area: `conv_transformer_d320_l12_corrvar_std`
- Phase: `single`
- Source path: `runpod_results/conv_transformer_d320_l12_corrvar_std`
- Metadata file: `runpod_results/conv_transformer_d320_l12_corrvar_std/conv_transformer_d320_l12_corrvar_std.json`
- Variant / architecture: `conv_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d320, l12, h8, ffn1280, k9`
- Config extras: `dropout=0.1; output_space=natural_range; num_parameters=16223099; model_size_mb=61.886`
- Best epoch: `16`
- Best val loss: `0.523375`
- Artifacts: total `436.7MB`, best checkpoints `186.0MB`, last checkpoints `186.0MB`, ONNX `63.7MB`
- Recorded metrics:
  - `train_loss` = `0.490706`
  - `train_l1_loss` = `0.385796`
  - `train_temporal_loss` = `0.108859`
  - `train_corr_loss` = `0.541678`
  - `train_var_loss` = `0.364298`
  - `train_grad_norm` = `0.286901`
  - `val_loss` = `0.523375`
  - `val_l1_loss` = `0.378265`
  - `val_temporal_loss` = `0.039076`
  - `val_corr_loss` = `0.875577`
  - `val_var_loss` = `0.23639`
  - `val_overall_mae` = `0.050086`
  - `val_mouth_mae` = `0.031807`
  - `val_jaw_open_mae` = `0.088431`
  - `val_mouth_close_mae` = `0.001005`
  - `val_smile_mae` = `0.080939`
  - `val_pose_mae` = `0.05511`
  - `val_jaw_open_corr` = `0.174331`
  - `val_mouth_close_corr` = `0.066997`
  - `val_smile_corr` = `0.347757`
  - `val_mouth_jaw_corr_mean` = `0.154994`
  - `val_overall_blendshape_corr_mean` = `0.218953`
  - `learning_rate` = `0.000125`

### `conv_transformer_d768_l20_b20_quick_v2`

- Area: `conv_transformer_d768_l20_b20_quick_v2`
- Phase: `single`
- Source path: `runpod_results/conv_transformer_d768_l20_b20_quick_v2`
- Variant / architecture: `n/a`
- Input -> output: `n/a`
- Architecture shape: `n/a`
- Best epoch: `5`
- Best val loss: `0.543007`
- Artifacts: total `7.0KB`, best checkpoints `0B`, last checkpoints `0B`, ONNX `0B`
- Recorded metrics:
  - `train_loss` = `0.598113`
  - `train_l1_loss` = `0.464285`
  - `train_temporal_loss` = `0.106781`
  - `train_corr_loss` = `0.696137`
  - `train_var_loss` = `0.481364`
  - `train_grad_norm` = `0.278135`
  - `val_loss` = `0.543007`
  - `val_l1_loss` = `0.389077`
  - `val_temporal_loss` = `0.034237`
  - `val_corr_loss` = `0.882352`
  - `val_var_loss` = `0.39731`
  - `val_overall_mae` = `0.051662`
  - `val_mouth_mae` = `0.031311`
  - `val_jaw_open_mae` = `0.087812`
  - `val_mouth_close_mae` = `0.001151`
  - `val_smile_mae` = `0.084548`
  - `val_pose_mae` = `0.060213`
  - `val_jaw_open_corr` = `0.124414`
  - `val_mouth_close_corr` = `0.027873`
  - `val_smile_corr` = `0.200154`
  - `val_mouth_jaw_corr_mean` = `0.122081`
  - `val_overall_blendshape_corr_mean` = `0.173928`
  - `learning_rate` = `0.00015`

### `conv_transformer_smoke`

- Area: `conv_transformer_smoke`
- Phase: `single`
- Source path: `runpod_results/conv_transformer_smoke`
- Metadata file: `runpod_results/conv_transformer_smoke/model.json`
- Variant / architecture: `conv_transformer`
- Input -> output: `80 -> 59`
- Architecture shape: `d320, l12, h8, ffn1280, k9`
- Config extras: `dropout=0.1; output_space=natural_range; num_parameters=16223099; model_size_mb=61.886`
- Best epoch: `1`
- Best val loss: `0.70455`
- Artifacts: total `436.4MB`, best checkpoints `185.9MB`, last checkpoints `185.9MB`, ONNX `63.7MB`
- Recorded metrics:
  - `train_loss` = `0.841703`
  - `train_l1_loss` = `0.656486`
  - `train_temporal_loss` = `0.297806`
  - `train_corr_loss` = `0.983754`
  - `train_var_loss` = `0.455266`
  - `train_grad_norm` = `0.720959`
  - `val_loss` = `0.70455`
  - `val_l1_loss` = `0.52794`
  - `val_temporal_loss` = `0.058666`
  - `val_corr_loss` = `0.94626`
  - `val_var_loss` = `0.634741`
  - `val_overall_mae` = `0.058267`
  - `val_mouth_mae` = `0.055976`
  - `val_jaw_open_mae` = `0.045407`
  - `val_mouth_close_mae` = `0.000584`
  - `val_smile_mae` = `0.218303`
  - `val_pose_mae` = `0.043721`
  - `val_jaw_open_corr` = `0.225107`
  - `val_mouth_close_corr` = `0.257683`
  - `val_smile_corr` = `0.061824`
  - `val_mouth_jaw_corr_mean` = `0.077044`
  - `val_overall_blendshape_corr_mean` = `0.043926`
  - `learning_rate` = `0`

### `tcn_model`

- Area: `legacy_local`
- Phase: `legacy`
- Source path: `V2A-over-training-old-nn/2_architecture_training/models`
- Metadata file: `V2A-over-training-old-nn/2_architecture_training/models/tcn_model.json`
- Variant / architecture: `TCN_Audio_to_Blendshapes`
- Input -> output: `80 -> 59`
- Architecture shape: `n/a`
- Config extras: `dropout=0.1`
- Best epoch: `3`
- Best val loss: `0.008304`
- Artifacts: total `10.1MB`, best checkpoints `0B`, last checkpoints `0B`, ONNX `1.5MB`
- Recorded metrics:
  - `total_loss` = `0.008304`
  - `base_loss` = `0.006962`
  - `temporal_loss` = `0.006711`
  - `silence_loss` = `0`
  - `pose_loss` = `0.013417`
  - `overall_mae` = `0.06489`
  - `mouth_mae` = `0.076518`
  - `jaw_open_mae` = `0.082849`
  - `lip_close_mae` = `0.185911`
  - `smile_mae` = `0.132127`
  - `pose_mae` = `0.078694`
  - `jaw_corr` = `0.209255`
  - `lip_corr` = `0.379834`
  - `smile_corr` = `0.241591`

### `tiny_transformer_full10s_l1`

- Area: `legacy_local`
- Phase: `legacy`
- Source path: `V2A-over-training-old-nn/2_architecture_training/models`
- Metadata file: `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_full10s_l1.json`
- Variant / architecture: `tiny_transformer_encoder`
- Input -> output: `80 -> 59`
- Architecture shape: `d128, l3, h4, ffn256`
- Config extras: `dropout=0.0; num_parameters=432443; model_size_mb=1.65`
- Best epoch: `14`
- Best val loss: `0.072877`
- Artifacts: total `7.3MB`, best checkpoints `5.0MB`, last checkpoints `0B`, ONNX `2.3MB`
- Recorded metrics:
  - `train_loss` = `0.063258`
  - `train_l1_loss` = `0.063258`
  - `train_temporal_loss` = `0`
  - `train_grad_norm` = `0.079144`
  - `val_loss` = `0.072877`
  - `val_l1_loss` = `0.072877`
  - `val_temporal_loss` = `0`
  - `learning_rate` = `0.000294`

### `tiny_transformer_full10s_l1temp`

- Area: `legacy_local`
- Phase: `legacy`
- Source path: `V2A-over-training-old-nn/2_architecture_training/plots`
- Variant / architecture: `n/a`
- Input -> output: `n/a`
- Architecture shape: `n/a`
- Best epoch: `8`
- Best val loss: `0.074392`
- Artifacts: total `5.0MB`, best checkpoints `5.0MB`, last checkpoints `0B`, ONNX `0B`
- Recorded metrics:
  - `train_loss` = `0.066209`
  - `train_l1_loss` = `0.066112`
  - `train_temporal_loss` = `0.004852`
  - `train_grad_norm` = `0.09489`
  - `val_loss` = `0.074392`
  - `val_l1_loss` = `0.074287`
  - `val_temporal_loss` = `0.005259`
  - `learning_rate` = `0.000293`
- Notes:
  - Metadata JSON is missing; only history/artifact data was available.
  - No matching `.onnx` export found in the legacy models folder.

### `tiny_transformer_l1`

- Area: `legacy_local`
- Phase: `legacy`
- Source path: `V2A-over-training-old-nn/2_architecture_training/models`
- Metadata file: `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1.json`
- Variant / architecture: `tiny_transformer_encoder`
- Input -> output: `80 -> 59`
- Architecture shape: `d128, l3, h4, ffn256`
- Config extras: `dropout=0.0; num_parameters=432443; model_size_mb=1.65`
- Best epoch: `58`
- Best val loss: `0.082626`
- Artifacts: total `7.4MB`, best checkpoints `5.0MB`, last checkpoints `0B`, ONNX `2.3MB`
- Recorded metrics:
  - `train_loss` = `0.048248`
  - `train_l1_loss` = `0.048248`
  - `train_temporal_loss` = `0`
  - `train_grad_norm` = `0.050549`
  - `val_loss` = `0.082626`
  - `val_l1_loss` = `0.082626`
  - `val_temporal_loss` = `0`
  - `learning_rate` = `0.000484`

### `tiny_transformer_l1temp`

- Area: `legacy_local`
- Phase: `legacy`
- Source path: `V2A-over-training-old-nn/2_architecture_training/models`
- Metadata file: `V2A-over-training-old-nn/2_architecture_training/models/tiny_transformer_l1temp.json`
- Variant / architecture: `tiny_transformer_encoder`
- Input -> output: `80 -> 59`
- Architecture shape: `d128, l3, h4, ffn256`
- Config extras: `dropout=0.0; num_parameters=432443; model_size_mb=1.65`
- Best epoch: `58`
- Best val loss: `0.082846`
- Artifacts: total `7.4MB`, best checkpoints `5.0MB`, last checkpoints `0B`, ONNX `2.3MB`
- Recorded metrics:
  - `train_loss` = `0.048288`
  - `train_l1_loss` = `0.048088`
  - `train_temporal_loss` = `0.010007`
  - `train_grad_norm` = `0.04201`
  - `val_loss` = `0.082846`
  - `val_l1_loss` = `0.082706`
  - `val_temporal_loss` = `0.007019`
  - `learning_rate` = `0.000484`

## Artifact-Only Entries

| Path | Kind | Size | Notes |
| --- | --- | --- | --- |
| `V2A-over-training-old-nn/models/best_tcn_model.pth` | `.pth` | `8.5MB` | No linked metrics file found in the repo. |
| `V2A-over-training-old-nn/2_architecture_training/models/best_tcn_model_train_50.pth` | `.pth` | `4.6MB` | No linked metrics file found in the repo. |
| `V2A-over-training-old-nn/2_architecture_training/models/best_tcn_model_test_35.pth` | `.pth` | `4.6MB` | No linked metrics file found in the repo. |
| `V2A-over-training-old-nn/2_architecture_training/models/best_tcn_model_train_15.pth` | `.pth` | `4.6MB` | No linked metrics file found in the repo. |
| `V2A-over-training-old-nn/models/face_landmarker_v2_with_blendshapes.task` | `.task` | `3.6MB` | No linked metrics file found in the repo. |
| `assets/tiny_transformer_full10s_l1.onnx` | `.onnx` | `2.3MB` | No linked metrics file found in the repo. |
| `assets/tiny_transformer_l1.onnx` | `.onnx` | `2.3MB` | No linked metrics file found in the repo. |
| `V2A-over-training-old-nn/2_architecture_training/models/best_tcn_model_train_50.onnx` | `.onnx` | `1.5MB` | No linked metrics file found in the repo. |
| `assets/best_tcn_model_train_50.onnx` | `.onnx` | `1.5MB` | No linked metrics file found in the repo. |
| `V2A-over-training-old-nn/outputhtml/models/best_tcn_model_train_50.onnx` | `.onnx` | `1.5MB` | No linked metrics file found in the repo. |
| `V2A-over-training-old-nn/2_architecture_training/models/best_tcn_model_train_50_optimized.onnx` | `.onnx` | `1.5MB` | No linked metrics file found in the repo. |
| `V2A-over-training-old-nn/2_architecture_training/models/tcn_model_optimized.onnx` | `.onnx` | `1.5MB` | No linked metrics file found in the repo. |
| `V2A-over-training-old-nn/outputhtml/models/best_tcn_model_train_50_optimized.onnx` | `.onnx` | `1.5MB` | No linked metrics file found in the repo. |
| `assets/tiny_transformer_full10s_l1.json` | `.json` | `1.1KB` | Metadata/export file without linked metrics. |
| `assets/tiny_transformer_l1.json` | `.json` | `1.1KB` | Metadata/export file without linked metrics. |
| `V2A-over-training-old-nn/outputhtml/models/best_tcn_model_train_50.json` | `.json` | `1.0KB` | Metadata/export file without linked metrics. |
| `V2A-over-training-old-nn/2_architecture_training/models/best_tcn_model_train_50.json` | `.json` | `440B` | Metadata/export file without linked metrics. |

## Suggested Deletion Order

1. Delete all completed `*_last.pth` files once you are sure you do not need to resume training from them.
2. Delete losing run folders after keeping this markdown report plus any winner `best.pth` / `.onnx` pair you still care about.
3. Remove duplicate exports in `assets/`, `outputhtml/models/`, and old legacy model folders if the same model already exists elsewhere in the repo.
4. Move any checkpoints you want to archive out of the git repo before you ever run `git add .` again.

