
## Overnight Sweep `overnight_20260321_205258`

- Start: 2026-03-21 20:52:58 UTC
- Deadline: 2026-03-22 06:52:58 UTC
- Dataset: `/workspace/v2a_pipeline/datasets/combined_long_10s_step500`
- Baseline reference:
  - best_epoch: `16`
  - best_val_loss: `0.523375`
  - val_mouth_mae: `0.031807`
  - val_mouth_jaw_corr_mean: `0.1550`
  - val_smile_corr: `0.3478`

| Phase | Experiment | Best val_loss | Mouth MAE | Mouth corr | Smile corr | Score | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

Pilot starting for `overnight_20260321_205258__conv_d320_l12_b16` at 2026-03-21 20:52:58 UTC.
| pilot | `overnight_20260321_205258__conv_d320_l12_b16` | 0.573302 | 0.030950 | 0.1236 | 0.2255 | 0.557817 | Control rerun with warmup and larger batch. |

Pilot starting for `overnight_20260321_205258__conv_d512_l16_b10` at 2026-03-21 20:56:15 UTC.
| pilot | `overnight_20260321_205258__conv_d512_l16_b10` | 0.576515 | 0.031232 | 0.1265 | 0.2165 | 0.561138 | Wider/deeper conv transformer. |

Pilot starting for `overnight_20260321_205258__convgated_d512_l16_b10` at 2026-03-21 21:01:55 UTC.
| pilot | `overnight_20260321_205258__convgated_d512_l16_b10` | 0.591581 | 0.031335 | 0.1199 | 0.2085 | 0.576587 | Conv mixer plus gated FFN hybrid. |

Pilot starting for `overnight_20260321_205258__conv_d640_l18_k15_b8` at 2026-03-21 21:07:58 UTC.
| pilot | `overnight_20260321_205258__conv_d640_l18_k15_b8` | 0.587657 | 0.031314 | 0.1350 | 0.1990 | 0.572104 | Deeper model with wider local kernel. |

Finalists selected: `overnight_20260321_205258__conv_d320_l12_b16`, `overnight_20260321_205258__conv_d512_l16_b10`

Full continuation starting for `overnight_20260321_205258__conv_d320_l12_b16` at 2026-03-21 21:16:01 UTC.
| full | `overnight_20260321_205258__conv_d320_l12_b16` | 0.549753 | 0.029998 | 0.1892 | 0.4141 | 0.523781 | Control rerun with warmup and larger batch. |

Full continuation starting for `overnight_20260321_205258__conv_d512_l16_b10` at 2026-03-21 21:31:41 UTC.
| full | `overnight_20260321_205258__conv_d512_l16_b10` | 0.554401 | 0.030381 | 0.1789 | 0.3804 | 0.530203 | Wider/deeper conv transformer. |

### Best Result
- Run: `overnight_20260321_205258__conv_d320_l12_b16`
- Phase reached: `full`
- best_val_loss: `0.549753`
- val_mouth_mae: `0.029998`
- val_mouth_jaw_corr_mean: `0.1892`
- val_smile_corr: `0.4141`
- End: 2026-03-21 21:57:34 UTC

## Overnight Sweep `overnight_20260322_053753`

- Start: 2026-03-22 05:37:53 UTC
- Deadline: 2026-03-22 07:37:53 UTC
- Dataset: `/workspace/v2a_pipeline/datasets/combined_long_10s_step500`
- Baseline reference:
  - best_epoch: `16`
  - best_val_loss: `0.523375`
  - val_mouth_mae: `0.031807`
  - val_mouth_jaw_corr_mean: `0.1550`
  - val_smile_corr: `0.3478`

| Phase | Experiment | Best val_loss | Mouth MAE | Mouth corr | Smile corr | Score | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

Pilot starting for `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber` at 2026-03-22 05:37:53 UTC.
| pilot | `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber` | 0.424152 | 0.032340 | 0.1180 | 0.2186 | 0.409218 | Large-batch control with NAdam and Huber. |

Pilot starting for `overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber` at 2026-03-22 05:41:15 UTC.
| pilot | `overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber` | 0.435818 | 0.032246 | 0.1427 | 0.2460 | 0.418319 | Hybrid conv+gated model with NAdam and Huber. |

Pilot starting for `overnight_20260322_053753__conv_d512_l16_b10_radam_huber` at 2026-03-22 05:47:36 UTC.
| pilot | `overnight_20260322_053753__conv_d512_l16_b10_radam_huber` | 0.445490 | 0.035472 | 0.1258 | 0.2250 | 0.429826 | Wider conv model with RAdam and Huber. |

Pilot starting for `overnight_20260322_053753__gated_d384_l12_b16_nadam` at 2026-03-22 05:53:21 UTC.
| pilot | `overnight_20260322_053753__gated_d384_l12_b16_nadam` | 0.606110 | 0.031782 | 0.0904 | 0.1521 | 0.594750 | Pure gated FFN transformer with NAdam. |

Finalists selected: `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber`, `overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber`

Full continuation starting for `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber` at 2026-03-22 05:56:33 UTC.
| full | `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber` | 0.406168 | 0.031580 | 0.1792 | 0.3555 | 0.382858 | Large-batch control with NAdam and Huber. |

Full continuation starting for `overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber` at 2026-03-22 06:15:19 UTC.
| full | `overnight_20260322_053753__convgated_d512_l16_b10_nadam_huber` | 0.422205 | 0.031852 | 0.1787 | 0.3330 | 0.399496 | Hybrid conv+gated model with NAdam and Huber. |

### Best Result
- Run: `overnight_20260322_053753__conv_d320_l12_b20_nadam_huber`
- Phase reached: `full`
- best_val_loss: `0.406168`
- val_mouth_mae: `0.031580`
- val_mouth_jaw_corr_mean: `0.1792`
- val_smile_corr: `0.3555`
- End: 2026-03-22 06:52:49 UTC

## Overnight Sweep `overnight_20260322_065411`

- Start: 2026-03-22 06:54:11 UTC
- Deadline: 2026-03-22 13:54:11 UTC
- Dataset: `/workspace/v2a_pipeline/datasets/combined_long_10s_step500`
- Baseline reference:
  - best_epoch: `16`
  - best_val_loss: `0.523375`
  - val_mouth_mae: `0.031807`
  - val_mouth_jaw_corr_mean: `0.1550`
  - val_smile_corr: `0.3478`

| Phase | Experiment | Best val_loss | Mouth MAE | Mouth corr | Smile corr | Score | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

Pilot starting for `overnight_20260322_065411__conformer_d384_l12_b12_huber` at 2026-03-22 06:54:11 UTC.
| pilot | `overnight_20260322_065411__conformer_d384_l12_b12_huber` | 0.445175 | 0.033485 | 0.1013 | 0.1859 | 0.432139 | Speech-oriented Conformer-style stack with Huber loss. |

Pilot starting for `overnight_20260322_065411__conformer_d512_l16_b8_nadam_huber` at 2026-03-22 07:01:23 UTC.
| pilot | `overnight_20260322_065411__conformer_d512_l16_b8_nadam_huber` | 0.458553 | 0.033016 | 0.1235 | 0.2302 | 0.442881 | Larger Conformer-style run with NAdam. |

Pilot starting for `overnight_20260322_065411__multiscale_d384_l12_b12_huber` at 2026-03-22 07:15:23 UTC.
| pilot | `overnight_20260322_065411__multiscale_d384_l12_b12_huber` | 0.425908 | 0.032171 | 0.1624 | 0.2835 | 0.405873 | Multi-scale local/global transformer fusion. |

Pilot starting for `overnight_20260322_065411__multiscale_d512_l16_b8_radam_huber` at 2026-03-22 07:21:39 UTC.
