## Ceilings (Phase 0)

### Leave-one-annotator-out consensus (151 images with 3 sets)

| rule | PQ | SQ | RQ |
|---|---|---|---|
| single | 0.3334 | 0.6318 | 0.5278 |
| union | 0.3335 | 0.6350 | 0.5253 |
| inter | 0.3176 | 0.6372 | 0.4985 |
| union_l8 | 0.3268 | 0.6348 | 0.5148 |
| inter_l8 | 0.3237 | 0.6354 | 0.5095 |
| matched | 0.3259 | 0.6512 | 0.5004 |
| matched_plus | 0.2888 | 0.6349 | 0.4550 |

### Annotator-aware oracle (296 images with 2+ sets)

PQ **0.6659** (SQ 0.8253, RQ 0.8069) at u > 0.05

model (oof_sub_v2_r34_1024+rgy040_1024+r34_1536+r34_2048+effv2s_1536.csv) on the 3-set images: PQ 0.4359

model (oof_sub_v2_r34_1024+rgy040_1024+r34_1536+r34_2048+effv2s_1536.csv) on the 2+-set images: PQ 0.4444
