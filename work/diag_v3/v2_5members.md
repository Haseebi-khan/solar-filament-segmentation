## v2_5members

`/root/kaggle/work/oof_sub_v2_r34_1024+rgy040_1024+r34_1536+r34_2048+effv2s_1536.csv`

PQ **0.4438**  SQ 0.6801  RQ 0.6524  TP 5046  FP 2223  FN 3153

### PQ lost by cause

| cause | count | PQ if removed | share |
|---|---|---|---|
| TP IoU deficit (SQ) | 5046 | +0.2087 | 56% |
| FN missed | 1708 | +0.0551 | 15% |
| FP merged_extent | 961 | +0.0294 | 8% |
| FN extent | 780 | +0.0236 | 6% |
| FP spurious | 750 | +0.0226 | 6% |
| FP fragment | 512 | +0.0152 | 4% |
| FN merged | 449 | +0.0133 | 4% |
| FN fragmented | 216 | +0.0063 | 2% |

### By GT / prediction size (px)

| group | PQ | GT | TP | FP | FN | recall | FN frag | FN merged | FN missed | FP spurious |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k-8k | 0.4983 | 2245 | 1561 | 534 | 684 | 0.695 | 132 | 73 | 200 | 83 |
| 500-2k | 0.4504 | 4390 | 2848 | 1280 | 1542 | 0.649 | 16 | 268 | 861 | 413 |
| <500 | 0.2915 | 1263 | 457 | 380 | 806 | 0.362 | 0 | 107 | 629 | 250 |
| >8k | 0.4949 | 301 | 180 | 29 | 121 | 0.598 | 68 | 1 | 18 | 4 |

### By radial zone r/R

| group | PQ | GT | TP | FP | FN | recall | FN frag | FN merged | FN missed | FP spurious |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5-0.8 | 0.4494 | 4151 | 2627 | 1179 | 1524 | 0.633 | 131 | 226 | 742 | 369 |
| >0.8 | 0.3730 | 954 | 455 | 257 | 499 | 0.477 | 7 | 28 | 381 | 135 |
| r<0.5 | 0.4562 | 3094 | 1964 | 787 | 1130 | 0.635 | 78 | 195 | 585 | 246 |

### By instrument site

| group | PQ | GT | TP | FP | FN | recall | FN frag | FN merged | FN missed | FP spurious |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 0.4445 | 1502 | 925 | 418 | 577 | 0.616 | 37 | 97 | 299 | 151 |
| C | 0.4374 | 1522 | 904 | 368 | 618 | 0.594 | 37 | 123 | 323 | 103 |
| L | 0.4344 | 1438 | 880 | 445 | 558 | 0.612 | 46 | 75 | 287 | 147 |
| M | 0.4455 | 1298 | 809 | 359 | 489 | 0.623 | 43 | 54 | 258 | 99 |
| T | 0.4502 | 1369 | 841 | 331 | 528 | 0.614 | 25 | 59 | 323 | 127 |
| U | 0.4537 | 1070 | 687 | 302 | 383 | 0.642 | 28 | 41 | 218 | 123 |

### By annotation sets per image

| group | PQ | GT | TP | FP | FN | recall | FN frag | FN merged | FN missed | FP spurious |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.4427 | 3206 | 1882 | 725 | 1324 | 0.587 | 80 | 171 | 817 | 211 |
| 2 | 0.4575 | 1987 | 1282 | 538 | 705 | 0.645 | 52 | 102 | 369 | 195 |
| 3 | 0.4359 | 3006 | 1882 | 960 | 1124 | 0.626 | 84 | 176 | 522 | 344 |

### Overlap degrees (1:n = predictions per GT, n:1 = GTs per prediction)

| overlapping partners | GT filaments (1:n) | predictions (n:1) |
|---|---|---|
| 0 | 1708 | 750 |
| 1 | 6225 | 6265 |
| 2 | 235 | 235 |
| 3+ | 31 | 18 |

### Decision (blueprint section 14)

fragmented + merged share of the lost PQ: **9%** -> M4 (query-based proposals) stays optional; largest lever: A (pixel quality / SQ).
