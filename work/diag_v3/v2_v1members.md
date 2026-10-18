## v2_v1members

`/root/kaggle/work/oof_sub_v2_r34_1024+rgy040_1024+r34_1536.csv`

PQ **0.4382**  SQ 0.6769  RQ 0.6474  TP 4939  FP 2121  FN 3260

### PQ lost by cause

| cause | count | PQ if removed | share |
|---|---|---|---|
| TP IoU deficit (SQ) | 4939 | +0.2091 | 56% |
| FN missed | 1845 | +0.0603 | 16% |
| FP merged_extent | 955 | +0.0293 | 8% |
| FN extent | 782 | +0.0237 | 6% |
| FP spurious | 683 | +0.0205 | 5% |
| FP fragment | 483 | +0.0143 | 4% |
| FN merged | 426 | +0.0126 | 3% |
| FN fragmented | 207 | +0.0060 | 2% |

### By GT / prediction size (px)

| group | PQ | GT | TP | FP | FN | recall | FN frag | FN merged | FN missed | FP spurious |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k-8k | 0.4948 | 2245 | 1537 | 509 | 708 | 0.685 | 130 | 64 | 248 | 72 |
| 500-2k | 0.4433 | 4390 | 2789 | 1241 | 1601 | 0.635 | 13 | 263 | 922 | 392 |
| <500 | 0.2820 | 1263 | 430 | 345 | 833 | 0.340 | 0 | 97 | 655 | 213 |
| >8k | 0.5020 | 301 | 183 | 26 | 118 | 0.608 | 64 | 2 | 20 | 6 |

### By radial zone r/R

| group | PQ | GT | TP | FP | FN | recall | FN frag | FN merged | FN missed | FP spurious |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5-0.8 | 0.4448 | 4151 | 2582 | 1133 | 1569 | 0.622 | 132 | 218 | 810 | 329 |
| >0.8 | 0.3652 | 954 | 438 | 247 | 516 | 0.459 | 4 | 27 | 392 | 133 |
| r<0.5 | 0.4500 | 3094 | 1919 | 741 | 1175 | 0.620 | 71 | 181 | 643 | 221 |

### By instrument site

| group | PQ | GT | TP | FP | FN | recall | FN frag | FN merged | FN missed | FP spurious |
|---|---|---|---|---|---|---|---|---|---|---|
| B | 0.4400 | 1502 | 899 | 390 | 603 | 0.599 | 34 | 84 | 336 | 135 |
| C | 0.4344 | 1522 | 896 | 353 | 626 | 0.589 | 31 | 111 | 349 | 103 |
| L | 0.4265 | 1438 | 853 | 430 | 585 | 0.593 | 50 | 76 | 318 | 139 |
| M | 0.4343 | 1298 | 791 | 372 | 507 | 0.609 | 47 | 50 | 278 | 104 |
| T | 0.4446 | 1369 | 831 | 323 | 538 | 0.607 | 25 | 67 | 318 | 112 |
| U | 0.4537 | 1070 | 669 | 253 | 401 | 0.625 | 20 | 38 | 246 | 90 |

### By annotation sets per image

| group | PQ | GT | TP | FP | FN | recall | FN frag | FN merged | FN missed | FP spurious |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.4361 | 3206 | 1843 | 704 | 1363 | 0.575 | 71 | 163 | 863 | 201 |
| 2 | 0.4489 | 1987 | 1250 | 532 | 737 | 0.629 | 55 | 100 | 403 | 185 |
| 3 | 0.4333 | 3006 | 1846 | 885 | 1160 | 0.614 | 81 | 163 | 579 | 297 |

### Overlap degrees (1:n = predictions per GT, n:1 = GTs per prediction)

| overlapping partners | GT filaments (1:n) | predictions (n:1) |
|---|---|---|
| 0 | 1845 | 683 |
| 1 | 6103 | 6137 |
| 2 | 225 | 222 |
| 3+ | 26 | 17 |

### Decision (blueprint section 14)

fragmented + merged share of the lost PQ: **9%** -> M4 (query-based proposals) stays optional; largest lever: A (pixel quality / SQ).
