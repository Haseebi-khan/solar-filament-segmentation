# Final submission checklist: Solar Filament Segmentation Challenge 2026

**Deadline: Sunday 16 November 2026, 09:59 PKT** (15 Nov, 23:59 CDT).
Kaggle team **"Umer Gulzar"** (haseebekhan, umergulzar).

## Status

| step | what | status |
|---|---|---|
| 1 | Kaggle: `submission_v3_final.csv` uploaded | **done**: public 0.39 (rank 188 / 1,068 on 19 Oct) |
| 2 | Report filled in (score, team) and compiled | **done**: `report/report.pdf` (3 pages) |
| 3 | Code on GitHub (Haseeb pushes, Umer merges) | **done** for everything up to this checklist |
| 4 | Kaggle: select the final submissions | **you** (2 min) |
| 5 | GitHub release with the M5 weights | **you** (zip is ready on the server) |
| 6 | Make the repository public | **you** (1 min) |
| 7 | Google form | **you** |
| 8 | Shut down the GPU server | **you** |

## What is in this folder, and where each file goes

| file | what it is | where it goes |
|---|---|---|
| `kaggle/submission_v3_final.csv` | final predictions (OOF PQ 0.4477, public 0.39) | Kaggle (already uploaded) |
| `report/report.pdf` | 4-page-limit report (3 pages), IEEE format | Google form |
| `report/report.tex` + `report/figures/` | LaTeX source of the PDF | repo (already there); Overleaf only if the organisers want their template |
| `notebook/reproduce_v3.ipynb` | rebuilds the CSV from the weights | repo (already there); link it in the form |
| `release/LICENSE_DINOv3.md`, `release/make_release_v3.sh` | DINOv3 licence, builds the weights zip | repo (already there) |
| `release/stagec_final/` | fragment linker + stacked scorer of the final system | repo (already there) |
| server: `/root/kaggle/release_v3/m5_cnxs_1536_v3_weights.zip` (+ `.sha256`) | M5 weights + licence, 1.2 GB | GitHub release (step 5) |
| server: `/root/kaggle/release_v3/submission_package_v3.zip` | this whole folder in one file, 25 MB | your computer (backup, form attachments) |

---

## Step 4. Select the final submissions on Kaggle

https://www.kaggle.com/competitions/filament-segmentation-2026/submissions: tick **Use for Final Score** on
`submission_v3_final.csv` (0.39) and on `submission_v3_rescorer_after_m3.csv` (0.38) as the backup.
Never select the `_erode1` / `_dilate1` files.

## Step 5. GitHub release with the M5 weights

1. Download the two files to your computer. In VS Code's Explorer, open `release_v3/`, right-click
   `m5_cnxs_1536_v3_weights.zip`, choose **Download...**, and do the same for `m5_cnxs_1536_v3_weights.zip.sha256`.
   It is 1.2 GB, so the download takes a while.
2. Logged in to GitHub as **umerkang66**: open https://github.com/umerkang66/solar-filament-segmentation/releases/new
   - **Choose a tag**: type `v3-final`, then **Create new tag: v3-final on publish**. Target: `master`.
   - **Release title**: `v3 final submission weights`
   - **Description**:
     `M5 (DINOv3 ConvNeXt-S U-Net, 5 folds) checkpoints of the final submission (nested OOF PQ 0.4477, public LB 0.39). Unzip inside work/models/ (see SETUP.md section 2). The fine-tuned DINOv3 weights are distributed under the DINOv3 License, included in the zip as LICENSE_DINOv3.md.`
   - Drag both files into **Attach binaries** (GitHub allows up to 2 GB per file) and wait until both finish uploading.
   - **Publish release**.
3. Check that https://github.com/umerkang66/solar-filament-segmentation/releases/tag/v3-final shows both files. README
   and SETUP already link to this address.

## Step 6. Make the repository public

https://github.com/umerkang66/solar-filament-segmentation/settings: at the bottom, **Danger Zone**, then **Change
repository visibility**, **Change to public**, and confirm.

## Step 7. Google form

Open the form from the competition page (**Overview**, **Rules** or the organisers' announcement). Have ready:

| field (likely) | answer |
|---|---|
| Kaggle team name | Umer Gulzar |
| members / Kaggle usernames | Haseeb Khan (haseebekhan), Muhammad Umer (umergulzar) |
| public repository | https://github.com/umerkang66/solar-filament-segmentation |
| report | `report/report.pdf` (from `submission_package_v3.zip`) |
| notebook | https://github.com/umerkang66/solar-filament-segmentation/blob/master/submission_package_v3/notebook/reproduce_v3.ipynb |
| weights | https://github.com/umerkang66/solar-filament-segmentation/releases/tag/v3-final |
| final score | public leaderboard 0.39; nested out-of-fold PQ 0.4477 |

If the form needs affiliations or e-mails, they are not in the report. Add them to `report/report.tex` (line 24),
then recompile:
`cd submission_package_v3/report && pdflatex report.tex && pdflatex report.tex`.

## Step 8. Shut down the GPU server

Once steps 5 and 7 are done (the zip downloads from this server), shut the instance down. It bills $1.99/h even
when idle.

Optional before shutting down (about 1 GPU-hour, ~$2): run `notebook/reproduce_v3.ipynb` once, to prove that the CSV
can be reproduced from the weights.
