# Final submission checklist: Solar Filament Segmentation Challenge 2026

**Deadline: Sunday 16 November 2026, 09:59 PKT** (15 Nov, 23:59 CDT).
Kaggle team **"Umer Gulzar"** (haseebekhan, umergulzar). Public score **0.39** (rank 188 / 1,068 on 19 Oct).

## Status

| step | what | status |
|---|---|---|
| 1 | Kaggle: `submission_v3_final.csv` uploaded | **done**: public 0.39 |
| 2 | Reproduction from the weights verified | **done**: 1,061 / 1,067 masks bit-identical, the rest IoU >= 0.95 |
| 3 | Code on GitHub (Haseeb pushes, Umer merges) | **done** |
| 4 | Kaggle: select the final submissions | **you** (2 min) |
| 5 | GitHub release with the M5 weights | **you**: zip downloaded; release `v3-final` not created yet (checked 19 Oct, 06:10 UTC) |
| 6 | Make the repository public | **done**: public; every key file on GitHub verified byte-identical to `master` |
| 7 | Report: e-mails in `main.tex`, compile in the organisers' Overleaf, download the PDF | **you** (10 min) |
| 8 | Google form | **you** |
| 9 | Shut down the GPU server | **you**: nothing left that needs it (steps 4, 5, 7, 8 run in the browser) |

## What is in this folder

| file | what it is | where it goes |
|---|---|---|
| `kaggle/submission_v3_final.csv` | the exact file submitted to Kaggle (284 KB) | form field "Submission File" |
| `report/main.tex`, `report/ours.bib`, `report/figures/` | 4-page report in the organisers' ACM template | organisers' Overleaf project (step 7) |
| `report/main_preview.pdf` | local preview (compiled with stand-ins of their preamble/main.bib) | only for checking; hand in the PDF from Overleaf |
| `report/stats.json` | bootstrap CIs and per-fold PQ used in the report | repo |
| `notebook/reproduce_v3.ipynb` | rebuilds the CSV from the weights (executed, outputs inside) | repo |
| `release/` | DINOv3 licence, weights-zip script, final linker + scorer | repo |
| server `/root/kaggle/release_v3/m5_cnxs_1536_v3_weights.zip` (+ `.sha256`) | M5 weights + licence, 1.2 GB | GitHub release (step 5) |
| server `/root/kaggle/release_v3/submission_package_v3.zip` | this folder in one file | your computer |

---

## Step 4. Select the final submissions on Kaggle

https://www.kaggle.com/competitions/filament-segmentation-2026/submissions: tick **Use for Final Score** on
`submission_v3_final.csv` (0.39) and on `submission_v3_rescorer_after_m3.csv` (0.38) as the backup.

## Step 5. GitHub release with the M5 weights

1. Use the `m5_cnxs_1536_v3_weights.zip` and `.sha256` you downloaded (SHA-256 `99cd9c52c20a2957c23097123704dfb93fe056d6576b52459f5aece224ce50dc`).
2. As **umerkang66**: https://github.com/umerkang66/solar-filament-segmentation/releases/new
   - **Choose a tag**: type `v3-final`, then **Create new tag: v3-final on publish**. Target: `master`.
   - **Title**: `v3 final submission weights`
   - **Description**: `M5 (DINOv3 ConvNeXt-S U-Net, 5 folds) checkpoints of the final submission (nested OOF PQ 0.4477, public LB 0.39). Unzip inside work/models/ (see SETUP.md section 2). The fine-tuned DINOv3 weights are distributed under the DINOv3 License, included in the zip as LICENSE_DINOv3.md.`
   - Drag both files into **Attach binaries**, wait for 100 %, then click **Publish release**.

## Step 6. Make the repository public

https://github.com/umerkang66/solar-filament-segmentation/settings: **Danger Zone**, **Change repository
visibility**, **Change to public**.
The form requires access "without additional access permissions", so this step is mandatory.

## Step 7. Report PDF in the organisers' Overleaf project

1. Open the Overleaf project the organisers provided (link on the competition page). Use **Menu**, **Copy Project**
   if it is read-only.
2. Replace the content of `main.tex` with our `submission_package_v3/report/main.tex`.
3. Upload `ours.bib`, `figures/examples.png` and `figures/rubric_final.png` (create a folder `figures`).
   Keep their `preamble.tex` and `main.bib`.
4. In `main.tex`, replace `EMAIL-HASEEB` and `EMAIL-UMER` with your real e-mails (the only black text the
   template lets you change). To add an institute, put `\institution{...}` and `\city{...}` above `\country{Pakistan}`.
5. Click **Recompile**. Check: the body (Sections 1-3) is at most 4 pages, no gray guide text, and the references
   show [1] MAGFiLO and [2] the challenge.
6. **Download PDF**. Name it, for example, `UmerGulzar_report.pdf`. It must be under 10 MB (the preview is 3 MB).

## Step 8. Google form

| form field | what to enter |
|---|---|
| Affiliated Institutes of All Team Members (optional) | leave empty, or `Institute A; Institute B` |
| Countries of All Team Members | `Pakistan; Pakistan` (or `Pakistan` if the form wants each country once) |
| GIT URL | `https://github.com/umerkang66/solar-filament-segmentation` |
| Submission File | upload `submission_package_v3/kaggle/submission_v3_final.csv` (byte-identical to the file scored 0.39) |
| Technical Report | upload the PDF from step 7 |
| other fields (team name, members, e-mails) | team `Umer Gulzar`; Haseeb Khan (haseebekhan), Muhammad Umer (umergulzar) |

Click **Submit** and screenshot the confirmation page. You can replace the files up to 10 times.

## Step 9. Shut down the GPU server

After the zip has finished downloading (step 5), shut the instance down. It bills $1.99/h even when idle.
