# Final submission checklist: Solar Filament Segmentation Challenge 2026

**Deadline: Sunday 16 November 2026, 09:59 PKT** (15 Nov, 23:59 CDT). Do the steps in this order: later steps need
links from earlier ones.

## What is in this folder, and where each file goes

| file | what it is | where it goes |
|---|---|---|
| `kaggle/submission_v3_final.csv` | final predictions (nested OOF PQ 0.4477) | Kaggle, "Submit Prediction" (step 1) |
| `report/report.tex` + `report/figures/` | 4-page report, IEEE format | Overleaf, then the PDF goes to the Google form (steps 5, 7) |
| `notebook/reproduce_v3.ipynb` | rebuilds the CSV from the weights | GitHub repo (and Kaggle Code if the form asks) (step 4) |
| `release/make_release_v3.sh` | packs the M5 weights with the DINOv3 licence | run on the server, upload the zip as a GitHub Release (step 3) |
| `release/LICENSE_DINOv3.md` | official DINOv3 licence (required next to DINOv3 weights) | inside the release zip and the repo |
| `release/stagec_final/` | fragment linker + stacked scorer of the final system | GitHub repo (already in this folder) |

---

## Step 1. Upload the predictions to Kaggle

1. Open https://www.kaggle.com/competitions/filament-segmentation-2026 and click **Submit Prediction**.
2. Upload `submission_package_v3/kaggle/submission_v3_final.csv`.
   Description: `v3 final: Stage C on r34_2048 + effv2s_1536 + M5 (OOF 0.4477)`.
3. When it is scored, write the public score down. Expect 0.38-0.39: the gain over earlier files is too small for two
   decimals.
4. Open the **Submissions** tab and select **two** final submissions:
   `submission_v3_final.csv` and, as a backup, `submission_v3_rescorer_after_m3.csv` (public 0.38).
   Never select the `_erode1` / `_dilate1` files.

## Step 2. Put the score into the report

- `report/report.tex`, line 13: replace `\textbf{[fill in]}` in `\finalLB` with the score (for example `0.38`).
- Same file, lines 14-15 and 23: Kaggle team name, both members' affiliation and e-mail. Check the author
  names; I used `Haseeb Khan and Muhammad Umer`.
- `REPORT.md` (repo root), results table: replace `[fill in after upload]`.

## Step 3. Release the M5 weights (GitHub refuses files over 100 MB)

The final blend needs the M5 weights (5 x 240 MB). `r34_2048` and `effv2s_1536` are already in git.

1. On the server:
   ```bash
   cd /root/kaggle && submission_package_v3/release/make_release_v3.sh
   ```
   It writes `release_v3/m5_cnxs_1536_v3_weights.zip` (~1.2 GB) and its `.sha256` file.
2. Copy both files to your computer (VS Code: right-click the files in `release_v3/`, then **Download**), or upload
   from the server with the GitHub web page open in a browser on your computer.
3. On GitHub, logged in as **umerkang66**: open https://github.com/umerkang66/solar-filament-segmentation, then
   **Releases** (right sidebar), **Draft a new release**.
   - Tag: `v3-final`, target: `master` (after step 4.2).
   - Title: `v3 final submission weights`.
   - Attach `m5_cnxs_1536_v3_weights.zip` and `m5_cnxs_1536_v3_weights.zip.sha256` (limit 2 GB per file).
   - Text: `M5 (DINOv3 ConvNeXt-S U-Net) fold checkpoints for the final submission. DINOv3 License: see LICENSE_DINOv3.md inside the zip.`
   - **Publish release**.

## Step 4. Code on GitHub (public)

1. Push this folder with the Haseeb account:
   ```bash
   cd /root/kaggle && git add -A && git commit -m "Submission package v3" && git push haseeb v3-budget-run
   ```
2. Merge with the Umer account: logged in as umerkang66, open
   https://github.com/umerkang66/solar-filament-segmentation/compare/master...Haseebi-khan:solar-filament-segmentation:v3-budget-run
   then **Create pull request** and **Merge pull request**.
3. Make the repository public: **Settings**, **General**, at the bottom **Danger Zone**, **Change visibility**,
   **Public**.
4. Recommended (about 1 GPU-hour, ~$2): run `notebook/reproduce_v3.ipynb` once on the server, to prove the CSV
   can be reproduced. The last cell should show the same number of filaments and a mean IoU of about 1. Commit the
   executed notebook (step 4.1 again). If the competition wants a **Kaggle notebook**: Kaggle, **Code**,
   **New Notebook**, **File**, **Import Notebook**, choose `reproduce_v3.ipynb`. On Kaggle it also needs the
   competition data and the weights added as inputs.

## Step 5. Report PDF (Overleaf)

1. Zip the `report/` folder (`report.tex` and `figures/rubric_final.png`).
2. https://www.overleaf.com, then **New Project**, **Upload Project**, choose the zip.
3. Menu: compiler **pdfLaTeX**, then **Recompile**. Check that it is at most 4 pages (if longer, shorten the
   "What did not work" paragraph first).
4. If the organisers give their own template: open it in Overleaf and paste everything from `\begin{abstract}` to
   `\end{thebibliography}` into it, and upload `figures/rubric_final.png`.
5. **Download PDF**. Suggested name: `report_<teamname>.pdf`.

## Step 6. Final checks before the form

- [ ] Kaggle: `submission_v3_final.csv` scored and selected (step 1).
- [ ] Repo public, `master` contains this folder, the release `v3-final` has the M5 zip (steps 3, 4).
- [ ] Report PDF compiled with the real score, team name and affiliations (steps 2, 5).
- [ ] No leaked data anywhere: the public MAGFiLO v1.0 Kaggle dataset was never used (also stated in the report).

## Step 7. Google form

Open the form from the competition page (**Overview** / **Rules** / discussion announcement). Keep these ready:

- Kaggle team name and every member's Kaggle username.
- Repository: https://github.com/umerkang66/solar-filament-segmentation
- Weights: the release link from step 3 (`.../releases/tag/v3-final`).
- Report: the PDF from step 5.
- Notebook: the repo path `submission_package_v3/notebook/reproduce_v3.ipynb` (or the Kaggle notebook link).
- Final score: nested out-of-fold PQ 0.4477, public leaderboard from step 1.

## Step 8. Stop paying

When everything is uploaded, **shut down the GPU instance**. It bills $1.99/h even when idle.
