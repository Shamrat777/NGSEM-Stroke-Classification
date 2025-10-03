# NGSEM-Stroke-Classification

**Paper:** *“An integrated ensemble fusion approach for accurate and interpretable stroke classification using CT imaging”*  
**Article type:** AI Application

---

## 1. Description
This repository reproduces the **Neuro Guardian Stroke Ensemble Matrix (NGSEM)** framework introduced in the manuscript.  

**Workflow summary:**
- **Preprocessing:** Grayscale → wavelet denoising (db1, level=2, soft threshold) → resize 512×512 → normalize [0,1]  
- **Augmentation:** Horizontal & vertical flips, rotations (±45°), zoom (1.5×), shifts (−30/+25 px), translation (~17, 6 px)  
- **Data split:** Stratified 65% train, 15% validation, 20% test on the *augmented dataset*  
- **Base learners:** CNN, MLP, Random Forest, Decision Tree, Gaussian Naive Bayes, Logistic Regression, XGBoost  
- **Ensemble:** Hard majority vote (≥4/7 classifiers)  
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, MCC, Cohen’s κ, AUC, Log Loss, FNR, FPR, FDR  
- **Explainability:** Grad-CAM overlays for CNN component  
- **Voting behavior:** Agreement rate of base learners with ensemble decision  

---

## 2. Dataset Information
We use the **Stroke CT dataset** released under the **TEKNOFEST-2021 AI in Healthcare Challenge**, curated by **TÜSEB** and the **Republic of Türkiye Ministry of Health**.  

- **Official source:** [https://acikveri.saglik.gov.tr/Home/DataSetDetail/1](https://acikveri.saglik.gov.tr/Home/DataSetDetail/1)  
- **Original classes:** *Normal*, *Ischemia*, *Bleeding*  
- **Binary setup:** (*Ischemia + Bleeding* → Stroke; Normal → Normal)  
- **Final counts after augmentation (per paper):**
  - Normal: 10,857  
  - Stroke: 6,650  
  - Total: ≈17,507  

**Expected folder structure:**
```
/path/to/Normal_CT/*.png
/path/to/Stroke_CT/*.png
```

All images are anonymized as released by the dataset host. Please comply with the dataset’s terms and license.

---

## 3. Code Information
- **`ngsem_paper_matched.py`** → End-to-end pipeline matching manuscript methodology  
- Outputs automatically saved to `./outputs/`:
  - `metrics_summary.json` — metrics for all models
  - `voting_contribution.json` — per-model ensemble agreement
  - `roc_ngsem.png` — ROC curve for NGSEM
  - `gradcam_samples/` — Grad-CAM overlays
  - `cnn_best.keras` — best CNN checkpoint

---

## 4. Requirements
- Python 3.10+  
- TensorFlow 2.15+  
- scikit-learn 1.4+  
- xgboost 2.0+  
- pywt 1.5+  
- OpenCV-python  
- numpy, matplotlib, tqdm  

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

## 5. Usage
1. Update dataset paths inside `ngsem_paper_matched.py`:
   ```python
   ROOT_NORMAL_DIR = "/absolute/path/Normal_CT"
   ROOT_STROKE_DIR = "/absolute/path/Stroke_CT"
   ```
2. Run the script:
   ```bash
   python ngsem_paper_matched.py
   ```
3. Outputs will appear in `./outputs/`.

---

## 6. Methodology (Summary)
The NGSEM framework integrates **traditional ML classifiers + deep learning (CNN)** into a **heterogeneous ensemble** with majority voting.  
- **Noise reduction:** Wavelet denoising validated via PSNR.  
- **Augmentation:** Improves class balance and diversity.  
- **Validation:** Stratified split ensures stable class distribution.  
- **Evaluation:** Uses 11+ robust metrics including MCC and Cohen’s κ.  
- **Interpretability:** Grad-CAM overlays provide transparency for CNN decisions.  
- **Ensemble analysis:** Agreement rates show contribution of each base learner.  

---

## 7. Citation
If you use this code or dataset, please cite the manuscript:  
> Shamrat FMJM, Kamal AHM, Idris MYI, Lu Z, Farid FA, Islam MS, Husen MN, Zhou X, Moni MA. (2025). *An integrated ensemble fusion approach for accurate and interpretable stroke classification using CT imaging.* PeerJ Computer Science (AI Application).

Dataset citation:  
> Koç C, et al. (2022). Artificial Intelligence in Healthcare Stroke CT Dataset. Republic of Türkiye Ministry of Health Open Data Portal. Available at: (https://acikveri.saglik.gov.tr/Home/DataSetDetail/1)

---

## 8. License
- **Code:** Released under MIT License for academic reproducibility.  
- **Dataset:** Governed by the Republic of Türkiye Ministry of Health Open Data Portal terms.  

---
