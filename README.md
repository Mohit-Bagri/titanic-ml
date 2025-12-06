# Titanic Survival Prediction — Production ML Project

An end-to-end, **production-ready machine learning system** built on the Titanic dataset.

This project demonstrates how a classical ML model is:
- trained using a proper pipeline
- evaluated and interpreted
- exported as a versioned artifact
- served via a real backend API
- consumed by a deployed web frontend

✅ Backend deployed on **Render**  
✅ Frontend hosted on **GitHub Pages**

---

## 🌐 Live Demo

- **Frontend (GitHub Pages)**  
  👉 https://mohit-bagri.github.io/titanic-ml/

- **Backend API (Render)**  
  👉 https://titanic-ml-api-dwyy.onrender.com

---

## 📌 What This Project Covers

- Data preprocessing with `scikit-learn` pipelines  
- Logistic Regression with cross-validation & hyperparameter tuning  
- Model interpretability using **SHAP**  
- Saving trained model & metadata as artifacts  
- Serving predictions via **FastAPI**  
- Frontend–backend integration over HTTP  
- Basic production-style monitoring concepts  
- Model documentation via a model card  

This mirrors how **real ML systems are shipped**, not notebook demos.

---

## 🧠 Model Overview

- **Algorithm:** Logistic Regression  
- **Pipeline Components:**
  - Numerical features: median imputation + standard scaling
  - Categorical features: most-frequent imputation + one-hot encoding
- **Evaluation:**
  - Stratified 5-fold cross-validation
  - ROC-AUC as the primary metric
- **Interpretability:**
  - SHAP summary bar plot

The trained pipeline and metadata are saved and versioned.

---

## 📂 Project Structure
```
titanic-ml/
│
├── api/              # Backend service
│ ├── app.py          # FastAPI inference server
│ ├── init.py
│ └── model/          # Versioned ML artifacts
│ ├── titanic_best_pipeline.joblib
│ ├── titanic_metadata.json
│ └── init.py
│
├── artifacts/        # Model artifacts
│ └── shap_summary_bar.png
│
├── data/
│ └── train.csv       # Training dataset
│
├── docs/             # GitHub Pages frontend
│ ├── index.html      # Static UI consuming live API
│ └── model_card.md   # Model documentation
│
├── main.py           # Training + evaluation pipeline
├── requirements.txt  # Python dependencies
├── render.yaml       # Render deployment config
└── README.md
```
---

## 🔄 System Flow
```
User (Browser)
↓
GitHub Pages (HTML + JS)
↓ REST API
Render FastAPI Service
↓
Saved ML Pipeline (joblib)
↓
Prediction Response (JSON)

```
The frontend **never accesses the model directly** — it only communicates with the backend API.

---

## 🚀 Running the Project Locally

### 1️⃣ Create a virtual environment
```
python -m venv .venv
source .venv/bin/activate
```
2️⃣ Install dependencies
```
pip install -r requirements.txt
```
3️⃣ Train the model (creates artifacts)
```
python main.py
```
This step will:

➤ train the model  
➤ perform cross-validation  
➤ generate SHAP plots  
➤ save the pipeline & metadata into `api/model/`


4️⃣ Run the API locally
```
uvicorn api.app:app --reload
```
Visit:
```
http://127.0.0.1:8000 → health check
```

POST /predict → predictions

🔮 API Usage
```
POST /predict
```
Request Body
```
{
  "data": [3, "male", 30, 0, 0, 7.25, "S"]
}
```
Response
```
{
  "survived": false,
  "probability_survived": 0.34
}
```
## 🧾 Why These Files Exist

| File | Purpose |
|------|--------|
| `main.py` | Training, cross-validation, tuning, SHAP, artifact saving |
| `api/app.py` | Production inference API |
| `*.joblib` | Serialized ML pipeline |
| `titanic_metadata.json` | Features, parameters, metrics, versioning |
| `render.yaml` | Render deployment configuration |
| `docs/index.html` | Static frontend UI |
| `model_card.md` | Model documentation |


## 📊 Monitoring (Basic)
Each prediction can be logged server-side with:
- input values
- timestamp
- predicted probability

This simulates basic production monitoring patterns.

## ⚠️ Limitations
- Model trained on historical Titanic data
- Assumes clean and well-formed inputs
- Not calibrated for fairness or bias analysis
- Intended for learning and demonstration purposes

## ✅ Project Status
- ✔ Model trained
- ✔ Interpretable
- ✔ Deployed
- ✔ Documented
- ✔ Accessible via UI

---
## 🔮 Future Improvements

- Improve UI and user experience  
- Add input validation and error handling  
- Add logging and basic monitoring  
- Try stronger models and recalibrate performance  
