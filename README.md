# 🩺 Interactive Symptom Checker (AI + 3D Human Body)

An AI-powered healthcare demo combining:

- **Interactive 3D human body** (React + Three.js) → select pain/discomfort location.
- **Symptom input form** → severity, duration, description.
- **Fine-tuned ClinicalBERT** → predicts possible diseases or conditions
- **FastAPI backend** → serves ML predictions.

⚠️ **For research/demo only. Not medical advice.**

---

## 🚀 Setup

### 1. Dataset
Download free Kaggle dataset:

```bash
pip install kaggle
kaggle datasets download -d itachi9604/disease-symptom-description-dataset -p ml/data/raw
unzip ml/data/raw/*.zip -d ml/data/raw

``` 

### 2. Run the preprocessing python script 
This python script will clean up and format the datasets used into a consistent format that the model can use to train. 

```bash 
cd ml-model 
python preprocess.py

```
### 3 . Train the model 
```
python train.py

```
### 4. FastAPI Backend
  
  ```
  cd ml-service
uvicorn app.main:app --reload --port 8000

  
  ```

### 5. React Frontend
  ```
cd frontend
npm install
npm start
  ```


## 🛠️ Tech Stack (subject to change) 

- **Frontend:** React, Three.js, React Three Fiber, Axios  
- **Backend:** FastAPI, Uvicorn  
- **ML:** Hugging Face Transformers, Bio/ClinicalBERT, PyTorch  
- **Infra:** Docker

---

## ✅ Features (subject to change)

- Interactive 3D body selection  
- Symptom form with severity & duration  
- Fine-tuned ClinicalBERT from Hugging Face library
- REST API serving predictions  
- Optional containerized deployment  

---

## ⚠️ Disclaimer

This project is for research/demo only and **not intended for medical use**.


  




