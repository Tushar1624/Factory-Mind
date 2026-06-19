# 🏭 Factory Mind

An AI-powered Predictive Maintenance System that uses Machine Learning to predict potential machine failures before they occur, helping reduce downtime, improve operational efficiency, and support proactive maintenance decisions.

## 🚀 Live Demo

**Application:** https://factory-mind.onrender.com/

**Source Code:** https://github.com/Tushar1624/Factory-Mind

---

# 📖 Overview

Factory Mind is a machine learning-powered web application designed to analyze machine operational data and predict equipment failures before they happen.

The project combines data preprocessing, machine learning, model persistence, and an interactive web dashboard to provide a complete predictive maintenance workflow.

The goal is to help industries move from reactive maintenance to proactive maintenance by identifying potential issues early.

---

# ✨ Features

* Predictive maintenance using Machine Learning
* Random Forest based failure prediction
* Interactive Streamlit dashboard
* Real-time prediction interface
* Data preprocessing pipeline
* Feature scaling and model training
* Saved model deployment using Joblib
* Cloud deployment with Render
* Beginner-friendly project structure
* Reproducible training workflow

---

# 🛠️ Tech Stack

## Machine Learning

* Scikit-Learn
* Random Forest Classifier
* Joblib

## Data Processing

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn

## Web Application

* Streamlit

## Deployment

* Render

---

# 📂 Project Structure

```text
Factory-Mind/
│
├── dashboard/
│   └── app.py
│
├── dataset/
│   └── machine_data.csv
│
├── docs/
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── check_data.py
│   ├── data_preprocessing.py
│   ├── predict.py
│   └── train_model.py
│
├── .gitignore
├── LICENSE
├── README.md
├── render.yaml
└── requirements.txt
```

---

# ⚙️ Installation Guide

Follow these steps to run the project locally.

## Step 1: Clone the Repository

```bash
git clone https://github.com/Tushar1624/Factory-Mind.git
```

---

## Step 2: Move into the Project Directory

```bash
cd Factory-Mind
```

---

## Step 3: Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5: Run the Application

```bash
streamlit run dashboard/app.py
```

---

## Step 6: Open in Browser

```text
http://localhost:8501
```

---

# 🧠 Machine Learning Workflow

The machine learning pipeline follows the following process:

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Feature Scaling
   ↓
Random Forest Training
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

---

# 🔄 Retraining the Model

If you want to retrain the machine learning model using the dataset:

## Run Data Preprocessing

```bash
python src/data_preprocessing.py
```

## Train the Model

```bash
python src/train_model.py
```

After training, new model files will be generated inside:

```text
models/
```

---

# 🌐 Deployment

The project is currently deployed on Render.

Live Application:

https://factory-mind.onrender.com/

## Render Configuration

Build Command:

```bash
pip install -r requirements.txt
```

Start Command:

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0
```

---

# 📦 Dependencies

Main project dependencies:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# 🤝 Contributing

Contributions are welcome.

To contribute:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Push the branch
6. Open a Pull Request

---

# 🗺️ Future Improvements

Planned enhancements:

* Model performance comparison
* XGBoost implementation
* REST API support
* Docker containerization
* User authentication
* Real-time monitoring dashboard
* Advanced analytics
* Cloud database integration

---

# 📜 License

This project is licensed under the MIT License.

See the LICENSE file for details.

---

# 👨‍💻 Author

Developed as a Machine Learning and Predictive Maintenance project for learning, experimentation, and real-world deployment.

If you find this project useful, consider giving it a ⭐ on GitHub.
