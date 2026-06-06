# 🎓 Predictive Modeling Using Machine Learning
### Internship Task 2 — Thiranex | Student Performance Dataset

---

## 📌 Project Overview

This project builds and evaluates Machine Learning models to predict student academic performance based on demographic and academic features. It covers the full ML pipeline — from data loading and EDA to model training, evaluation, and visualization.

---

## 🗂️ Project Structure

```
StudentML_Project/
│
├── student_ml_project.py       # Main Python script
│
├── eda_distributions.png       # Score distribution plots
├── linear_regression_plot.png  # Actual vs Predicted scores
├── dt_confusion_matrix.png     # Decision Tree confusion matrix
├── rf_confusion_matrix.png     # Random Forest confusion matrix
├── roc_curve.png               # ROC AUC curve
├── feature_importance.png      # Feature importance chart
│
└── README.md                   # Project documentation
```

---

## 📊 Dataset

**Student Performance Dataset** — 500 student records with the following features:

| Feature | Description |
|--------|-------------|
| `gender` | Student gender (male/female) |
| `race_ethnicity` | Group A to E |
| `parental_education` | Highest education level of parent |
| `lunch` | Standard or free/reduced lunch |
| `test_preparation` | Completed or none |
| `math_score` | Math exam score (0–100) |
| `reading_score` | Reading exam score (0–100) |
| `writing_score` | Writing exam score (0–100) |
| `average_score` | Mean of all three scores *(derived)* |
| `pass_fail` | 1 = Pass (avg ≥ 60), 0 = Fail *(target)* |

---

## 🔧 Tech Stack

- **Language:** Python 3.x
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🪜 Steps Followed

1. **Import Libraries** — pandas, numpy, sklearn, matplotlib, seaborn
2. **Load Dataset** — 500 student records, 10 features
3. **Exploratory Data Analysis (EDA)** — distributions, missing values, class balance
4. **Data Preprocessing** — Label encoding of categorical columns
5. **Model Training & Evaluation**
   - Linear Regression → predict average score
   - Decision Tree Classifier → predict Pass/Fail
   - Random Forest Classifier → predict Pass/Fail
6. **ROC Curve** — model performance visualization
7. **Feature Importance** — identify key predictors

---

## 📈 Results

| Model | Metric | Score |
|-------|--------|-------|
| Linear Regression | R² Score | 1.0000 |
| Decision Tree | Accuracy | 81.00% |
| **Random Forest** | **Accuracy** | **90.00%** ✅ |
| Random Forest | AUC (ROC) | 0.9665 |

> ✅ **Random Forest** is the best performing model with **90% accuracy** and **AUC of 0.97**

---

## 📊 Output Visualizations

| Plot | Description |
|------|-------------|
| `eda_distributions.png` | Histogram of Math, Reading, Writing scores |
| `linear_regression_plot.png` | Actual vs Predicted average score |
| `dt_confusion_matrix.png` | Decision Tree — True/False predictions |
| `rf_confusion_matrix.png` | Random Forest — True/False predictions |
| `roc_curve.png` | ROC curve with AUC score |
| `feature_importance.png` | Top features ranked by importance |

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/StudentML_Project.git

# Navigate to folder
cd StudentML_Project

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the project
python student_ml_project.py
```

All 6 plots will be saved automatically in the same folder.

---

## 👩‍💻 Author

**Hasini**
B.Tech Information Technology — 3rd Year
J.B. Institute of Engineering and Technology, Hyderabad
Thiranex Internship — Task 2

---

## 📄 License

This project is submitted as part of an internship task and is intended for educational purposes.
