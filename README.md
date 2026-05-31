# syntecxhub_salaryprediction
# 💼 Salary Prediction — ML Regression App

An **interactive, browser-based ML dashboard** that trains and compares Linear Regression models on a salary dataset — featuring data exploration, model benchmarking, coefficient analysis, and a live salary predictor. All in a single self-contained HTML file.

![ML Pipeline](https://img.shields.io/badge/ML-Linear%20Regression-00e5a0?style=flat-square)
![R²](https://img.shields.io/badge/R%C2%B2-0.924-00e5a0?style=flat-square)
![RMSE](https://img.shields.io/badge/RMSE-%245%2C694-ffd166?style=flat-square)
![Samples](https://img.shields.io/badge/Dataset-200%20samples-4f8fff?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-64748b?style=flat-square)

-----

## 📸 Demo

Open `salary_prediction_app.html` directly in any modern browser — no server, no install, no dependencies (Chart.js loaded from CDN).

-----

## 🚀 Features

- **4-tab interactive dashboard** — Overview, Data Explorer, Model Comparison, Live Predictor
- **3 models benchmarked** — single feature, two numeric, and full model with categorical encoding
- **Live salary predictor** — select experience, test score, education, and department for instant estimates
- **Salary component breakdown** — see how each feature contributes to the final prediction
- **EDA charts** — salary by education, by department, histogram, and scatter with regression line
- **Residual plot + Actual vs. Predicted** — visualize model quality beyond just R²
- **Fully responsive dark UI** — works on desktop and mobile

-----

## 🗂️ Dashboard Tabs

### 1 — Overview

- Key metric cards: dataset size, average salary, best R², best RMSE
- Model performance summary table with rankings
- Top feature coefficient bar chart

### 2 — Data Explorer

- Average salary by education level (High School → PhD)
- Average salary by department (Engineering, Marketing, Sales, HR, Finance)
- Salary distribution histogram (200 samples)
- Experience vs. salary scatter plot with regression line

### 3 — Model Comparison

- R² score comparison across all 3 models
- RMSE comparison (lower = better)
- Actual vs. Predicted scatter plot — full model
- Residuals plot — errors distributed around zero

### 4 — Live Predictor

- Input: years of experience, test score, education level, department
- Output: estimated annual salary with 95% confidence range
- Salary component breakdown (base + experience contribution + education bonus + department bonus)
- Gauge chart showing prediction vs. population range

-----

## 📊 Model Performance

|Model           |Features               |RMSE      |R² Score |Rank    |
|----------------|-----------------------|----------|---------|--------|
|Single feature  |experience only        |$13,098   |0.596    |Weak    |
|Two numeric     |experience + test_score|$13,283   |0.585    |OK      |
|**Full model ✦**|**all features + OHE** |**$5,694**|**0.924**|**Best**|


> Adding One-Hot Encoded categorical features (education + department) cut RMSE by **57%** and nearly doubled the R² score.

-----

## 🔍 Key Findings

|Feature               |Coefficient|Effect                            |
|----------------------|-----------|----------------------------------|
|`education_PhD`       |+$23,134   |Strongest positive predictor      |
|`experience`          |+$14,578   |Each year adds ~$14.5k            |
|`education_Master`    |+$8,995    |Solid mid-tier premium            |
|`test_score`          |+$4,139    |Positive but secondary            |
|`department_HR`       |−$10,637   |Largest negative department effect|
|`education_HighSchool`|−$10,158   |Significant salary penalty        |
|`department_Marketing`|−$8,947    |Below-average pay                 |
|`department_Sales`    |−$5,589    |Moderate negative                 |
|`department_Finance`  |−$1,702    |Near baseline                     |

-----

## 🗂️ Dataset

- **200 synthetic samples** modeled on real-world salary patterns
- **Target**: Annual salary (USD)

|Feature     |Type       |Description                               |
|------------|-----------|------------------------------------------|
|`experience`|Numeric    |Years of work experience                  |
|`test_score`|Numeric    |Aptitude/skills test score (50–100)       |
|`education` |Categorical|High School, Bachelor, Master, PhD        |
|`department`|Categorical|Engineering, Marketing, Sales, HR, Finance|

**Preprocessing:**

- `StandardScaler` on numeric features
- `OneHotEncoder` on categorical features (drop first to avoid multicollinearity)
- Combined via `sklearn` `ColumnTransformer` pipeline

-----

## 🛠️ Tech Stack

|Layer       |Technology                                 |
|------------|-------------------------------------------|
|UI          |HTML5, CSS3 (custom properties, animations)|
|Logic       |Vanilla JavaScript (ES6+)                  |
|Charts      |Chart.js 4.4.1                             |
|Scatter/SVG |Inline SVG (hand-drawn, no library)        |
|Fonts       |DM Mono, Syne (Google Fonts)               |
|ML (backend)|Python 3.11 · scikit-learn                 |


> Model coefficients are pre-computed from a trained sklearn pipeline and embedded in the JS for browser-side inference.

-----

## 📁 File Structure

```
salary-prediction/
└── salary_prediction_app.html    # Entire app — self-contained
```

-----

## ⚡ Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/salary-prediction.git

# Open in browser — no build step needed
open salary_prediction_app.html
```

-----

## 🧩 Python Backend (reproduce the model)

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df = pd.read_csv('salary_data.csv')

numeric_features     = ['experience', 'test_score']
categorical_features = ['education', 'department']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

X = df[numeric_features + categorical_features]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print(f"R²   : {r2_score(y_test, y_pred):.3f}")
print(f"RMSE : ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
```

-----

## 📄 License

MIT — free to use, modify, and distribute.

-----

## 🙌 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) — ML pipeline & preprocessing
- [Chart.js](https://www.chartjs.org/) — interactive charts
- Dataset inspired by common HR analytics benchmarks