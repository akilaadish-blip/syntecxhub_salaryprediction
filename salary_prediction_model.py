"""
Salary Prediction - Regression Model
Project 2: Build regression models to predict salary from experience, test scores, and other features.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. DATASET GENERATION
# ─────────────────────────────────────────────
np.random.seed(42)
n = 200

experience    = np.random.uniform(0, 20, n)
test_scores   = np.random.uniform(50, 100, n)
education     = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n,
                                  p=[0.15, 0.45, 0.30, 0.10])
department    = np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], n)

edu_map   = {'High School': 0, 'Bachelor': 10000, 'Master': 20000, 'PhD': 35000}
dept_map  = {'Engineering': 15000, 'Marketing': 5000, 'Sales': 8000, 'HR': 2000, 'Finance': 10000}

salary = (
    30000
    + experience * 2500
    + (test_scores - 50) * 300
    + np.array([edu_map[e]  for e in education])
    + np.array([dept_map[d] for d in department])
    + np.random.normal(0, 5000, n)
)

df = pd.DataFrame({
    'experience':  experience,
    'test_score':  test_scores,
    'education':   education,
    'department':  department,
    'salary':      salary
})

print("=" * 60)
print("         SALARY PREDICTION — REGRESSION PROJECT")
print("=" * 60)
print(f"\nDataset shape : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nStatistics:\n{df.describe().round(2)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor('#0f172a')
for ax in axes.flat:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

ACCENT = '#38bdf8'
ACCENT2 = '#f472b6'

# Experience vs Salary
axes[0,0].scatter(df['experience'], df['salary']/1000, alpha=0.6, color=ACCENT, s=25)
z = np.polyfit(df['experience'], df['salary']/1000, 1)
p = np.poly1d(z)
x_line = np.linspace(df['experience'].min(), df['experience'].max(), 100)
axes[0,0].plot(x_line, p(x_line), color=ACCENT2, lw=2)
axes[0,0].set_title('Experience vs Salary', color='white', fontsize=12, pad=10)
axes[0,0].set_xlabel('Years of Experience', color='#94a3b8')
axes[0,0].set_ylabel('Salary (k$)', color='#94a3b8')

# Test Score vs Salary
axes[0,1].scatter(df['test_score'], df['salary']/1000, alpha=0.6, color='#34d399', s=25)
z2 = np.polyfit(df['test_score'], df['salary']/1000, 1)
p2 = np.poly1d(z2)
x_line2 = np.linspace(df['test_score'].min(), df['test_score'].max(), 100)
axes[0,1].plot(x_line2, p2(x_line2), color=ACCENT2, lw=2)
axes[0,1].set_title('Test Score vs Salary', color='white', fontsize=12, pad=10)
axes[0,1].set_xlabel('Test Score', color='#94a3b8')
axes[0,1].set_ylabel('Salary (k$)', color='#94a3b8')

# Salary distribution
axes[0,2].hist(df['salary']/1000, bins=30, color=ACCENT, edgecolor='#0f172a', alpha=0.85)
axes[0,2].set_title('Salary Distribution', color='white', fontsize=12, pad=10)
axes[0,2].set_xlabel('Salary (k$)', color='#94a3b8')
axes[0,2].set_ylabel('Count', color='#94a3b8')

# Education boxplot
edu_order = ['High School','Bachelor','Master','PhD']
edu_data  = [df[df['education']==e]['salary'].values/1000 for e in edu_order]
bp = axes[1,0].boxplot(edu_data, patch_artist=True, labels=edu_order)
colors = ['#38bdf8','#34d399','#f472b6','#fb923c']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for element in ['whiskers','caps','medians','fliers']:
    plt.setp(bp[element], color='white')
axes[1,0].set_title('Salary by Education', color='white', fontsize=12, pad=10)
axes[1,0].set_xlabel('Education', color='#94a3b8')
axes[1,0].set_ylabel('Salary (k$)', color='#94a3b8')
axes[1,0].tick_params(axis='x', labelrotation=15, colors='#94a3b8')

# Department bar
dept_avg = df.groupby('department')['salary'].mean().sort_values(ascending=False) / 1000
bars = axes[1,1].bar(dept_avg.index, dept_avg.values, color=[ACCENT,'#34d399','#f472b6','#fb923c','#a78bfa'])
axes[1,1].set_title('Avg Salary by Department', color='white', fontsize=12, pad=10)
axes[1,1].set_xlabel('Department', color='#94a3b8')
axes[1,1].set_ylabel('Avg Salary (k$)', color='#94a3b8')
axes[1,1].tick_params(axis='x', labelrotation=20, colors='#94a3b8')
for bar in bars:
    axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   f'{bar.get_height():.0f}k', ha='center', va='bottom', color='white', fontsize=9)

# Correlation heatmap (numeric)
corr = df[['experience','test_score','salary']].corr()
im = axes[1,2].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[1,2].set_xticks(range(len(corr))); axes[1,2].set_yticks(range(len(corr)))
axes[1,2].set_xticklabels(corr.columns, color='#94a3b8', rotation=20)
axes[1,2].set_yticklabels(corr.columns, color='#94a3b8')
for i in range(len(corr)):
    for j in range(len(corr)):
        axes[1,2].text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
axes[1,2].set_title('Correlation Matrix', color='white', fontsize=12, pad=10)

fig.suptitle('Salary Dataset — Exploratory Data Analysis', color='white', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/home/claude/salary_prediction/eda_plots.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("\n[✓] EDA plot saved.")

# ─────────────────────────────────────────────
# 3. PREPROCESSING & SPLITTING
# ─────────────────────────────────────────────
X = df.drop('salary', axis=1)
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

num_features  = ['experience', 'test_score']
cat_features  = ['education', 'department']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
])

# ─────────────────────────────────────────────
# 4. MODEL 1 — Single Feature (experience only)
# ─────────────────────────────────────────────
single_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['experience'])
])
single_pipe = Pipeline([
    ('pre', single_preprocessor),
    ('model', LinearRegression())
])
single_pipe.fit(X_train[['experience']], y_train)
y_pred_single = single_pipe.predict(X_test[['experience']])
rmse_single = np.sqrt(mean_squared_error(y_test, y_pred_single))
r2_single   = r2_score(y_test, y_pred_single)

print("\n── Model 1: Single Feature (experience) ──")
print(f"   RMSE : ${rmse_single:,.0f}")
print(f"   R²   : {r2_single:.4f}")

# ─────────────────────────────────────────────
# 5. MODEL 2 — Two Numeric Features
# ─────────────────────────────────────────────
two_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features)
])
two_pipe = Pipeline([
    ('pre', two_preprocessor),
    ('model', LinearRegression())
])
two_pipe.fit(X_train[num_features], y_train)
y_pred_two = two_pipe.predict(X_test[num_features])
rmse_two = np.sqrt(mean_squared_error(y_test, y_pred_two))
r2_two   = r2_score(y_test, y_pred_two)

print("\n── Model 2: Two Numeric Features ──")
print(f"   RMSE : ${rmse_two:,.0f}")
print(f"   R²   : {r2_two:.4f}")

# ─────────────────────────────────────────────
# 6. MODEL 3 — Full Multiple Regression (all features + categorical)
# ─────────────────────────────────────────────
full_pipe = Pipeline([
    ('pre', preprocessor),
    ('model', LinearRegression())
])
full_pipe.fit(X_train, y_train)
y_pred_full = full_pipe.predict(X_test)
rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_full))
r2_full   = r2_score(y_test, y_pred_full)

print("\n── Model 3: Full Model (all features + categorical OHE) ──")
print(f"   RMSE : ${rmse_full:,.0f}")
print(f"   R²   : {r2_full:.4f}")

# Feature importance
ohe_cats  = full_pipe.named_steps['pre'].named_transformers_['cat'].get_feature_names_out(cat_features)
feat_names = num_features + list(ohe_cats)
coefs      = full_pipe.named_steps['model'].coef_
coef_df    = pd.DataFrame({'Feature': feat_names, 'Coefficient': coefs}).sort_values('Coefficient', key=abs, ascending=False)
print(f"\nTop 10 Feature Coefficients:\n{coef_df.head(10).to_string(index=False)}")

# ─────────────────────────────────────────────
# 7. COMPARISON SUMMARY
# ─────────────────────────────────────────────
results = pd.DataFrame({
    'Model':        ['Single (experience)', 'Two Numeric', 'Full Model'],
    'Features':     [1, 2, len(feat_names)],
    'RMSE ($)':     [rmse_single, rmse_two, rmse_full],
    'R² Score':     [r2_single, r2_two, r2_full]
})
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results.to_string(index=False))

best_idx   = results['R² Score'].idxmax()
best_model = results.loc[best_idx, 'Model']
print(f"\n🏆  Best Model: {best_model}  (R²={results.loc[best_idx,'R² Score']:.4f})")

# ─────────────────────────────────────────────
# 8. RESULT VISUALIZATIONS
# ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.patch.set_facecolor('#0f172a')
for ax in axes2.flat:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

# RMSE Comparison
model_labels = ['Single\n(experience)', 'Two\nNumeric', 'Full\nModel']
rmse_vals    = [rmse_single, rmse_two, rmse_full]
bar_colors   = ['#f472b6', '#fb923c', '#38bdf8']
bars2 = axes2[0,0].bar(model_labels, [r/1000 for r in rmse_vals], color=bar_colors, alpha=0.85, edgecolor='#0f172a')
axes2[0,0].set_title('RMSE Comparison (lower = better)', color='white', fontsize=12, pad=10)
axes2[0,0].set_ylabel('RMSE (k$)', color='#94a3b8')
for bar in bars2:
    axes2[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f'${bar.get_height():.1f}k', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

# R² Comparison
r2_vals = [r2_single, r2_two, r2_full]
bars3   = axes2[0,1].bar(model_labels, r2_vals, color=bar_colors, alpha=0.85, edgecolor='#0f172a')
axes2[0,1].set_title('R² Score Comparison (higher = better)', color='white', fontsize=12, pad=10)
axes2[0,1].set_ylabel('R² Score', color='#94a3b8')
axes2[0,1].set_ylim(0, 1.05)
for bar in bars3:
    axes2[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

# Actual vs Predicted (full model)
axes2[1,0].scatter(y_test/1000, y_pred_full/1000, alpha=0.6, color=ACCENT, s=30, label='Predictions')
lims = [min(y_test.min(), y_pred_full.min())/1000, max(y_test.max(), y_pred_full.max())/1000]
axes2[1,0].plot(lims, lims, '--', color=ACCENT2, lw=2, label='Perfect Fit')
axes2[1,0].set_title('Actual vs Predicted — Full Model', color='white', fontsize=12, pad=10)
axes2[1,0].set_xlabel('Actual Salary (k$)', color='#94a3b8')
axes2[1,0].set_ylabel('Predicted Salary (k$)', color='#94a3b8')
axes2[1,0].legend(facecolor='#334155', labelcolor='white')

# Residuals
residuals = y_test - y_pred_full
axes2[1,1].scatter(y_pred_full/1000, residuals/1000, alpha=0.6, color='#34d399', s=30)
axes2[1,1].axhline(0, color=ACCENT2, lw=2, linestyle='--')
axes2[1,1].set_title('Residual Plot — Full Model', color='white', fontsize=12, pad=10)
axes2[1,1].set_xlabel('Predicted Salary (k$)', color='#94a3b8')
axes2[1,1].set_ylabel('Residuals (k$)', color='#94a3b8')

fig2.suptitle('Model Evaluation & Comparison', color='white', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/salary_prediction/model_evaluation.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("\n[✓] Evaluation plot saved.")

# ─────────────────────────────────────────────
# 9. SAVE BEST MODEL
# ─────────────────────────────────────────────
joblib.dump(full_pipe, '/home/claude/salary_prediction/best_salary_model.pkl')
print("[✓] Best model saved → best_salary_model.pkl")

# ─────────────────────────────────────────────
# 10. SAMPLE PREDICTIONS
# ─────────────────────────────────────────────
sample = pd.DataFrame({
    'experience': [2, 8, 15],
    'test_score': [60, 75, 90],
    'education':  ['Bachelor', 'Master', 'PhD'],
    'department': ['Marketing', 'Engineering', 'Finance']
})
preds = full_pipe.predict(sample)
sample['predicted_salary'] = preds.round(2)
print("\nSample Predictions:")
print(sample.to_string(index=False))
print("\n[✓] All done!")