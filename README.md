# Heart-Disease-Analysis-Clinical-Risk-Insights

##  About This Project

Created as a **Python EDA + ML assessment project** to demonstrate:
- Clinical data exploration and medical feature interpretation
- Multi-section analytical storytelling with visual dashboards
- Statistical hypothesis testing (T-test for group significance)
- End-to-end ML pipeline: preprocessing → training → evaluation → visualization
- Translating model outputs into actionable clinical recommendations

A Python-based **Exploratory Data Analysis (EDA) + Predictive Modelling** project built in Jupyter Notebook, analyzing **303 patients across 14 clinical features** to identify heart disease risk factors and build an early-warning classification model using Logistic Regression.

---

##  Dataset & Key KPIs

| Metric | Value |
|---|---|
| Total Patients | 303 |
| Clinical Features | 14 |
| Patients WITH Heart Disease | ~54% |
| Male Patients | ~68% |
| Average Age | ~54 years |
| Average Cholesterol | ~246 mg/dL |
| Average Resting Blood Pressure | ~131 mmHg |
| Average Max Heart Rate | ~149 bpm |
| Exercise-Induced Angina Rate | ~32% |
| Age–Cholesterol Correlation | Weak positive |

---

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data loading, cleaning, aggregation |
| NumPy | Numerical operations |
| Matplotlib | Multi-panel dashboard charts |
| Seaborn | Heatmaps, statistical plots |
| Scikit-learn | Logistic Regression, train/test split, StandardScaler, metrics |
| SciPy | T-test for statistical significance testing |
| Jupyter Notebook | Development & presentation environment |

---

##  Predictive Model — Logistic Regression

| Component | Detail |
|---|---|
| Model | Logistic Regression (`liblinear` solver) |
| Features Used | All 13 clinical features (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) |
| Target | `target` (1 = Heart Disease, 0 = No Disease) |
| Train/Test Split | 80% / 20% (random_state=42) |
| Preprocessing | StandardScaler on 5 continuous variables |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC |

The model achieves **strong AUC performance** using only routine clinical variables — making it viable for real-world clinical triage and early warning applications.

---

##  Key Findings

**Thalassemia (reversible defect) is the #1 predictor** — Patients with `thal=3` (reversible defect) show the strongest positive correlation with heart disease in the dataset, ahead of all other features.

**Asymptomatic chest pain carries the highest hidden risk** — Chest pain type 3 (asymptomatic) has the highest oldpeak values and is strongly associated with disease — making it a dangerous silent indicator.

**Major vessels (ca) dramatically escalate risk** — Each additional blocked major vessel (`ca ≥ 1`) sharply increases disease likelihood. Fluoroscopy findings are among the most decisive clinical signals.

**Exercise-induced angina suppresses max heart rate** — Patients with `exang=1` show significantly reduced maximum heart rate, indicating cardiac stress response and marking them as high-risk candidates.

**ST depression (oldpeak) is a continuous risk signal** — Higher oldpeak correlates strongly with disease presence. It is one of the top features in both EDA and model coefficients.

**Male patients have significantly higher disease rates** — Gender is a statistically meaningful predictor, with male patients far more likely to present with positive diagnoses.

**Age and declining heart rate form a dual warning signal** — As age increases, max heart rate (thalach) falls. This divergence is visible in scatter plots and is a powerful non-invasive screening indicator.

**Blood pressure differs significantly by sex (T-test)** — The two-sample T-test confirms a statistically significant difference in resting blood pressure between male and female patients (p < 0.05).

---

##  Dashboard Output Files

The notebook generates **8 saved visualization figures**:

| Output File | Contents |
|---|---|
| `demographics.png` | Disease split donut, gender distribution, age histogram |
| `clinical_indicators.png` | Cholesterol, max heart rate, blood pressure comparisons |
| `chest_thal_vessels.png` | Chest pain types, thalassemia, major vessels, ST slope |
| `trends.png` | Age vs max heart rate scatter, ECG pattern trends |
| `correlations.png` | Full 14×14 heatmap + target correlation bar chart |
| `model_evaluation.png` | Confusion matrix, ROC-AUC curve, feature coefficients |
| `dashboard_page1.png` | Master clinical risk dashboard — 3×3 grid |
| `dashboard_page2.png` | Advanced clinical deep-dive dashboard |

---

##  Clinical Recommendations

1. **Prioritise thal=3 patients for immediate cardiac workup** — Reversible thalassemia defect is the strongest single predictor in this dataset. Automatic escalation to a cardiologist is warranted for all such cases.

2. **Flag exang=1 patients for stress testing** — Exercise-induced angina is both a strong predictor and a signal of reduced cardiac reserve. These patients need stress echocardiogram or nuclear stress test.

3. **Make fluoroscopy (ca) central to risk scoring** — The number of major vessels coloured by fluoroscopy is among the most decisive features. It should be a required input in any clinical risk scoring tool built on this model.

4. **Monitor oldpeak continuously in at-risk groups** — ST depression during exercise is a continuous, trackable signal. Patients with rising oldpeak over successive visits should trigger escalated review.

5. **Implement routine screening for asymptomatic chest pain (CP type 3)** — These patients feel no obvious discomfort yet carry the highest hidden risk. Symptom-driven care models will systematically miss them.

6. **Adopt gender-specific screening intervals** — Male patients face significantly higher disease rates. Clinics should apply earlier and more frequent cardiac screening protocols for male patients in the 45–65 age range.

7. **Use thalach-age divergence as a non-invasive screen** — The age vs max heart rate scatter is a low-cost, non-invasive indicator that can be plotted from routine vitals — no additional testing required.

8. **Deploy the logistic regression model as a triage layer** — With strong AUC from only 13 routine clinical variables, this model is immediately usable for pre-visit risk stratification or automated triage flagging in outpatient settings.

---



## Author

**[Mohd Faiz]**

- **Role:** Business and Data Analyst
- **GitHub:** [https://github.com/faizikbal01-lab](https://github.com/your-github-handle)
