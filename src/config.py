"""Shared constants describing the Heart Disease dataset's schema.

Column semantics (UCI Heart Disease / Cleveland, id=45):
    age       continuous  age in years
    sex       categorical 1 = male, 0 = female
    cp        categorical chest pain type (4 values)
    trestbps  continuous  resting blood pressure (mm Hg)
    chol      continuous  serum cholesterol (mg/dl)
    fbs       categorical fasting blood sugar > 120 mg/dl (1 = true)
    restecg   categorical resting ECG results (3 values)
    thalach   continuous  max heart rate achieved
    exang     categorical exercise-induced angina (1 = yes)
    oldpeak   continuous  ST depression induced by exercise relative to rest
    slope     categorical slope of the peak exercise ST segment (3 values)
    ca        categorical number of major vessels colored by fluoroscopy (0-3)
    thal      categorical thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
    num       original multi-class target (0 = no disease, 1-4 = increasing severity)
    target    binarized target used for classification (0 = no disease, 1 = disease present)
"""

RAW_TARGET_COL = "num"
TARGET_COL = "target"

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

FEATURE_COLUMNS = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
