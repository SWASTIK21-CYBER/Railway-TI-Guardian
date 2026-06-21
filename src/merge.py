import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. SMART PATH SETUP (Prevents FileNotFoundError)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(os.path.dirname(current_dir), 'data')

train_path = os.path.join(data_folder, 'KDDTrain+.txt')
test_path = os.path.join(data_folder, 'KDDTest+.txt')

# ==========================================
# 2. LOADING THE NSL-KDD DATASET
# ==========================================
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'label', 'difficulty_level'
]

print("Loading datasets...")
# Reads only your 6 chosen features plus the label/difficulty columns
train_data = pd.read_csv(train_path, names=columns, header=None, usecols=[0, 1, 2, 3, 4, 5, 41, 42])
test_data = pd.read_csv(test_path, names=columns, header=None, usecols=[0, 1, 2, 3, 4, 5, 41, 42])

# Map labels to binary values: 0 for normal network traffic, 1 for a threat attack
train_data['target'] = train_data['label'].apply(lambda x: 0 if x.strip() == 'normal' else 1)
test_data['target'] = test_data['label'].apply(lambda x: 0 if x.strip() == 'normal' else 1)

# Split features from target labels
X_train = train_data.drop(columns=['label', 'difficulty_level', 'target'])
y_train = train_data['target']

X_test = test_data.drop(columns=['label', 'difficulty_level', 'target'])
y_test = test_data['target']

# ==========================================
# 3. DATA PREPROCESSING (Encoding & Scaling)
# ==========================================
categorical_cols = ['protocol_type', 'service', 'flag']
numerical_cols = ['duration', 'src_bytes', 'dst_bytes']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

print("Preprocessing features...")
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# ==========================================
# 4. HYBRID STAGE 1: Isolation Forest Scoring
# ==========================================
print("Training Isolation Forest to extract anomaly scores...")
iso_forest = IsolationForest(n_estimators=200, contamination=0.42, random_state=42, n_jobs=-1)
iso_forest.fit(X_train_encoded)

# Generate raw outlier structural scores (-1 means highly anomalous, 1 means structured normal)
train_anomaly_scores = iso_forest.decision_function(X_train_encoded).reshape(-1, 1)
test_anomaly_scores = iso_forest.decision_function(X_test_encoded).reshape(-1, 1)

# Combine the original features with the new structural anomaly column
X_train_hybrid = np.hstack((X_train_encoded, train_anomaly_scores))
X_test_hybrid = np.hstack((X_test_encoded, test_anomaly_scores))

# ==========================================
# 5. HYBRID STAGE 2: Supervised Random Forest
# ==========================================
print("Training Random Forest on hybrid dataset features...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_hybrid, y_train)

# ==========================================
# 6. MODEL PERFORMANCE EVALUATION
# ==========================================
print("\nGenerating Performance Reports...")
y_pred = rf_model.predict(X_test_hybrid)

accuracy = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

print("\n=================== HYBRID MODEL EVALUATION ===================")
print(f"Final Combined Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix Layout:")
print(f"True Normals Correctly Flagged:  {conf[0][0]}")
print(f"False Alarms (Normal as Threat): {conf[0][1]}")
print(f"Missed Threats (Threat as Normal): {conf[1][0]}")
print(f"True Threats Correctly Flagged:  {conf[1][1]}\n")

print("Detailed Classification Matrix:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Threat (1)']))
print("===============================================================")