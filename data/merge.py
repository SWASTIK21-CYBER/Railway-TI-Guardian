import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. PATH SETUP & SMART FILE CHECK
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(os.path.dirname(current_dir), 'data')

train_path = os.path.join(data_folder, 'KDDTrain+.txt')
test_path = os.path.join(data_folder, 'KDDTest+.txt')

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'label', 'difficulty_level'
]

print("Loading datasets...")
# Load the training data (we know this file exists!)
train_data = pd.read_csv(train_path, names=columns, header=None, usecols=[0, 1, 2, 3, 4, 5, 41, 42])

# Check if the test file actually exists
if os.path.exists(test_path):
    print("-> Found KDDTest+.txt! Loading test file...")
    test_data = pd.read_csv(test_path, names=columns, header=None, usecols=[0, 1, 2, 3, 4, 5, 41, 42])
else:
    print("-> KDDTest+.txt not found. Automatically splitting KDDTrain+.txt for testing...")
    # Dynamically split the training data into 80% train and 20% test
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)

# ==========================================
# 2. LABEL MAPPING
# ==========================================
# Map labels to binary values: 0 for normal traffic, 1 for a threat
train_data['target'] = train_data['label'].apply(lambda x: 0 if x.strip() == 'normal' else 1)
test_data['target'] = test_data['label'].apply(lambda x: 0 if x.strip() == 'normal' else 1)

# Split features from targets
X_train = train_data.drop(columns=['label', 'difficulty_level', 'target'])
y_train = train_data['target']

X_test = test_data.drop(columns=['label', 'difficulty_level', 'target'])
y_test = test_data['target']

# ==========================================
# 3. DATA PREPROCESSING
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

# Generate raw anomaly scores
train_anomaly_scores = iso_forest.decision_function(X_train_encoded).reshape(-1, 1)
test_anomaly_scores = iso_forest.decision_function(X_test_encoded).reshape(-1, 1)

# Append the anomaly score as a brand new feature feature column
X_train_hybrid = np.hstack((X_train_encoded, train_anomaly_scores))
X_test_hybrid = np.hstack((X_test_encoded, test_anomaly_scores))

# ==========================================
# 5. HYBRID STAGE 2: Random Forest Classifier
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
# 6. PERFORMANCE EVALUATION
# ==========================================
print("\nGenerating Performance Reports...")
y_pred = rf_model.predict(X_test_hybrid)

accuracy = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

print("\n=================== HYBRID MODEL EVALUATION ===================")
print(f"Final Combined Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(f"True Normals Correctly Flagged:  {conf[0][0]}")
print(f"False Alarms (Normal as Threat): {conf[0][1]}")
print(f"Missed Threats (Threat as Normal): {conf[1][0]}")
print(f"True Threats Correctly Flagged:  {conf[1][1]}\n")

print("Detailed Classification Matrix:")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Threat (1)']))
print("===============================================================")
import joblib

print("Saving models to disk...")
# 1. Save your Random Forest model
joblib.dump(rf_model, 'hybrid_rf_model.joblib')

# 2. Save your Isolation Forest model (using its real name: iso_forest)
joblib.dump(iso_forest, 'isolation_forest.joblib')

joblib.dump(preprocessor, 'preprocessor.joblib')

print("Models saved successfully!")