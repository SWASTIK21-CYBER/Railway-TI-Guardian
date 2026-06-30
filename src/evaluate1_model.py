import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler  

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'label', 'difficulty_level'
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data', 'KDDTrain+.txt')

df = pd.read_csv(data_path, names=columns, header=None, usecols=[0, 1, 2, 3, 4, 5, 41, 42])

df['ground_truth'] = df['label'].apply(lambda x: 1 if x.strip() == 'normal' else -1)

features = df[['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes']]

X_encoded = pd.get_dummies(features, columns=['protocol_type', 'service', 'flag'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

y_true = df['ground_truth']

num_threats = np.sum(y_true == -1)
total_samples = len(y_true)
contamination_rate = num_threats / total_samples

print("\nTraining upgraded Isolation Forest model... Please wait...")

model = IsolationForest(
    n_estimators=200, 
    contamination=contamination_rate, 
    random_state=42, 
    n_jobs=-1
)

model.fit(X_scaled)

print("Evaluating model predictions...")
y_pred = model.predict(X_scaled)

accuracy = accuracy_score(y_true, y_pred)
conf = confusion_matrix(y_true, y_pred, labels=[1, -1])

print("\n================ UPGRADED EVALUATION REPORT ================")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(f"True Normals Correctly Flagged:  {conf[0][0]}")
print(f"False Alarms (Normal as Threat): {conf[0][1]}")
print(f"Missed Threats (Threat as Normal): {conf[1][0]}")
print(f"True Threats Correctly Flagged:  {conf[1][1]}")

print("\nDetailed Classification Performance:")
print(classification_report(y_true, y_pred, target_names=['Threat (-1)', 'Normal (1)']))
print("============================================================")