import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'label', 'difficulty_level'
]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'data', 'KDDTrain+.txt')

# Load the dataset using the dynamic path
df = pd.read_csv(data_path, names=columns, header=None)

df['ground_truth'] = df['label'].apply(lambda x: 1 if x.strip() == 'normal' else -1)

numerical_features = ['duration', 'src_bytes', 'dst_bytes'] 
X = df[numerical_features]
y_true = df['ground_truth']
# Calculate the contamination rate (percentage of anomalies)
num_threats = np.sum(y_true == -1)
total_samples = len(y_true)
contamination_rate = num_threats / total_samples

print("\nTraining Isolation Forest model... Please wait...")
# Initialize and fit the model
model = IsolationForest(contamination=contamination_rate, random_state=42, n_jobs=-1)
model.fit(X)

print("Evaluating model predictions...")
# Predict flags (-1 for threat, 1 for normal)
y_pred = model.predict(X)

# Compute evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
conf = confusion_matrix(y_true, y_pred, labels=[1, -1])

# Print the final results
print("\n================ EVALUATION REPORT ================")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(f"True Normals Correctly Flagged:  {conf[0][0]}")
print(f"False Alarms (Normal as Threat): {conf[0][1]}")
print(f"Missed Threats (Threat as Normal): {conf[1][0]}")
print(f"True Threats Correctly Flagged:  {conf[1][1]}")

print("\nDetailed Classification Performance:")
print(classification_report(y_true, y_pred, target_names=['Threat (-1)', 'Normal (1)']))
print("===================================================")