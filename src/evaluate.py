import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    train_data = pd.read_csv('../data/train_toy.csv')
    test_data = pd.read_csv('../data/test_toy.csv')
except FileNotFoundError:
    train_data = pd.read_csv('data/train_toy.csv')
    test_data = pd.read_csv('data/test_toy.csv')

X_train = train_data.drop(columns=['label'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

categorical_cols = ['protocol_type', 'service', 'flag']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_encoded, y_train)


y_pred = rf_model.predict(X_test_encoded)

accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 50)

print("Detailed Performance Report:")
print(classification_report(y_test, y_pred, target_names=['Normal Traffic (0)', 'Railway Threat (1)']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Normal', 'Predicted Threat'],
            yticklabels=['Actual Normal', 'Actual Threat'])
plt.ylabel('True Network Category')
plt.xlabel('Model Prediction')
plt.title('Railway Threat Detection: Confusion Matrix')
plt.show()





















