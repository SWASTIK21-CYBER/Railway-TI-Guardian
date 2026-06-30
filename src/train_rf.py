import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols) 
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1 
)

rf_model.fit(X_train_encoded, y_train)


y_pred = rf_model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")

unique_classes = sorted(y_test.unique())
target_names = ['Normal', 'Threat'] if len(unique_classes) == 2 else None
print(classification_report(y_test, y_pred, target_names=target_names))