import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# 1. Load your NSL-KDD data
# Ensure your CSV has columns named: 'protocol_type', 'service', 'flag', ..., 'label'
df = pd.read_csv("KDDTrain+.csv") 

# 2. Preprocessing: Encode Categorical Features
# You MUST save these encoders to apply them to live traffic later!
encoders = {}
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 3. Handle Labels (Binary Classification: Normal=0, Attack=1)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# 4. Separate Features (X) and Target (y)
X = df.drop('label', axis=1)
y = df['label']

# 5. Train Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 6. Save the Model and Encoders
joblib.dump(model, 'model.joblib')
joblib.dump(encoders, 'encoders.joblib')
print("Model and encoders saved successfully!")