import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_railway_toy_dataset(n_samples=1000, anomaly_ratio=0.05):
    np.random.seed(42)
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies
    
    normal_data = {
        'duration': np.random.exponential(scale=2, size=n_normal),
        'src_bytes': np.random.normal(loc=500, scale=100, size=n_normal),
        'dst_bytes': np.random.normal(loc=1200, scale=200, size=n_normal),
        'count': np.random.poisson(lam=10, size=n_normal),
        'srv_serror_rate': np.random.uniform(low=0.0, high=0.05, size=n_normal)
    }
    df_normal = pd.DataFrame(normal_data)
    df_normal['ground_truth'] = 1
    
    anomaly_data = {
        'duration': np.random.uniform(low=50, high=300, size=n_anomalies),
        'src_bytes': np.random.exponential(scale=8000, size=n_anomalies),
        'dst_bytes': np.random.normal(loc=50, scale=10, size=n_anomalies),
        'count': np.random.randint(low=200, high=512, size=n_anomalies),
        'srv_serror_rate': np.random.uniform(low=0.8, high=1.0, size=n_anomalies)
    }
    df_anomaly = pd.DataFrame(anomaly_data)
    df_anomaly['ground_truth'] = -1
    
    toy_df = pd.concat([df_normal, df_anomaly]).sample(frac=1).reset_index(drop=True)
    return toy_df

if __name__ == "__main__":
    print("Step 1: Generating synthetic railway packet toy dataset...")
    contamination_rate = 0.05
    df = generate_railway_toy_dataset(n_samples=1000, anomaly_ratio=contamination_rate)
    
    X = df.drop(columns=['ground_truth'])
    y_true = df['ground_truth']
    
    print("\nStep 2: Training unsupervised Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination=contamination_rate, 
        random_state=42
    )
    
    iso_forest.fit(X)
    
    print("\nStep 3: Predicting anomalies and scoring model metrics...")
    y_pred = iso_forest.predict(X)
    
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, -1])
    
    print("\n================== RESULTS ==================")
    print(f"Overall Accuracy Score: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix:")
    print("                Predicted Normal   Predicted Threat")
    print(f"Actual Normal       {conf_matrix[0][0]:<18} {conf_matrix[0][1]}")
    print(f"Actual Threat       {conf_matrix[1][0]:<18} {conf_matrix[1][1]}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Threat (-1)', 'Normal (1)']))