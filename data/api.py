from fastapi import FastAPI, Response # 1. Import Response
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load models...
rf_model = joblib.load('hybrid_rf_model.joblib')
iso_forest = joblib.load('isolation_forest.joblib')
preprocessor = joblib.load('preprocessor.joblib')

@app.post("/predict")
def predict_traffic(duration: float, src_bytes: float, dst_bytes: float, protocol_type: str, service: str, flag: str):
    data = pd.DataFrame([[duration, protocol_type, service, flag, src_bytes, dst_bytes]], 
                        columns=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes'])
    
    encoded_data = preprocessor.transform(data)
    anomaly_score = iso_forest.decision_function(encoded_data)
    hybrid_features = np.hstack((encoded_data, anomaly_score.reshape(-1, 1)))
    prediction = rf_model.predict(hybrid_features)
    
    # 2. Logic to tell NGINX what to do
    if prediction[0] == 1:
        return Response(status_code=403) # Block the traffic
    else:
        return Response(status_code=200) # Allow the traffic
    
    print(f"Prediction: {prediction[0]}")