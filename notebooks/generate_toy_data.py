import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

def generate_nsl_kdd_toy_data(num_samples=200):
    np.random.seed(42)
    
    # Key NSL-KDD categorical features
    protocols = ['tcp', 'udp', 'icmp']
    services = ['http', 'ftp', 'smtp', 'private', 'domain']
    flags = ['SF', 'S0', 'REJ', 'RSTR']
    
    data = {
        # Continuous and discrete numerical features
        'duration': np.random.exponential(scale=10, size=num_samples),
        'protocol_type': np.random.choice(protocols, size=num_samples),
        'service': np.random.choice(services, size=num_samples),
        'flag': np.random.choice(flags, size=num_samples),
        'src_bytes': np.random.randint(0, 5000, size=num_samples),
        'dst_bytes': np.random.randint(0, 5000, size=num_samples),
        'land': np.random.choice([0, 1], size=num_samples, p=[0.99, 0.01]),
        'wrong_fragment': np.random.choice([0, 1, 3], size=num_samples, p=[0.95, 0.03, 0.02]),
        'urgent': np.random.choice([0, 1], size=num_samples, p=[0.99, 0.01]),
        'count': np.random.randint(1, 512, size=num_samples),
        'srv_count': np.random.randint(1, 512, size=num_samples),
        'serror_rate': np.random.uniform(0.0, 1.0, size=num_samples),
        'srv_serror_rate': np.random.uniform(0.0, 1.0, size=num_samples),
        'same_srv_rate': np.random.uniform(0.0, 1.0, size=num_samples),
        'diff_srv_rate': np.random.uniform(0.0, 1.0, size=num_samples),
        'dst_host_count': np.random.randint(1, 256, size=num_samples),
        'dst_host_srv_count': np.random.randint(1, 256, size=num_samples),
        'dst_host_same_srv_rate': np.random.uniform(0.0, 1.0, size=num_samples),
        'dst_host_diff_srv_rate': np.random.uniform(0.0, 1.0, size=num_samples),
        
        # Target label: 0 for normal (safe railway operations), 1 for threat (anomaly)
        'label': np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
    }
    
    return pd.DataFrame(data)

# Generate train and test toy sets
train_df = generate_nsl_kdd_toy_data(300)
test_df = generate_nsl_kdd_toy_data(100)

# Save to your data directory
train_df.to_csv('../data/train_toy.csv', index=False)
test_df.to_csv('../data/test_toy.csv', index=False)

print("Toy datasets successfully saved to the 'data/' directory!")