"""Train the disturbance-record anomaly model (Isolation Forest) on the real record features.

    python train_record_model.py Fixed_Master_Relay_Dataset.xlsx

Input columns: Max_Fault_Current_A, Pre_Fault_Noise_Index (one row per disturbance record).
The data itself is not stored in the repository; only the trained model and its score distribution are."""
import sys, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
import relayguard_core as rc

df = pd.read_excel(sys.argv[1]) if sys.argv[1].endswith('xlsx') else pd.read_csv(sys.argv[1])
X = rc.record_matrix(df['Max_Fault_Current_A'].to_numpy(float), df['Pre_Fault_Noise_Index'].to_numpy(float))
m = IsolationForest(n_estimators=300, max_samples=256, contamination='auto', random_state=0).fit(X)
s = -m.score_samples(X)                       # higher = more unusual
bundle = {'model': m, 'train_scores': np.sort(s), 'n_records': int(len(X)),
          'feature_mean': X.mean(0), 'feature_cov': np.cov(X.T), 'version': 1}
pickle.dump(bundle, open('record_anomaly_model.pkl', 'wb'))
print(f"trained on {len(X)} records; score 50/90/99th percentile: {np.percentile(s, [50, 90, 99]).round(3)}")
