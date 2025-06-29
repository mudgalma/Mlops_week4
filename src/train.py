import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ✅ Load data from CSV
df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["target"]).values
y = df["target"].values

# ✅ Encode labels (in case target isn't numeric in future)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# ✅ Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ✅ Save model and encoder
dump(model, "model.joblib")
dump(encoder, "encoder.joblib")

# ✅ Save test data for evaluation later
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("✅ Model trained and files saved.")

