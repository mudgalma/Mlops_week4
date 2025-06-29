import numpy as np
from joblib import load


def test_model_prediction_shape():
    model = load("model.joblib")
    X_test = np.load("X_test.npy")
    y_pred = model.predict(X_test)
    assert len(y_pred) == len(X_test)
