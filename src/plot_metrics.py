import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from joblib import load

# Load model and test data
model = load("model.joblib")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Predict
y_pred = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot and save
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("metrics.png")

# Write report
with open("report.md", "w") as f:
    f.write("## 📊 Model Evaluation\n")
    f.write(f"Accuracy: {model.score(X_test, y_test):.2f}\n")
    f.write("![Confusion Matrix](metrics.png)\n")

print("✅ Confusion matrix and report saved.")
