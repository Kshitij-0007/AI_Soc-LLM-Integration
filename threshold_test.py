
import pandas as pd
from sklearn.metrics import confusion_matrix

path = "models/netflow/demo/edge_case_predictions.csv"
df = pd.read_csv(path)

actual = df["label"].str.lower() == "malicious"
probability = df["malicious_probability"]

print("Threshold | Precision | Recall | False positives | False negatives")
print("-" * 72)

for threshold in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]:
    predicted = probability >= threshold

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[False, True]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(
        f"{threshold:9.2f} | {precision:9.3f} | "
        f"{recall:6.3f} | {fp:14d} | {fn:14d}"
    )
