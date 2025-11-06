import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

with mlflow.start_run(run_name="Eksperimen K-Fold Cross Validation"):
    mlflow.log_param("n_splits", n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        with mlflow.start_run(nested=True, run_name=f"Fold-{fold+1}"):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model.fit(X_train_fold, y_train_fold)

            y_pred_fold = model.predict(X_val_fold)
            accuracy = accuracy_score(y_val_fold, y_pred_fold)
            scores.append(accuracy)
            mlflow.log_metric("accuracy", accuracy)

    avg_acc = np.mean(scores)
    mlflow.log_metric("average_accuracy", avg_acc)
    print(f"Rata-rata akurasi k-Fold: {avg_acc}")