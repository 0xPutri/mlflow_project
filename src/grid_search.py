from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.datasets import load_iris
import mlflow
from base import train_and_evaluate_model

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

with mlflow.start_run(run_name="Eksperimen Grid Search"):
    for params in ParameterGrid(param_grid):
        train_and_evaluate_model(params, X_train, y_train, X_test, y_test)