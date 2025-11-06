from sklearn.model_selection import ParameterSampler, train_test_split
from scipy.stats import randint
from sklearn.datasets import load_iris
import mlflow
from base import train_and_evaluate_model

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 11)
}

with mlflow.start_run(run_name="Eksperimen Random Search"):
    for params in ParameterSampler(param_distributions, n_iter=10):
        train_and_evaluate_model(params, X_train, y_train, X_test, y_test)