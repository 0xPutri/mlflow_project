from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from base import train_and_evaluate_model

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 200, 1)),
    'max_depth': hp.choice('max_depth', [None, scope.int(hp.quniform('max_depth_int', 10, 20, 1))]),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1))
}

def objective(params):
    with mlflow.start_run(nested=True):
        accuracy = train_and_evaluate_model(params, X_train, y_train, X_test, y_test)
        return {'loss': -accuracy, 'status': STATUS_OK}

with mlflow.start_run(run_name="Eksperimen Bayesian Optimization"):
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
    best_params = {}
    for k, v in trials.best_trial['misc']['vals'].items():
        if isinstance(v, list) and len(v) > 0:
            best_params[k] = v[0]
        elif not isinstance(v, list):
            best_params[k] = v
    mlflow.log_params(best_params)