"""Classical baseline training."""

from time import perf_counter

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_classical_model(x_train, y_train, seed: int = 42):
    start = perf_counter()
    model = make_pipeline(
        StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed)
    )
    model.fit(x_train, y_train)
    return model, perf_counter() - start
