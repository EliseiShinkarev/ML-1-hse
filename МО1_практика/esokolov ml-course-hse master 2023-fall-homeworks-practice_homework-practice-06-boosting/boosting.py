from __future__ import annotations

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        base_model = self.base_model_class(**self.base_model_params)
        derivative = -self.loss_derivative(y, predictions)

        size = int(x.shape[0] * self.subsample)
        indices = np.random.choice(x.shape[0], size=size)
        base_model.fit(x[indices], derivative[indices])
        new_gamma = self.find_optimal_gamma(y, predictions, base_model.predict(x))

        self.gammas.append(new_gamma)
        self.models.append(base_model)


    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        rounds_cnt = 0

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(x_valid)
            train_loss = self.loss_fn(train_predictions, y_train)
            self.history['train_loss'].append(train_loss)
            val_loss = self.loss_fn(y_valid, valid_predictions)
            self.history['val_loss'].append(val_loss)

            if self.early_stopping_rounds is not None:
                score = score(x_valid, y_valid)
                print("round number {} with score = {}".format(rounds_cnt, score))
                rounds_cnt += 1
                if rounds_cnt > self.early_stopping_rounds:
                    break


        if self.plot:
            plt.figure(figsize=(12, 6))
            for index, (loss, value) in enumerate(self.history.items()):
                plt.subplot(1, 2, index + 1)
                plt.xlabel('step')
                plt.ylabel('loss value')
                plt.title(loss)
                plt.plot(value)

    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])

        for gamma, model in zip(self.gammas, self.models):
            predictions += self.learning_rate * gamma * model.predict(x)
        
        probs = np.zeros((x.shape[0], 2))
        probs[:, 1] = self.sigmoid(predictions)
        probs[:, 0] = 1 - probs[:, 1]

        return probs


    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        pass
