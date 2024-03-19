from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        prev_w = self.w
        self.w = prev_w - self.lr.__call__() * gradient
        return self.w - prev_w

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        grad = np.transpose(y - x @ self.w) @ x
        l = x.shape[0]
        return 2 * grad / l

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        l = x.shape[0]
        loss = (1 / l) * np.transpose(y - x @ self.w) @ (y - x @ self.w)

        if self.loss_function is LossFunction.LogCosh:
            loss = (1 / l) * np.log(np.cosh(x @ self.w - y)).sum()
        
        if self.loss_function is LossFunction.MAE:
            loss = (1 / l) * np.abs(x @ self.w - y).sum()
                
        if self.loss_function is LossFunction.Huber:
            delta = 0.5
            indicator = np.abs(x @ self.w - y) <= delta
            cur_x = x[indicator]
            cur_y = y[indicator]
            # w = self.w[indicator]
            w = self.w
            loss = 0
            loss += ((cur_x @ w - cur_y) ** 2).sum() / 2
            indicator = np.abs(x @ self.w - y) > delta
            cur_x = x[indicator]
            cur_y = y[indicator]
            # w = self.w[indicator]
            loss += delta * (np.abs(cur_x @ w - cur_y) - delta / 2).sum()
        return loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        prev_w = self.w
        self.w = prev_w - self.lr.__call__() * gradient
        return self.w - prev_w

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        l = x.shape[0]
        if self.loss_function is LossFunction.LogCosh:
            return (1 / l) * x.T @ np.tanh(x @ self.w - y)
        
        if self.loss_function is LossFunction.MAE:

            return (1 / l) * x.T @ np.sign(x @ self.w - y)
        
        if self.loss_function is LossFunction.Huber:
            delta = 0.5
            indicator = np.abs(x @ self.w - y) <= delta
            cur_x = x[indicator]
            cur_y = y[indicator]
            # w = self.w[indicator]
            w = self.w
            grad = (1 / l) * cur_x.T @ (cur_x @ w - cur_y)
            indicator = np.abs(x @ self.w - y) > delta
            cur_x = x[indicator]
            cur_y = y[indicator]
            # w = self.w[indicator]
            grad += (delta / l) * cur_x.T @ np.sign(cur_x @ w - cur_y)
            return grad
            # if indicator:
            #     return (1 / l) * x.T @ (x @ self.w - y)
            # else:
            #     return (delta / l) * x.T @ np.sign(x @ self.w - y)

        grad = x.T @ (x @ self.w - y) * 2 / l

        return grad


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        mask = np.random.randint(x.shape[0], size=self.batch_size)
        cur_x = x[mask]
        cur_y = y[mask]
        return super().calc_gradient(cur_x, cur_y)


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        prev_w = self.w

        self.h = self.alpha * self.h + self.lr.__call__() * gradient

        self.w = prev_w - self.h

        return self.w - prev_w


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
            
        prev_w = self.w
        eta_t = self.lr.__call__()
        g_t = gradient
        np.add(self.beta_1 * self.m, (1 - self.beta_1) * g_t, out=self.m)
        np.add(self.beta_2 * self.v, (1 - self.beta_2) * g_t * g_t, out=self.v)
        m_hat = self.m / (1 - self.beta_1 ** self.iteration)
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)
        
        np.add(self.w, -1 * eta_t * m_hat / ((v_hat) ** 0.5 + self.eps) , out=self.w)

        return -1 * eta_t * m_hat / ((v_hat) ** 0.5 + self.eps)


class Nadam(Adam):
    """
    NADAM, or Nesterov-accelerated Adaptive Moment Estimation, combines Adam and Nesterov Momentum. 
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
            
        prev_w = self.w
        eta_t = self.lr.__call__()
        g_t = gradient
        np.add(self.beta_1 * self.m, (1 - self.beta_1) * g_t, out=self.m)
        np.add(self.beta_2 * self.v, (1 - self.beta_2) * g_t * g_t, out=self.v)
        m_hat = self.beta_1 * self.m + (1 - self.beta_1) * g_t / (1 - self.beta_1 ** self.iteration)
        v_hat = self.v / (1 - self.beta_2 ** self.iteration)
        np.add(self.w, -1 * eta_t * m_hat / ((v_hat) ** 0.5 + self.eps) , out=self.w)

        return -1 * eta_t * m_hat / ((v_hat) ** 0.5 + self.eps)


class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = self.w
        l2_gradient[-1] = 0

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """

class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

class NadamReg(BaseDescentReg, Nadam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'nadam': Nadam if not regularized else NadamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
