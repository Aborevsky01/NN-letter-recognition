import math
from Runge_Kutta import runge_kutta_system



def active_function(y_t, weights_coef):
    # f(y, theta, t)
    inp = y_t * weights_coef
    if -8 < inp < 7.5:
        return (1 - math.exp(-inp)) / (1 + math.exp(-inp))
    elif inp <= -8:
        return -1
    return 1


def adjoint_function(y_t, a_t, weights_coef):
    # a'(t) = -a(t) * 2e^-y(t)theta(t) / (1+e^-y(t)theta(t))^2
    input_y = y_t * weights_coef
    if -10 <= a_t * input_y <= 10:
        return -a_t * 2 * math.exp(-input_y) * weights_coef / pow(1 + math.exp(-input_y), 2)
    return 0


class HiddenNeuron:
    def __init__(self, step):  # нужно добавлять время видимо
        self.weights_prev = []
        self.weights_next = []
        self.z_values = []
        self.a_values = []
        self.result_z = 0
        self.result_a = 0
        self.step = step

    def calculate(self, direction):
        flag = 1 if direction == 'forward' else -1
        self.result_z, self.result_a = \
            runge_kutta_system(self.a_values, self.z_values, self.weights_prev, adjoint_function, active_function,
                               flag * self.step)


class InputNeuron():
    def __init__(self, step):
        self.weights_prev = []
        self.z_values = []
        self.a_values = []
        self.weights_next = []
        self.result_z = 0
        self.result_a = 0
        self.step = step

    def calculate(self, direction):
        flag = 1 if direction == 'forward' else -1
        self.result_z, self.result_a = \
            runge_kutta_system(self.a_values, self.z_values, self.weights_prev, adjoint_function, active_function,
                               flag * self.step)


class OutputNeuron:
    def __init__(self, step):  # нужно добавлять время видимо
        self.weights_prev = []
        self.weights_next = []
        self.z_values = []
        self.a_values = []
        self.result_z = 0
        self.result_a = 0
        self.step = step

    def calculate(self, direction):
        flag = 1 if direction == 'forward' else -1
        self.result_z, self.result_a = \
            runge_kutta_system(self.a_values, self.z_values, self.weights_prev, adjoint_function, active_function,
                               flag * self.step)
