import math
import numpy as np

def runge_kutta_system(adjoint, y_initial, weights_coef, function_a, function_y, step):
    # a'(t) = f_a(a, y, t) - k
    # y'(t) = f_y(a, y, t) - l
    a_t = adjoint
    y_t = y_initial
    for t in range(20):
        k_1 = np.array([function_a(y_t[i], a_t[i], weights_coef[i]) for i in range(len(y_t))])
        l_1 = np.array([function_y(y_t[i], weights_coef[i]) for i in range(len(y_t))])

        k_2 = np.array([function_a(y_t[i] + l_1[i] / 2, a_t[i] + k_1[i] / 2, weights_coef[i]) for i in range(len(y_t))])
        l_2 = np.array([function_y(y_t[i] + l_1[i] / 2, weights_coef[i]) for i in range(len(y_t))])

        k_3 = np.array([function_a(y_t[i] + l_2[i] / 2, a_t[i] + k_2[i] / 2, weights_coef[i]) for i in range(len(y_t))])
        l_3 = np.array([function_y(y_t[i] + l_2[i] / 2, weights_coef[i]) for i in range(len(y_t))])

        k_4 = np.array([function_a(y_t[i] + l_3[i], a_t[i] + k_3[i], weights_coef[i]) for i in range(len(y_t))])
        l_4 = np.array([function_y(y_t[i] + l_3[i], weights_coef[i]) for i in range(len(y_t))])

        a_t += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * step / 6
        y_t += (l_1 + 2 * l_2 + 2 * l_3 + l_4) * step / 6

    res_y = 0
    res_a = 0
    for i in range(len(y_t)):
        res_y += y_t[i] * weights_coef[i]
        res_a += a_t[i]

    return res_y, res_a
