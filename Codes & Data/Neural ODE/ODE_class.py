import torch
import math
import random
import numpy as np
from numpy import linalg
from numpy.linalg import LinAlgError
from Neurons import HiddenNeuron, OutputNeuron, InputNeuron


def der_func_theta(zw_value, z_value, a_value):
    return a_value * 2 * math.exp(-zw_value) * z_value / pow(1 + math.exp(-zw_value), 2)


def generator(t, i, j):
    result = math.cos(t) * i / 4 - j / 4
    return random.uniform(2, 7) if result == 0 else result


class ODEFunc:
    def __init__(self, time, input_size):
        self.time = time  # время t0-tn
        self.struct = []
        self.runge_steps_count = 20  # количество шагов между слоями для рунге-кутта
        self.time_middle = len(self.time) // 2  # время t* откуда идут метод пристрелки
        self.create_structure(self, input_size)
        self.create_weights(self)
        self.desired = []
        self.prepare_shooting(self, 'user')
        self.matrix = np.zeros((2 * input_size, 2 * input_size))

    @staticmethod
    def create_structure(self, input_size):
        self.steps = float((self.time[1] - self.time[0]) / self.runge_steps_count)  # длина шага в рунге-кутта
        self.struct = [[InputNeuron(self.steps) for _ in range(input_size)]]  # задаем входной слой
        for i in range(1, len(self.time) - 1):
            length = input_size if i == self.time_middle else random.randint(1, 4)
            self.struct.append([HiddenNeuron(self.steps) for _ in range(length)])  # задаем скрытые слои
        self.struct.append([OutputNeuron(self.steps) for _ in range(input_size)])  # задаем выходной слой

    @staticmethod
    def create_weights(self):
        for position in range(len(self.time) - 1):
            for i in range(len(self.struct[position])):
                weight = []
                for j in range(len(self.struct[position + 1])):
                    weight.append(generator(self.time[position], i + 1, j + 1))
                self.struct[position][i].weights_next = weight  # задаем нейронам веса к следующему слою

    @staticmethod
    def prepare_shooting(self, mode, *args):
        if mode == 'user':
            self.param = [[], []]
            print("Please, enter input of length", len(self.struct[self.time_middle]),
                  "for initial adjoint values")
            for i in range(len(self.struct[self.time_middle])):
                self.struct[self.time_middle][i].result_a = float(input())
            print("Please, enter input of length", len(self.struct[self.time_middle]),
                  "for initial z_function values")
            for i in range(len(self.struct[self.time_middle])):
                self.struct[self.time_middle][i].result_z = float(input())
        elif mode == 'update':
            for i in range(len(self.struct[self.time_middle])):
                self.struct[self.time_middle][i].result_a -= args[0][0][i]
                self.struct[self.time_middle][i].result_z -= args[0][0][i + len(args[0] // 2)]

    def shooting_back(self, binary_data):
        for i in range(self.time_middle, 0, -1):  # от t*-tn
            for j in range(len(self.struct[i])):  # левый слой
                right_neuron = self.struct[i][j]  # конкретный нейрон левого слоя
                for y in range(len(self.struct[i - 1])):  # передали z(t) * weight следующему слою
                    self.struct[i - 1][y].z_values.append(right_neuron.result_z)
                    self.struct[i - 1][y].a_values.append(right_neuron.result_a)
                    #  правый слой имеет все веса и значения функций
            for j in range(len(self.struct[i - 1])):
                self.struct[i - 1][j].weights_prev = self.struct[i - 1][j].weights_next
                self.struct[i - 1][j].calculate('back')  # посчитали значения вперед
        return binary_data - np.array([self.struct[0][i].result_z for i in range(len(self.struct[0]))])

    def shooting_forward(self):
        for i in range(self.time_middle, len(self.struct) - 1):  # от t*-tn
            for j in range(len(self.struct[i])):  # левый слой
                left_neuron = self.struct[i][j]  # конкретный нейрон левого слоя
                for y in range(len(self.struct[i + 1])):  # передали z(t) * weight следующему слою
                    self.struct[i + 1][y].weights_prev.append(left_neuron.weights_next[y])
                    self.struct[i + 1][y].z_values.append(left_neuron.result_z)
                    self.struct[i + 1][y].a_values.append(left_neuron.result_a)
                    #  правый слой имеет все веса и значения функций
            for j in range(len(self.struct[i + 1])):
                self.struct[i + 1][j].calculate('forward')  # посчитали значения вперед
        return np.array([self.struct[-1][i].result_a + self.struct[-1][i].result_z - self.desired[i]
                         for i in range(len(self.struct[-1]))])

    @staticmethod
    def clear(self):
        for i in range(len(self.struct)):
            for j in range(len(self.struct[i])):
                self.struct[i][j].a_values = []
                self.struct[i][j].z_values = []

    # TODO: self.desired and self.binary data
    # TODO: how to unite newton update into one code
    # TODO: убрать вычисления a(T) назад

    @staticmethod
    def newton_update(self, board_z, board_a, binary_data):
        noise = 1
        for i in range(len(self.struct[self.time_middle])):
            self.struct[self.time_middle][i].result_a += noise
            lb_noise = self.shooting_back(binary_data)
            for j in range(len(lb_noise)):
                lb_noise[j] = (lb_noise[j] - board_z[j])
            rb_noise = self.shooting_forward()
            for j in range(len(rb_noise)):
                rb_noise[j] = (rb_noise[j] - board_a[j])
            self.matrix[i] = np.hstack([lb_noise, rb_noise])
            self.struct[self.time_middle][i].result_a -= noise
            self.clear(self)
        for i in range(len(self.struct[self.time_middle])):
            self.struct[self.time_middle][i].result_z += noise
            lb_noise = self.shooting_back(binary_data)
            for j in range(len(lb_noise)):
                lb_noise[j] = (lb_noise[j] - board_z[j])
            rb_noise = self.shooting_forward()
            for j in range(len(rb_noise)):
                rb_noise[j] = (rb_noise[j] - board_a[j])
            self.matrix[i + len(self.struct[self.time_middle])] = np.hstack([lb_noise, rb_noise])
            self.struct[self.time_middle][i].result_z -= noise
            self.clear(self)
        try:
            self.matrix = np.linalg.inv(self.matrix)
        except LinAlgError:
            print(self.matrix)
            exit(1)
            return self.newton_method(binary_data, 0)
        return self.matrix.dot(np.reshape(np.hstack([board_a, board_z]), (len(board_a) * 2, 1)))

       def correct_weights(self):
        for i in range(len(self.struct) - 1):
            sum_z_start = []
            sum_a_start = []
            sum_z_finish = []
            sum_a_finish = []
            for j in range(len(self.struct[i])):
                sum_z_start.append(sum(self.struct[i][j].z_values))
                sum_a_start.append(sum(self.struct[i][j].a_values))
                sum_z_finish.append(sum(self.struct[i][j].resukt_z))
                sum_a_finish.append(sum(self.struct[i][j].result_a))
            for j in range(len(self.struct[i+1])):
                right_neuron = self.struct[i+1][j]
                zw_t1 = np.array(right_neuron.coef) * np.array(sum_z_finish)
                zw_t0 = np.array(right_neuron.coef) * np.array(sum_z_start)
                for y in range(len(self.struct[i])):
                    t_0 = der_func_theta(zw_t0, sum_z_start[y], sum_a_start)
                    t_1 = der_func_theta(zw_t1, sum_z_finish[y], sum_a_finish)
                    self.struct[i][y].weights_next[j] -= 0.1 * (t_1-t_0)

    def newton_method(self, binary_data, counter):
        right_board_a = self.shooting_forward()
        left_board_z = self.shooting_back(binary_data)
        print("№{number}: left board - {value} and right board - {value2}".format
              (number=counter, value=left_board_z, value2=right_board_a))
        self.clear(self)
        correcting = np.reshape(self.newton_update(self, left_board_z, right_board_a, binary_data),
                                (1, 2 * len(self.struct[-1])))
        self.clear(self)
        if counter > 15:
            print("Number of Newton iterations exceeded the limit. Choose another initial values")
            self.prepare_shooting(self, 'user')
            return self.newton_method(binary_data, 0)
        self.prepare_shooting(self, 'update', correcting)
        if linalg.norm(correcting) > 0.1:
            self.newton_method(binary_data, counter + 1)
        self.shooting_forward()
        self.shooting_back(binary_data)
        self.correct_weights()

    # TODO: расширить на все буквы алфавита и чтобы ввод не менее 26 символов

    def request(self, binary_data, letter):
        self.desired = [1, 0, 1, 0] if letter[0] == 'A' else [0, 1, 0, 1]
        binary_data = np.reshape(binary_data, (1, len(self.struct[0])))[0]
        self.newton_method(binary_data, 0)
        print("Done")
        print(self.struct[-1][0].result_z, self.struct[-1][1].result_z)


time_array = torch.tensor([math.pi / 2 * i for i in range(5)])
neural = ODEFunc(time_array, 2)
neural.request([[0, 1]], "O")
