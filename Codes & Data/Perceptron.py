from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import shutil
%matplotlib inline
import random
from random import randint
from cmath import exp

# scan the names of the initial images and split them into the two groups: train and test datasets
universe = []

for file in os.scandir(r'A'):
    if file.name.endswith(".png"):
        universe.insert(-1, 'A/' + file.name)

for file in os.scandir(r'O'):
    if file.name.endswith(".png"):
        universe.insert(-1, 'O/' + file.name)

train, test = train_test_split(universe, test_size=0.3, train_size=0.7)

# create new folders for the train & test datasets
shutil.rmtree("train/A")
shutil.rmtree("train/O")
os.mkdir("train/A")
os.mkdir("train/O")

shutil.rmtree("test/A")
shutil.rmtree("test/O")
os.mkdir("test/A")
os.mkdir("test/O")

# minimize images to make them all of one particular size for the NN
for file in train:
    letter = Image.open(os.path.abspath(file))
    letter.thumbnail((10, 10), Image.ANTIALIAS)
    letter.save('train/{id}'.format(id=file), quality=95)

for file in test:
    letter = Image.open(os.path.abspath(file))
    letter.thumbnail((10, 10), Image.ANTIALIAS)
    letter.save('test/{id}'.format(id=file), quality=95)

separator=255
samples = []
data = np.zeros((10, 10))

#interpret of the train images as the binary arrays
for file in train:
    letter = Image.open(os.path.abspath('train/' + file))
    data = np.array(letter.convert('L'))
    data = data.astype('int')
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = 1 if data[i][j] != separator else -1
    samples.insert(-1, [data, file])


# calculate F'(sum) = F(sum) * (1 - F(sum)) = y(1 - y) for the sigmoid function
def derivative(value):
    temp = function(value)
    return (temp * (1 - temp))


# calculate the sigmoid function
def function(value):
    if (-8 < value < 7.5):
        return ((1 / (1 + exp(-value))).real)
    elif (value > -8):
        return 1
    return 0


# Neurons in the hidden layers initially accept a list of random values, representing the initial weights
class HiddenNeuron():
    def __init__(self, *args):
        self.weights = args[0]
        self.input = 0
        self.result = 0
        self.sigma = 0

    # Function add() is used to update neuron's sum - s
    def add(self, injection):
        self.input += injection

    # Function calculate() assigns F(s) to the unit's output - y
    def calculate(self):
        self.result = function(self.input)


# Input neurons initially accept not only the random weights, but also the value of assigned image's sector
class InputNeuron():
    def __init__(self, *args):
        self.input = 0
        self.weights = args[0]
        self.result = 0

    # Input neurons have a state of activation equal to the accepted value
    def calculate(self):
        self.result = self.input

    def assign(self, value):
        self.input = value


# Neurons in the output layer initially don't accept anything
class OutputNeuron():
    def __init__(self):
        self.input = 0
        self.input = 0
        self.result = 0
        self.sigma = 0

    # The following functions are identical to those in the HiddenNeuron class
    def add(self, injection):
        self.input += injection

    def calculate(self):
        self.result = function(self.input)

    # Neural network, when being created, accepts a frame of hidden layers - (*data)


class NeuralNetwork:
    def __init__(self, *data):
        self.layers = [[OutputNeuron()]]
        self.assumption = ''
        self.counter = 0
        self.cycles = 0
        self.max = [0, 0, '']
        self.correct = 0

        # One by one, hidden layers of set length are added to the network
        for i in range(len(data) - 1):
            self.layers.insert(0,
                               [HiddenNeuron([random.uniform(-1, 1) for y in range(len(self.layers[0]))]) for j in
                                range(data[i + 1])])

        input_layer = []
        for i in range(data[0]):
            input_layer.insert(-1, InputNeuron([random.uniform(-1, 1) for y in range(len(self.layers[0]))]))
        self.layers.insert(0, input_layer)

    # Image's name and binary array are sent to the network through this function
    def request(self, data, image_name):

        # Input layer is created and added to the neural network
        for i in range(len(data)):
            for j in range(len(data[0])):
                self.layers[0][len(data) * i + j].assign(data[i][j])
        # After the desired answer is defined, the learning process starts
        desired = 1 if image_name[0] == 'A' else 0
        self.forward_propagate(desired)
        self.counter += 1
        if self.max[0] < self.cycles:
            self.max[0] = self.cycles
            self.max[1] = self.counter
            self.max[2] = image_name
        self.clear()
        self.cycles = 0

    # First part of training - forward propagation without any changes in weights
    def forward_propagate(self, desired):
        self.cycles += 1
        for i in range(len(self.layers) - 1):
            for j in range(len(self.layers[i])):
                left_neuron = self.layers[i][j]
                left_neuron.calculate()
                for y in range(len(self.layers[i + 1])):
                    self.layers[i + 1][y].add(left_neuron.result * left_neuron.weights[y])
        self.layers[-1][0].calculate()
        self.assumption = 'A' if self.layers[-1][0].result == 1 else 'O'

        # Comparing the network's output with the desired answer
        # If they're unequal, back-propagation is initiated
        # The process repeats till the correct answer is given
        if abs(self.layers[-1][0].result - desired) > 0.1:
            self.back_propagate(desired)
            self.forward_propagate(desired)

    # Function updates the weight of last hidden layer and then transfers the rest of the process to hidden_propagation()
    def back_propagate(self, desired):
        last_one = self.layers[-1][0]
        last_one.sigma = (desired - last_one.result) * derivative(last_one.result)
        for i in range(len(self.layers[-2])):
            self.layers[-2][i].weights[0] += 1 * last_one.sigma * self.layers[-2][i].result
        self.hidden_propagation(len(self.layers) - 2)

    # The propagated error is calculated for the each neuron in a layer
    # After, the connections with the previous layer are updated
    def hidden_propagation(self, left):
        if left == 0: return
        for i in range(len(self.layers[left])):
            updated = self.layers[left][i]
            error = 0
            for j in range(len(updated.weights)):
                error += updated.weights[j] * self.layers[left + 1][j].sigma
            updated.sigma = derivative(updated.result) * error

        left -= 1
        for i in range(len(self.layers[left])):
            activated = self.layers[left][i]
            for j in range(len(activated.weights)):
                activated.weights[j] += 1 * self.layers[left + 1][j].sigma * activated.result

        return self.hidden_propagation(left)

    # NN's structure needs to be refreshed with all values set to 0 from time to time
    def clear(self):
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i])):
                self.layers[i][j].input = 0

    # Trying the NN into the real task: no training allowed, one chance for each request
    def testing(self, data, image_name):
        for i in range(len(data)):
            for j in range(len(data[0])):
                self.layers[0][len(data) * i + j].assign(data[i][j])

        answer = self.forward_test()
        self.assumption = 'A' if answer == 1 else 'O'
        self.counter += 1
        if image_name[0] == 'F':
            print("Testing № " + str(self.counter) + " completed. The requested image "
                  + image_name[6:] + " contained letter " + self.assumption + '.')
            if self.assumption == image_name[6]: self.correct += 1
        else:
            print("Testing № " + str(self.counter) + " completed. The requested image "
                  + image_name[2:] + " contained letter " + self.assumption + '.')
            if self.assumption == image_name[0]: self.correct += 1
        self.clear()

    def forward_test(self):
        for i in range(len(self.layers) - 1):
            for j in range(len(self.layers[i])):
                left_neuron = self.layers[i][j]
                left_neuron.calculate()
                for y in range(len(self.layers[i + 1])):
                    self.layers[i + 1][y].add(left_neuron.result * left_neuron.weights[y])
        self.layers[-1][0].calculate()
        return self.layers[-1][0].result

# Such a structure appeared to be one of the most realible
network = NeuralNetwork(100, 60, 60)

# During the working process it was essential to understand the maximum length of study
random.shuffle(train)
for sample in samples:
    network.request(sample[0], sample[1])
print(network.max)
print('---------')
network.counter = 0

separator=255
trials = []
data = np.zeros((10, 10))

#interpret of the train images as the binary arrays
for file in test:
    letter = Image.open(os.path.abspath('test/' + file))
    data = np.array(letter.convert('L'))
    data = data.astype('int')
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = 1 if data[i][j] != separator else -1
    trials.insert(-1, [data, file])

for trial in trials:
    network.testing(trial[0], trial[1])
print('Out of 30 trials ' + str(network.correct) + ' were correct!')
network.correct = 0