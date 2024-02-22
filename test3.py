import numpy as np
import random
import time
import keyboard
import jsonpickle

start_time = time.time()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def der_sigmoid(x):
    ex = np.exp(-x)
    return ex / (1 + ex) ** 2


def leaky_ReLu(x):
    return x * 0.01 if x < 0 else x


def der_leaky_ReLu(x):
    return 0.01 if x < 0 else 1


activations = {"sigmoid":(np.vectorize(sigmoid), np.vectorize(der_sigmoid)), "leaky_relu":(np.vectorize(leaky_ReLu), np.vectorize(der_leaky_ReLu))}

class Layer:
    def __init__(self, input_size, output_size, activation="sigmoid"):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.rand(output_size, input_size) * 0.5 - 0.25
        self.biases = np.random.rand(output_size) * 0.5 - 0.25

        self.activation, self.der_activation = activations[activation]

        self.weight_derivative = np.zeros((output_size, input_size))
        self.bias_derivative = np.zeros(output_size)

        self.z = 0
        self.values = np.zeros(output_size)

    def output(self, inputs):
        self.z = np.matmul(self.weights, inputs) + self.biases
        self.values = self.activation(self.z)
        return self.values

    def backpropagate(self, layer_der_cost, prev_layer_values):
        der_z = self.der_activation(self.z)

        self.bias_derivative += der_z * layer_der_cost
        self.weight_derivative += np.array(
            [[prev_layer_values[k] * der_z[j] * layer_der_cost[j] for k in range(self.input_size)] for j
             in range(self.output_size)])

        return np.array([np.sum(self.weights[:, j] * der_z * layer_der_cost) for j in range(self.input_size)])

    def change(self, batch_size, learning_rate):
        self.weights -= self.weight_derivative / batch_size * learning_rate
        self.biases -= self.bias_derivative / batch_size * learning_rate

        self.weight_derivative = np.zeros((self.output_size, self.input_size))
        self.bias_derivative = np.zeros(self.output_size)

class Net:
    def __init__(self, input_size):
        self.topology = [input_size]
        self.layers = []

    def get_output(self, input: np.ndarray):
        if len(input) != self.topology[0]:
            raise ValueError("Expected input size {0} but got {1}".format(self.topology[0], len(input)))

        values = input
        for layer in self.layers:
            values = layer.output(values)

        return values

    def cost_function(self, last_values, expected_output):
        return 2 * (last_values - expected_output)

    def with_backpropagation(self, input: np.ndarray, expected_output: np.ndarray):
        if len(input) != self.topology[0]:
            raise ValueError("Expected input size {0} but got {1}".format(self.topology[0], len(input)))
        if len(expected_output) != self.topology[-1]:
            raise ValueError("Expected output size {0} but got {1}".format(self.topology[1], len(expected_output)))

        values = input
        for layer in self.layers:
            values = layer.output(values)

        values_vs_cost = self.cost_function(values, expected_output)
        for i in range(len(self.layers) - 1, 0, -1):
            values_vs_cost = self.layers[i].backpropagate(values_vs_cost, self.layers[i-1].values)

    def train(self, training_data, epochs=500, batch_size=20, learning_rate=0.1):
        for epoch in range(epochs):
            print("epoch=", epoch)
            for _ in range(batch_size):
                input, output = training_data[random.randint(0, len(training_data) - 1)]
                self.with_backpropagation(np.array(input), np.array(output))

            for layer in self.layers:
                layer.change(batch_size, learning_rate)


    def train2(self, training_data, test_data, epochs=500, batch_size=20):
        # keyboard.on_press_key("g", self.save_data)

        for epoch in range(epochs):
            print("epoch=", epoch)
            permutation = np.random.permutation(training_data)

            for i in range(len(training_data)):
                print("|", end="")

                input, output = permutation[i]

                self.with_backpropagation(np.array(input), np.array(output))

                if (i != 0 and i % batch_size == 0) or i == len(training_data) - 1:
                    for layer in self.layers:
                        layer.change()


            test_sample = test_data[random.randint(0, len(test_data) - 1)]
            expectedoutput_test = test_sample[1]
            actualoutput = self.get_output(test_sample[0])

            error = sum([(expectedoutput_test[i] - actualoutput[i]) ** 2 for i in range(len(expectedoutput_test))])
            print()
            print("Epoch {0}: Error {1}      Expected {2} and Got {3}".format(epoch, error, expectedoutput_test,
                                                                              actualoutput))

            self.save_data()

        self.save_data()

    def save_data(self, evt=None):
        print("Initialising saving...")
        json_net = jsonpickle.encode(self)
        open('./save_net.json', 'w').close()
        with open("./save_net.json", 'w') as f:
            f.write(json_net)
        print('Net succesfully saved')

    def add_layer(self, size, activation):
        self.layers.append(Layer(self.topology[-1], size, activation=activation))
        self.topology.append(size)



nums = [0] * 10


def format_data(old_version):
    lr = np.arange(10)

    desired_number = old_version[0]
    nums[int(desired_number)] += 1
    inputs = old_version[1:].tolist()

    return [inputs, (lr == desired_number).astype(np.int).tolist()]


if __name__ == "__main__":

    test = Net(3)
    test.add_layer(3, "sigmoid")
    test.add_layer(2, "sigmoid")
    examples = [[[0, 0, 1], [0, 1]], [[1, 0, 1], [1, 1]], [[0, 1, 1], [1, 1]], [[0, 0, 0], [0, 0]], [[1, 0, 0], [1, 0]], [[1, 1, 0], [0, 0]]]



    test.train(examples, epochs=12000, learning_rate=1)

    #test.train(examples)

    print("After")
    print(test.get_output(np.array([1, 1, 1])))
    print(test.get_output(np.array([1, 1, 0])))
    print(test.get_output(np.array([0, 1, 0])))
    print(test.get_output(np.array([0, 1, 1])))
    print(test.get_output(np.array([1, 0, 0])))

    print("Took {} seconds".format(time.time() - start_time))