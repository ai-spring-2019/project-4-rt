"""

Author: Richard Teerlink
Project: Classification with Neural Networks
Description: Implements classifications utilizing neural networks. The training of the neural network's weights are done using back 
             propogation of errors. Currently the code is set up to test classification problems using k cross validation. You can
             run the code by typing something of the format under Usage into the command line. To test 3-bit incrementer I just
             had my network ouput it's data into a list an compared that to the input and intended output.

Usage: python3 project4.py DATASET.csv learning_rate k epochs

"""

import csv, sys, random, math

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

class NeuralNetworkNode():
    """ Creates a neural network node with weights of all links into itself, 
        its level, error and its activation. """

    def __init__(self, weight_num, level, activation):
        self.weights = []
        for i in range(0, weight_num):
            self.weights.append(random.random())
        self.level = level
        self.error = 0
        self.activation = activation

class NeuralNetwork():
    def __init__(self, neural_data_list):
        """ Given a list of neural network data creates our neural network. Contains methods for 
            forward propagation, back propagation, predict class, and back propagation training. """

        if len(neural_data_list) < 2:
            print("Invalid amount of layers.")
            exit()
        else:
            self.layers = []
            level_count = 0
            for i in range(0, len(neural_data_list)):
                layer = []
                for j in range(0,neural_data_list[i]):
                    if i == 0:
                        layer.append(NeuralNetworkNode(0, level_count, 0))
                    else:
                        layer.append(NeuralNetworkNode(neural_data_list[i-1], level_count, 0))
                self.layers.append(layer)
                level_count += 1

    def forward_propagate(self, input_list):
        """ Runs given inputs through our neural network. """

        # Set input values for our inputs, assuming first given value is 1 (dummy value)
        for i in range(0,len(input_list)):
            self.layers[0][i].activation = input_list[i]

        # Then for the rest of the layers
        for j in range(1, len(self.layers)):

            # Create an activation vector of all activations from previous layer
            activation_vector = []
            for input_node in self.layers[j-1]:
                activation_vector.append(input_node.activation)

            # Then to calculate the activation of a node we simply take the dot product
            # of the weights and activation and apply the logistic function
            for node in self.layers[j]:
                node.activation = logistic(dot_product(node.weights, activation_vector))

    def back_propagate(self, output_list, learning_rate):
        """ Back_propogate, updates our weights once. """

        # For our given outputs, calculate errors for our output layer
        for i in range(0, len(output_list)):
            node = self.layers[len(self.layers)-1][i]
            act = node.activation
            node.error = act+(1-act)*(output_list[i] - act)

        # Now calculate errors for the rest of the layers
        for layer_num in range(len(self.layers)-2, -1, -1):
            error_vector = []
            for i in self.layers[layer_num+1]:
                error_vector.append(i.error)
            for node_num in range(0, len(self.layers[layer_num])-1):
                weight_vector = []
                for k in self.layers[layer_num+1]:
                    weight_vector.append(k.weights[node_num])
                node = self.layers[layer_num][node_num]
                node.error = node.activation*(1-node.activation)*dot_product(error_vector, weight_vector)

        # Adjust all of the weights, layers 2 to L
        for layer_num in range(len(self.layers)-1,0,-1):
            for node in self.layers[layer_num]:
                for weight_num in range(0, len(node.weights)):
                    node.weights[weight_num] = node.weights[weight_num] + learning_rate*self.layers[layer_num-1][weight_num].activation*node.error


    def back_propagation_training(self, training_list, epochs, learning_rate):
        """ Training for our whole network. """

        # Train our data for our given number of epochs
        for epoch in range(0,epochs):
            # For all of the training data given run our input through and then back propagate the error
            for example in training_list:
                input_list, output_list = example
                self.forward_propagate(input_list)
                self.back_propagate(output_list, learning_rate)
                print(self.layers[len(self.layers)-1][0].activation)
                print(self.layers[len(self.layers)-1][0].weights)

    def predict_class(self):
        """ Returns a predicted class for classification problems. """

        # Get a list of our outputs
        output_list = []
        for i in self.layers[len(self.layers) - 1]:
            output_list.append(i.activation)

        max_value = 0
        max_pos = 0

        # Return our class closest to 1
        for j in range(0, len(output_list) - 1):
            if output_list[j] > max_value:
                max_value = output_list[j]
                max_pos = j
        return max_pos

def split(training_data, k):
    """ Function for splitting our list in approximately k equal subsets. Used in our k cross-validation. 
        Credit to: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length """

    # We first divide our training data to get our subset size and remainder
    integer, rem = divmod(len(training_data), k)
    subsets = []

    # Create our subsets
    for i in range(k):

        # While we are below our remainder we add on an extra element, once we have accounted for the
        # extra elements we go back to our original list length. This is because of the min(i, rem)
        # and min(i+1, rem), since if we are below rem then we are taking an additional element on the end.
        values = (i * integer + min(i, rem), (i + 1) * integer + min(i + 1, rem))
        subsets.append(values)
 
    return subsets

def cross_validation(neural_network, training_data, k, learning_rate, epochs):
    """ Cross validation for our network and given training data. Assumes problems are classification. """

    # Get the position of our subsets within our training data
    data_subsets_pos = split(training_data, k)

    total_accuracy = 0
    # Iterate k times
    for test_range in range(0,k):

        # Get the start and stop value of a subset we have not used yet and use it as test data and 
        # all others as our training data
        start, stop = data_subsets_pos[test_range]
        test_data = training_data[start:stop]
        new_training_data = training_data[0:start] + training_data[stop:len(training_data)]

        # Traing our network on our training data
        neural_network.back_propagation_training(new_training_data, epochs, learning_rate)

        # Print our accuracy
        print(accuracy(neural_network, test_data))
        total_accuracy += accuracy(neural_network, test_data)
    print("Average Accuracy: ", total_accuracy/k)

def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]
    # Check out the data:
    for example in training:
        print(example)

    # Get all of our input from user
    learning_rate = float(sys.argv[2])
    k_val = int(sys.argv[3])
    epochs = int(sys.argv[4])

    # Create our network
    nn = NeuralNetwork([3, 6, 2])

    # run cross validation
    cross_validation(nn, training, k_val, learning_rate, epochs)

if __name__ == "__main__":
    main()
