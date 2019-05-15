import numpy as np
import random
import math
import matplotlib.pyplot as plt

class ParameterError(Exception):
    pass

'''
Sigmoidal function for switch neurons
'''
def sigmoid_switch(x, lam, beta):
    return 1.0/(1+ np.exp(-1 * lam * (x - beta)))

'''
Derivative of sigmoidal function for switch neurons
'''
def sigmoid_switch_prime(input, lam, beta):
    return (lam*np.exp(-lam*(input-beta))) / (np.exp(-lam*(input-beta)) + 1) ** 2

'''
sigmoidal function for multi-layer perceptron neurons
'''
def sigmoid(s):
    # activation function
    return 1/(1+np.exp(-s))

'''
Derivative of sigmoidal function for multi-layer perceptron neurons
'''
def sigmoid_prime(s):
    #derivative of sigmoid
    return s * (1 - s)

'''
Cost function for backprop
'''
def cost(true, predicted):
    return .5 * (true - predicted) ** 2
'''
Derivative cost function for backprop
'''
def cost_prime(true, predicted):
    return (true - predicted)

'''Class for multi-layer perceptron that includes neurons with hysteresis'''
class Neural_Network(object):
    '''
    Constructor.
    layer_sizes: list of sizes of each layer of network, including input and output layer_sizes
    regression: boolean value. If true, the problem is a regression problem.
    switches: of the form [activation, self-excitatory weight, [(layer num, neuron num, weight)] for neurons the switch is connected to]
    lam: lam parameter for sigmoid_switch
    beta: beta parameter for sigmoid switch
    '''
    def __init__(self, layer_sizes, regression, switches, lam, beta):
        self.weights = []
        self.biases = []
        self.layers = []
        self.layers_unsquashed = []
        self.switches = switches
        self.num_layers = len(layer_sizes)
        self.input_size = layer_sizes[0]
        self.regression_problem = regression
        self.lam = lam
        self.beta = beta
        last_size = layer_sizes[0]

        #generate weight matricies with random values from normal distribution
        for i in range(1, len(layer_sizes)):
            self.weights.append(np.true_divide(np.random.randn(last_size, layer_sizes[i]), 10))
            last_size = layer_sizes[i]

        #generate biases for each neuron with random values from normal distribution
        for i in range(self.num_layers):
            layer_biases = []
            for j in range(layer_sizes[i]):
                b = np.random.normal(0, .1)
                layer_biases.append(b)
            self.biases.append(layer_biases)

    '''
    Forward propogates given an input list.
    input size must match the size of input layers
    with_switches: boolean value switch indicates if switches have an effect. Switches are not used furing training
    '''
    def forward_propogate(self, input, with_switches):
        if len(input) is not self.input_size:
            raise ParameterError('input does not match input size')

        self.layers = []
        self.layers_unsquashed = []

        #set self.layers and self.layer_unsquashed to all 0's
        for num_neurons in self.biases:
            self.layers.append([0] * len(num_neurons))
            self.layers_unsquashed.append([0] * len(num_neurons))

        self.layers[0] = input
        self.layers_unsquashed[0] = input

        #update switches and add their effect to activations of neurons in the multi-layer perceptron
        if with_switches is True:
            for i in range(len(self.switches)):
                activation_old = self.switches[i][0]
                self_excitatory_weight = self.switches[i][1]
                self.switches[i][0] = sigmoid_switch(activation_old * self_excitatory_weight, self.lam, self.beta)
                activation = self.switches[i][0]
                for c in self.switches[i][2]:
                    layer = c[0]
                    neuron_index = c[1]
                    weight = c[2]
                    unsquashed = activation * weight
                    print('activation value:' + str(activation))
                    self.layers_unsquashed[layer][neuron_index] += unsquashed
                print('Unsquased Layers:')
                print(self.layers_unsquashed)
                print()

        last_layer = input

        #forward propogate multi-layer perceptron
        for i in range(1, self.num_layers):
            #matrix multiplication
            product = np.dot(last_layer, self.weights[i - 1])
            for j in range(len(self.layers_unsquashed[i])):
                self.layers_unsquashed[i][j] += product[j]

            #add biases
            for j in range(len(self.layers_unsquashed[i])):
                self.layers_unsquashed[i][j] = self.layers_unsquashed[i][j] + self.biases[i][j]

            #apply sigmoidal squishification
            if self.regression_problem is True and i is self.num_layers - 1:
                for j in range(len(self.layers[i])):
                    self.layers[i][j] += self.layers_unsquashed[i][j]
            else:
                for j in range(len(self.layers[i])):
                    self.layers[i][j] += sigmoid(self.layers_unsquashed[i][j])

            last_layer = self.layers[i]


        return self.layers[len(self.layers) - 1]

    '''
    Steps for backpropagation
    Divide the training examples into batches
    do the following for each training example in one batch:
    1. make 2d arrays for dc/da, da/dz, and dz/dw
    2. loop through all output neurons using index i and calculate 2(self.layers[self.num_layers-1][i] - expected[i]), and store it in dc/da
    3. loop through layers_unsquashed[self.num_layers-1] using index i and calculate sigmoid_prime(layers_unsquashed[self.numlayers-1][i]), and store it in da/dz
    4. loop through the last layer of weights and store the activation of the neuron that feeds into the connection in dz/dw
    5. calculate the derivative of the cost wrt each weight, which is dc/da * da/dz * dz/dw. Store this in dc/dw
    6. calculate the derivative of the cost wrt each bias, which is dc/da * da/dz * 1. Store this in dc/db
    7. for each neuron in layer L-1, loop through all neurons in layer L and calculate dc/da(L) * da(L)/dz * dz/da(L-1). Sum these values and store them in the next layer of dc/da.
    8. Repeat steps 3-7 until you get to the bottom layer.
    Then sum the resulting dc/dw and dc/db values for each weight and bias.
    Define a constant c to use with gradient descent
    Add c*dc/dw to each weight and c*dc/db for each bias
    Repeat this whole process for each batch of training examples
    '''
    def back_propagate(self, inputArray, expected, DcDw, DcDb):
        DcDa = []

        lastLayer = self.layers[self.num_layers-1]
        for i in range(len(lastLayer)): #step 2
            DcDa.append(cost_prime(expected[i], lastLayer[i]))

        final_layer = True
        #print('New Sample******************')
        #print('Weights:' + str(self.weights))
        for layerNum in range(self.num_layers - 1, 0, -1):
            #print("layerNum: " + str(layerNum))
            lenLayer = len(self.layers[layerNum])
            lenLayerPrev = len(self.layers[layerNum - 1])
            DaDz = []

            #calculate Da/Dz for current layer
            for i in range(lenLayer): #step 3
                 DaDz.append(sigmoid_prime(self.layers[layerNum][i]))

            #calculate Dc/Dw for weights going into current layer
            for i in range(lenLayerPrev):
                 for k in range(len(self.weights[layerNum-1][i])):
                     DzDw_weight = self.layers[layerNum-1][i] #step 4
                     if self.regression_problem is True and final_layer is True:
                         DcDw[layerNum-1][i][k] += DcDa[k] * DzDw_weight #step 5
                     else:
                         DcDw[layerNum-1][i][k] += DaDz[k] * DcDa[k] * DzDw_weight #step 5

            #calculate Dc/Db for current layer
            for i in range(lenLayer):
                 if self.regression_problem is True and final_layer is True:
                     DcDb[layerNum][i] += DcDa[i] #step 6. Note that DcDb is a 2D array
                 else:
                     DcDb[layerNum][i] += DaDz[i] * DcDa[i] #step 6. Note that DcDb is a 2D array


            #calculate DcDa for previous layer
            DcDa_prev_layer = [0] * lenLayerPrev

            for i in range(lenLayer):
                DcDa_old = DcDa[i] #current layer
                for j in range(lenLayerPrev):
                    DzDa_neuron_prev = self.weights[layerNum-1][j][i] #dz/da(L-1) = w(L)
                    if self.regression_problem is True and final_layer is True:
                        DcDa_prev_layer[j] += DcDa_old * DzDa_neuron_prev
                    else:
                        DcDa_prev_layer[j] += DcDa_old * DaDz[i] * DzDa_neuron_prev

            DcDa = DcDa_prev_layer

            final_layer = False

    '''
    Sets DaDz and DzDw to a list of zeros corresponding to each layer of
    the neural net.
    '''
    def zeroify(self, DcDw, DcDb):
        for num_neurons in self.biases:
            DcDb.append([0] * len(num_neurons))

        for layer_num in range(len(self.weights)):
            DcDw.append([])
            #DcDw[layer_num].append([] * len(self.weights[layer_num]))
            for neuron in range(len(self.weights[layer_num])):
                DcDw[layer_num].append([0] * len(self.weights[layer_num][neuron]))
                #DcDw[layer_num][neuron].append(9)

    '''
    creates set of random batches from training data
    '''
    def randomize_batches(self, training_samples, num_batches, batch_min_len):
        #divide training samples into batches
        batch_len = int(len(training_samples) / num_batches)
        random.shuffle(training_samples)
        batches = [training_samples[x:x+batch_len] for x in range(0, len(training_samples), batch_len)]

        #If last batch is < batch_min_length, concatenate last 2 batches
        if len(batches[len(batches) - 1]) < batch_min_len and len(batches) >= 2:
            batches[len(batches) - 2] = batches[len(batches) - 2] + batches[len(batches) - 1]
            batches.pop(len(batches) - 1)

        return batches

    '''
    For each batch, runs backprop algorithm for each sample in batch, then updates values accordingly.
    This process repeats for the given number of epochs
    '''
    def train(self, training_samples, learning_rate_w, learning_rate_b, num_batches, batch_min_len, epochs):
        counter = 0
        for iteration in range(epochs):
            print("Epoch:", iteration)
            #list of lists, one for each layer. We will maintain a running average
            batches = self.randomize_batches(training_samples, num_batches, batch_min_len)

            for batch in batches:
                DcDw = []
                DcDb = []
                self.zeroify(DcDw, DcDb)

                for sample in batch:
                    #run forward and back propogation for each sample in batch
                    inputs = sample[0]
                    expected = sample[1]

                    self.forward_propogate(inputs, False)
                    self.back_propagate(inputs, expected, DcDw, DcDb)
                counter += 1

                #average results to get DcDb and DcDw
                batch_size = len(batch)

                for i in range(len(DcDw)):
                    for j in range(len(DcDw[i])):
                        for k in range(len(DcDw[i][j])):
                            DcDw[i][j][k] = DcDw[i][j][k] / batch_size

                for i in range(len(DcDb)):
                    for j in range(len(DcDb[i])):
                        DcDb[i][j] = DcDb[i][j] / batch_size

                #apply learning rule for weights
                for layer_w_num in range(len(self.weights)):
                    for i in range(0, len(self.weights[layer_w_num])):
                        for j in range(0, len(self.weights[layer_w_num][i])):
                            self.weights[layer_w_num][i][j] += DcDw[layer_w_num][i][j] * learning_rate_w

                #apply learning rule for biases
                for layer in range(len(self.biases)):
                    for neuron in range(len(self.biases[layer])):
                        self.biases[layer][neuron] += DcDb[layer][neuron] * learning_rate_b


def main():
    '''
    #debugging code for Neural_Network constructor
    #network = Neural_Network([3, 5, 1])
    switch1 = [0, 2, [(0,0,1)]] #activation, self-excitatory weight, [(layer num, neuron num, weight)] for neurons the switch is connected to
    switches = [switch1]
    network = Neural_Network([3,2,1], True, [], 4, .5)

    #inputList = np.asarray([1,3,7,2,14,9]) #answer is mod 5 the sum of the input
    #expected = np.asarray([0,1,0,0,0,0])

    #print weights matricies
    print('Weights:')
    for weights_matrix in network.weights:
        print(weights_matrix)
    print()

    print('Biases:')
    #print bias list for each layer
    for bias_list in network.biases:
        print(bias_list)
    print()

    #debugging code for sigmoidal squishification
    #print(sigmoid(2, 4, .5))

    #debugging code for forward propogation
    x = network.forward_propogate([0, .5, 1])
    #network.forward_propogate(inputList)

    print('Layers:')
    for layer in network.layers:
        print(layer)

    print("Output:")
    print(x)
    print()

    #print('******************************************************************************************************************')
    '''

    training_samples = []
    for x in range(0, 1000):
        training_samples.append(([x / 100], [math.sin(x / 100)]))

    switch1 = [0, 2, [(2,0,3)]] #[activation, self-excitatory weight, [(layer num, neuron num, weight)] for neurons the switch is connected to]
    switches = [switch1]
    network = Neural_Network([1, 5, 1], True, switches, 4, 1)

    #print weights matricies
    print('Weights before training:')
    for weights_matrix in network.weights:
        print(weights_matrix)
    print()

    print('Biases before training:')
    #print bias list for each layer
    for bias_list in network.biases:
        print(bias_list)
    print()

    #train(self, training_samples, learning_rate_w, learning_rate_b, num_batches, batch_min_len, epochs):
    network.train(training_samples, .1, .1, 10, 100, 1000)

    #print weights matricies
    print('Weights after training:')
    for weights_matrix in network.weights:
        print(weights_matrix)
    print()

    print('Biases after training:')
    #print bias list for each layer
    for bias_list in network.biases:
        print(bias_list)
    print()

    #debugging code for sigmoidal squishification
    #print(sigmoid(2, 4, .5))
    x_values = []
    y_values = []
    x_values_sin = []
    y_values_sin = []

    for value in range(-300, 300, 1):
        x_values.append(value/10)
        y_values.append(network.forward_propogate([value/10], True))
        x_values_sin.append(value/10)
        y_values_sin.append(math.sin(value/10))

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.plot(x_values, y_values)
    plt.plot(x_values_sin, y_values_sin)
    #plt.axis([-16, 16, 0, 10])
    plt.show()


    network.forward_propogate([value], False)
    print('Layers:')
    for layer in network.layers:
        print(layer)

main()
