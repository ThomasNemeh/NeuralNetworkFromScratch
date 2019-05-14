import numpy as np
import random
import math
import matplotlib.pyplot as plt

class ParameterError(Exception):
    pass

'''
sigmoidal function for switch neurons
'''
def sigmoid_switch(x, lam, beta):
    return 1.0/(1+ np.exp(-1 * lam * (x - beta)))

'''
derivative of sigmoidal function for switch neurons
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
derivative of sigmoidal function for multi-layer perceptron neurons
'''
def sigmoid_prime(s):
    #derivative of sigmoid
    return s * (1 - s)

'''
cost function for backprop
'''
def cost(true, predicted):
    return .5 * (true - predicted) ** 2
'''
derivative cost function for backprop
'''
def cost_prime(true, predicted):
    return (true - predicted)

'''Class to created neural network with one hidden layer'''
class Neural_Network(object):
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
    def forward_propogate(self, input):
        if len(input) is not self.input_size:
            raise ParameterError('input does not match input size')

        self.layers = []
        self.layers_unsquashed = []

        self.layers.append(input)
        self.layers_unsquashed.append(input)
        last_layer = input

        #should I maintain values before applying sigmoidal squishification?
        #Yes. This is necessary to compute dz/da
        for i in range(1, self.num_layers):
            #matrix multiplication
            self.layers_unsquashed.append(np.dot(last_layer, self.weights[i - 1]))
#            if i == self.num_layers - 1:
#                print("@@@@@@")
#                print(last_layer)
#                print(self.weights[i-1])
#                print(self.layers_unsquashed[-1])
#                print("%%%%%")

            #add biases
            for j in range(len(self.layers_unsquashed[i])):
                self.layers_unsquashed[i][j] = self.layers_unsquashed[i][j] + self.biases[i][j]

            #apply sigmoidal squishification
            self.layers.append([])
            #self.layers[i] = [sigmoid(z, 4, .5) for z in self.layers_unsquashed[i]]
            #with new sigmoid function
            if self.regression_problem is True and i is self.num_layers - 1:
                self.layers[i] = self.layers_unsquashed[i]
            else:
                self.layers[i] = [sigmoid(z) for z in self.layers_unsquashed[i]]

            last_layer = self.layers[i]


        return self.layers[len(self.layers) - 1]
    '''

    def forward_propogate(self, input, with_switches):
        if len(input) is not self.input_size:
            raise ParameterError('input does not match input size')

        self.layers = []
        self.layers_unsquashed = []

        for num_neurons in self.biases:
            self.layers.append([0] * len(num_neurons))
            self.layers_unsquashed.append([0] * len(num_neurons))

        self.layers[0] = input
        self.layers_unsquashed[0] = input

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

        #should I maintain values before applying sigmoidal squishification?
        #Yes. This is necessary to compute dz/da
        for i in range(1, self.num_layers):
            #matrix multiplication
            product = np.dot(last_layer, self.weights[i - 1])
            for j in range(len(self.layers_unsquashed[i])):
                self.layers_unsquashed[i][j] += product[j]

            #add biases
            for j in range(len(self.layers_unsquashed[i])):
                self.layers_unsquashed[i][j] = self.layers_unsquashed[i][j] + self.biases[i][j]

            #apply sigmoidal squishification
            #self.layers[i] = [sigmoid(z, 4, .5) for z in self.layers_unsquashed[i]]
            '''with new sigmoid function'''
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
            #print(expected[i], lastLayer[i], DcDa)
            #print('True!!!!!!: ' + str(lastLayer[i]))
            #print('Expected!!:' + str(expected[i]))

        final_layer = True
        #print('New Sample!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #print('Weights:' + str(self.weights))
        #should there be a 1 at the end
        for layerNum in range(self.num_layers - 1, 0, -1):
            #print("layerNum: " + str(layerNum))
            lenLayer = len(self.layers[layerNum])
            lenLayerPrev = len(self.layers[layerNum - 1])
            DaDz = []
            #i is each neuron in current layer!!!!!
            for i in range(lenLayer): #step 3
                 #print("i: " + str(i))
                 #DaDz.append(sigmoid_prime(self.layers_unsquashed[layerNum][i],4,0.5))
                 '''with new sigmoid function'''
                 '''Change HERE!'''
                 DaDz.append(sigmoid_prime(self.layers[layerNum][i]))
            for i in range(lenLayerPrev):
                 #print("i: " + str(i))
                 for k in range(len(self.weights[layerNum-1][i])):
                     #print("k: " + str(k))
                     DzDw_weight = self.layers[layerNum-1][i] #step 4
                     if self.regression_problem is True and final_layer is True:
                         #print('DcDw +: '  + str(DcDa[k] * DzDw_weight))
                         DcDw[layerNum-1][i][k] += DcDa[k] * DzDw_weight #step 5
                     else:
                         #print('DcDw +: ' + str(DaDz[k] * DcDa[k] * DzDw_weight))
                         #print('DcDw AGAIN +: '  + str(sigmoid_prime(self.layers[layerNum][k]) * DcDa[k] * DzDw_weight))
                         DcDw[layerNum-1][i][k] += DaDz[k] * DcDa[k] * DzDw_weight #step 5
                         #DcDw[layerNum-1][i][k] += sigmoid_prime(self.layers_unsquashed[layerNum][k]) * DcDa[k] * DzDw_weight #step 5
            for i in range(lenLayer):
                 if self.regression_problem is True and final_layer is True:
                     DcDb[layerNum][i] += DcDa[i] #step 6. Note that DcDb is a 2D array
                 else:
                     DcDb[layerNum][i] += DaDz[i] * DcDa[i] #step 6. Note that DcDb is a 2D array


            #calculate DcDa for previous layer
            DcDa_prev_layer = [0] * lenLayerPrev
            ##print('DcDa old: ' + str(DcDa) + '***************************************')
            for i in range(lenLayer):
                DcDa_old = DcDa[i] #current layer
                for j in range(lenLayerPrev):
                    DzDa_neuron_prev = self.weights[layerNum-1][j][i] #dz/da(L-1) = w(L)
                    if self.regression_problem is True and final_layer is True:
                        DcDa_prev_layer[j] += DcDa_old * DzDa_neuron_prev
                    else:
                        DcDa_prev_layer[j] += DcDa_old * DaDz[i] * DzDa_neuron_prev
            #print('Layer num old: ' + str(layerNum))
            #print('DcDa old: ' + str(DcDa))
            DcDa = DcDa_prev_layer
            #print()
            #print('Layer num new: ' + str(layerNum - 1))
            #print('DcDa new: ' + str(DcDa))
            #print()
            '''Is this right?????'''
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
    epoch: number of times entire data set is passed through neural net
    The number of batches is equal to the number of iterations for one epoch
    Batches are not randomized after each epoch?
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

                '''
                #batches debugging statement
                #print('batches: ')
                #print(batches)
                #print()
                '''

                #debugging statements
                #print('DcDw initial: ')
                #print(DcDw)
                #print()
                #print('DcDb initial: ')
                #print(DcDb)
                #print()

                for sample in batch:
                    #run forward and back propogation for each sample in batch
                    inputs = sample[0]
                    expected = sample[1]
                    #print('input!!!!: ' + str(inputs))
                    self.forward_propogate(inputs, False)
                    self.back_propagate(inputs, expected, DcDw, DcDb)
                counter += 1
                #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                #average results to get DcDb and DcDw
                batch_size = len(batch)

                #print('DcDw before averaging. batch_size = ' + str(batch_size))
                #print(DcDw)
                #print()
                #print('DcDb before averaging:')
                #print(DcDb)
                #print()


                '''
                for i in range(len(DcDw)):
                    for j in range(len(DcDw[i])):
                        DcDw[i][j] = map(lambda x: x / batch_size, DcDw[i][j])
                for i in range(len(DcDb)):
                    DcDb[i] = map(lambda x: x / batch_size, DcDb[i])
                '''

                for i in range(len(DcDw)):
                    for j in range(len(DcDw[i])):
                        for k in range(len(DcDw[i][j])):
                            DcDw[i][j][k] = DcDw[i][j][k] / batch_size

                for i in range(len(DcDb)):
                    for j in range(len(DcDb[i])):
                        DcDb[i][j] = DcDb[i][j] / batch_size

                #print('DcDw after averaging. batch_size = ' + str(batch_size))
                #print(DcDw)
                #print()
                #print('DcDb after averaging:')
                #print(DcDb)
                #print()

                '''
                apply learning rule. Use activation of neuron in higher layer as expected.
                for biases, activation not taken into account. Can use different learning rate,
                should be lower.
                '''

                #print('Weights before learning rule.')
                #print(self.weights)
                #print()
                #print('biases before learning rule:')
                #print(self.biases)
                #print()

                for layer_w_num in range(len(self.weights)):
                    for i in range(0, len(self.weights[layer_w_num])):
                        for j in range(0, len(self.weights[layer_w_num][i])):
                            #self.weights[layer_num][i][j] -= DcDw[layer_num][i][j] * learning_rate_w * self.layers[layer_num][j]
                            self.weights[layer_w_num][i][j] += DcDw[layer_w_num][i][j] * learning_rate_w

                for layer in range(len(self.biases)):
                    for neuron in range(len(self.biases[layer])):
                        self.biases[layer][neuron] += DcDb[layer][neuron] * learning_rate_b
#                        if layer == len(self.biases) - 1:
#                            print(neuron, DcDb[layer][neuron])

                #print('Weights after learning rule.')
                #print(self.weights)
                #print()
                #print('biases after learning rule:')
                #print(self.biases)
                #print()

                #if counter is 1:
                #    exit()

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
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.show()

    training_samples = []
    for x in range(0, 1000):
#        training_samples.append(([x/100], [math.sin(x/100)]))
        training_samples.append(([x / 100], [math.sin(x / 100)]))

    switch1 = [0, 2, [(2,0,3)]] #activation, self-excitatory weight, [(layer num, neuron num, weight)] for neurons the switch is connected to
    switches = [switch1]
    network = Neural_Network([1, 5, 1], True, switches, 4, .5)

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

    for value in range(-160, 160, 1):
        x_values.append(value/10)
        y_values.append(network.forward_propogate([value/10], True))

    plt.plot(x_values, y_values)
    #plt.axis([-16, 16, 0, 10])
    plt.show()


    network.forward_propogate([value], True)
    print('Layers:')
    for layer in network.layers:
        print(layer)

main()
