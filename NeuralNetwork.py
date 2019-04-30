import numpy as np

class ParameterError(Exception):
    pass

def sigmoid(x, lam, beta):
    return 1.0/(1+ np.exp(-1 * lam * (x - beta)))

def sigmoid_prime(self, input, lam, beta):
    return (lam*np.exp(-lam*(input-beta))) / (np.exp(-lam*(input-beta)) + 1) ** 2

'''Class to created neural network with one hidden layer'''
class Neural_Network(object):
  def __init__(self, layer_sizes):
    self.weights = []
    self.biases = []
    self.layers = []
    self.layers_unsquashed = []
    self.num_layers = len(layer_sizes)
    self.input_size = layer_sizes[0]
    last_size = layer_sizes[0]

    #generate weight matricies with random values from normal distribution
    for i in range(1, len(layer_sizes)):
        self.weights.append(np.random.randn(last_size, layer_sizes[i]))
        last_size = layer_sizes[i]

    #generate biases for each neuron with random values from normal distribution
    for i in range(self.num_layers):
        layer_biases = []
        for j in range(layer_sizes[i]):
            b = np.random.normal(0, 1)
            layer_biases.append(b)
        self.biases.append(layer_biases)


def forward_propogate(self, input):
    if len(input) is not self.input_size:
        raise ParameterError('input does not match input size')

    self.layers.append(input)
    self.layers_unsquashed.append(input)
    last_layer = input

    #should I maintain values before applying sigmoidal squishification?
    #Yes. This is necessary to compute dz/da
    for i in range(1, self.num_layers):
        #matrix multiplication
        self.layers_unsquashed.append(np.dot(last_layer, self.weights[i - 1]))

        #add biases
        for j in range(len(self.layers_unsquashed[i])):
            self.layers_unsquashed[i][j] = self.layers_unsquashed[i][j] + self.biases[i][j]

        #apply sigmoidal squishification
        self.layers.append([])
        self.layers[i] = [sigmoid(z, 4, .5) for z in self.layers_unsquashed[i]]

        last_layer = self.layers[i]


    return self.layers[len(self.layers) - 1]

#steps for backpropagation
#do the following for each training example:
#make 2d arrays for dc/da, da/dz, and dw/da
#calculate the expected output values and stored them in an array, expected
#loop through all output neurons using index i and calculate 2(self.layers[self.num_layers-1][i] - expected[i]), and store it in dc/da
#loop through layers_unsquashed[self.num_layers-1] using index i and calculate sigmoid_prime(layers_unsquashed[self.numlayers-1][i]), and store it in da/dz
#loop through the last layer of weights and store the activation of the neuron that feeds into the connection in dz/dw
#calculate the derivative of the cost wrt each weight, which is dc/da * da/dz * dz/dw. Store this in dc/dw
#calculate the derivative of the cost wrt each bias, which is dc/da * da/dz * 1. Store this in dc/db


def main():
    '''debugging code for Neural_Network constructor'''
    network = Neural_Network([3, 5, 1])

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

    '''debugging code for sigmoidal squishification'''
    #print(sigmoid(2, 4, .5))

    '''debugging code for forward propogation'''
    x = network.forward_propogate([0, .5, 1])

    print('Layers:')
    for layer in network.layers:
        print(layer)

    print("Output:")
    print(x)
    print()



main()
