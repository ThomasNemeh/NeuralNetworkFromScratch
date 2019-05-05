import numpy as np
import random

class ParameterError(Exception):
    pass

def sigmoid(x, lam, beta):
    return 1.0/(1+ np.exp(-1 * lam * (x - beta)))

def sigmoid_prime(input, lam, beta):
    return (lam*np.exp(-lam*(input-beta))) / (np.exp(-lam*(input-beta)) + 1) ** 2

#does ordering of actual and expected matter?
def cost(actual, expected):
    return (actual - expected) ** 2

def cost_prime(actual, expected):
    return 2 * (actual - expected)

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
#Divide the training examples into batches
#do the following for each training example in one batch:
#1. make 2d arrays for dc/da, da/dz, and dz/dw
#2. loop through all output neurons using index i and calculate 2(self.layers[self.num_layers-1][i] - expected[i]), and store it in dc/da
#3. loop through layers_unsquashed[self.num_layers-1] using index i and calculate sigmoid_prime(layers_unsquashed[self.numlayers-1][i]), and store it in da/dz
#4. loop through the last layer of weights and store the activation of the neuron that feeds into the connection in dz/dw
#5. calculate the derivative of the cost wrt each weight, which is dc/da * da/dz * dz/dw. Store this in dc/dw
#6. calculate the derivative of the cost wrt each bias, which is dc/da * da/dz * 1. Store this in dc/db
#7. for each neuron in layer L-1, loop through all neurons in layer L and calculate dc/da(L) * da(L)/dz * dz/da(L-1). Sum these values and store them in the next layer of dc/da.
#8. Repeat steps 3-7 until you get to the bottom layer.
#Then sum the resulting dc/dw and dc/db values for each weight and bias.
#Define a constant c to use with gradient descent
#Add c*dc/dw to each weight and c*dc/db for each bias
#Repeat this whole process for each batch of training examples

    def back_propagate(self, inputArray, expected, DcDw, DcDb):
        #only 1 training batch for now
        DcDa = []        #step 1
        DaDz = []

        lastLayer = self.layers[self.num_layers-1]
        for i in range(len(lastLayer)): #step 2
            DcDa[i] = cost_prime(lastLayer[i],expected[i])

        for l in range(self.num_layers-1):
            layerNum = self.num_layers - l
            lenLayer = len(self.layers[layerNum])
            for i in range(lenLayer): #step 3
                 DaDz[i] = sigmoid_prime(self.layers_unsquashed[layerNum][i],4,0.5)
                 for k in range(len(self.weights[layerNum-1][i])):
                     DzDw_weight = self.layers[layerNum-1][k] #step 4
                     DcDw[layerNum-1][i][k] += DaDz[i] * DcDa[i] * DzDw_weight #step 5
                     DcDb[layerNum-1][i][k] += DaDz[i] * DcDa[i] #step 6
            for k in range(len(self.layers[layerNum-1])):
                DcDa_old = DcDa[i] #current layer
                for i in range(lenLayer):
                    DzDa_neuron_prev = self.weights[layerNum-1][i][k] #dz/da(L-1) = w(L)
                    DcDa[k] += DcDa_old * DaDz[i] * DzDa_neuron_prev

    '''
    Sets DaDz and DzDw to a list of zeros corresponding to each layer of
    the neural net.
    '''
    def zeroify(self, DaDz, DzDw):
        for num_neurons in self.biases:
            DaDz.append([0] * len(num_neurons))
            DzDw.append([0] * len(num_neurons))

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
    def train(self, training_samples, learning_rate, num_batches, batch_min_len, epochs):
        batches = self.randomize_batches(training_samples, num_batches, batch_min_len)

        #batches debugging statement
        print(batches)

        for iteration in epochs:
            #list of lists, one for each layer. We will maintain a running average
            DcDw = []
            DcDb = []
            self.zeroify(DcDw, DcDb)

            for batch in batches:
                #run back propogation for each sample in batch
                inputs = batch[0]
                expected = batch[1]
                self.back_propogate(inputs, expected, DcDw, DcDb)

                #average results to get DcDb and DcDw
                batch_size = len(batch)
                for layer in DcDw:
                    map(lambda x: x / batch_size, layer)
                for layer in DcDb:
                    map(lambda x: x / batch_size, layer)





def main():
    '''debugging code for Neural_Network constructor'''
    #network = Neural_Network([3, 5, 1])
    network = Neural_Network([6,6,6])

    inputList = np.asarray([1,3,7,2,14,9]) #answer is mod 5 the sum of the input
    expected = np.asarray([0,1,0,0,0,0])
    
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
    #x = network.forward_propogate([0, .5, 1])
    network.forward_propogate(inputList)

    print('Layers:')
    for layer in network.layers:
        print(layer)

    #print("Output:")
    #print(x)
    print()
    print('*********************************************************')

    training_samples = [([0], [1]), ([34], [2]), ([0], [5]), ([10], [1]), ([0], [132]), ([0], [1]), ([23], [1]), ([0], [1]), ([12], [1]), ([0], [1])]
    #need to put in epochs as a parameter
    network.train(training_samples, .5, 3, 2, 5)

main()
