# Toy educational code for an Artificial Neural Network
import numpy as np


class Layer(object):
    # All bias, weighted sums and activations for the layer stored as array
    # rather than Neuron objects that need to be assembled into vectors to
    # do linear algebra manipulations
    def __init__(self, size):
        self.size = size
        self.b = np.zeros(size) # Bias
        self.z = np.zeros(size) # Weighted sum of inputs
        self.a = np.zeros(size) # Activation of Neuron
        self.set_activation_ReLU() # ReLU by default
        return

    def set_activation_identity(self):
        self.activation_function = lambda x: x
        self.activation_function_derivative = lambda x: 1.0
    
    def set_activation_ReLU(self):
        self.activation_function = lambda x: x * (x > 0.0)
        self.activation_function_derivative = lambda x: 1.0 * (x > 0.0)

    def set_activation_leakyReLU(self, leak):
        self.activation_function = lambda x: leak * x * (x < 0.0) + x * (x > 0.0)
        self.activation_function_derivative = lambda x: leak * (x < 0.0) + 1.0 * (x > 0.0)
    
    def set_activation_GeLU(self):
        return 0 # TODO
    
    def set_activation_softplus(self):
        return 0 # TODO
    
    def set_activation_logistic(self):
        self.activation_function = lambda x: 1.0 / (1.0 + np.exp(-x))
        self.activation_function_derivative = lambda x: self.activation_function(x) * (1.0 - self.activation_function(x))


class ANN(object):
    def __init__(self, size_in, size_hidden, size_out):
        self.size_in = size_in
        self.size_hidden = size_hidden # List of sizes of all hidden layers
        self.num_hidden = len(size_hidden)
        self.size_out = size_out

        # Create layers
        # Input layer is only a vector, not assocaited to a Layer object
        # Layer list index 0 corresponds to first hidden layer (l=1)
        # Layer list index num_hidden corresponds to output hidden layer (l=num_hidden+1)
        self.layers = [Layer(size) for size in size_hidden]
        self.layers += [Layer(size_out)]
        self.layers[-1].set_activation_logistic() # TODO: Sigmoid by default rn

        # Create Weight matrices W_ij. i < size_next_layer ; j < size_prev_layer
        # Stored as vector: size_next_layer concatenated vectors of length size_prev_layer
        self.weight_matrices = []
        size_prev = size_in
        for size_next in list(size_hidden)+[size_out]:
            bound_xavier = np.sqrt(6.0 / (size_next + size_prev)) # TODO: Default: Xavier Normalized Initialization
            self.weight_matrices += [bound_xavier * (2*np.random.rand(size_next, size_prev) - 1.0)]
            size_prev = size_next
        return

    def set_output_activation(self, func_name):
        if (func_name == "identity"):
            self.layers[-1].set_activation_identity()
        elif (func_name == 'logistic'):
            self.layers[-1].set_activation_logistic()
                
        else:
            return
    
    def set__all_hidden_activation(self, func_name, param=0.01):
        if (func_name == "leakyReLU"):
            for layer in self.layers[:-1]:
                layer.set_activation_leakyReLU(param)
        elif (func_name == "logistic"):
            for layer in self.layers[:-1]:
                layer.set_activation_logistic()
        else:
            return

    # Compute activation of layer (l) from activation of layer (l-1) recursively
    def compute_activation_recursive(self, layer_index, x_i):
        if (layer_index == -1):
            return x_i

        layer = self.layers[layer_index]
        b = layer.b
        w = self.weight_matrices[layer_index]
        a_prev = self.compute_activation_recursive(layer_index - 1, x_i)
        z = np.matmul(w, a_prev) + b
        layer.z = z
        layer.a = layer.activation_function(z)
        return layer.a

    def compute_forward(self, x_i):
        self.compute_activation_recursive(self.num_hidden, x_i)
        return self.layers[-1].z, self.layers[-1].a
    
    # Compute the delta vector recursively
    def delta_backward_propagation(self, layer_index, delta_next):
        layer = self.layers[layer_index]
        z = layer.z
        w_next = self.weight_matrices[layer_index + 1]
        delta = layer.activation_function_derivative(z) * delta_next * w_next # TODO: shape correctly and make matrix product that yields a column vector
        return delta

    def compute_cost_single_example(self, x_i, y_i):
        z_i, yhat_i = self.compute_forward(x_i)
        diff = yhat_i - y_i
        return np.dot(diff, diff) # TODO: default is least squares
    
    def compute_cost_batch(self, batch_input, batch_output):
        # Batch example data is given as a matrix for the multiple realizations
        # one example per row, column = size of the first/last input/output layer
        # TODO: does not verify input and output have the same length
        # TODO: check that incoming batch is 2D
        cost = 0.0
        for x_i, y_i in zip(batch_input, batch_output):
            cost += self.compute_cost_single_example(x_i, y_i)
        return cost / len(batch_input) # TODO: maybe leave the normalization task to anther function

    # The parameters of the ANN are not vectorized in this academic example
    # This leads to batch calculation of the gradient to look a little clunky
    def compute_gradient_single_example(self, x_i, y_i):
        # Compute the gradient of the output layer wrt weights and biases
        # Gradients of bias given a list (1 entry per Layer) of vector (size of layer)        
        grad_b_i = []
        # Gradients of weights given as list (1 entry per Layer) of matrices (size of weight matrix for that Layer)
        grad_w_i = []
        # Compute forward to obtain the activations for the given input
        z_i, yhat_i = self.compute_forward(x_i)
        dcost_dyhat = 2.0 * (yhat_i - y_i) # TODO: default is least squares
        # Backward propagation
        # Initialize with last layer
        delta_next = dcost_dyhat * self.layers[-1].activation_function_derivative(z_i)
        a_prev = self.layers[-2].a
        grad_b_i += [delta_next]
        grad_w_i += [np.outer(delta_next, np.transpose(a_prev))]

        for layer_index in reversed(range(self.num_hidden)):
            layer = self.layers[layer_index]
            w_next = self.weight_matrices[layer_index + 1]
            z = layer.z
            delta = np.matmul(np.transpose(w_next), delta_next) * layer.activation_function_derivative(z)
            a_prev = x_i if layer_index == 0 else self.layers[layer_index - 1].a
            grad_b_i += [delta]
            grad_w_i += [np.outer(delta, a_prev)]
            delta_next = delta
        grad_b_i.reverse(), grad_w_i.reverse()
        return grad_b_i, grad_w_i

    def compute_gradient_batch(self, batch_input, batch_output):
        # Batch example data is given as a matrix for the multiple realizations
        # one example per row, column = size of the first/last input/output layer
        # TODO: does not verify input and output have the same length
        # TODO: check that incoming batch is 2D
        invn = 1.0 / len(batch_input) # TODO: maybe leave the normalization task to anther function
        grad_b = [np.zeros(size) for size in list(self.size_hidden)+[self.size_out]]
        grad_w = [np.zeros(w.shape) for w in self.weight_matrices]

        for x_i, y_i in zip(batch_input, batch_output):
            grad_b_i, grad_w_i = self.compute_gradient_single_example(x_i, y_i)
            for l, (grad_b_i_layer_l, grad_w_i_layer_l) in enumerate(zip(grad_b_i, grad_w_i)):
                grad_b[l] += grad_b_i_layer_l
                grad_w[l] += grad_w_i_layer_l
        for l in range(self.num_hidden + 1):
            grad_b[l] *= invn
            grad_w[l] *= invn
        return grad_b, grad_w

    def update_weights_and_biases(self, increment_biases, increment_weights):
        for l, (db_l, dw_l) in enumerate(zip(increment_biases, increment_weights)):
            self.layers[l].b += db_l
            self.weight_matrices[l] += dw_l
        return


def stochastic_gradient_descent(ann, examples, learning_rate, batch_size, epoch_max):
    x_examples = examples[0]
    y_examples = examples[1]
    m = len(examples[0])
    counter_example = 0
    epoch = 0
    if (len(examples[0].shape) != 2):
        print("WARNING: Cannot batch single example. batch_size will be ignored")
        # Resize single example into 2D array
        x_examples.resize((1, m))
        y_examples.resize((1, m))
        batch_size = 1
    if (batch_size > m):
        batch_size = m
        print("WARNING: Batch size larger than examples size. batch_size will be reduced")


    cost_prev = ann.compute_cost_batch(x_examples, y_examples)
    history = [cost_prev]
    grad_b_history = [np.zeros(size) for size in list(ann.size_hidden)+[ann.size_out]]
    grad_w_history = [np.zeros(w.shape) for w in ann.weight_matrices]
    forget = 0.9
    increment_b = [np.zeros(size) for size in list(ann.size_hidden)+[ann.size_out]]
    increment_w = [np.zeros(w.shape) for w in ann.weight_matrices]
    while (epoch < epoch_max):
        batch_index = (counter_example + np.arange(batch_size))%m
        counter_example += batch_size
        # Compute gradient
        grad_batch = ann.compute_gradient_batch(x_examples[batch_index,:], y_examples[batch_index,:])
        # RMSProp update
        for l, (db_l, dw_l) in enumerate(zip(grad_batch[0], grad_batch[1])):
            grad_b_history[l] = forget * grad_b_history[l] + (1 - forget) * db_l * db_l
            grad_w_history[l] = forget * grad_w_history[l] + (1 - forget) * dw_l * dw_l
            increment_b[l] = -learning_rate * db_l / (np.sqrt(grad_b_history[l]) + 1e-8)
            increment_w[l] = -learning_rate * dw_l / (np.sqrt(grad_w_history[l]) + 1e-8)
        ann.update_weights_and_biases(increment_b, increment_w)
        # Only compute cost at end of epoch
        if (counter_example // m > epoch):
            # Compute total cost
            # Todo: check for descent and convergence
            cost = ann.compute_cost_batch(x_examples, y_examples)
            epoch += 1
            print("epoch: ", epoch, "/", epoch_max)
            history.append(cost)
            cost_prev = cost
    return history


def main():
    from matplotlib import pyplot as plt

    # Create network
    neural_net = ANN(size_in = 1, size_hidden = [5, 5], size_out = 1)

    # Regress continuous function on given interval
    neural_net.set_output_activation('identity')
    neural_net.set__all_hidden_activation('leakyReLU')
    func_to_learn = lambda x: np.sqrt(x)
    # Create random examples
    m = 1000
    xmin = 0.0
    xmax = 1.0
    n_all = 1000
    x_all = np.linspace(xmin, xmax, n_all)
    x_examples = xmin + (xmax - xmin) * np.random.rand(m).reshape(m,1)
    y_examples = func_to_learn(x_examples)

    # Train the network
    n_batches = 10
    learning_rate = 1e-2
    max_epoch = 1000
    hist = stochastic_gradient_descent(neural_net, [x_examples, y_examples], learning_rate, int(m/n_batches), max_epoch)

    # Show results
    plt.figure(1)
    plt.semilogy(hist)
    plt.xlabel('epoch')
    plt.ylabel('cost')

    plt.figure(3)
    for i, xi in enumerate(x_all.reshape(n_all,1)):
        plt.plot(xi, neural_net.compute_forward(xi)[1], marker='.', linestyle='none')
    plt.plot(x_all, func_to_learn(x_all), 'r')
    plt.show()

    return 0

if __name__ == "__main__":
    main()
        