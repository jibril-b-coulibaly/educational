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
        self.activation_function = lambda x: x*(x > 0.0) # ReLU by default, same for a given layer
        self.activation_function_derivative = lambda x: 1.0*(x >= 0.0) + 1e-4*(x < 0.0) # to avoid singularity
        return

    def set_activation_identity(self):
        self.activation_function = lambda x: x
        self.activation_function_derivative = lambda x: 1.0
    
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
            print(self.weight_matrices[-1])
            size_prev = size_next
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
        cost = 0.0
        for x_i, y_i in zip(batch_input, batch_output):
            cost += self.compute_cost_single_example(x_i, y_i)
        return cost / len(batch_input) 

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
        print("dcost_dyhat = ", dcost_dyhat)
        print("z_i = ", z_i)
        # Backward propagation
        # Initialize with last layer
        delta_next = dcost_dyhat * self.layers[-1].activation_function_derivative(z_i)
        print("delta_next = ", delta_next)
        a_prev = self.layers[-2].a
        grad_b_i += [delta_next]
        grad_w_i += [np.outer(delta_next, np.transpose(a_prev))]

        for layer_index in reversed(range(self.num_hidden)):
            print("layer_index = ", layer_index)
            layer = self.layers[layer_index]
            w_next = self.weight_matrices[layer_index + 1]
            z = layer.z
            print("delta_next = ", delta_next)
            print("w_next = ", w_next)
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
        invn = 1.0 / len(batch_input) 
        grad_b = [np.zeros(size) for size in self.list(self.size_hidden)+[self.size_out]]
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

def main():
    # 1 input 1 output, 1xN hidden layers
    hidden_layers_size = [5, 5]
    neural_net = ANN(1, hidden_layers_size, 1)
    print(neural_net.layers[0].a)
    print(neural_net.layers[1].a)
    print(neural_net.layers[2].a)
    
    print(neural_net.weight_matrices[0])
    print(neural_net.weight_matrices[1])
    print(neural_net.weight_matrices[2])
    print(neural_net.compute_forward(np.array([2])))

    # Compute gradient
    print(neural_net.compute_cost_single_example(np.array([2]), np.array([1])))
    print(neural_net.compute_gradient_single_example(np.array([2]), np.array([10])))
    return 0



if __name__ == "__main__":
    main()
        