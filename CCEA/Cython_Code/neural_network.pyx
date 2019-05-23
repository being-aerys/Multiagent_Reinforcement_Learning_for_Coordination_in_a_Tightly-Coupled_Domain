import numpy as np
from parameters import Parameters as p


cdef class NeuralNetwork:
    # Declare variables
    cdef int n_rovers
    cdef int n_inputs
    cdef int n_outputs
    cdef int n_nodes
    cdef int n_weights
    cdef double input_bias
    cdef double hidden_bias
    cdef public double[:, :] weights
    cdef public double[:, :] in_layer
    cdef public double[:, :] hid_layer
    cdef public double[:, :] out_layer

    def __cinit__(self):
        self.n_rovers = int(p.num_rovers * p.num_types)
        self.n_inputs = int(p.num_inputs)
        self.n_outputs = int(p.num_outputs)
        self.n_nodes = int(p.num_nodes)  # Number of nodes in hidden layer
        self.n_weights = int((self.n_inputs + 1)*self.n_nodes + (self.n_nodes + 1)*self.n_outputs)
        self.input_bias = 1.0
        self.hidden_bias = 1.0
        self.weights = np.zeros((self.n_rovers, self.n_weights), dtype=np.float64)
        self.in_layer = np.zeros((self.n_rovers, self.n_inputs), dtype=np.float64)
        self.hid_layer = np.zeros((self.n_rovers, self.n_nodes), dtype=np.float64)
        self.out_layer = np.zeros((self.n_rovers, self.n_outputs), dtype=np.float64)

    cpdef reset_nn(self):  # Clear current network
        self.weights = np.zeros((self.n_rovers, self.n_weights), dtype=np.float64)
        self.in_layer = np.zeros((self.n_rovers, self.n_inputs), dtype=np.float64)
        self.hid_layer = np.zeros((self.n_rovers, self.n_nodes), dtype=np.float64)
        self.out_layer = np.zeros((self.n_rovers, self.n_outputs), dtype=np.float64)

    cdef get_inputs(self, state_vec, rov_id):  # Get inputs from state-vector
        cdef int i
        for i in range(self.n_inputs):
            self.in_layer[rov_id, i] = state_vec[i]

    cdef get_weights(self, nn_weights, rov_id):  # Get weights from CCEA population
        cdef int i

        for i in range(self.n_weights):
            self.weights[rov_id, i] = nn_weights[i]

    cdef reset_layers(self, rov_id):  # Clear hidden layers and output layers
        cdef int i, j

        for i in range(self.n_nodes):
            self.hid_layer[rov_id, i] = 0.0
        for j in range(self.n_outputs):
            self.out_layer[rov_id, j] = 0.0

    cdef get_outputs(self, rov_id):
        cdef int count, i, j

        count = 0  # Keeps count of which weight is being applied
        self.reset_layers(rov_id)

        for i in range(self.n_inputs):  # Pass inputs to hidden layer
            for j in range(self.n_nodes):
                self.hid_layer[rov_id, j] += self.in_layer[rov_id, i] * self.weights[rov_id, count]
                count += 1

        for j in range(self.n_nodes):  # Add Biasing Node
            self.hid_layer[rov_id, j] += (self.input_bias * self.weights[rov_id, count])
            count += 1

        for i in range(self.n_nodes):  # Pass through sigmoid
            self.hid_layer[rov_id, i] = self.sigmoid(self.hid_layer[rov_id, i])

        for i in range(self.n_nodes):  # Pass from hidden layer to output layer
            for j in range(self.n_outputs):
                self.out_layer[rov_id, j] += self.hid_layer[rov_id, i] * self.weights[rov_id, count]
                count += 1

        for j in range(self.n_outputs):  # Add biasing node
            self.out_layer[rov_id, j] += (self.hidden_bias * self.weights[rov_id, count])
            count += 1

        for i in range(self.n_outputs):  # Pass through sigmoid
            self.out_layer[rov_id, i] = self.sigmoid(self.out_layer[rov_id, i]) - 0.5

    cdef tanh(self, double inp):  # Tanh function as activation function
        cdef double tanh

        tanh = (2/(1 + np.exp(-2*inp)))-1
        return tanh

    cdef sigmoid(self, double inp):  # Sigmoid function as activation function
        cdef double sig

        sig = 1/(1 + np.exp(-inp))
        return sig

    cpdef run_neural_network(self, state_vec, weight_vec, rover_id):
        self.get_inputs(state_vec, rover_id)
        self.get_weights(weight_vec, rover_id)
        self.get_outputs(rover_id)
