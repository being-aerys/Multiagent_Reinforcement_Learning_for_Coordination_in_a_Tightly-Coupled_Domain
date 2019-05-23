import numpy as np
from parameters import Parameters as p


class NeuralNetwork:

    def __init__(self):
        self.n_rovers = p.num_rovers * p.num_types
        self.n_inputs = p.num_inputs
        self.n_outputs = p.num_outputs
        self.n_nodes = p.num_nodes  # Number of nodes in hidden layer
        self.n_weights = (self.n_inputs + 1)*self.n_nodes + (self.n_nodes + 1)*self.n_outputs
        self.input_bias = 1.0
        self.hidden_bias = 1.0
        self.weights = np.zeros((self.n_rovers, self.n_weights), dtype=np.float64)
        self.in_layer = np.zeros((self.n_rovers, self.n_inputs), dtype=np.float64)
        self.hid_layer = np.zeros((self.n_rovers, self.n_nodes), dtype=np.float64)
        self.out_layer = np.zeros((self.n_rovers, self.n_outputs), dtype=np.float64)

    def reset_nn(self):  # Clear current network
        self.weights = np.zeros((self.n_rovers, self.n_weights), dtype=np.float64)
        self.in_layer = np.zeros((self.n_rovers, self.n_inputs), dtype=np.float64)
        self.hid_layer = np.zeros((self.n_rovers, self.n_nodes), dtype=np.float64)
        self.out_layer = np.zeros((self.n_rovers, self.n_outputs), dtype=np.float64)

    def get_inputs(self, state_vec, rov_id):  # Get inputs from state-vector
        for i in range(p.num_inputs):
            self.in_layer[rov_id, i] = state_vec[i]

    def get_weights(self, nn_weights, rov_id):  # Get weights from CCEA population
        for i in range(self.n_weights):
            self.weights[rov_id, i] = nn_weights[i]

    def reset_layers(self, rov_id):  # Clear hidden layers and output layers
        for i in range(self.n_nodes):
            self.hid_layer[rov_id, i] = 0.0
        for j in range(self.n_outputs):
            self.out_layer[rov_id, j] = 0.0

    def get_outputs(self, rov_id):
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

    def tanh(self, inp):  # Tanh function as activation function
        tanh = (2/(1 + np.exp(-2*inp)))-1
        return tanh

    def sigmoid(self, inp):  # Sigmoid function as activation function
        sig = 1/(1 + np.exp(-inp))
        return sig

    def run_neural_network(self, state_vec, weight_vec, rover_id):
        self.get_inputs(state_vec, rover_id)
        self.get_weights(weight_vec, rover_id)
        self.get_outputs(rover_id)
