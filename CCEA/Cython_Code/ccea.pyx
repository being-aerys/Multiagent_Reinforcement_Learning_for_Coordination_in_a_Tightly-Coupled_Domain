import numpy as np
from parameters import Parameters as p
import random


cdef class Ccea:
    # Declare variables
    cdef double mut_prob
    cdef double epsilon
    cdef int n_populations
    cdef public int population_size
    cdef int n_inputs
    cdef int n_outputs
    cdef int n_nodes
    cdef int policy_size
    cpdef public double[:, :, :] pops
    cpdef public double[:, :] fitness
    cpdef public double[:, :] team_selection

    def __cinit__(self):
        self.mut_prob = p.mutation_rate
        self.epsilon = p.epsilon
        self.n_populations = int(p.num_rovers * p.num_types ) # One population for each rover
        self.population_size = int(p.pop_size * 2)  # Number of policies in each pop
        self.n_inputs = int(p.num_inputs)
        self.n_outputs = int(p.num_outputs)
        self.n_nodes = int(p.num_nodes)  # Number of nodes in hidden layer
        self.policy_size = int((self.n_inputs + 1)*self.n_nodes + (self.n_nodes + 1) * self.n_outputs)  # Number of weights for NN
        self.pops = np.zeros((self.n_populations, self.population_size, self.policy_size))
        self.fitness = np.zeros((self.n_populations, self.population_size))
        self.team_selection = np.zeros((self.n_populations, self.population_size))

    cpdef reset_populations(self):  # Re-initializes CCEA populations for new run
        cdef int pop_index, policy_index, w

        self.team_selection = np.zeros((self.n_populations, self.population_size))
        for pop_index in range(self.n_populations):
            for policy_index in range(self.population_size):
                for w in range(self.policy_size):
                    self.pops[pop_index, policy_index, w] = random.uniform(-1, 1)

    cpdef select_policy_teams(self):  # Create policy teams for testing
        cdef int pop_id, j, k, rpol

        self.team_selection = np.zeros((self.n_populations, self.population_size))
        for pop_id in range(self.n_populations):
            for j in range(self.population_size):
                rpol = random.randint(0, (self.population_size - 1))  # Select a random policy from pop
                k = 0
                while k < j:  # Check for duplicates
                    if rpol == self.team_selection[pop_id, k]:
                        rpol = random.randint(0, (self.population_size - 1))
                        k = -1
                    k += 1
                self.team_selection[pop_id, j] = rpol  # Assign policy to team

    cpdef mutate(self):
        cdef int half_pop_length, pop_index, policy_index, target
        cdef double rnum

        half_pop_length = int(self.population_size/2)

        for pop_index in range(self.n_populations):
            policy_index = half_pop_length
            while policy_index < self.population_size:
                rnum = random.uniform(0, 1)
                if rnum <= self.mut_prob:
                    target = random.randint(0, (self.policy_size - 1))  # Select random weight to mutate
                    self.pops[pop_index, policy_index, target] = random.uniform(-1, 1)
                policy_index += 1

    cpdef epsilon_greedy_select(self):  # Replace the bottom half with parents from top half
        cdef int half_pop_length, policy_id, k, pop_id, parent
        cdef double rnum

        half_pop_length = int(self.population_size/2)
        for pop_id in range(self.n_populations):
            policy_id = half_pop_length
            while policy_id < self.population_size:
                rnum = random.uniform(0, 1)
                if rnum >= self.epsilon:  # Choose best policy
                    for k in range(self.policy_size):
                        self.pops[pop_id, policy_id, k] = self.pops[pop_id, 0, k]  # Best policy
                else:
                    parent = random.randint(0, half_pop_length)  # Choose a random parent
                    for k in range(self.policy_size):
                        self.pops[pop_id, policy_id, k] = self.pops[pop_id, parent, k]  # Random policy
                policy_id += 1

    cpdef down_select(self):  # Create a new offspring population using parents from top 50% of policies
        cdef int pop_id, j, k

        # Reorder populations in terms of fitness (top half = best policies)
        for pop_id in range(self.n_populations):
            for j in range(self.population_size):
                k = j + 1
                while k < self.population_size:
                    if self.fitness[pop_id, j] < self.fitness[pop_id, k]:
                        self.fitness[pop_id, j], self.fitness[pop_id, k] = self.fitness[pop_id, k], self.fitness[pop_id, j]
                        self.pops[pop_id, j], self.pops[pop_id, k] = self.pops[pop_id, k], self.pops[pop_id, j]
                    k += 1

        self.epsilon_greedy_select()  # Select parents for offspring population
        self.mutate()  # Mutate offspring population
