import numpy as np

class OUNoise:
    """Ornstein-Uhelnbeck noise generator

        Parameters:
            action_dimension (int): Dimension of the action space
            scale (float): OU process scale
            mu (float): Mean for OU process
            theta (float): theta param for OU process
            sigma (float): Variance for OU process

        Returns:
            None
    """
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        """Sample noise from the OU generator

            Parameters:

            Returns:
                noise (ndarray): Noise from the OU generator
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


def get_list_generators(num_generators, action_dim):
    """Get a list of OU generators with varying parameters

        Parameters:
            num_generators (int): Number of OU generators
            action_dim (int): Dimension of the action space


        Returns:
            noise generators (list): A list of OU noise generators
    """

    NUM_REPLICATES = 1 #Number of policy anchors (expoloration beams) to start rollouts from

    noise_gens = []
    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.1, mu = 0.0, theta=0.15, sigma=0.2))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.15, mu = 0.0, theta=0.15, sigma=0.5))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.3, mu = 0.0, theta=0.15, sigma=0.2))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.1, mu = 0.0, theta=0.15, sigma=0.9))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.1, mu = 0.0, theta=0.15, sigma=0.5))

    for _ in range(NUM_REPLICATES): noise_gens.append(OUNoise(action_dim, scale=0.2, mu = 0.0, theta=0.15, sigma=0.1))


    #IF anything left
    for i in range(num_generators - len(noise_gens)):
        noise_gens.append(noise_gens.append(OUNoise(action_dim, scale=0.1, mu = 0.0, theta=0.15, sigma=0.2)))

    return noise_gens