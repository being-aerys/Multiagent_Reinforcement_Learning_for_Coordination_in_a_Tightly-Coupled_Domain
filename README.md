# AutoEncoder for Latent Representation Learning and Multi-Agent Coordination using DDPG
- First we use an autoencoder to compress the state representation to one-fourth the original size.
- Then we train DDPG agents to learn joint optimal policies to observe points of interests in a rover domain using the compressed state representation.
  - We use **difference reward** as the reward function.
  - The learned policies perform **as good as** the policies learned with the original state representation.

## Demo
![Game Process](https://github.com/being-aerys/Multiagent_Reinforcement_Learning_for_Coordination_in_a_Tightly-Coupled_Domain/blob/master/Visualization_Code/demo.gif)
