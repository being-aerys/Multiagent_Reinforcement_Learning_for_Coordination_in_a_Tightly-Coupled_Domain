import torch.nn as nn
import torch.nn.functional as F
import torch

class Actor(nn.Module):
    """Actor model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        l1 = 100; l2 = 50

        # Construct Hidden Layer 1
        self.f1 = nn.Linear(args.state_dim, l1)
        self.ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.f2 = nn.Linear(l1, l2)
        self.ln2 = nn.LayerNorm(l2)

        #Out
        self.w_out = nn.Linear(l2, args.action_dim)

    def forward(self, input):
        """Method to forward propagate through the actor's graph

            Parameters:
                  input (tensor): states

            Returns:
                  action (tensor): actions


        """

        #Hidden Layer 1
        out = F.elu(self.f1(input))
        out = self.ln1(out)

        #Hidden Layer 2
        out = F.elu(self.f2(out))
        out = self.ln2(out)

        #Out
        return torch.tanh(self.w_out(out))