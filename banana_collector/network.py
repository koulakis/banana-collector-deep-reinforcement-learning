from torch import nn
import torch.nn.functional as F


class FullyConnected(nn.Module):
    """A feed forward, fully connected network."""
    def __init__(self, state_size: int, action_size: int, network_width: int = 64, number_of_hidden_layers: int = 2):
        """The network has a number of hidden layers with the a fixed number of neurons per layer."""
        super().__init__()
        self.input_layer = nn.Linear(state_size, network_width)
        self.fc_layers = [nn.Linear(network_width, network_width) for _ in range(number_of_hidden_layers)]
        self.output_layer = nn.Linear(network_width, action_size)

    def forward(self, state):
        """Map an environment state to an action."""
        net = F.relu(self.input_layer(state))
        for fc in self.fc_layers:
            net = F.relu(fc(net))

        return F.softmax(self.output_layer(net))
