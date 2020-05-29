from typing import List, Optional

from torch import nn
import torch.nn.functional as F


class FullyConnectedNetwork(nn.Module):
    """A feed forward, fully connected network."""
    def __init__(self, input_size: int, output_size: int, hidden_layer_widths: Optional[List[int]] = None):
        """The network is initialized with a list of hidden layer sizes."""
        super().__init__()
        if hidden_layer_widths is None:
            hidden_layer_widths = [64, 64, 64]

        input_layer_output_size = hidden_layer_widths[0] if any(hidden_layer_widths) else output_size
        output_layer_input_size = hidden_layer_widths[-1] if any(hidden_layer_widths) else input_size

        self.input_layer = nn.Linear(input_size, input_layer_output_size)
        self.fc_layers = nn.ModuleList([
            nn.Linear(width0, width1)
            for width0, width1
            in zip(hidden_layer_widths[:-1], hidden_layer_widths[1:])])
        self.output_layer = nn.Linear(output_layer_input_size, output_size)

    def forward(self, state):
        """Map an environment state to an action."""
        net = F.relu(self.input_layer(state))
        for fc in self.fc_layers:
            net = F.relu(fc(net))

        return self.output_layer(net)


class DuelingNetwork(nn.Module):
    """A dueling network architecture which splits the computation of the Q-function
    to the computation of the value and advantage."""
    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_layer_widths: Optional[List[int]] = None):
        super().__init__()
        if hidden_layer_widths is None:
            hidden_layer_widths = [64, 64, 64]

        latent_size = hidden_layer_widths[-1]
        self.backbone = FullyConnectedNetwork(input_size, latent_size, hidden_layer_widths)
        self.value = FullyConnectedNetwork(latent_size, 1, [latent_size // 2])
        self.advantage = FullyConnectedNetwork(latent_size, output_size, [latent_size // 2])

    def forward(self, state):
        latent = self.backbone(state)
        value = self.value(latent)
        advantage = self.advantage(latent)

        zero_advantage = advantage - advantage.mean(dim=1, keepdim=True)[0]

        return value + zero_advantage
