from typing import List, Optional

from torch import nn
import torch.nn.functional as F


class FullyConnectedNetwork(nn.Module):
    """A feed forward, fully connected network."""
    def __init__(self, state_size: int, action_size: int, hidden_layer_widths: Optional[List[int]] = None):
        """The network is initialized with a list of hidden layer sizes."""
        super().__init__()
        if hidden_layer_widths is None:
            hidden_layer_widths = [128, 64, 64]

        input_layer_output_size = hidden_layer_widths[0] if any(hidden_layer_widths) else action_size
        output_layer_input_size = hidden_layer_widths[-1] if any(hidden_layer_widths) else state_size

        self.input_layer = nn.Linear(state_size, input_layer_output_size)
        self.fc_layers = nn.ModuleList([
            nn.Linear(width0, width1)
            for width0, width1
            in zip(hidden_layer_widths[:-1], hidden_layer_widths[1:])])
        self.output_layer = nn.Linear(output_layer_input_size, action_size)

    def forward(self, state):
        """Map an environment state to an action."""
        net = F.relu(self.input_layer(state))
        for fc in self.fc_layers:
            net = F.relu(fc(net))

        return self.output_layer(net)
