import torch
import torch.nn as nn

class GRUCellStack(nn.Module):
    """Multi-layer stack of GRU cells"""

    def __init__(self, input_size, hidden_size, num_layers, cell_type):
        super().__init__()
        self.num_layers = num_layers
        layer_size = hidden_size // num_layers
        assert layer_size * num_layers == hidden_size, "Must be divisible"
        if cell_type == 'gru':
            cell = nn.GRUCell
        else:
            assert False, f'Unknown cell type {cell_type}'
        layers = [cell(input_size, layer_size)] 
        layers.extend([cell(layer_size, layer_size) for _ in range(num_layers - 1)])
        self.layers = nn.ModuleList(layers)

    def forward(self, input: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        input_states = state.chunk(self.num_layers, -1)
        output_states = []
        x = input
        for i in range(self.num_layers):
            x = self.layers[i](x, input_states[i])
            output_states.append(x)
        return torch.cat(output_states, -1)
