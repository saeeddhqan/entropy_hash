import random

import numpy
import torch
import torch.nn as nn

seed = 1234


def set_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)


class EntropyHashNetwork(nn.Module):
    def __init__(
        self, input_dim: int = 1024, output_dim: int = 256, device: str = "cuda"
    ):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        self.model = nn.Linear(input_dim, output_dim, bias=False)
        self.apply(self.norm_weights)

    def norm_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)

    def forward(self, x: torch.Tensor, ndocs: int) -> torch.Tensor:
        x = self.model(x)  # Shape: (N, output_dim)
        x = x.view(ndocs, -1, self.output_dim).mean(dim=1)  # Shape: (ndocs, output_dim)
        return x


def initialize_network(device: str, context_window: int, dim: int):
    network = EntropyHashNetwork(
        input_dim=context_window, output_dim=dim, device=device
    )
    network.to(device)
    return network


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    network = initialize_network(device, 1024, 256)
    sample_input = torch.randn(2, 1024).to(device)
    output = network(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output tensor:\n{output}")
