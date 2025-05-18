import torch
from torch import nn

class OccipitalLobe(nn.Module):
    """Simple MLP over CLIP embeddings producing 128-dim visual features."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    @torch.no_grad()
    def process(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)

if __name__ == "__main__":
    lobe = OccipitalLobe()
    dummy = torch.zeros(1, 512)
    print(lobe.process(dummy).shape)
