import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import SimplePatchifier, TwoLayerNN


class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=8):
        super().__init__()
        self.num_edges = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.in_layer2 = TwoLayerNN(in_features)
        self.out_layer2 = TwoLayerNN(in_features)
        self.fc = nn.Linear(in_features*2, in_features)

    def forward(self, x):
        B, N, C = x.shape

        sim = x @ x.transpose(-1, -2)
        graph = sim.topk(5, dim=-1).indices

        shortcut = x
        x = self.in_layer1(x.view(B * N, -1)).view(B, N, -1)

        # aggregation
        neibor_features = x[torch.arange(
            B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph]
        x = torch.stack(
            [x, (neibor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1)

        # update
        # TODO: Should be multi-head
        # PyTorch has nn.MultiheadAttention, maybe useful.
        x = self.fc(x.view(B * N, -1)).view(B, N, -1)

        x = self.out_layer1(F.relu(x).view(B * N, -1)).view(B, N, -1)
        x = x + shortcut

        x = self.out_layer2(F.relu(self.in_layer2(
            x.view(B * N, -1)))).view(B, N, -1) + x

        return x


class VGNN(nn.Module):
    def __init__(self, in_features=256*3, num_patches=196, num_ViGBlocks=16):
        super().__init__()

        self.patchifier = SimplePatchifier()
        self.patch_embedding = TwoLayerNN(in_features)
        self.pose_embedding = nn.Parameter(
            torch.rand(num_patches, in_features))

        self.blocks = nn.Sequential(
            *[ViGBlock(in_features) for _ in range(num_ViGBlocks)])

    def forward(self, x):
        x = self.patchifier(x)
        B, N, C, H, W = x.shape
        x = self.patch_embedding(x.view(B * N, -1)).view(B, N, -1)
        x = x + self.pose_embedding

        x = self.blocks(x)

        return x
