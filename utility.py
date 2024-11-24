from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layer(x) + x


class SimplePatchifier(nn.Module):

    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).unfold(1, self.patch_size, self.patch_size)\
            .unfold(2, self.patch_size, self.patch_size).contiguous()\
            .view(B, -1, C, self.patch_size, self.patch_size)
        return x

class WeightedPatchifier(nn.Module):

    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):

        # B, C, H, W = x.shape
        # image = x[:, :3, :, :]
        # mask = x[:, 3:, :, :]
        #
        # patches = image.permute(0, 2, 3, 1).unfold(1, self.patch_size, self.patch_size)\
        #     .unfold(2, self.patch_size, self.patch_size).contiguous()\
        #     .view(B, -1, C, self.patch_size, self.patch_size)
        #
        # mask_patches = mask.permute(0, 2, 3, 1).unfold(1, self.patch_size, self.patch_size)\
        #     .unfold(2, self.patch_size, self.patch_size).contiguous()\
        #     .view(B, -1, C, self.patch_size, self.patch_size)
        #
        # # Calculate intersection (sum of mask values in each patch)
        # intersection = mask_patches.sum(dim=(-2, -1))  # Sum over patch height and width
        #
        # # Calculate weights based on overlap (normalize by patch area)
        # weights = intersection / (self.patch_size * self.patch_size)
        #
        # # Reshape weights to match patch dimensions for broadcasting
        # weights = weights.view(B, -1, 1, 1, 1)  # (B, N, 1, 1, 1)
        #
        # # Apply weights to patches (scaling each patch)
        # weighted_patches = patches * weights
        #
        # return weighted_patches

        B, C, H, W = x.shape
        assert C == 4, "Input must have 4 channels (image + mask)."

        # Calculate padding
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size

        # Pad image and mask
        x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad (W_left, W_right, H_top, H_bottom)

        # Separate image and mask
        image = x[:, :3, :, :]  # First 3 channels: Image
        mask = x[:, 3, :, :].unsqueeze(1)  # Fourth channel: Mask, keep channel dim

        # Patchify the image
        patches = image.permute(0, 2, 3, 1).unfold(1, self.patch_size, self.patch_size) \
            .unfold(2, self.patch_size, self.patch_size).contiguous() \
            .view(B, -1, C, self.patch_size, self.patch_size)

        # Patchify the mask
        mask_patches = mask.unfold(2, self.patch_size, self.patch_size) \
            .unfold(3, self.patch_size, self.patch_size).contiguous() \
            .view(B, -1, self.patch_size, self.patch_size)

        # Calculate intersection (sum of mask values in each patch)
        intersection = mask_patches.sum(dim=(-2, -1))  # Sum over patch height and width

        # Calculate weights based on overlap (normalize by patch area)
        weights = intersection / (self.patch_size * self.patch_size)  # Shape: (B, N)

        # Reshape weights to match patch dimensions for broadcasting
        weights = weights.view(B, -1, 1, 1, 1)  # (B, N, 1, 1, 1)

        # Apply weights to patches (scaling each patch)
        weighted_patches = patches * weights

        return weighted_patches

