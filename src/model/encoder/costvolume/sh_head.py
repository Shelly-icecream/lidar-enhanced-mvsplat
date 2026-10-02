"""Independent geometry and SH heads, initialized from the joint predictor."""
from copy import deepcopy

import torch
from torch import nn


class SplitGaussianHead(nn.Module):
    def __init__(self, joint, channels_per_surface):
        super().__init__()
        self.channels_per_surface = channels_per_surface
        channels = joint[2].out_channels
        indices = torch.arange(channels).reshape(-1, channels_per_surface)
        geometry_indices = indices[:, :9].flatten()
        sh_indices = indices[:, 9:].flatten()
        self.register_buffer("geometry_indices", geometry_indices, persistent=False)
        self.register_buffer("sh_indices", sh_indices, persistent=False)
        self.geometry = self._branch(joint, geometry_indices)
        self.sh = self._branch(joint, sh_indices)

    @staticmethod
    def _branch(joint, indices):
        branch = deepcopy(joint)
        old = joint[2]
        last = nn.Conv2d(old.in_channels, len(indices), old.kernel_size,
                         old.stride, old.padding, device=old.weight.device,
                         dtype=old.weight.dtype)
        with torch.no_grad():
            last.weight.copy_(old.weight[indices])
            last.bias.copy_(old.bias[indices])
        branch[2] = last
        return branch

    def forward(self, features):
        geometry, sh = self.geometry(features), self.sh(features)
        b, _, h, w = geometry.shape
        return torch.cat((geometry.reshape(b, -1, 9, h, w),
                          sh.reshape(b, -1, self.channels_per_surface - 9, h, w)),
                         dim=2).flatten(1, 2)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Migrate joint-head checkpoints; split checkpoints load normally.
        if prefix + "0.weight" in state_dict:
            for parameter in ("weight", "bias"):
                first = state_dict.pop(prefix + "0." + parameter)
                last = state_dict.pop(prefix + "2." + parameter)
                for name, indices in (("geometry", self.geometry_indices),
                                      ("sh", self.sh_indices)):
                    state_dict[prefix + name + ".0." + parameter] = first.clone()
                    state_dict[prefix + name + ".2." + parameter] = last[indices.to(last.device)].clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
