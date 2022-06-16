"""
Extensions for stylegan3 visualizer
"""

import torch
import copy
from typing import List
from training.networks_stylegan3 import SynthesisNetwork, SynthesisLayer, Generator, MappingNetwork
from torch_utils import misc


def insert_layer(gen: Generator, layer: torch.nn.Module, after: str, name: str):
    syn: SynthesisNetwork = gen.synthesis
    mapping: MappingNetwork = gen.mapping

    # Insert into sythesis network
    name = f'{name}'
    setattr(syn, name, layer)
    idx = syn.layer_names.index(after) + 1
    syn.layer_names.insert(idx, name)

    # Increase number of ws
    syn.num_ws += 1
    mapping.num_ws += 1
    gen.num_ws += 1


class ChannelScalingLayer(torch.nn.Module):
    channels: List[int]
    factor: int

    def __init__(self,
                 syn: SynthesisNetwork,
                 after_layer: str,
                 channels: List[int],
                 factor: float = 0.,
                 ):
        super().__init__()
        self.channels = copy.copy(channels)
        self.factor = factor

        ref: SynthesisLayer = getattr(syn, after_layer)
        # Out is not typo! It sits after ref and thus should have the output dimensions
        self.in_channels = getattr(ref, "out_channels", getattr(ref, "channels", None))
        self.out_channels = getattr(ref, "out_channels", getattr(ref, "channels", None))
        self.in_size = getattr(ref, "out_size", getattr(ref, "size", None))
        self.out_size = getattr(ref, "out_size", getattr(ref, "size", None))

    def forward(self, x, *_, **__):
        if len(self.channels) == 0:
            return x
        misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        size = x.size()
        mask_size = list(size)
        mask_size[1] = len(self.channels)
        mask = torch.ones(size)
        mask[:, self.channels] = torch.full(mask_size, self.factor)
        mask = mask.cuda()
        x = x * mask

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])

        return x * mask
