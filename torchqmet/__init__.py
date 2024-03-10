from typing import *

import abc

import torch
import torch.nn as nn

from .transforms import TransformBase, make_transform
from .reductions import ReductionBase, make_reduction


class QuasimetricBase(nn.Module, metaclass=abc.ABCMeta):
    input_size: int  # dimensionality of input latent space
    num_components: int  # number of components to be combined to form the latent quasimetric
    discount: Optional[float]  # if set, output the discounted quasimetric, `discount ** d`
    guaranteed_quasimetric: bool  # whether this is guaranteed to satisfy quasimetric constraints

    transforms: nn.Sequential  # Sequential[TransformBase]
    reduction: ReductionBase

    scale: torch.Tensor  # some non-trainable scale that can be set, if needed

    def __init__(self, input_size: int, num_components: int, *,
                 warn_if_not_quasimetric: bool = True, guaranteed_quasimetric: bool,
                 transforms: Collection[str], reduction: str, discount: Optional[float] = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_components = num_components
        self.guaranteed_quasimetric = guaranteed_quasimetric
        self.discount = discount
        self.register_buffer('scale', torch.ones(()))

        _transforms: List[TransformBase] = []
        for transform in transforms:
            _transforms.append(make_transform(transform, num_components))
            num_components = _transforms[-1].output_num_components
        self.transforms = nn.Sequential(*_transforms)
        self.reduction = make_reduction(reduction, num_components, discount)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'scale' not in state_dict:
            state_dict[prefix + 'scale'] = torch.ones(())  # BC
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @abc.abstractmethod
    def compute_components(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r'''
        Inputs:
            x (torch.Tensor): Shape [..., input_size]
            y (torch.Tensor): Shape [..., input_size]

        Output:
            d (torch.Tensor): Shape [..., num_components]
        '''
        pass

    def forward(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        assert x.shape[-1] == y.shape[-1] == self.input_size
        d = self.compute_components(x, y, **kwargs)
        d: torch.Tensor = self.transforms(d)
        scale =  self.scale
        if not self.training:
            scale = scale.detach()
        return self.reduction(d) * scale

    def __call__(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # Manually define for typing
        # https://github.com/pytorch/pytorch/issues/45414
        return super().__call__(x, y, **kwargs)

    def extra_repr(self) -> str:
        return f"guaranteed_quasimetric={self.guaranteed_quasimetric}\ninput_size={self.input_size}, num_components={self.num_components}" + (
            ", discount=None" if self.discount is None else f", discount={self.discount:g}"
        )


from .pqe import PQE, PQELH, PQEGG
from .iqe import IQE, IQE2
from .mrn import MRN, MRNFixed
from .neural_norms import DeepNorm, WideNorm

__all__ = ['PQE', 'PQELH', 'PQEGG', 'IQE', 'MRN', 'MRNFixed', 'DeepNorm', 'WideNorm']
__version__ = "0.1.0"
