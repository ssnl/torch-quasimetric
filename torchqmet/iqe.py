r'''
Inteval Quasimetric Embedding (IQE)
https://arxiv.org/abs/2211.15120
'''

from typing import *

import torch
import math

from . import QuasimetricBase

# The PQELH function.

@torch.jit.script
def f_PQELH(h: torch.Tensor):  # PQELH: strictly monotonically increasing mapping from [0, +infty) -> [0, 1)
    return -torch.expm1(-h)

def iqe_tensor_delta(x: torch.Tensor, y: torch.Tensor, delta: torch.Tensor, div_pre_f: torch.Tensor, mul_kind: str,
                     fake_grad: bool = True) -> torch.Tensor:
    D = x.shape[-1]  # D: component_dim

    # ignore pairs that x >= y
    valid = (x < y)

    # sort to better count
    xy = torch.cat(torch.broadcast_tensors(x, y), dim=-1)
    sxy, ixy = xy.sort(dim=-1)

    # neg_inc: the **negated** increment of **input** of f at sorted locations
    # inc = torch.gather(delta * valid, dim=-1, index=ixy % D) * torch.where(ixy < D, 1, -1)
    neg_inc = torch.gather(delta * valid, dim=-1, index=ixy % D) * torch.where(ixy < D, -1, 1)

    # neg_incf: the **negated** increment of **output** of f at sorted locations
    neg_f_input = torch.cumsum(neg_inc, dim=-1) / div_pre_f[:, None]

    if fake_grad:
        neg_f_input__grad_path = neg_f_input.clone()
        neg_f_input__grad_path.data.clamp_(max=17)  # fake grad
        neg_f_input = neg_f_input__grad_path + (
            neg_f_input - neg_f_input__grad_path
        ).detach()

    neg_f = torch.expm1(neg_f_input)
    neg_incf = torch.cat([neg_f.narrow(-1, 0, 1), torch.diff(neg_f, dim=-1)], dim=-1)

    # reduction
    if neg_incf.ndim == 3:
        comp = torch.einsum('bkd,bkd->bk', sxy, neg_incf)
    else:
        comp = (sxy * neg_incf).sum(-1)

    if mul_kind == 'undiv':
        comp = comp * div_pre_f
    elif mul_kind == 'normdiv':
        comp = comp / f_PQELH(D / 8 / div_pre_f)
    elif mul_kind == 'normdeltadiv':
        comp = comp / f_PQELH(delta.expand(x.shape[-2:]).sum(-1) / 8 / div_pre_f)
    elif mul_kind == 'normdiv_half':
        comp = comp * 2 / f_PQELH(D / 8 / div_pre_f)
    else:
        assert mul_kind == 'none'
    return comp



def iqe(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    D = x.shape[-1]  # D: dim_per_component

    # ignore pairs that x >= y
    valid = x < y

    # sort to better count
    xy = torch.cat(torch.broadcast_tensors(x, y), dim=-1)
    sxy, ixy = xy.sort(dim=-1)

    # f(c) = indic( c > 0 )
    # at each location `x` along the real line, get `c` the number of intervals covering `x`, and apply `f`:
    #     \int f(c(x)) dx

    # neg_inc_copies: the **negated** increment of **input** of f at sorted locations, in terms of **#copies of delta**
    neg_inc_copies = torch.gather(valid, dim=-1, index=ixy % D) * torch.where(ixy < D, -1, 1)

    # neg_incf: the **negated** increment of **output** of f at sorted locations
    neg_inp_copies = torch.cumsum(neg_inc_copies, dim=-1)

    # delta = inf
    # f input: 0 -> 0, x -> -inf.
    # neg_inp = torch.where(neg_inp_copies == 0, 0., -delta)
    # f output: 0 -> 0, x -> 1.
    neg_f = (neg_inp_copies < 0) * (-1.)
    neg_incf = torch.cat([neg_f.narrow(-1, 0, 1), torch.diff(neg_f, dim=-1)], dim=-1)

    # reduction
    return (sxy * neg_incf).sum(-1)


if torch.__version__ >= '2.0.1' and False:  # well, broken process pool in notebooks
    iqe = torch.compile(iqe)
    iqe_tensor_delta = torch.compile(iqe_tensor_delta)
    # iqe = torch.compile(iqe, dynamic=True)
else:
    iqe = torch.jit.script(iqe)
    iqe_tensor_delta = torch.jit.script(iqe_tensor_delta)


class IQE(QuasimetricBase):
    r'''
    Inteval Quasimetric Embedding (IQE):
    https://arxiv.org/abs/2211.15120

    One-line Usage:

        IQE(input_size: int, dim_per_component: int = 16, ...)


    Default arguments implement IQE-maxmean. Set `reduction="sum"` to create IQE-sum.

    IQE-Specific Args:
        input_size (int): Dimension of input latent vectors
        dim_per_component (int): IQE splits latent vectors into chunks, where each chunk computes gives an IQE component.
                                 This is the number of latent dimensions assigned to each chunk. This number must
                                 perfectly divide ``input_size``. IQE paper recommends at least ``8``.
                                 Default: ``16``.

    Common Args (Exist for all quasimetrics, **Keyword-only**, Default values may be different for different quasimetrics):
        transforms (Collection[str]): A sequence of transforms to apply to the components, before reducing them to form
                                      the final latent quasimetric.
                                      Supported choices:
                                        + "concave_activation": Concave activation transform from Neural Norms paper.
                                      Default: ``()`` (no transforms).
        reduction (str): Reduction method to aggregate components into final quasimetric value.
                         Supported choices:
                           + "sum": Sum of components.
                           + "max": Max of components.
                           + "mean": Average of components.
                           + "maxmean": Convex combination of max and mean. Used in original Deep Norm, Wide Norm, and IQE.
                           + "deep_linear_net_weighted_sum": Weighted sum with weights given by a deep linear net. Used in
                                                             original PQE, whose components have a limited range [0, 1).
                         Default: ``"maxmean"``.
        discounted (Optional[float]): If not ``None``, this module instead estimates discounted distances with the
                                      base as ``discounted``.
                                      Default ``None``.
        warn_if_not_quasimetric (bool): If ``True``, issue a warning if this module does not always obey quasimetric
                                        constraints.  IQEs always obey quasimetric constraints.
                                        Default: ``True``.

    Shape:
        - Input: Two broadcastable tensors of shape ``(..., input_size)``
        - Output: ``(...)``

    Non-Module Attributes:
        input_size (int)
        num_components (int): Number of components to be combined to form the latent quasimetric. For IQEs, this is
                              ``input_size // dim_per_component``.
        discount (Optional[float])
        guaranteed_quasimetric (bool): Whether this is guaranteed to satisfy quasimetric constraints.

    Module Attributes:
        transforms (nn.Sequential[TransformBase]): Transforms to be applied on quasimetric components.
        reduction (ReductionBase): Reduction methods to aggregate components.

    Examples::

        >>> iqe = IQE(128, dim_per_component=16)
        >>> print(iqe)
        IQE(
          guaranteed_quasimetric=True
          input_size=128, num_components=8, discount=None
          (transforms): Sequential()
          (reduction): MaxMean(input_num_components=8)
        )
        >>> x = torch.randn(5, 128, requires_grad=True)
        >>> y = torch.randn(5, 128, requires_grad=True)
        >>> print(iqe(x, y))
        tensor([3.3045, 3.8072, 3.9671, 3.3521, 3.7831],, grad_fn=<LerpBackward1>)
        >>> print(iqe(y, x))
        tensor([3.3850, 3.8457, 4.0870, 3.1757, 3.9459], grad_fn=<LerpBackward1>)
        >>> print(iqe(x[:, None], x))  # pdist
        tensor([[0.0000, 3.8321, 3.7907, 3.5915, 3.3326],
                [3.9845, 0.0000, 4.0173, 3.8059, 3.7177],
                [3.7934, 4.3673, 0.0000, 4.0536, 3.6068],
                [3.1764, 3.4881, 3.5300, 0.0000, 2.9292],
                [3.7184, 3.8690, 3.8321, 3.5905, 0.0000]], grad_fn=<ReshapeAliasBackward0>)
    '''

    def __init__(self, input_size: int, dim_per_component: int = 16, *,
                 transforms: Collection[str] = (), reduction: str = 'maxmean',
                 discount: Optional[float] = None, warn_if_not_quasimetric: bool = True):
        assert dim_per_component > 0, "dim_per_component must be positive"
        assert input_size % dim_per_component == 0, \
            f"input_size={input_size} is not divisible by dim_per_component={dim_per_component}"
        num_components = input_size // dim_per_component
        super().__init__(input_size, num_components, guaranteed_quasimetric=True, warn_if_not_quasimetric=warn_if_not_quasimetric,
                         transforms=transforms, reduction=reduction, discount=discount)
        self.latent_2d_shape = torch.Size([num_components, dim_per_component])

    def compute_components(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return iqe(
            x=x.unflatten(-1, self.latent_2d_shape),
            y=y.unflatten(-1, self.latent_2d_shape),
        )


class IQE2(IQE):
    component_dropout_thresh: Tuple[float, float]  # multiplied with 1/num_components
    dropout_p_thresh: Tuple[float, float]
    dropout_batch_frac: float
    div_init_mul: float
    ema_weight: float
    ema_usage: torch.Tensor

    raw_delta: torch.Tensor
    raw_div: torch.Tensor
    mul_kind: str
    fake_grad: bool

    last_components: torch.Tensor
    last_drop_p: torch.Tensor

    def __init__(self, input_size: int, dim_per_component: int = 16, *,
                 transforms: Collection[str] = (), reduction: str = 'maxmean',
                 discount: Optional[float] = None, warn_if_not_quasimetric: bool = True,
                 learned_delta: bool = False, learned_div: bool = False,
                 div_init_mul: Optional[float] = None,  # exp( mul * dim_per_comp )
                 mul_kind: str = 'undiv',
                 fake_grad: bool = False,
                 component_dropout_thresh: Tuple[float, float] = (0.5, 2),
                 dropout_p_thresh: Tuple[float, float] = (0.005, 0.995),
                 dropout_batch_frac: float = 0.2,
                 ema_weight: float = 0.95):
        super().__init__(input_size, dim_per_component, transforms=transforms, reduction=reduction,
                         discount=discount, warn_if_not_quasimetric=warn_if_not_quasimetric)
        self.component_dropout_thresh = tuple(component_dropout_thresh)
        self.dropout_p_thresh = tuple(dropout_p_thresh)
        self.dropout_batch_frac = float(dropout_batch_frac)
        self.fake_grad = fake_grad
        assert 0 <= self.dropout_batch_frac <= 1
        self.ema_weight = float(ema_weight)
        assert 0 <= self.ema_weight <= 1
        self.register_buffer('ema_usage', torch.ones(self.num_components) / self.num_components)

        if learned_delta:
            # self.register_parameter(
            #     'raw_delta',
            #     torch.nn.Parameter(
            #         torch.zeros(self.latent_2d_shape).sub_(math.log(dim_per_component)).requires_grad_()
            #     )
            # )
            self.register_parameter(
                'raw_delta',
                torch.nn.Parameter(
                    torch.zeros(self.latent_2d_shape).requires_grad_()
                )
            )
        else:
            self.register_buffer(
                'raw_delta',
                torch.zeros(()),
            )
            # assert not learned_div
            # self.register_parameter(
            #     'raw_delta',
            #     None,
            # )

        if learned_div:
            if div_init_mul is None:
                div_init_mul = 2

            self.register_parameter(
                'raw_div',
                torch.nn.Parameter(torch.zeros(self.num_components).requires_grad_())
            )
        else:
            self.register_buffer(
                'raw_div',
                torch.zeros(1),
            )
            if div_init_mul is None:
                div_init_mul = 1 / dim_per_component

        with torch.no_grad():
            self.raw_div.add_(math.log(div_init_mul * dim_per_component))

        self.div_init_mul = div_init_mul
        self.mul_kind = mul_kind
        self.last_components = None
        self.last_drop_p = None


    def compute_components(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # if self.raw_delta is None:
        #     components = super().compute_components(x, y)
        # else:
        delta = self.raw_delta.exp()
        div_pre_f = self.raw_div.exp()
        # cap each component total to 1e3 to avoid overflow, fake gradient
        delta.data.clamp_(max=1e3 / (self.latent_2d_shape[-1] / 8))
        div_pre_f.data.clamp_(min=1e-3)

        components = iqe_tensor_delta(
            x=x.unflatten(-1, self.latent_2d_shape),
            y=y.unflatten(-1, self.latent_2d_shape),
            delta=delta,
            div_pre_f=div_pre_f,
            mul_kind=self.mul_kind,
            fake_grad=self.fake_grad,
        )

        # scale = self.ema_usage ** (-0.5)
        # scale /= scale.mean()

        # components = components * scale

        if self.training and False:
            bshape = components.shape[:-1]
            bsz = components[..., 0].numel()
            components = components.reshape(bsz, self.num_components)

            # 1. compute argmax before droput
            comp_indices = components.argmax(dim=-1)

            # # 2. do dropout
            # drop_bsz = int(bsz * self.dropout_batch_frac)
            # with torch.no_grad():
            #     drop_p = (self.ema_usage * self.num_components - self.component_dropout_thresh[0]).div(
            #         self.component_dropout_thresh[1] - self.component_dropout_thresh[0] + 1e-5
            #     ).clamp(*self.dropout_p_thresh)
            #     self.last_drop_p = drop_p
            #     keep_p = 1 - drop_p

            #     drop_mask = torch.empty_like(components)
            #     drop_mask[drop_bsz:].fill_(1)
            #     torch.bernoulli(
            #         keep_p.expand(drop_bsz, self.num_components),
            #         out=drop_mask[:drop_bsz],
            #     )
            #     drop_mask = drop_mask[torch.randperm(bsz, device=x.device)]
            #     # keep_bool.scatter_(
            #     #     # index=comp.min(dim=-1).indices[..., None],
            #     #     index=torch.randint(dist_est.num_components, size=(bsz, 1), device=device),
            #     #     dim=-1,
            #     #     value=True,
            #     # )
            #     # keep_bool[reg_bidx] = True

            # components = components * drop_mask

            # 3. update ema_usage
            with torch.no_grad():
                freq = torch.bincount(comp_indices, minlength=self.num_components) / bsz
                self.ema_usage.lerp_(
                    end=freq, weight=(1 - self.ema_weight),
                )

            components = components.reshape(*bshape, self.num_components)

        self.last_components = components
        return components

    def extra_repr(self) -> str:
        return super().extra_repr() + rf"""
component_dropout_thresh={self.component_dropout_thresh},
dropout_p_thresh={self.dropout_p_thresh},
dropout_batch_frac={self.dropout_batch_frac:g},
ema_weight={self.ema_weight:g},
learned_delta={self.raw_delta is not None},
learned_div={self.raw_div.requires_grad},
div_init_mul={self.div_init_mul:g},
mul_kind={self.mul_kind},
fake_grad={self.fake_grad},"""
