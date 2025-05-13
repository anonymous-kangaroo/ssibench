from __future__ import annotations
from typing import TYPE_CHECKING

from torch import Tensor
import torch

from deepinv.loss.sure import SureGaussianLoss
from deepinv.physics.mri import MRI, MRIMixin
from deepinv.physics.inpainting import Inpainting
if TYPE_CHECKING:
    from deepinv.physics.generator.base import PhysicsGenerator
    from deepinv.models.base import Reconstructor
    from deepinv.physics.forward import Physics

class ENSURELoss(SureGaussianLoss):
    r"""
    ENSURE loss for image reconstruction in Gaussian noise.

    The loss function extends :class:`.SUREGaussianLoss` and is designed for the following noise model:

    .. math::

        y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A(x).

    The loss is computed as

    .. math::

        \frac{1}{m}\|W^{-1}P(A^{\dagger}y - \inverse{y})\|_2^2 +\frac{2\sigma^2}{m\tau}b^{\top} \left(\inverse{A^{\top}y+\tau b_i} -
        \inverse{A^{\top}y}\right)

    where :math:`R` is the trainable network, :math:`A` is the forward operator,
    :math:`y` is the noisy measurement vector of size :math:`m`, :math:`W^{-1}P` is a weighted projection operator determined by the set of measurement operators,
    :math:`b\sim\mathcal{N}(0,I)` and :math:`\tau\geq 0` is a hyperparameter controlling the
    Monte Carlo approximation of the divergence.

    The ENSURE loss was proposed in `Aggarwal et al <https://arxiv.org/abs/2010.10631>`_.

    .. warning::

        We currently only provide an implementation for :class:`single-coil MRI <deepinv.physics.MRI>` and :class:`inpainting <deepinv.physics.Inpainting>`,
        where `A^\top=A^\dagger` such that :math:`P=A^{\dagger}A,W^2=\mathbb{E}\left[P\right]`.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param deepinv.physics.generator.PhysicsGenerator physics_generator: random physics generator used to compute the weighting :math:`W`.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence.
    :param torch.Generator rng: Optional random number generator. Default is None.
    """

    def __init__(
        self,
        sigma: float,
        physics_generator: PhysicsGenerator,
        tau: float = 0.01,
        rng: torch.Generator = None,
    ):
        super().__init__(sigma=sigma, tau=tau, rng=rng)
        d = physics_generator.step(batch_size=2000)["mask"].mean(0, keepdim=True)
        self.dsqrti = 1 / d.sqrt()

    def div(
        self, x_net: Tensor, y: Tensor, f: Reconstructor, physics: Physics
    ) -> Tensor:
        r"""
        Monte-Carlo estimation for the divergence of f(x).

        :param torch.Tensor x_net: Reconstructions.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module f: Reconstruction network.
        """
        b = torch.empty_like(y).normal_(generator=self.rng)
        x2 = f(physics.A(physics.A_adjoint(y) + b * self.tau), physics)
        return (b * (x2 - x_net) / self.tau).reshape(y.size(0), -1).mean(1)

    def forward(
        self, y: Tensor, x_net: Tensor, physics: Physics, model: Reconstructor, **kwargs
    ) -> Tensor:
        if isinstance(physics, MRI):
            metric = lambda y: MRIMixin().kspace_to_im(y * self.dsqrti)
        elif isinstance(physics, Inpainting):
            metric = lambda y: y * self.dsqrti
        else:
            raise ValueError(
                "ENSURE loss is currently only implemented for single-coil MRI."
            )

        y1 = physics.A(x_net)
        div = 2 * self.sigma2 * self.div(x_net, y, model, physics)
        mse = metric(y1 - y).pow(2).reshape(y.size(0), -1).mean(1)
        return mse + div - self.sigma2
