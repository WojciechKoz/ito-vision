from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from ito_vision.discretizations import Discretization
from ito_vision.parametrizations import IdentityParametrization, Parametrization
from ito_vision.samplers import Sampler
from ito_vision.schedulers.scheduler import Scheduler


class IterativeRefinementMethod:
    """
    We consider all linear SDEs with form:
    dx = [a(t)x + b(t)y]dt + g(t)dw
    """

    def __init__(
        self,
        scheduler: Scheduler,
        clip: Optional[Tuple[float, float]] = (-1, 1),
        parametrization: Optional[Parametrization] = None,
    ):
        self.scheduler = scheduler
        self.clip = clip
        self.parametrization = (
            parametrization
            if parametrization is not None
            else IdentityParametrization()
        )

        self.parametrization.set_method_reference(self)

    """ unique functions for each diffusion-based method """

    def a(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Method 'a' must be implemented in subclasses.")

    def b(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Method 'b' must be implemented in subclasses.")

    def g(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Method 'g' must be implemented in subclasses.")

    def transition_lambda_x(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Method 'transition_lambda_x' must be implemented in subclasses."
        )

    def transition_lambda_y(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Method 'transition_lambda_y' must be implemented in subclasses."
        )

    def transition_std(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Method 'transition_std' must be implemented in subclasses."
        )

    def base_distribution(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Method 'base_distribution' must be implemented in subclasses."
        )

    """ functions that all methods share """

    def f(
        self, xt: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.a(t) * xt + self.b(t) * (y if y is not None else 0)

    def sample_time(self, x0: torch.Tensor) -> torch.Tensor:
        return torch.rand((x0.shape[0], *[1 for _ in x0.shape[1:]]), device=x0.device)

    def transition_mu(
        self, x0: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if y is None:
            return self.transition_lambda_x(t) * x0
        return self.transition_lambda_x(t) * x0 + self.transition_lambda_y(t) * y

    def transition_sample(
        self, x0: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.transition_mu(x0, t, y)
        std = self.transition_std(t)
        epsilon = torch.randn_like(mu)
        return mu + std * epsilon, epsilon

    def score(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.pred_x0(model, xt, t, y, **kwargs)
        return (self.transition_mu(x0, t, y) - xt) / self.transition_std(t) ** 2, x0

    def pred_x0(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        x0 = self.parametrization.get_x0(model, xt, t, y, **kwargs)

        if self.clip is not None:
            x0 = x0.clamp(*self.clip)

        return x0

    def loss(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        t = self.sample_time(x0)

        xt, epsilon = self.transition_sample(x0, t, y)

        ground_truth = self.parametrization.get_ground_truth(x0, xt, epsilon, t)

        pred = self.parametrization(model, xt, t, y, **kwargs)

        loss = (pred - ground_truth).square().mean(
            dim=list(range(1, pred.ndim))
        ) * self.parametrization.loss_weight(t)

        return loss.mean()

    @torch.no_grad()
    def sample(
        self,
        discretization: Discretization,
        sampler: Sampler,
        model: nn.Module,
        x1: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        ts = discretization(sampler.N, x1.device)
        return sampler(self, model, x1, ts, y, return_trajectory, **kwargs)
