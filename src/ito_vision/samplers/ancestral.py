from typing import TYPE_CHECKING, Any, Iterator, Optional, Tuple

import torch

if TYPE_CHECKING:
    from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod

from ito_vision.samplers.sampler import Sampler


class AncestralSampler(Sampler):
    def __init__(self, N: int = 30, quiet: bool = False):
        super().__init__(N, quiet)

    def sample_xt(
        self,
        method: "IterativeRefinementMethod",
        x: torch.Tensor,
        pred_x0: torch.Tensor,
        y: Optional[torch.Tensor],
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        return self.sample_from_posterior(method, x, pred_x0, y, t_next, t)

    def sample_from_posterior(
        self,
        method: "IterativeRefinementMethod",
        x: torch.Tensor,
        pred_x0: torch.Tensor,
        y: Optional[torch.Tensor],
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = torch.randn_like(x)

        phi_x_0s = method.transition_lambda_x(s, s=0)
        phi_x_st = method.transition_lambda_x(t, s=s)

        phi_y_0s = method.transition_lambda_y(s, s=0)
        phi_y_st = method.transition_lambda_y(t, s=s)

        sigma_0s = method.transition_std(s, s=0)
        sigma_st = method.transition_std(t, s=s)

        variance = (sigma_st**2 * sigma_0s**2) / (
            phi_x_st**2 * sigma_0s**2 + sigma_st**2
        )

        lambda_hat_xt = variance * phi_x_st / sigma_st**2
        lambda_hat_x0 = variance * phi_x_0s / sigma_0s**2
        lambda_hat_y = variance * (
            phi_y_0s / sigma_0s**2 - phi_x_st * phi_y_st / sigma_st**2
        )

        return (
            lambda_hat_xt * x
            + lambda_hat_x0 * pred_x0
            + lambda_hat_y * (y if y is not None and y.shape == x.shape else 0)
            + torch.sqrt(variance) * epsilon
        )

    def __iter__(
        self,
        method: "IterativeRefinementMethod",
        model: torch.nn.Module,
        x1: torch.Tensor,
        ts: torch.Tensor,
        y: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        dts = torch.diff(ts)
        x = x1.clone()

        for t, dt in zip(ts[:-1], dts):
            pred_x0 = method.pred_x0(model, x, t, y, **kwargs)

            x = self.sample_from_posterior(method, x, pred_x0, y, t + dt, t)

            yield x, pred_x0

        final_prediction = method.pred_x0(model, x, ts[-1], y, **kwargs)
        yield final_prediction, final_prediction
