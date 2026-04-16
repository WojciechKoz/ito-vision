from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Optional, Tuple

import torch

if TYPE_CHECKING:
    from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod
from ito_vision.samplers.sampler import Sampler


class RKDDIMSampler(Sampler):
    def __init__(
        self,
        N: int = 30,
        variant: Literal["heun", "ralston", "midpoint"] = "heun",
        sigma: Optional[Callable] = None,
        quiet: bool = False,
    ):
        super().__init__(N, quiet)
        self.sigma = sigma if sigma is not None else lambda t: torch.zeros_like(t)
        if variant == "heun":
            self.alpha = 1.0
        elif variant == "ralston":
            self.alpha = 2.0 / 3.0
        elif variant == "midpoint":
            self.alpha = 0.5
        else:
            raise ValueError(
                f"Supported RK-DDIM variants are ['heun', 'ralston', 'midpoint'] but got {variant}."
            )

    def sample_xt(
        self,
        method: "IterativeRefinementMethod",
        x: torch.Tensor,
        pred_x0: torch.Tensor,
        y: Optional[torch.Tensor],
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        return self.sample_from_DDIM(method, x, pred_x0, y, t_next, t)

    def sample_from_DDIM(
        self,
        method: "IterativeRefinementMethod",
        x: torch.Tensor,
        pred_x0: torch.Tensor,
        y: Optional[torch.Tensor],
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        l_s = method.transition_lambda_x(s)
        l_t = method.transition_lambda_x(t)

        b_s = method.transition_lambda_y(s) * (
            y
            if y is not None and y.shape == x.shape
            else 0
        )
        b_t = method.transition_lambda_y(t) * (
            y
            if y is not None and y.shape == x.shape
            else 0
        )

        sigma_s = method.transition_std(s)
        sigma_t = method.transition_std(t)

        epsilon_t = (x - l_t * pred_x0 - b_t) / sigma_t

        epsilon = torch.randn_like(x)

        return (
            l_s * pred_x0 +
            b_s +
            sigma_s * (
                torch.sqrt(1 - self.sigma(t) ** 2) * epsilon_t +
                self.sigma(t) * epsilon
            )
        )

    def __iter__(
        self,
        method: "IterativeRefinementMethod",
        model: torch.nn.Module,
        x1: torch.Tensor,
        ts: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        dts = torch.diff(ts)
        x = x1.clone()

        w1 = 1.0 - 1.0 / (2.0 * self.alpha)
        w2 = 1.0 / (2.0 * self.alpha)

        for t, dt in zip(ts[:-1], dts):
            pred_x0 = method.pred_x0(model, x, t, y, **kwargs)

            t_mid = t + self.alpha * dt
            x_mid = self.sample_from_DDIM(method, x, pred_x0, y, t_mid, t)

            pred_x0_mid = method.pred_x0(model, x_mid, t_mid, y, **kwargs)

            pred_mixed = w1 * pred_x0 + w2 * pred_x0_mid

            x = self.sample_from_DDIM(method, x, pred_mixed, y, t + dt, t)

            yield x, pred_mixed

        final_prediction = method.pred_x0(model, x, ts[-1], y, **kwargs)
        yield final_prediction, final_prediction
