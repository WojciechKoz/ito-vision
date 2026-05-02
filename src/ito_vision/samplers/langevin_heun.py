from typing import TYPE_CHECKING, Any, Iterator, Optional, Tuple

import torch

if TYPE_CHECKING:
    from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod
from ito_vision.samplers.sampler import Sampler


# Deterministic Runge-Kutta 2nd order method for sampling with langevin exploration
class LangevinHeunSampler(Sampler):
    def __init__(self, s: float = 0.33, N: int = 30, quiet: bool = False):
        super().__init__(N, quiet=quiet)
        self.s = s

    def sample_xt(
        self,
        method: "IterativeRefinementMethod",
        x: torch.Tensor,
        pred_x0: torch.Tensor,
        y: Optional[torch.Tensor],
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        dt = t_next - t
        score = (method.transition_mu(pred_x0, t, y) - x) / method.transition_std(t) ** 2
        f_value = self.f_bar_deterministic(method, x, t, score, y)
        return x + f_value * dt

    def f_bar_deterministic(
        self,
        method: "IterativeRefinementMethod",
        xt: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return method.f(xt, t, y) - 0.5 * method.g(t) ** 2 * score

    def f_bar_stochastic(
        self,
        method: "IterativeRefinementMethod",
        xt: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return method.f(xt, t, y) - method.g(t) ** 2 * score
        # return -method.g(t)**2 * score

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

        for t, dt in zip(ts[:-1], dts):
            eps = torch.randn_like(x)

            # Stochastic exploration
            t_hat = t + self.s * dt

            score, _ = method.score(model, x, t, y, **kwargs)
            d = self.f_bar_stochastic(method, x, t, score, y)

            x_hat = (
                x + d * (t_hat - t) + method.g(t) * torch.sqrt((t_hat - t).abs()) * eps
            )

            # Deterministic first Heun step
            t_next = t + dt

            score, _ = method.score(model, x_hat, t_hat, y, **kwargs)
            d_hat = self.f_bar_deterministic(method, x_hat, t_hat, score, y)

            x_next = x_hat + d_hat * (t_next - t_hat)

            # Deterministic second Heun step
            score, pred_x0 = method.score(model, x_next, t_next, y, **kwargs)
            d_prime = self.f_bar_deterministic(method, x_next, t_next, score, y)

            x = x_hat + 0.5 * (d_hat + d_prime) * (t_next - t_hat)

            yield x, pred_x0

        final_prediction = method.pred_x0(model, x, ts[-1], y, **kwargs)
        yield final_prediction, final_prediction
