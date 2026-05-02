from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional, Tuple

import torch

if TYPE_CHECKING:
    from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod
from ito_vision.samplers.sampler import Sampler


# Deterministic Runge-Kutta 2nd order method for sampling
class RungeKutta2Sampler(Sampler):
    def __init__(
        self,
        N: int = 30,
        variant: Literal["midpoint", "heun", "ralston"] = "heun",
        quiet: bool = False,
    ):
        super().__init__(N, quiet)
        if variant == "midpoint":
            self.alpha = 0.5
        elif variant == "heun":
            self.alpha = 1.0
        elif variant == "ralston":
            self.alpha = 2.0 / 3.0
        else:
            raise ValueError(
                f"Supported RK2 variants are ['midpoint', 'heun', 'ralston'] but got {variant}."
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
        dt = t_next - t
        score = (method.transition_mu(pred_x0, t, y) - x) / method.transition_std(t) ** 2
        f_value = self.f_bar(method, x, t, score, y)
        return x + f_value * dt

    def f_bar(
        self,
        method: "IterativeRefinementMethod",
        xt: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return method.f(xt, t, y) - 0.5 * method.g(t) ** 2 * score

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
            score_1, _ = method.score(model, x, t, y, **kwargs)

            slope_start = self.f_bar(method, x, t, score_1, y)

            score_2, pred_x0 = method.score(
                model,
                x + self.alpha * dt * slope_start,
                t + self.alpha * dt,
                y,
                **kwargs,
            )

            slope_mid = self.f_bar(
                method,
                x + self.alpha * dt * slope_start,
                t + self.alpha * dt,
                score_2,
                y,
            )

            x += (
                (1.0 - 1.0 / (2.0 * self.alpha)) * slope_start
                + (1.0 / (2.0 * self.alpha)) * slope_mid
            ) * dt

            yield x, pred_x0

        final_prediction = method.pred_x0(model, x, ts[-1], y, **kwargs)
        yield final_prediction, final_prediction
