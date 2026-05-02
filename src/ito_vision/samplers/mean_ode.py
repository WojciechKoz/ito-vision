from typing import TYPE_CHECKING, Any, Iterator, Optional, Tuple

import torch

if TYPE_CHECKING:
    from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod
from ito_vision.samplers.sampler import Sampler


class MeanODESampler(Sampler):
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
        **kwargs: Any,
    ) -> torch.Tensor:
        return method.f(xt, t, y) - method.g(t) ** 2 * score

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
            score, pred_x0 = method.score(model, x, t, y, **kwargs)

            f_value = self.f_bar(method, x, t, score, y)
            dx = f_value * dt
            x += dx

            yield x, pred_x0

        final_prediction = method.pred_x0(model, x, ts[-1], y, **kwargs)
        yield final_prediction, final_prediction
