from typing import TYPE_CHECKING, Any, Iterator, Optional, Tuple

import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod


class Sampler:
    def __init__(self, N: int = 30, quiet: bool = False):
        self.N = N
        self.quiet = quiet

    def sample_xt(
        self,
        method: "IterativeRefinementMethod",
        x: torch.Tensor,
        pred_x0: torch.Tensor,
        y: Optional[torch.Tensor],
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Sampler must implement sample_xt method.")

    def __iter__(
        self,
        method: "IterativeRefinementMethod",
        model: torch.nn.Module,
        x1: torch.Tensor,
        ts: torch.Tensor,
        y: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError("Sampler must implement __iter__ method.")

    def __call__(
        self,
        method: "IterativeRefinementMethod",
        model: torch.nn.Module,
        x1: torch.Tensor,
        ts: torch.Tensor,
        y: Optional[torch.Tensor],
        return_trajectory: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        iterator = self.__iter__(method, model, x1, ts, y, **kwargs)
        xs, predictions = [], []
        for xt, pred_x0 in tqdm(
            iterator, disable=self.quiet, desc="Sampling", total=len(ts)
        ):
            last_item = xt

            if return_trajectory:
                xs.append(xt.cpu().clone())
                predictions.append(pred_x0.cpu().clone())

        if return_trajectory:
            return last_item, torch.stack(xs, dim=0), torch.stack(predictions, dim=0)
        return last_item, None, None
