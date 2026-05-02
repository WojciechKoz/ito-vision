from typing import Any, Literal, Optional, Union

import torch

from ito_vision.parametrizations.parametrization import Parametrization


class IdentityParametrization(Parametrization):
    def __init__(
        self,
        target: Literal["score", "epsilon", "x0", "v", "diff"] = "x0",
        loss_weight_type: Literal["t_inv", "var_inv", "uniform"] = "uniform",
        epsilon: float = 1e-4,
    ):
        super().__init__(target)
        self.loss_weight_type = loss_weight_type
        self.eps = epsilon

    def __call__(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return model(xt, t, y, **kwargs)  # type: ignore

    def loss_weight(self, t: torch.Tensor) -> Union[torch.Tensor, float]:
        if self.loss_weight_type == "t_inv":
            return 1.0 / (t + self.eps)  # type: ignore
        if self.loss_weight_type == "var_inv":
            return 1.0 / (self.method.transition_std(t) ** 2 + self.eps)  # type: ignore
        if self.loss_weight_type == "uniform":
            return 1.0
        raise ValueError(f"Unknown loss weight type: {self.loss_weight_type}")
