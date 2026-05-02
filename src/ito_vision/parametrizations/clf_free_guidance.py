from typing import Any, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ito_vision.parametrizations.identity import IdentityParametrization


class ClassifierFreeGuidanceParametrization(IdentityParametrization):
    def __init__(
        self,
        guidance_scale: float = 7.5,
        target: Literal["score", "epsilon", "x0", "v", "diff"] = "x0",
        loss_weight_type: Literal["t_inv", "var_inv", "uniform"] = "uniform",
        epsilon: float = 1e-4,
    ):
        super().__init__(target, loss_weight_type, epsilon)
        self.guidance_scale = guidance_scale

    def get_x0(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        model_output = self(
            model, 
            torch.cat([xt, xt], dim=0),
            t, 
            y, 
            **kwargs
        )

        out_cond, out_uncond = model_output.chunk(2)
        model_output = out_uncond + self.guidance_scale * (out_cond - out_uncond)

        if self.target == "score":
            y_term = self.method.transition_lambda_y(t) * y if y is not None else 0
            x0 = (
                xt + self.method.transition_std(t) ** 2 * model_output - y_term
            ) / self.method.transition_lambda_x(t)

        elif self.target == "epsilon":
            y_term = self.method.transition_lambda_y(t) * y if y is not None else 0

            x0 = (
                xt - self.method.transition_std(t) * model_output - y_term
            ) / self.method.transition_lambda_x(t)

        elif self.target == "v" and y is None:
            x0 = self.method.transition_lambda_x(t) * xt - self.method.transition_std(t) * model_output

        elif self.target == "diff":
            x0 = xt - model_output * t

        elif self.target == "x0":
            x0 = model_output

        else:
            raise ValueError(f"Unknown prediction target: {self.target}")

        return x0
