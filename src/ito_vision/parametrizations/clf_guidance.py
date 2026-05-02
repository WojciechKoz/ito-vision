from typing import Any, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ito_vision.parametrizations.identity import IdentityParametrization


class ClassifierGuidanceParametrization(IdentityParametrization):
    def __init__(
        self,
        classifier: nn.Module,
        guidance_scale: float = 1.0,
        target: Literal["score", "epsilon", "x0"] = "x0",
        loss_weight_type: Literal["t_inv", "var_inv", "uniform"] = "uniform",
        epsilon: float = 1e-4,
    ):
        super().__init__(target, loss_weight_type, epsilon)
        self.classifier = classifier
        self.guidance_scale = guidance_scale

        if target != "epsilon":
            raise ValueError("Classifier guidance only supports 'epsilon' target.")

    def cond_function(self, x: torch.Tensor, t: torch.Tensor, class_cond: torch.Tensor, **kwargs) -> torch.Tensor:
        self.classifier.to(x.device)
        self.classifier.eval() 

        with torch.set_grad_enabled(True):
            x_in = x.detach().clone().requires_grad_(True)
            logits = self.classifier(x_in, t)

            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[torch.arange(len(logits), device=logits.device), class_cond.view(-1)]

            grad = torch.autograd.grad(selected.sum(), x_in, retain_graph=False, create_graph=False)[0]
            return grad * self.guidance_scale

    def get_x0(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        model_output = self(model, xt, t, y, **kwargs)

        if self.target == "score":
            y_term = self.method.transition_lambda_y(t) * y if y is not None else 0
            x0 = (
                xt + self.method.transition_std(t) ** 2 * model_output - y_term
            ) / self.method.transition_lambda_x(t)

        elif self.target == "epsilon":
            y_term = self.method.transition_lambda_y(t) * y if y is not None else 0

            model_output = model_output - self.method.transition_std(t) * self.cond_function(xt, t, **kwargs)

            x0 = (
                xt - self.method.transition_std(t) * model_output - y_term
            ) / self.method.transition_lambda_x(t)

        elif self.target == "x0":
            x0 = model_output

        else:
            raise ValueError(f"Unknown prediction target: {self.target}")

        return x0
