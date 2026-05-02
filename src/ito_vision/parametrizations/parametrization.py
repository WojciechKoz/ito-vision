from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import torch

if TYPE_CHECKING:
    from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod


class Parametrization:
    def __init__(self, target: Literal["score", "epsilon", "x0", "v", "diff"] = "x0"):
        self.target = target

    def set_method_reference(self, method: "IterativeRefinementMethod") -> None:
        self.method = method

    def __call__(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError("Parametrization must implement __call__ method.")

    def loss_weight(self, t: torch.Tensor) -> Union[torch.Tensor, float]:
        raise NotImplementedError("Parametrization must implement loss_weight method.")

    def get_x0(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        model_output = self(model, xt, t, y, **kwargs)
        y_used = y is not None and y.shape == xt.shape

        if self.target == "score":
            y_term = self.method.transition_lambda_y(t) * y if y_used else 0
            x0 = (
                xt + self.method.transition_std(t) ** 2 * model_output - y_term
            ) / self.method.transition_lambda_x(t)

        elif self.target == "epsilon":
            y_term = self.method.transition_lambda_y(t) * y if y_used else 0
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

    def get_ground_truth(
        self, 
        x0: torch.Tensor, 
        xt: torch.Tensor, 
        epsilon: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        if self.target == "score":
            return -epsilon / (self.method.transition_std(t) + 1e-4)
        if self.target == "epsilon":
            return epsilon
        if self.target == "x0":
            return x0
        if self.target == "v":
            return (
                (self.method.transition_lambda_x(t) * xt - x0) / self.method.transition_std(t)
            )
        if self.target == "diff":
            return (xt - x0) / t

        raise ValueError(f"Unknown prediction target: {self.target}")
