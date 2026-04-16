import torch

from ito_vision.discretizations.discretization import Discretization


class CustomDiscretization(Discretization):
    def __init__(self, steps: torch.tensor):
        self.steps = steps

    def __call__(self, N: int, device: torch.device) -> torch.Tensor:
        if len(self.steps) != N:
            raise ValueError("wrong number of steps in Custom Discretization")

        return self.steps.to(device)
