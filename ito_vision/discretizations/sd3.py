import torch

from ito_vision.discretizations.discretization import Discretization


class StableDiffusion3Discretization(Discretization):
    def __init__(self, t_min: float = 0.003, t_max: float = 1.0, shift: float = 3.0):
        self.t_max = t_max
        self.t_min = t_min
        self.shift = shift

    def __call__(self, N: int, device: torch.device) -> torch.Tensor:
        xs = torch.linspace(self.t_max, self.t_min, N, device=device)
        return self.shift * xs / (1 + (self.shift - 1) * xs)
