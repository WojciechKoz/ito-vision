import torch


class Discretization:
    def __init__(self) -> None:
        pass

    def __call__(self, N: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError("Discretization must implement __call__ method.")
