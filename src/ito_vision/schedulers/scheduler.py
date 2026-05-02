from typing import Optional, Tuple, Union

import torch


class Scheduler:
    def __init__(self) -> None:
        pass

    def __call__(
        self, t: Union[torch.Tensor, float], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        raise NotImplementedError("Scheduler must implement __call__ method.")

    def riemann_integral(
        self,
        t: Union[torch.Tensor, float],
        lower_bound: Union[torch.Tensor, float] = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Scheduler must implement riemann_integral method.")

    def geometric_integral(
        self,
        t: Union[torch.Tensor, float],
        exponent: float = 1,
        lower_bound: Union[torch.Tensor, float] = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return torch.exp(
            exponent * self.riemann_integral(t, lower_bound, device=device)
        )

    def tensorize_inputs(
        self,
        t: Union[torch.Tensor, float],
        lower_bound: Union[torch.Tensor, float],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(t, torch.Tensor)
            and not isinstance(lower_bound, torch.Tensor)
            and device is None
        ):
            raise ValueError(
                "Either 't' or 'lower_bound' must be a torch.Tensor or 'device' must be specified."
            )

        if isinstance(t, (int, float)):
            if isinstance(lower_bound, torch.Tensor):
                t = torch.full_like(lower_bound, t)
            else:
                t = torch.tensor([t], dtype=torch.float32, device=device)

        if isinstance(lower_bound, (int, float)):
            if isinstance(t, torch.Tensor):
                lower_bound = torch.full_like(t, lower_bound)
            else:
                lower_bound = torch.tensor(
                    [lower_bound], dtype=torch.float32, device=device
                )

        return t, lower_bound

    def tensorize_input(
        self, t: Union[torch.Tensor, float], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        if isinstance(t, (int, float)):
            if device is None:
                raise ValueError("If 't' is a scalar, 'device' must be specified.")

            t = torch.tensor([t], dtype=torch.float32, device=device)
        return t
