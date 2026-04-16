from typing import Any, Callable, Literal, Optional, Union

import torch

from ito_vision.parametrizations.parametrization import Parametrization
from ito_vision.schedulers.scheduler import Scheduler


def ve_alpha(_: Scheduler) -> Callable[[torch.Tensor], torch.Tensor]:
    def inner(_: torch.Tensor) -> torch.Tensor:
        return torch.tensor(1.0)

    return inner


def ve_sigma(scheduler: Scheduler) -> Callable[[torch.Tensor], torch.Tensor]:
    def inner(t: torch.Tensor) -> torch.Tensor:
        return scheduler.riemann_integral(t) ** 0.5

    return inner


def vp_alpha(scheduler: Scheduler) -> Callable[[torch.Tensor], torch.Tensor]:
    def inner(t: torch.Tensor) -> torch.Tensor:
        return scheduler.geometric_integral(t, exponent=-1)

    return inner


def vp_sigma(scheduler: Scheduler) -> Callable[[torch.Tensor], torch.Tensor]:
    def inner(t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1 - scheduler.geometric_integral(t, exponent=-2))

    return inner


class DDBMParametrization(Parametrization):
    def __init__(
        self,
        var_0: float,
        var_1: float,
        cov_01: float,
    ):
        super().__init__(target="x0")

        self.var_0 = var_0
        self.var_1 = var_1
        self.cov_01 = cov_01

    def set_karras_utils_functions(
        self, variant: Literal["VE", "VP"], scheduler: Scheduler
    ) -> None:
        alpha = None
        sigma = None

        if variant == "VE":
            alpha = ve_alpha(scheduler)
            sigma = ve_sigma(scheduler)
        elif variant == "VP":
            alpha = vp_alpha(scheduler)
            sigma = vp_sigma(scheduler)
        else:
            raise NotImplementedError(
                f"DDBM reperametrization supports: 'VE' and 'VP' variants. Got {variant}."
            )

        snr: Callable[[torch.Tensor], torch.Tensor] = (  # noqa: E731
            lambda t: alpha(t) ** 2 / sigma(t) ** 2
        )

        self.a: Callable[[torch.Tensor], torch.Tensor] = (
            lambda t: alpha(t)
            / alpha(torch.ones_like(t))
            * snr(torch.ones_like(t))
            / snr(t)
        )
        self.b: Callable[[torch.Tensor], torch.Tensor] = lambda t: alpha(t) * (
            1 - snr(torch.ones_like(t)) / snr(t)
        )
        self.c: Callable[[torch.Tensor], torch.Tensor] = lambda t: sigma(t) ** 2 * (
            1 - snr(torch.ones_like(t)) / snr(t)
        )

    def c_in(self, t: torch.Tensor) -> torch.Tensor:
        return (
            self.a(t) ** 2 * self.var_1
            + self.b(t) ** 2 * self.var_0
            + 2 * self.a(t) * self.b(t) * self.cov_01
            + self.c(t)
        ) ** (-0.5)

    def c_out(self, t: torch.Tensor) -> torch.Tensor:
        return self.c_in(t) * torch.sqrt(
            self.a(t) ** 2 * (self.var_0 * self.var_1 - self.cov_01**2)
            + self.var_0 * self.c(t)
        )

    def c_skip(self, t: torch.Tensor) -> torch.Tensor:
        return self.c_in(t) ** 2 * (self.b(t) * self.var_0 + self.a(t) * self.cov_01)

    def loss_weight(self, t: torch.Tensor) -> Union[torch.Tensor, float]:
        return 1.0 / (self.c_out(t) ** 2 + 1e-4)  # type: ignore

    def c_noise(self, t: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.log(t + 1e-4)

    def __call__(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.c_skip(t) * xt + self.c_out(t) * model(  # type: ignore
            self.c_in(t) * xt, self.c_noise(t), y, **kwargs
        )
