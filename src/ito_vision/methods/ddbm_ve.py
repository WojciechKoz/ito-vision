from typing import Optional, Tuple

from ito_vision.methods.bbdm import BBDM
from ito_vision.parametrizations import DDBMParametrization, Parametrization
from ito_vision.schedulers.scheduler import Scheduler


# BBDM and DDBM VE share the same implementation except for the scheduler as a parameter
class DDBM_VE(BBDM):
    def __init__(
        self,
        scheduler: Scheduler,
        clip: Optional[Tuple[float, float]] = (-1, 1),
        parametrization: Optional[Parametrization] = None,
    ):
        super().__init__(
            scheduler=scheduler, clip=clip, parametrization=parametrization
        )

        if isinstance(self.parametrization, DDBMParametrization):
            self.parametrization.set_karras_utils_functions(
                variant="VE", scheduler=scheduler
            )
