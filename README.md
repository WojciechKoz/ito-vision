# ito_vision
Diffusion-based generative models formulated through stochastic differential equations. 
Here you will find many stochastic processes used in modern image enhancement and generation.
This library ensures minimalistic implementations that are 1:1 with math formulations of the methods. 
The library is mainly for scientists whos research is focused on diffusion models.
We encourage the scientists to try this library for
- fast modification of the existing sota methods
- use implemented methods as your baselines
- combine and test different variants of processes, samplers, paretrizations and more...

## Methods 
In this library you will find such methods
- [DDPM](https://arxiv.org/pdf/2011.13456)
- [Flow Matching](https://arxiv.org/pdf/2210.02747)
- [Mean Flow](https://arxiv.org/pdf/2505.13447)
- [InDI](https://arxiv.org/pdf/2303.11435)
- [ResShift](https://arxiv.org/pdf/2307.12348)
- [IR-SDE](https://arxiv.org/pdf/2301.11699)
- [BBDM](https://arxiv.org/pdf/2205.07680)
- [DDBM (VE & VP)](https://arxiv.org/pdf/2309.16948)
- [I$^2$SB](https://arxiv.org/pdf/2302.05872)
- [GOUB](https://arxiv.org/pdf/2312.10299)
- [UniDB](https://arxiv.org/abs/2502.05749)
- [IMF](https://arxiv.org/pdf/2303.16852) (partially)

## Samplers
Each method can be used with the following samplers 
- Ancestral [arxiv](https://arxiv.org/pdf/2011.13456)
- DDIM [arxiv](https://arxiv.org/pdf/2010.02502)
- Euler-Maruyama [wikipedia](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
- 2nd Heun with Langevin dynamics [arxiv](https://arxiv.org/pdf/2206.00364)
- Mean ODE [arxiv](https://arxiv.org/pdf/2312.10299)
- 2nd order Runge-Kutta [wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
- 2nd order DDIM (ours, experimental)

## More
- Different discretization strategies (EDM, DDBM, Stable Diffusion 3.5)
- Different backbone parametrization 
- Support for classifier guidance and classifier-free guidance
- Stable Diffusion support

## Installation
As a pip package:
``` pip install ito-vision ```

