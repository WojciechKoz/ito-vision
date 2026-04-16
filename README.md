# Ito Vision
Ito Vision is a research-oriented library for diffusion-based generative models formulated through stochastic differential equations (SDEs).

It provides clean, minimalistic implementations that closely follow their mathematical formulations, making it particularly suitable for researchers and practitioners working on modern generative modeling, especially in Image Enhancement.

The primary goal of this project is to enable rapid experimentation, reproducibility, and extensibility across a wide range of diffusion-based methods.

## Key Features
- Mathematically faithful implementations — designed to match original formulations as closely as possible
- Modular design — easily swap processes, samplers, and parameterizations
- Research-first approach — ideal for prototyping and benchmarking new ideas
- Extensive method coverage — includes many state-of-the-art diffusion and flow-based techniques

## Installation
As a pip package:
``` pip install ito-vision ```

## Methods 
The library includes implementations of:
- [DDPM](https://arxiv.org/pdf/2011.13456)
- [Flow Matching](https://arxiv.org/pdf/2210.02747)
- [Mean Flow](https://arxiv.org/pdf/2505.13447)
- [InDI](https://arxiv.org/pdf/2303.11435)
- [ResShift](https://arxiv.org/pdf/2307.12348)
- [IR-SDE](https://arxiv.org/pdf/2301.11699)
- [BBDM](https://arxiv.org/pdf/2205.07680)
- [DDBM (VE & VP)](https://arxiv.org/pdf/2309.16948)
- [I2SB](https://arxiv.org/pdf/2302.05872)
- [GOUB](https://arxiv.org/pdf/2312.10299)
- [UniDB](https://arxiv.org/abs/2502.05749)
- [IMF](https://arxiv.org/pdf/2303.16852) (partially)

## Samplers
Each method can be paired with a variety of sampling strategies:
- Ancestral [arxiv](https://arxiv.org/pdf/2011.13456)
- DDIM [arxiv](https://arxiv.org/pdf/2010.02502)
- Euler-Maruyama [wikipedia](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
- 2nd Heun with Langevin dynamics [arxiv](https://arxiv.org/pdf/2206.00364)
- Mean ODE [arxiv](https://arxiv.org/pdf/2312.10299)
- 2nd order Runge-Kutta [wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
- 2nd order DDIM (ours, experimental)

## More
- Different discretization strategies ([EDM](https://arxiv.org/pdf/2206.00364), [DDBM](https://arxiv.org/pdf/2309.16948), [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large))
- Different backbone parametrization ($\epsilon$, $x_0$, $\nabla_x \log p_t$, $v$, flow matching-based)
- Support for:
    - Classifier guidance
    - Classifier-free guidance
- Integration with Stable Diffusion pipelines
