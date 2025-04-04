# Leveraging Influence Functions for Resampling Data in Physics-Informed Neural Networks

This repository is the official implementation of [Leveraging Influence Functions for Resampling Data in Physics-Informed Neural Networks](TODO:_add_arxiv_link). 

Authors: Jonas R. Naujoks<sup>1</sup>, Aleksander Krasowski<sup>1</sup>, Moritz Weckbecker, Galip Ümit Yolcu, Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek, René P. Klausen

<sup>1</sup> Equal contribution.

### Abstract 

Physics-informed neural networks (PINNs) offer a powerful approach to solving partial differential equations (PDEs), which are ubiquitous in the quantitative sciences. Applied to both forward and inverse problems across various scientific domains, PINNs have recently emerged as a valuable tool in the field of scientific machine learning. A key aspect of their training is that the data — spatio-temporal points sampled from the PDE’s input domain — are readily available. Influence functions (IFs), a tool from the field of explainable AI (XAI), approximate the effect of individual training points on the model, enhancing interpretability. In the present work, we explore the application of IF-based sampling approaches for the training data. Our results indicate that such targeted resampling based on data attribution methods has the potential to enhance prediction accuracy in physics-informed neural networks.

## Installation

Using pip (`Python>=3.10`):

```bash
pip install git+https://github.com/aleks-krasowski/pinnfluence_resampling.git
```

Note that we use a slightly modified versions of [Captum](https://github.com/aleks-krasowski/captum) and [DeepXDE](https://github.com/aleks-krasowski/deepxde) for calculating Influence Functions.

Furthermore please set the `DeepXDE` backend to pytorch:

```bash 
python -m deepxde.backend.set_default_backend pytorch
```

## Overview

We introduce a novel resampling method based on [**PINNfluence**](https://arxiv.org/abs/2409.08958) (Influence Functions for PINNs). Influence Functions (IFs) quantify the effect of removing or **adding** individual training points on the model's predictions. PINNfluence-based resampling thus prioritizes most impactful points. 

The influence of adding a point $\boldsymbol x^+$ onto the loss of the test set $\mathcal{L}_{\text{test}}$ given by:

![Eq1](https://latex.codecogs.com/svg.image?\operatorname{Inf}_{\mathcal{L}_{\text{test}}(\hat{\theta})}(\boldsymbol{x}^+):=\nabla_{\theta}\mathcal{L}_{\text{test}}(\hat{\theta})^\top\mathcal{H}_{\hat{\theta}}^{-1}\nabla_{\theta}\mathcal{L}(\boldsymbol{x}^+;\hat{\theta}))

where $\mathcal{L}(\cdot; \hat\theta)$ denotes the PINN loss given model parameters $\hat\theta$ and $\mathcal H_{\hat\theta}$ the Hessian w.r.t. model parameters.

We score importance of a candidate point $\boldsymbol x^+$ is given by the absolute influence:


![Eq2](https://latex.codecogs.com/svg.image?S_{\text{Inf}}(\boldsymbol&space;x^&plus;)=\left|\operatorname{Inf}_{\mathcal{L}_{\text{test}}(\hat\theta)}(\boldsymbol&space;x^&plus;)\right|)


In addition to using **PINNfluence** to score importance we also include:

**Residual Adaptive Resampling (RAR)** ([Lu Lu et al., 2019](https://ml4physicalsciences.github.io/2019/files/NeurIPS_ML4PS_2019_2.pdf); [Wu et al., 2022](https://arxiv.org/abs/2207.10289))

![Eq3](https://latex.codecogs.com/svg.image?\mathcal{S}_{\text{RAR}}(\boldsymbol&space;x^&plus;)=\|\mathcal{N}[\phi(\boldsymbol{x}^&plus;;\hat\theta)]\|_2)

**Grad-Dot** ([Charpiat et al., 2019](https://arxiv.org/abs/2102.05262))

![Eq4](https://latex.codecogs.com/svg.image?\mathcal{S}_{\text{grad-dot}}(\boldsymbol{x}^&plus;)=\nabla_{\theta}\mathcal{L}_\text{test}(\hat{\theta})^\top\nabla_{\theta}\mathcal{L}(\boldsymbol{x}^&plus;;\hat{\theta}))

**Prediction Gradient**

![Eq5](https://latex.codecogs.com/svg.image?\mathcal{S}_{\text{output-grad}}(\boldsymbol{x})=\left\|\nabla_{\boldsymbol&space;x}\phi(\boldsymbol{x};\hat{\theta})\right\|_2)

**Loss Gradient**

![Eq6](https://latex.codecogs.com/svg.image?\mathcal{S}_{\text{loss-grad}}(\boldsymbol{x})=\left\|\nabla_{\theta}\mathcal{L}(\boldsymbol{x};\hat\theta)\right\|_2)

For evaluation we include 4 different partial derivative equations (PDEs):
- Allen-Cahn Equation (`"allen_cahn"`)
- Burgers' Equation (`"burgers"`)
- Diffusion Equation (`"diffusion"`)
- Wave Equation (`"wave"`)

## Pretraining

Before applying resampling methods, you need to pretrain a PINN model. Use the following code to pretrain models for different PDEs:

```bash
python -m pinnfluence_resampling.pretrain \
                  --problem allen_cahn \
                  --n_iterations 15_000 \
                  --layers 2 32 32 32 1 \
                  --num_domain 1_000 \
                  --seed 42 \
                  --save_path model_zoo/allen_cahn
```

The pretraining will store both the trained model (`.pt`) and training history (`.csv`) under the specified `save_path`.

For more information please take a look at [pinnfluence_resampling/pretrain.py](./pinnfluence_resampling/pretrain.py).

If you wish to implement more PDEs we recommend doing so in the [pinnfluence_resampling/problem_factory.py](./pinnfluence_resampling/problem_factory.py).

## Finetuning with resampling

You can evaluate a resampling method as follows:

```bash
python -m pinnfluence_resampling.run_experiment \
                  --problem allen_cahn \
                  --n_iterations 15_000 \
                  --layers 2 32 32 32 1 \
                  --num_domain 1_000 \
                  --seed 42 \
                  --save_path model_zoo/allen_cahn \
                  --scoring_strategy PINNfluence \
                  --sampling_strategy distribution \
                  --distribution_k 2 \
                  --distribution_c 0 \
                  --n_samples 10 \
                  --training_strategy incremental \
                  --n_iterations_finetune 1_000
```

> ⚠️ Note: You need to match the pretraining arguments exactly.

For more information please take a look at [pinnfluence_resampling/run_experiment.py](./pinnfluence_resampling/run_experiment.py).

> ⚠️ Note: For evaluation of randomly initialized models please run the pretrain script with `n_iterations 0`.

Take a look at [eval](./eval) for some notebooks showcasing options to evaluate respective methods.

## Citing 

Please use the following citation when referencing resampling using PINNfluence in literature:

```bibtex
@misc{
      placeholder
}
```

## License

This project is licensed under the BSD-3-Clause License. Please see the [LICENSE](./LICENSE) file for more information.
