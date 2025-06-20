# Leveraging Influence Functions for Resampling Data in Physics-Informed Neural Networks

This repository is the official implementation of [Leveraging Influence Functions for Resampling Data in Physics-Informed Neural Networks](TODO:_add_arxiv_link). 

Authors: Jonas R. Naujoks<sup>1</sup>, Aleksander Krasowski<sup>1</sup>, Moritz Weckbecker, Galip Ümit Yolcu, Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek, René P. Klausen

<sup>1</sup> Equal contribution.

### Abstract 

Physics-informed neural networks (PINNs) offer a powerful approach to solving partial differential equations (PDEs), which are ubiquitous in the quantitative sciences. Applied to both forward and inverse problems across various scientific domains, PINNs have recently emerged as a valuable tool in the field of scientific machine learning. A key aspect of their training is that the data - spatio-temporal points sampled from the PDE’s input domain - are readily available. Influence functions, a tool from the field of explainable AI (XAI), approximate the effect of individual training points on the model, enhancing interpretability. In the present work, we explore the application of influence function-based sampling approaches for the training data. Our results indicate that such targeted resampling based on data attribution methods has the potential to enhance prediction accuracy in physics-informed neural networks, demonstrating a practical application of an XAI method in PINN training.

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

The influence of adding a point $\boldsymbol x^+$ onto the loss of the test set $\mathcal{L}_{\text{test}} = \sum_{x \in \mathcal X_{\text{test}}} L(\boldsymbol x, \theta)$ given by:

$$
\operatorname{Inf}_{\mathcal{L}_\text{test}(\hat{\theta})}(\boldsymbol{x}^+)\coloneqq \nabla_{\theta}\mathcal{L}_\text{test}(\hat{\theta})^\top\mathcal{H}_{\hat{\theta}}^{-1}\ \nabla_{\theta} L(\boldsymbol{x}^+;\hat{\theta}),
$$

where $L(\cdot; \hat\theta)$ denotes the PINN loss given model parameters $\hat\theta$ and $\mathcal H_{\hat\theta} = \frac{1}{N_{\text{train}}} \nabla^2_{\theta} \sum_{\boldsymbol x \in \mathcal{X}_{\text{train}}}L(\boldsymbol x, \hat\theta)$ the Hessian of the training loss w.r.t. model parameters.

We score importance of a candidate point $\boldsymbol x^+$ is given by the absolute influence:

$$
S_{\text{Inf}}(\boldsymbol x^+) = \left| \operatorname{Inf}_{\mathcal{L}_{\text{test}}(\hat\theta)}(\boldsymbol x^+) \right|
$$

In addition to using **PINNfluence** to score importance we also include:

**Residual Adaptive Resampling (RAR)** ([Lu Lu et al., 2019](https://ml4physicalsciences.github.io/2019/files/NeurIPS_ML4PS_2019_2.pdf); [Wu et al., 2022](https://arxiv.org/abs/2207.10289))
$$
\mathcal{S}_{\text{RAR}}(\boldsymbol x^+) = \left| \mathcal{N}[\phi(\boldsymbol{x}^+; \hat\theta)] \right|
$$

**Grad-Dot** ([Charpiat et al., 2019](https://arxiv.org/abs/2102.05262))

$$
\mathcal{S}_{\text{grad-dot}}(\boldsymbol{x}^+) = \left| \nabla_{\theta}\mathcal{L}_\text{test}(\hat{\theta})^\top \nabla_{\theta}\mathcal{L}(\boldsymbol{x}^+;\hat{\theta}) \right|
$$

**Prediction Gradient**

$$
\mathcal{S}_{\text{output-grad}}(\boldsymbol{x}) = \left\| \nabla_{\boldsymbol x} \phi(\boldsymbol{x}; \hat{\theta}) \right\|_2
$$

**Loss Gradient**

$$
\mathcal{S}_{\text{loss-grad}}(\boldsymbol{x}) = \left\| \nabla_{\theta} \mathcal{L}(\boldsymbol{x}; \hat\theta) \right\|_2
$$

For evaluation we include 5 different partial derivative equations (PDEs):
- Diffusion Equation (`"diffusion"`)
- Allen-Cahn Equation (`"allen_cahn"`)
- Burgers' Equation (`"burgers"`)
- Wave Equation (`"wave"`)
- Drift-Diffusion Equation (`"drift_diffusion"`)

## Finetuning with resampling

You may start an 

```bash
python -m pinnfluence_resampling.run_experiment \
                  # PDE
                  --problem allen_cahn \
                  # network arch
                  --layers 2 64 64 64 1 \
                  # size of X_train
                  --num_domain 1_000 \
                  # pretraining steps (run pinnfluence_resampling.pretrain before)
                  --n_iterations 0 \
                  # target path for model saving
                  --save_path model_zoo/allen_cahn \
                  # scoring strategy PINNfluence, RAR, ...
                  --scoring_method PINNfluence \
                  # sampling strategy distribution or top-k
                  --sampling_strategy distribution \
                  # distribution paremeters
                  --distribution_k 2 \
                  --distribution_c 0 \
                  # incrementally adding points
                  --pertubation_strategy add \
                  # number of points to add per iter
                  --n_samples 10 \
                  # number of adam iterations in each cycle
                  --n_iterations_finetune 1_000 \
                  # number of LBFGS iterations in each cycle
                  --n_iterations_lbfgs_finetune 1_000 \
                  # use 64 bit precision
                  --float64 \
                  # RNG seed
                  --seed 1 
```

For more information on all arguments, please take a look at [pinnfluence_resampling/run_experiment.py](./pinnfluence_resampling/run_experiment.py) or run

```bash
python -m pinnfluence_resampling.run_experiment --help
```

Take a look at [eval](./eval) for some notebooks showcasing options to evaluate respective methods.

## Pretraining

We omitted pretraining in the publication but feel free to utilize it before applying resampling. 

> ⚠️ Note: When calling `python -m pinnfluence_resampling.run_experiment` on a pretrained model, make sure to match the arguments exactly. The script should be able to recover your pretrained model.

```bash
python -m pinnfluence_resampling.pretrain \
                  --problem allen_cahn \
                  --n_iterations 15_000 \
                  --layers 2 64 64 64 1 \
                  --num_domain 1_000 \
                  --seed 42 \
                  --save_path model_zoo/allen_cahn
```

The pretraining will store both the trained model (`.pt`) and training history (`.csv`) under the specified `save_path`.

For more information please take a look at [pinnfluence_resampling/pretrain.py](./pinnfluence_resampling/pretrain.py).

If you wish to implement more PDEs we recommend doing so in the [pinnfluence_resampling/utils/problems.py](./pinnfluence_resampling/utils/problems.py).


## Citing 

Please use the following citation when referencing resampling using PINNfluence in literature:

```bibtex
@misc{
      placeholder
}
```

## License

This project is licensed under the BSD-3-Clause License. Please see the [LICENSE](./LICENSE) file for more information.
