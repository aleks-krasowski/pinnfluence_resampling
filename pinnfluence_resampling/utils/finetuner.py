import deepxde as dde
import json
from pathlib import Path
import torch

from .callbacks import BestModelCheckpoint, EvalMetricCallback
from .defaults import DEFAULTS
from .sampling import Sampler, Scorer
from .. import problem_factory


class Finetuner:
    """Finetuner class for improved training of physics-informed neural networks.

    This class implements various fine-tuning strategies for physics-informed neural networks
    by adaptively sampling new training points based on scoring metrics and retraining the model.
    """

    def __init__(
        self,
        model: dde.Model,
        scorer: Scorer,
        sampler: Sampler,
        strategy: str = "replace",
        n_iterations_finetune: int = 25_000,
        n_iterations_lbfgs_finetune: int = 0,
        model_name: str = None,
        save_path: str = DEFAULTS["model_zoo"],
        lr: float = DEFAULTS["lr"],
        recover_run: bool = False,
    ):
        assert strategy in [
            "replace",
            "add",
            "incremental",
            "incremental_replace",
        ]
        self.strategy = strategy

        self.model = model
        self.scorer = scorer
        self.sampler = sampler
        self.n_iterations_finetune = n_iterations_finetune
        self.n_iterations_lbfgs_finetune = n_iterations_lbfgs_finetune
        self.model_name = model_name
        self.lr = lr
        self.n_cycles = 1 if not ("incremental" in self.strategy) else 100

        self.callbacks = []

        experiment_name = f"{model_name}_finetuned_{self.strategy}_iter_{n_iterations_finetune}_samples_{sampler.num_samples}_scoring_{scorer.strategy}_{scorer.sign}_distribution_{sampler.strategy == 'distribution'}"

        config = {
            "problem": model_name.split("_")[0],
            "model_name": model_name,
            "training_strategy": self.strategy,
            "n_iterations_finetune": n_iterations_finetune,
            "n_iterations_lbfgs_finetune": n_iterations_lbfgs_finetune,
            "lr": lr,
            "n_candidate_points": scorer.n_candidate_points,
            "n_samples": sampler.num_samples,
            "distribution_k": sampler.distribution_k,
            "distribution_c": sampler.distribution_c,
            "scoring_strategy": scorer.strategy,
            "scoring_sign": scorer.sign,
            "sampling_strategy": sampler.strategy,
        }

        if recover_run:
            print("Recovering run")
            model_zoo_src = Path(DEFAULTS["model_zoo_src"])
            for potential_run in model_zoo_src.rglob(f"{model_name}_full.pt"):
                print(potential_run)
                potential_dir = potential_run.parent
                if (potential_dir / "config.json").exists():
                    with open(potential_dir / "config.json", "r") as f:
                        potential_config = json.load(f)
                        chkpt = torch.load(potential_run, weights_only=False)
                        if (potential_config == config) and (chkpt["epoch"] == 50_000):
                            self.model = problem_factory.compile_model(
                                self.model.net,
                                self.model.data,
                                model=self.model,
                                optimizer="adam",
                            )
                            self.model.net.load_state_dict(chkpt["model_state_dict"])
                            self.model.opt.load_state_dict(
                                chkpt["optimizer_state_dict"]
                            )
                            self.model.data.train_x_all = chkpt["train_x_all"]
                            self.model.data.train_x = chkpt["train_x"]
                            self.model.data.train_x_bc = chkpt["train_x_bc"]
                            self.model.data.test_x = chkpt["test_x"]
                            self.model.data.test_y = chkpt["test_y"]
                            self.model.data.holdout_test_x = chkpt["holdout_test_x"]
                            self.model.data.holdout_test_y = chkpt["holdout_test_y"]
                            self.model.train_state.epoch = chkpt["epoch"]
                            print(f"Recovered run from {potential_dir}")
                            break

        print(config)

        with open(f"{save_path}/config.json", "w") as f:
            json.dump(config, f)

        if self.model_name is not None:
            self.callbacks = [
                # best model on train
                BestModelCheckpoint(
                    filepath=f"{save_path}/{experiment_name}.pt",
                    monitor="test loss",
                    save_better_only=True,
                    verbose=True,
                ),
                BestModelCheckpoint(
                    filepath=f"{save_path}/{experiment_name}_train.pt",
                    monitor="train loss",
                    save_better_only=True,
                    verbose=True,
                ),
                BestModelCheckpoint(
                    filepath=f"{save_path}/{model_name}_full.pt",
                    verbose=False,
                    save_better_only=False,
                ),
                EvalMetricCallback(
                    filepath=f"{save_path}/{experiment_name}_eval.csv",
                    verbose=1,
                ),
            ]

    def apply_sampled_data(
        self,
    ):
        candidate_points, scores = self.scorer()
        sample = self.sampler(candidate_points, scores)

        if self.strategy == "replace" or self.strategy == "incremental_replace":
            self.model.data.replace_with_anchors(sample)
        elif self.strategy == "add" or self.strategy == "incremental":
            self.model.data.add_anchors(sample)
        print(f"New dataset size: {len(self.model.data.train_x_all)}")

    def compile(self, optimizer="adam"):
        self.model = problem_factory.compile_model(
            self.model.net,
            self.model.data,
            lr=self.lr if optimizer != "L-BFGS" else None,
            model=self.model,
            optimizer=optimizer,
        )

    def train_cycle(
        self,
    ):
        self.compile(optimizer="adam")
        print(f"Training with {len(self.model.data.train_x_all)} samples")
        self.model.train(
            self.n_iterations_finetune,
            display_every=100,
            verbose=0,
            callbacks=self.callbacks,
        )

        if self.n_iterations_lbfgs_finetune > 0:
            dde.optimizers.config.set_LBFGS_options(
                maxiter=self.n_iterations_lbfgs_finetune
            )
            self.compile(optimizer="L-BFGS")

            self.model.train(
                self.n_iterations_finetune,
                display_every=100,
                verbose=0,
                callbacks=self.callbacks,
            )

    def __call__(self):
        for _ in range(self.n_cycles):
            self.apply_sampled_data()
            self.train_cycle()
            self.scorer.model = self.model
