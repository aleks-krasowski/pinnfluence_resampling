import json
from pathlib import Path

import deepxde as dde
import torch

from .. import problem_factory
from .callbacks import BestModelCheckpoint, EvalMetricCallback
from .defaults import DEFAULTS
from .sampling import Sampler, Scorer


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
        strategy: str = "add",
        n_iterations_finetune: int = 25_000,
        n_iterations_lbfgs_finetune: int = 0,
        model_name: str = None,
        save_path: str = DEFAULTS["model_zoo"],
        lr: float = DEFAULTS["lr"],
        recover_run: bool = False,
        n_cycles_finetune: int = 1,
    ):

        if not Path(save_path).exists():
            Path(save_path).mkdir(parents=True, exist_ok=True)

        assert strategy in [
            "add",
            "replace",
        ]
        self.strategy = strategy

        self.model = model
        self.scorer = scorer
        self.sampler = sampler
        self.n_iterations_finetune = n_iterations_finetune
        self.n_iterations_lbfgs_finetune = n_iterations_lbfgs_finetune
        self.model_name = model_name
        self.lr = lr
        self.n_cycles = n_cycles_finetune

        self.resume_mid_cycle = False
        self.resume_adam_iter = 0
        self.resume_lbfgs_iter = 0

        self.callbacks = []

        experiment_name = f"{model_name}_finetuned_{self.strategy}_iter_{n_iterations_finetune}_samples_{sampler.num_samples}_scoring_{scorer.strategy}_{scorer.sign}_distribution_{sampler.strategy == 'distribution'}"

        config = {
            "problem": model_name.split("_")[0],
            "model_name": model_name,
            "pertubation_strategy": self.strategy,
            "n_iterations_finetune": n_iterations_finetune,
            "n_iterations_lbfgs_finetune": n_iterations_lbfgs_finetune,
            "lr": lr,
            "n_candidate_points": scorer.n_candidate_points,
            "n_samples": sampler.num_samples,
            "distribution_k": sampler.distribution_k,
            "distribution_c": sampler.distribution_c,
            "scoring_method": scorer.strategy,
            "scoring_sign": scorer.sign,
            "sampling_strategy": sampler.strategy,
        }

        if recover_run:
            print("Recovering run")
            model_zoo_src = Path(DEFAULTS["model_zoo_src"])
            print(
                f"Looking for runs in {model_zoo_src} under {experiment_name}_full.pt"
            )
            for potential_run in model_zoo_src.rglob(f"*/{experiment_name}_full.pt"):
                potential_dir = potential_run.parent
                print(potential_dir)
                if (potential_dir / "config.json").exists():
                    with open(potential_dir / "config.json", "r") as f:
                        potential_config = json.load(f)
                        chkpt = torch.load(potential_run, weights_only=False)
                        if potential_config == config:

                            # determine last used optimizer
                            optimizer = "adam"
                            if (
                                "H_diag"
                                in chkpt["optimizer_state_dict"]["state"][0].keys()
                            ):
                                optimizer = "L-BFGS"
                            print(f"Found run with optimizer: {optimizer}")

                            # recompile model with given optimizer and params
                            self.model = problem_factory.compile_model(
                                self.model.net,
                                self.model.data,
                                model=self.model,
                                optimizer=optimizer,
                            )

                            # load state
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

                            recovered_epoch = chkpt["epoch"]
                            self.model.train_state.epoch = recovered_epoch
                            save_path = potential_dir

                            # determine if we are resuming mid-cycle
                            total_iter_per_cycle = (
                                self.n_iterations_finetune
                                + self.n_iterations_lbfgs_finetune
                            )
                            completed_cycles = recovered_epoch // total_iter_per_cycle
                            remaining_iter_in_cycle = (
                                recovered_epoch % total_iter_per_cycle
                            )

                            self.n_cycles -= completed_cycles

                            if remaining_iter_in_cycle > 0 and total_iter_per_cycle > 0:
                                self.resume_mid_cycle = True
                                self.n_cycles -= 1  # resumed cycle counts as one

                                if remaining_iter_in_cycle < self.n_iterations_finetune:
                                    self.resume_adam_iter = (
                                        self.n_iterations_finetune
                                        - remaining_iter_in_cycle
                                    )
                                    self.resume_lbfgs_iter = (
                                        self.n_iterations_lbfgs_finetune
                                    )
                                else:
                                    self.resume_adam_iter = 0
                                    lbfgs_done = (
                                        remaining_iter_in_cycle
                                        - self.n_iterations_finetune
                                    )
                                    self.resume_lbfgs_iter = (
                                        self.n_iterations_lbfgs_finetune - lbfgs_done
                                    )

                                print(
                                    f"Recovered run mid-cycle at epoch {recovered_epoch}."
                                )
                                print(
                                    f"Will complete the cycle with {self.resume_adam_iter} Adam and {self.resume_lbfgs_iter} L-BFGS iterations."
                                )
                            print(
                                f"Current dataset size: {len(self.model.data.train_x_all)}"
                            )
                            print(f"Total cycles left to run: {self.n_cycles}")
                            break

        with open(f"{save_path}/config.json", "w") as f:
            json.dump(config, f)

        if self.model_name is not None:
            self.callbacks = [
                # best model on train
                BestModelCheckpoint(
                    filepath=f"{save_path}/{experiment_name}_full.pt",
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

        # NOTE: this doesn't update data.num_domain, num_initial, num_boundary
        if self.strategy == "replace":
            self.model.data.replace_with_anchors(sample)
        elif self.strategy == "add":
            self.model.data.add_anchors(sample)
        print(f"New dataset size: {len(self.model.data.train_x_all)}")

    def compile(self, optimizer="adam"):
        if optimizer not in ["adam", "L-BFGS"]:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        if optimizer == "L-BFGS":
            # to make sure lbfgs reports progress every 100 iterations
            # else CSV logger would differ between optimizers
            dde.optimizers.config.LBFGS_options["iter_per_step"] = 100

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
        if self.continued_run_tmp:
            print("Continuing run with previous optimizer")
            if self.continued_optimizer == "adam":
                n_iterations_finetune_tmp = self.n_iterations_finetune
                # for continued runs reduce iterations by already done in current cycle
                self.n_iterations_finetune = self.n_iterations_finetune - int(
                    self.model.train_state.epoch
                    / (self.n_iterations_finetune + self.n_iterations_lbfgs_finetune)
                )
                self.train_adam()
                self.n_iterations_finetune = n_iterations_finetune_tmp
            elif self.continued_optimizer == "L-BFGS":
                n_iterations_lbfgs_finetune_tmp = self.n_iterations_lbfgs_finetune
                # for continued runs reduce iterations by already done in current cycle
                self.n_iterations_lbfgs_finetune = (
                    self.n_iterations_lbfgs_finetune
                    - int(
                        self.model.train_state.epoch
                        / (
                            self.n_iterations_finetune
                            + self.n_iterations_lbfgs_finetune
                        )
                    )
                )
                self.train_lbfgs()
                self.n_iterations_lbfgs_finetune = n_iterations_lbfgs_finetune_tmp
            return

        else:
            self.train_adam()
            if self.n_iterations_lbfgs_finetune > 0:
                self.train_lbfgs()

    def train_adam(self, iterations: int):
        if iterations <= 0:
            return
        print(f"Starting Adam training for {iterations} iterations...")
        self.compile(optimizer="adam")
        self.model.train(
            iterations=iterations,
            display_every=100,
            callbacks=self.callbacks,
            verbose=0,  # don't print as handled by callbacks - display_every still used to trigger them
        )

    def train_lbfgs(self, iterations: int):
        if iterations <= 0:
            return
        print(f"Starting L-BFGS training for {iterations} iterations...")
        dde.optimizers.config.set_LBFGS_options(maxiter=iterations)
        self.compile(optimizer="L-BFGS")
        self.model.train(
            display_every=100,
            callbacks=self.callbacks,
            verbose=0,  # don't print as handled by callbacks - display_every still used to trigger them
        )

    def __call__(self):
        total_cycles_to_run = self.n_cycles
        if self.resume_mid_cycle:
            total_cycles_to_run += 1

        for i in range(total_cycles_to_run):
            print(f"\n--- Starting Cycle {i+1}/{total_cycles_to_run} ---")

            # Determine iterations for this specific cycle
            adam_iters = self.n_iterations_finetune
            lbfgs_iters = self.n_iterations_lbfgs_finetune

            if i == 0 and self.resume_mid_cycle:
                print("This is a resumed cycle. Skipping data sampling.")
                adam_iters = self.resume_adam_iter
                lbfgs_iters = self.resume_lbfgs_iter
            else:
                self.apply_sampled_data()

            self.train_adam(iterations=adam_iters)
            self.train_lbfgs(iterations=lbfgs_iters)

            self.scorer.model = self.model
            print(f"--- Finished Cycle {i+1}/{total_cycles_to_run} ---")
