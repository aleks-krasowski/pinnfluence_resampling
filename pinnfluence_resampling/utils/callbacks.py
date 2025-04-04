from deepxde.callbacks import Callback
from deepxde.metrics import l2_relative_error, mean_squared_error
import deepxde as dde
import numpy as np
import os
import time
import torch


class BestModelCheckpoint(Callback):
    """
    Callback to save the best model based on a specified metric.
    """

    def __init__(
        self,
        filepath,
        verbose=0,
        save_better_only=False,
        restore_best=True,
        monitor="test loss",
        min_delta=0.0,
    ):
        super().__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.restore_best = restore_best
        self.monitor = monitor
        self.min_delta = min_delta
        self.best = torch.inf
        self.best_epoch = 0
        self.start_time = None

    def on_train_begin(self):
        self.best = self.get_monitor_value()
        self._save_model(0, self.best)
        self.start_time = time.time()

    def on_epoch_end(self):
        """Save the model at the end of an epoch."""
        current = self.get_monitor_value()
        epoch = self.model.train_state.epoch

        if current is None:
            raise ValueError(f"Metric '{self.monitor}' is not available in logs.")

        if self.save_better_only:
            if current < (self.best - self.min_delta):
                self.best = current
                self._save_model(epoch, current)
        else:
            self._save_model(epoch, current)

        if (self.verbose) and ((epoch % 1000) == 0):
            print(
                f"Epoch {epoch}: {self.monitor} = {current:.2e} -- Best @ {self.best_epoch}: {self.best:.2e}"
            )

    def on_train_end(self):
        if self.verbose > 0:
            print(
                f"Training completed. Best {self.monitor} = {self.best:.2e} @ {self.best_epoch}"
            )
            print(f"Training time: {time.time() - self.start_time:.2f}s")

        if self.restore_best:
            checkpoint = torch.load(self.filepath, weights_only=False)
            self.model.net.load_state_dict(checkpoint["model_state_dict"])

    def _save_model(self, epoch, current):
        """Save the model to the specified filepath."""
        if self.verbose > 1:
            print(
                f"Epoch {epoch}: {self.monitor} improved to {current:.2e}, saving model to {self.filepath}"
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.net.state_dict(),
            "optimizer_state_dict": self.model.opt.state_dict(),
            "train_x_all": self.model.data.train_x_all,
            "train_x": self.model.data.train_x,
            "train_x_bc": self.model.data.train_x_bc,
            "test_x": self.model.data.test_x,
            "test_y": self.model.data.test_y,
            "holdout_test_x": self.model.data.holdout_test_x,
            "holdout_test_y": self.model.data.holdout_test_y,
            "epoch": epoch,
        }
        torch.save(checkpoint, self.filepath)

        self.best_epoch = epoch

    def get_monitor_value(self):
        if self.monitor == "train loss":
            result = sum(self.model.train_state.loss_train)
        elif self.monitor == "test loss":
            # self.model._test()
            result = sum(self.model.train_state.loss_test)
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result


class EvalMetricCallback(Callback):
    """
    Evaluate the model on the test data after every epoch.
    """

    def __init__(self, verbose=0, filepath=None, verbose_period=1):
        super().__init__()
        self.verbose = verbose
        self.last_test_loss_state = None
        self.filepath = filepath
        self.verbose_period = verbose_period
        self.file = None

    def on_train_begin(self):
        self.X = self.model.data.holdout_test_x
        self.y = self.model.data.holdout_test_y

        if self.filepath is not None:
            file_exists = os.path.exists(self.filepath)
            self.file = open(self.filepath, "a")
            if not file_exists:
                self.file.write(
                    "epoch,train_loss,valid_loss,test_loss,l2_relative_error,mse,mean_residual,mean_abs_residual\n"
                )

        y_pred = self.model.predict(self.X)
        l2re = l2_relative_error(self.y, y_pred)
        self.model.train_state.l2re = l2re
        mse = mean_squared_error(self.y, y_pred)
        self.model.train_state.mse = mse

        row = (
            0,
            np.nan,
            np.nan,
            np.nan,
            l2re,
            mse,
        )

        if self.filepath is not None:
            self.file.write(",".join(map(str, row)) + "\n")

        if self.verbose == 1:
            print(
                f"Epoch 0 \t\t Train loss = {np.nan:.2e}, Valid loss = {np.nan:.2e}, Test loss = {np.nan:.2e} L2 relative error = {l2re:.2e}, MSE = {mse:.2e}"
            )

    def on_epoch_end(self):

        if self.model.train_state.loss_test != self.last_test_loss_state:
            self.last_test_loss_state = self.model.train_state.loss_test

            epoch = self.model.train_state.epoch
            train_loss = self.model.train_state.loss_train
            if len(train_loss) == 1:
                train_loss = train_loss[0]
            valid_loss = self.model.train_state.loss_test
            if len(valid_loss) == 1:
                valid_loss = valid_loss[0]

            test_loss = np.mean(
                np.square(self.model.predict(self.X, operator=self.model.data.pde))
            )

            y_pred = self.model.predict(self.X)
            l2re = l2_relative_error(self.y, y_pred)
            self.model.train_state.l2re = l2re
            mse = mean_squared_error(self.y, y_pred)
            self.model.train_state.mse = mse

            row = (
                epoch,
                train_loss,
                valid_loss,
                test_loss,
                l2re,
                mse,
            )

            if self.filepath is not None:
                self.file.write(",".join(map(str, row)) + "\n")

            if self.verbose == 1 and (epoch % self.verbose_period) == 0:
                print(
                    f"Epoch {epoch} \t\t Train loss = {train_loss:.2e}, Valid loss = {valid_loss:.2e}, Test loss = {test_loss:.2e} L2 relative error = {l2re:.2e}, MSE = {mse:.2e}"
                )

            self.model.loss_history = dde.model.LossHistory()

    def on_train_end(self):

        if self.file is not None:
            self.file.close()
