import os

import tensorflow as tf
from tensorflow import keras
from keras import callbacks
import numpy as np
import json



class CallbackList:
    """ Container abstracting a list of callbacks."""
    def __init__(
        self,
        callbacks=None,
        add_history=True,
        add_printlog=True,
        model=None
    ):
        self.callbacks = callbacks or []
        self._add_default_callbacks(add_history, add_printlog)

        if model:
            self.set_model(model)

    def _add_default_callbacks(self, add_history, add_progbar):

        self._progbar = None
        self._history = None
        for cb in self.callbacks:
            if isinstance(cb, PrintLog):
                self._progbar = cb
            elif isinstance(cb, History):
                self._history = cb

        if self._history is None and add_history:
            self._history = History()
            self.callbacks.append(self._history)

        if self._progbar is None and add_progbar:
            self._progbar = PrintLog()
            self.callbacks.append(self._progbar)
        
    def set_model(self, model):
        self.model = model
        if self._history:
            model.history = self._history
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_begin(self, logs=None):
        logs = self.update_log(logs)

        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = self.update_log(logs)

        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        logs = self.update_log(logs)

        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = self.update_log(logs)
            
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def tf_to_list_or_float(self, x):
        if {isinstance(x, tf.Tensor) or isinstance(x, tf.Variable)}:
            x = x.numpy().astype(np.float32).tolist()
            if isinstance(x, list) and len(x) == 1:
                x = x[0]
            return x
        else:
            return x
    
    def update_log(self, logs):
        logs = logs or {}
        logs = {k: self.tf_to_list_or_float(v) for k, v in logs.items()}
        return logs

class GoalBasedStopping(callbacks.Callback):
    """Callback that stops training when the target loss is reached or the deadline is hit.

    Args:
        filepath: path under which the best model is saved.
        goal: target loss; training stops once the loss drops below this value.
        threshold: minimum loss required at check_epoch.
        check_epoch: epoch number at which the threshold is enforced.
    """
    def __init__(self, filepath=None, goal=0.0001, threshold=None, check_epoch=None):
        super().__init__()
        self.filepath = filepath
        self.goal = goal
        self.threshold = threshold
        self.check_epoch = check_epoch

    def on_train_begin(self, logs=None):
        self.best = np.inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get('loss')
        
        # Run once, on the first call
        if self.best_weights is None:
            self.best_weights = self.model.get_weights()

        # Stop training if the loss has not dropped below threshold by check_epoch
        if self.check_epoch is not None:
            if (epoch+1) == self.check_epoch and self.best > self.threshold:
                self.model.stop_training = True
                print("stopped by GoalBasedStopping: threshold not met at deadline")
                return

        # Update best if the current loss improved
        if current < self.best:
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        
        # Stop training once the goal is reached.
        # Checked after the best update so that the best weights are saved and training stops immediately.
        if self.best < self.goal:
            self.model.set_weights(self.best_weights)
            if self.filepath:
                self.model.save_weights(self.filepath.format(epoch=self.best_epoch), overwrite=True)
            self.model.stop_training = True
            print(f"stopped by GoalBasedStopping: goal {self.goal} achieved at epoch {self.best_epoch}")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        if self.filepath:
            self.model.save_weights(self.filepath.format(epoch=self.best_epoch), overwrite=True)

#obsolete
class EarlyStopping(callbacks.Callback):
    def __init__( self, filepath=None, baseline=0.0001, threshold=None, check_epoch=None):
        super().__init__()
        self.filepath = filepath
        self.baseline = baseline
        self.threshold = threshold
        self.check_epoch = check_epoch

    def on_train_begin(self, logs=None):
        # self.stopped_epoch = 0
        self.best = np.inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get('loss')
        
        # Run once, on the first call
        if self.best_weights is None:
            self.best_weights = self.model.get_weights()

        # Stop training if the loss has not dropped below threshold by check_epoch
        if self.check_epoch is not None:
            if (epoch+1) == self.check_epoch and self.best > self.threshold:
                self.model.stop_training = True
                print("stopped by EarlyStopping")

        if current < self.best:
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            return
        
        if self.best < self.baseline:
            self.model.set_weights(self.best_weights)
            self.model.save_weights(self.filepath.format(epoch=self.best_epoch), overwrite=True)
            self.model.stop_training = True
    
    def on_train_end(self, epoch, logs=None):
        self.model.set_weights(self.best_weights)
        self.model.save_weights(self.filepath.format(epoch=self.best_epoch), overwrite=True)

class History(callbacks.Callback):

    def __init__(self, filepath=None):
        super().__init__()
        self.history = {}
        self.filepath = filepath
    
    def on_train_begin(self, logs=None):
        # Save the pre-training log
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.model.history = self

        if self.filepath:
            # Create the directory if it does not exist
            history_dir, filename = os.path.split(self.filepath)
            os.makedirs(history_dir, exist_ok=True)

    def on_train_end(self, logs=None):       
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            # Create an empty list if the key is new; otherwise append the value
            self.history.setdefault(k, []).append(v)
        self.model.history = self # pyright: ignore[reportOptionalMemberAccess]



class ModelCheckpoint(callbacks.Callback):

    def __init__(
        self,
        filepath,
        save_every=500,
        start_from_epoch=0
    ):
        super().__init__()
        self.save_dir = os.path.split(filepath)[0]
        self.filepath = filepath
        self.save_every = save_every
        self.start_from_epoch = start_from_epoch

        if filepath:
            os.makedirs(self.save_dir, exist_ok=True)

    def on_train_begin(self, logs=None):
        self.model.save_weights(self.save_dir + '/initial_weight.weights.h5', overwrite=True) # type:ignore 

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) < self.start_from_epoch:
            return
        
        if (epoch+1) % self.save_every == 0:
            self.model.save_weights(self.filepath.format(epoch=(epoch+1)//100), overwrite=True) #type:ignore

    def on_train_end(self, logs=None):
        self.model.save_weights(self.save_dir + '/last_weight.weights.h5', overwrite=True)   #type:ignore


class PrintLog(callbacks.Callback):

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        # Print loss and spectral radius
        str = f'\rIteration: {epoch+1}'
        if 'loss' in logs.keys():
            str += f', loss:{logs["loss"]:.4}'
        if 'sp_r' in logs.keys():
            str += f', sp radius:{logs["sp_r"]:.6}'
        print(str, end='')

            