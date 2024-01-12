from keras.callbacks import Callback
import keras.backend as K

class LinearLearningRateScheduler(Callback):
    def __init__(self, start_lr, end_lr, num_epochs):
        super(LinearLearningRateScheduler, self).__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_epochs = num_epochs
        self.lr_increment = (end_lr - start_lr) / (num_epochs - 1)

    def on_epoch_begin(self, epoch, logs=None):
        new_lr = self.start_lr + epoch * self.lr_increment
        K.set_value(self.model.optimizer.lr, new_lr)
        print(f"Epoch {epoch+1}: Adjusting learning rate to {new_lr:.6f}.")

import math

class SinusoidalLearningRateScheduler(Callback):
    def __init__(self, base_lr, max_lr, num_epochs):
        super(SinusoidalLearningRateScheduler, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.num_epochs = num_epochs

    def on_epoch_begin(self, epoch, logs=None):
        cycle = math.floor(1 + epoch / (2 * 10))
        x = abs(epoch / 10 - 2 * cycle + 1)
        new_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        K.set_value(self.model.optimizer.lr, new_lr)
        print(f"Epoch {epoch+1}: Adjusting learning rate to {new_lr:.6f}.")
