import numpy as np
import torch
import skimage.io as io

from utils import rgb_to_class_channels, rgb_to_class_map


class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, optim, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            # INFO: Don't pass labels to the device yet, because they have to be transformed as below.

            # self.optim.zero_grad()  # INFO: mine; TODO: check if parameter gradients
            #  should be zeroed before forward prop or only before backprop

            # Forward propagation
            # TODO: check if forward prop needs to be done with torch.set_grad_enabled()
            # torch.set_grad_enabled(True)
            outputs = self.model(inputs)

            # INFO: mine; converts mask from 3-channel to class-channel
            # labels = rgb_to_class_channels(labels)  # shape: Tensor 1, 12, 360, 480; range: -20 to 20

            # INFO: mine
            # Converts labels from 3-channel to 0-channel.
            # But moves the tensors to the CPU
            labels = rgb_to_class_map(labels)
            # So they need to be put back to GPU (if there is one)
            labels = labels.to(self.device)

            # Loss computation
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
