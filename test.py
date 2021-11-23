import torch
import numpy as np
import skimage.io as io
from utils import save_masks, rgb_to_class_channels, class_channels_to_rgb, rgb_to_class_map
from utils import batch_transform


class Test:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        self.model.eval()
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)  # shape: Tensor (batch_size, 3, 360, 480); range: 0 to 1
            labels = batch_data[1]#.to(self.device)  # shape: Tensor (batch_size, 3, 360, 480); range: 0 to 255
            # INFO: Don't pass labels to the device yet, because they have to be transformed as below.

            # INFO: mine
            # Converts labels from 3-channel to 0-channel with each class num as pixel value.
            # This is needed in order to pass labels along with outputs to the loss computation.
            labels = rgb_to_class_map(labels)

            # INFO: mine
            # Converts labels from 3-channel to class-channel. Turned out, it's a wrong approach.
            # labels = rgb_to_class_channels(labels)  # shape: Tensor 1, 12, 360, 480; range: -20 to 20

            labels = labels.to(self.device)

            with torch.no_grad():
                # Forward propagation
                outputs = self.model(inputs)  # shape: Tensor (batch_size, 12, 360, 480); range: -17 to 13 (varies per channel)

                # INFO: mine
                # This converts outputs from class-channel to 3-channel.
                # Also can save resulting images (plain masks and overlayed masks)
                # rgb_outputs = class_channels_to_rgb(inputs, outputs, labels)

                # Loss computation
                loss = self.criterion(outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of evaluation the metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
