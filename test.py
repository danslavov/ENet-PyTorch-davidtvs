import torch
import numpy as np
import skimage.io as io
from utils import save_masks, rgb_to_class_channels
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
            inputs = batch_data[0].to(self.device)  # shape: Tensor 1, 3, 360, 480; range: 0 to 1
            labels = batch_data[1].to(self.device)  # shape: Tensor 1, 3, 360, 480; range: 0 to 255

            #TODO: mine; improve for batch processing
            labels = rgb_to_class_channels(labels[0], 1, 0, 12)  # shape: Tensor 1, 12, 360, 480; range: -20 to 20

            with torch.no_grad():
                # Forward propagation
                outputs = self.model(inputs)  # shape: Tensor 1, 12, 360, 480; range: -17 to 13 (varies per channel)






            # # TODO: MY CODE BEGIN
            #
            # tmp_dir = 'C:/Users/Admin/Desktop/tmp'
            # file_ext = 'png'
            #
            # inp = inputs[0].cpu().numpy()
            # inp = np.transpose(inp, (1, 2, 0))
            # io.imsave('{}/inp.{}'.format(tmp_dir, file_ext), inp)
            #
            # lab = labels[0].cpu().numpy()
            # lab = np.transpose(lab, (1, 2, 0))
            # io.imsave('{}/lab.{}'.format(tmp_dir, file_ext), lab)
            #
            # with torch.no_grad():
            #     heatmaps = outputs[0].cpu().numpy()
            #
            #     for i in range(12):
            #         print(i)
            #         single_map = heatmaps[i]
            #         min_value = np.min(single_map)
            #         max_value = np.max(single_map)
            #         print('min {}'.format(min_value))
            #         print('max {}'.format(max_value))
            #         print('---------------')
            #         io.imsave('{}/outp_{}.{}'.format(tmp_dir, i, file_ext), single_map)
            # exit()
            #
            # # extract the path to the image in order to use its name and extension
            # img_path = self.data_loader.dataset.test_data[step]
            # # create class map (i.e. mask) with greyscale pixels close to black
            # class_map_numpy = outputs[0].max(0)[1].byte().cpu().data.numpy()
            # save_masks(img_path, class_map_numpy)
            # exit()
            #
            # # TODO: MY CODE END









            # Loss computation
            loss = self.criterion(outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            # Keep track of evaluation the metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()
