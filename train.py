import numpy as np
import torch
import skimage.io as io


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

            # Forward propagation
            outputs = self.model(inputs)
            # outputs_1 = torch.argmax(outputs, dim=1)  # TODO



            # TODO: MY CODE BEGIN

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
            #
            # print()
            # exit()

            # TODO: MY CODE END




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
