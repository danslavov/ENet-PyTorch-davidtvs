from collections import OrderedDict

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from args import get_arguments
from torchvision import transforms
from PIL import Image
from data.camvid import get_color_encoding


color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        # ('road_marking', (255, 69, 0)),  #  remove the road_marking class from the CamVid dataset as it's merged with the road class
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ])

# Color encoding for Cityscapes.
# This is the same as the ordered dict color_encoding in cityscapes.py, so TODO: reuse it.
palette = [[128, 64, 128],
           [244, 35, 232],
           [70, 70, 70],
           [102, 102, 156],
           [190, 153, 153],
           [153, 153, 153],
           [250, 170, 30],
           [220, 220, 0],
           [107, 142, 35],
           [152, 251, 152],
           [70, 130, 180],
           [220, 20, 60],
           [255, 0, 0],
           [0, 0, 142],
           [0, 0, 70],
           [0, 60, 100],
           [0, 80, 100],
           [0, 0, 230],
           [119, 11, 32],
           [0, 0, 0]]


# TODO: check what this function does
def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.

    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``

    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``

    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)

    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def save_checkpoint(model, optimizer, epoch, miou, args):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - args (``ArgumentParser``): An instance of ArgumentParser which contains
    the arguments used to train ``model``. The arguments are written to a text
    file in ``args.save_dir`` named "``args.name``_args.txt".

    """
    name = args.name
    save_dir = args.save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)

    # Save arguments
    summary_filename = os.path.join(save_dir, name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(miou))


def load_checkpoint(model, optimizer, folder_dir, filename):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.

    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.

    """
    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)  # TODO: orig -- to load on GPU
    # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # modified to load on CPU
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    return model, optimizer, epoch, miou


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 11] = 240
    img[img == 10] = 220
    img[img == 9] = 200
    img[img == 8] = 180
    img[img == 7] = 160
    img[img == 6] = 140
    img[img == 5] = 120
    img[img == 4] = 100
    img[img == 3] = 80
    img[img == 2] = 60
    img[img == 1] = 40
    img[img == 0] = 20
    img[img == 255] = 0
    return img


def save_masks(img_path, class_map_numpy):  # TODO: mine

    # Get the arguments
    args = get_arguments()

    # extract image name and extension from image path
    img_name_and_extension = img_path.split('\\')[-1]

    # create class maps (i.e. masks): more distinguishable greyscale and color (3 channels R, G, B)
    class_map_numpy_grey = relabel(class_map_numpy.astype(np.uint8))  # from almost black to more distinguishable
    class_map_numpy_color = np.zeros((args.height, args.width, 3),
                                     dtype=np.uint8)  # from greyscale (1 channel) to color (3 channels)
    for idx in range(args.num_classes):
        [r, g, b] = palette[idx]
        class_map_numpy_color[class_map_numpy == idx] = [b, g, r]

    # save masks
    dir_grey = args.save_dir_results_CamVid_grey
    dir_color = args.save_dir_results_CamVid_color
    cv2.imwrite(os.path.join(dir_grey, img_name_and_extension), class_map_numpy_grey)
    cv2.imwrite(os.path.join(dir_color, img_name_and_extension), class_map_numpy_color)


def rgb_to_class_channels(batch, high_value=20, low_value=-20):
    """ Converts RGB ground truth mask to multichannel tensor in order to compare it with the model output. """

    colors = get_color_encoding()
    batch_size = list(batch.size())
    class_count = len(colors)
    batch_size[1] = class_count
    multichannel_tensor = torch.full(batch_size, low_value, dtype=torch.float32)   # the same size as the model output
    mask_number = 0

    for mask in batch:
        for class_number in range(len(colors)):
            color = list(colors.items())[class_number][1]
            red_value = color[0]
            green_value = color[1]
            blue_value = color[2]

            cond_red = mask[0] == red_value
            cond_green = mask[1] == green_value
            cond_blue = mask[2] == blue_value

            cond_image = torch.zeros(batch_size[1:], dtype=torch.bool)
            cond_image[0] = cond_red
            cond_image[1] = cond_green
            cond_image[2] = cond_blue
            cond_image_1_channel = torch.logical_and(torch.logical_and(cond_image[0], cond_image[1]), cond_image[2])

            multichannel_tensor[mask_number][class_number][cond_image_1_channel] = high_value  # for a single image

        mask_number += 1

    return multichannel_tensor


def class_channels_to_rgb(input_batch, output_batch):

    """ Converts multichannel tensor to RGB image -- i.e. model output to final mask. """
    colors = get_color_encoding()
    rgb_batch_size = list(output_batch.size())
    rgb_batch_size[1] = 3  # 3 channels for RGB result
    rgb_batch = torch.zeros(rgb_batch_size)
    counter = 0  # TODO: get input image name and use it for masks names

    for input_image, multichannel_tensor in zip(input_batch, output_batch):

        # for each pixel:
        # get the channel number where this pixel has largest value (argmax by dim=0)
        # and write this number as pixel value in the resulting 1-channel ndarray
        class_map = torch.argmax(multichannel_tensor, 0).byte().cpu().data.numpy()
        cv2.imwrite('data/CamVid/results/grey/{}.png'.format(counter), class_map)

        # convert the class_map to RGB image
        class_map_color = np.zeros((input_image.shape[1], input_image.shape[2], input_image.shape[0]), dtype=np.uint8)
        for class_number in range(len(colors)):
            color = list(colors.items())[class_number][1]
            [r, g, b] = color
            class_map_color[class_map == class_number] = [b, g, r]
        cv2.imwrite('data/CamVid/results/color/{}.png'.format(counter), class_map_color)
        counter += 1

        # TODO:
        # overlayed = cv2.addWeighted(input_image, 0.5, class_map_color, 0.5, 0)
        # cv2.imwrite('data/CamVid/results/overlayed/{}.png'.format(counter), overlayed)

        # put the RGB image into the resulting tensor; TODO: reorder dimensions
        rgb_batch[counter] = torch.from_numpy(class_map_color)

    return rgb_batch








