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
from data.camvid import get_color_encoding as get_color_encoding_CamVid
from data.cityscapes import get_color_encoding as get_color_encoding_Cityscapes
from data.elements import get_color_encoding as get_color_encoding_Elements


args = get_arguments()

# Make list of color_encoding values (to form a palette)
if args.dataset == 'camvid':
    color_encoding = get_color_encoding_CamVid()
elif args.dataset == 'cityscapes':
    color_encoding = get_color_encoding_Cityscapes()
elif args.dataset == 'elements':
    color_encoding = get_color_encoding_Elements()
colors = list(color_encoding.values())


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


# INFO: If an optimizer saved state is needed, pass optimizer's structure through the signature
def load_checkpoint(model, folder_dir, filename):
    """Loads the model from a specified directory with a specified name.

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
    checkpoint = torch.load(model_path)  # INFO: orig -- to load on GPU
    # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # INFO: to load on CPU
    model.load_state_dict(checkpoint['state_dict'])

    # INFO: uncomment if need to load optimizer state.
    # Also, an optimizer structure should be passed by the function signature.
    # optimizer.load_state_dict(checkpoint['optimizer'])

    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    # return model, optimizer, epoch, miou
    return model, epoch, miou


# Only for visualization -- more distinguishable greyscale images
def relabel(img):
    if args.dataset == 'camvid':
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
    elif args.dataset == 'elements':
        img[img == 3] = 200  # resistor
        img[img == 2] = 150  # capacitor ceramic
        img[img == 1] = 50   # capacitor electrolytic
        # background remains 0

    return img


def save_masks(img_path, class_map_numpy):  # INFO: mine, for visualization

    # extract image name and extension from image path
    img_name_and_extension = img_path.split(os.path.sep)[-1]

    # create class maps (i.e. masks): more distinguishable greyscale and color (3 channels R, G, B)
    class_map_numpy_grey = relabel(class_map_numpy.astype(np.uint8))  # from almost black to more distinguishable
    class_map_numpy_color = np.zeros((args.height, args.width, 3),
                                     dtype=np.uint8)  # from greyscale (1 channel) to color (3 channels)
    for idx in range(args.num_classes):
        [r, g, b] = colors[idx]
        class_map_numpy_color[class_map_numpy == idx] = [b, g, r]

    # save masks
    dir_grey = args.save_dir_results_CamVid_grey
    dir_color = args.save_dir_results_CamVid_color
    cv2.imwrite(os.path.join(dir_grey, img_name_and_extension), class_map_numpy_grey)
    cv2.imwrite(os.path.join(dir_color, img_name_and_extension), class_map_numpy_color)


def rgb_to_class_channels(batch, high_value=20, low_value=-20):
    """ Converts RGB ground truth mask to multichannel tensor in order to compare it with the model output. """

    # colors = get_color_encoding_CamVid()
    colors = get_color_encoding_Elements()
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


def rgb_to_class_map(batch):
    """ Converts RGB ground truth mask to grey-level class map (0-channel tensor)
     where each color combination becomes the corresponding class number. """

    # colors = get_color_encoding_CamVid()
    colors = get_color_encoding_Elements()
    class_count = len(colors)
    batch_size = list(batch.size())
    del batch_size[1]  # remove the channel dimension, as it won't be needed for loss function computation
    class_map = torch.full(batch_size, class_count-1, dtype=torch.int64)
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

            cond_matrix = torch.logical_and(torch.logical_and(cond_red, cond_blue), cond_green)
            # cond_matrix = torch.unsqueeze(cond_matrix, 0)  # add the channel dimension
            class_map[mask_number][cond_matrix] = class_number  # for a single image
        mask_number += 1

    return class_map


def class_channels_to_rgb(input_batch, output_batch, label_batch):

    """ Converts multichannel tensor to RGB image -- i.e. model output to final mask. """
    # colors = get_color_encoding_CamVid()
    colors = get_color_encoding_Elements()
    rgb_batch_size = list(output_batch.size())
    rgb_batch_size[1] = 3  # 3 channels for RGB result
    rgb_batch = torch.zeros(rgb_batch_size)
    counter = 0  # TODO: get input image name and use it for masks names

    for input_tensor, output_tensor, label_tensor in zip(input_batch, output_batch, label_batch):

        # for each pixel:
        # get the channel number where this pixel has largest value (argmax by dim=0)
        # and write this number as pixel value in the resulting 1-channel ndarray
        class_map_tensor = torch.argmax(output_tensor, 0)
        class_map_ndarray = class_map_tensor.byte().cpu().data.numpy()
        # cv2.imwrite('data/CamVid/results/grey/{}.png'.format(counter), class_map)

        # convert the class_map to RGB image
        # TODO: make color class map from class_map_tensor (not class_map_ndarray)
        class_map_color = np.zeros((input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[0]), dtype=np.uint8)
        for class_number in range(len(colors)):
            color = list(colors.items())[class_number][1]
            [r, g, b] = color
            class_map_color[class_map_ndarray == class_number] = [b, g, r]
        cv2.imwrite('data/tmp/result/{}.png'.format(counter), class_map_color)

        # save ground-truth mask overlayed with output mask
        # label_image = label_tensor.byte().cpu().data.numpy()
        # label_image = np.moveaxis(label_image, 0, -1)
        # cv2.imwrite('data/CamVid/results/label/{}.png'.format(counter), label_image)
        # overlayed = cv2.addWeighted(label_image, 0.5, class_map_color, 0.5, 0)
        # cv2.imwrite('data/CamVid/results/overlayed/{}.png'.format(counter), overlayed)


        # put the RGB image into the resulting tensor
        class_map_color = np.moveaxis(class_map_color, -1, 0)
        rgb_batch[counter] = torch.from_numpy(class_map_color)

        counter += 1

    return rgb_batch


# INFO: mine
# Freeze some modules.
# Default: the whole encoder part, i.e. form 0.initial_block to 22.dilated3_7 including
def freeze_encoder(model, start_module=0, end_module=23):
    module_list = [module for module in model.children()][start_module:end_module]
    for module in module_list:
        freeze_parameters_recursively(module)


def freeze_parameters_recursively(module):
    # freeze all parameters of current module
    for parameter in module.parameters():
        parameter.requires_grad = False
    # call the same function on current module's children
    for submodule in module.children():
        freeze_parameters_recursively(submodule)


def print_trainable_state(model):
    module_list = [module for module in model.named_children()]
    for name, module in module_list:
        print(name)
        for parameter in module.parameters():
            print(parameter.requires_grad)
        print_trainable_state(module)
