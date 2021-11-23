from argparse import ArgumentParser


def get_arguments():
    """
    Defines command-line arguments, and parses them.
    """
    parser = ArgumentParser()

    # Execution mode
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full'],
        default='train',
        help=("train: performs training and validation; test: tests the model "
              "found in \"--load-dir\" with name \"--name\" on \"--dataset\"; "
              "full: combines train and test modes. Default: train"))
    parser.add_argument(
        "--resume",
        action='store_true',
        default=False,
        help=("The model found in \"--load-dir/--name/\" and filename "
              "\"--name.h5\" is loaded."))

    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="The batch size. Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=9999,
        help="Number of training epochs. Default: 300")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.005,
        help="The learning rate. Default: 5e-4")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.5")
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=100,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 100")
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4")

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=['camvid', 'cityscapes', 'elements'],
        default='elements',
        help="Dataset to use. Default: camvid")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/Elements",
        # default="data/tmp",  # to check single output as RGB image
        help="Path to the root directory of the selected dataset. "
        "Default: data/CamVid")
    parser.add_argument(
        "--input-image-extension",
        type=str,
        default=".png",
        help="File extension of the input images of the selected dataset. "
             "Default: .png")
    parser.add_argument(
        "--height",
        type=int,
        default=350,
        help="The image height. Default: 360")
    parser.add_argument(
        "--width",
        type=int,
        default=350,
        help="The image width. Default: 480")
    parser.add_argument(
        "--weighing",
        choices=['enet', 'mfb', 'none'],
        default='none',
        help="The class weighing technique to apply to the dataset. "
        "Default: enet")
    parser.add_argument(
        "--with-unlabeled",
        dest='ignore_unlabeled',
        action='store_false',
        default=False,
        help="The unlabeled class is not ignored.")

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of subprocesses to use for data loading. Default: 4")
    parser.add_argument(
        "--print-step",
        action='store_true',
        default=True,
        help="Print loss every step")
    parser.add_argument(
        "--imshow-batch",
        action='store_true',
        # default=True,
        # INFO: When imshow-batch is True, it tries to perform label_to_rgb transform, but gives exception
        help=("Displays batch images when loading the dataset and making "
              "predictions."))
    parser.add_argument(
        "--device",
        default='cuda',
        help="Device on which the network will be trained. Default: cuda")

    # Storage settings
    parser.add_argument(
        "--name",
        type=str,
        default='ENet',
        help="Name given to the model when saving. Default: ENet")
    parser.add_argument(
        "--save-dir",
        type=str,
        default='save/ENet_Elements',
        help="The directory where models are saved. Default: save/ENet_Elements")
    parser.add_argument(
        "--load-dir-pretrained",
        type=str,
        default='load/pretrained/ENet_CamVid',
        help="The directory where the pre-trained models are loaded from."
             "Downloaded from: https://github.com/davidtvs/PyTorch-ENet/tree/master/save/ENet_CamVid."
             "Default: load/pretrained/ENet_CamVid")
    parser.add_argument(
        "--load-dir",
        type=str,
        default='load/new_training',
        help="The directory where models are loaded from. Default: load/best-so-far/ENet_Elements")

    return parser.parse_args()
