import os, os.path
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import transforms as ext_transforms
from models.enet import ENet
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils

from data import elements

# Get the arguments
args = get_arguments()

device = torch.device(args.device)


def load_dataset(dataset):
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        ext_transforms.PILToLongTensor()
    ])

    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # Load the validation set as tensors
    val_set = dataset(
        args.dataset_dir,
        mode='val',
        transform=image_transform,
        label_transform=label_transform)
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Load the test set as tensors
    test_set = dataset(
        args.dataset_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    # Get encoding between pixel values in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # INFO: start logging
    log_file = open('save/ENet_Elements/log.txt', 'a')
    log_file.write('Number of classes: {}\nTraining dataset size: {}\nValidation dataset size: {}\n'
                   .format(num_classes, len(train_set), len(val_set)))

    # Get a batch of samples to display
    if args.mode.lower() == 'test':
        images, labels = iter(test_loader).next()
    else:
        images, labels = iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    log_file.write('Image size: {}, label size: {}\nClass-color encoding:\n{}\n'
                   .format(images.size(), labels.size(), class_encoding))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = utils.batch_transform(labels, label_to_rgb)
        utils.imshow_batch(images, color_labels)

    # Get class weights from the selected weighing technique
    class_weights = 0
    print("\nWeighing technique:", args.weighing)

    log_file.write('Weighing technique: {}\n'.format(args.weighing))

    if args.weighing.lower() == 'none':
        class_weights = None
    else:
        print("Computing class weights...")
        print("(this can take a while depending on the dataset size)")
        if args.weighing.lower() == 'enet':
            class_weights = enet_weighing(train_loader, num_classes)  # TODO: need to understand whether and why to use class_weights !!!
        elif args.weighing.lower() == 'mfb':
            class_weights = median_freq_balancing(train_loader, num_classes)

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    log_file.write('Class weights:\n{}\n'.format(class_weights))

    log_file.close()

    return (train_loader, val_loader,
            test_loader), class_weights, class_encoding


def train(train_loader, val_loader, class_weights, class_encoding):

    # train_init_start = time.time()

    print("\nTraining...\n")

    num_classes = len(class_encoding)

    # Intialize ENet
    # model = ENet(num_classes).to(device)

    # INFO: mine
    # If a pretrained model on CamVid is needed (for transfer learning),
    # it should be initialized with 12 classes:
    model = ENet(12).to(device)

    # Check if the network architecture is correct
    # print(model)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequently used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ENet authors used Adam as the optimizer

    # optimizer = optim.Adam(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     weight_decay=args.weight_decay)  # INFO: orig

    # INFO: mine
    # Since I will freeze some layers by setting requires_grad=False of their parameters,
    # I also need to configure the optimizer so that it doesn't compute their gradients.
    # Otherwise, it will still calculate them, despite they are not used in the training.
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # INFO: mine
    # Load the pre-trained model state to the ENet model
    # downloaded from https://github.com/davidtvs/PyTorch-ENet/tree/master/save
    # model = utils.load_checkpoint(model, optimizer, args.load_dir_pretrained, args.name)[0]

    # INFO: mine
    # If model is pretrained on CamVid, the size of the final layer should be changed
    # according to the current number of classes
    model.transposed_conv = nn.ConvTranspose2d(
        16,
        num_classes,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False)
    model = model.to(device)

    # INFO: Continue writing into the log file
    log_file = open('save/ENet_Elements/log.txt', 'a')

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.load_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))

        log_file.write('Resuming from model: Start epoch = {0} | Best mean IoU = {1:.4f}\n'
                       .format(start_epoch, best_miou))

    else:
        start_epoch = 0
        best_miou = 0

    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, metric, device)
    val = Test(model, val_loader, criterion, metric, device)

    # train_init_end = time.time()
    # print('Train initialization time: {}'.format(train_init_end - train_init_start))

    # INFO: mine
    # To avoid calling train() followed by freeze_layers() in the beginning of each epoch,
    # call them here and then after each validation.
    model.train()
    model = utils.freeze_layers(model)

    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        epoch_start = time.time()
        epoch_loss, (iou, miou) = train.run_epoch(args.print_step)
        lr_updater.step()
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f} | Time: {3:.4f}".
              format(epoch, epoch_loss, miou, epoch_time))

        log_file.write('Epoch: {0:d} | Avg. loss: {1:.4f} | Mean IoU: {2:.4f} | Time: {3:.4f}\n'
                       .format(epoch, epoch_loss, miou, epoch_time))

        # INFO: Validate each 4 epochs; orig: each 10 epochs
        if (epoch + 1) % 4 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            val_start = time.time()
            loss, (iou, miou) = val.run_epoch(args.print_step)

            # INFO: mine
            # Because val.run_epoch() calls model.eval(), it needs to be set back to train()
            # but followed by freezing.
            model.train()
            model = utils.freeze_layers(model)

            val_end = time.time()
            val_duration = val_end - val_start

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f} | Time: {3:.4f}".
                  format(epoch, loss, miou, val_duration))

            log_file.write('VALIDATION Avg. loss: {1:.4f} | Mean IoU: {2:.4f} | Time: {3:.4f}\n'
                           .format(epoch, loss, miou, val_duration))

            # Print per class IoU on last epoch or if best iou -- INFO: mine; print per class IoU unconditionally
            # if epoch + 1 == args.epochs or miou > best_miou:
            for key, class_iou in zip(class_encoding.keys(), iou):
                print("{0}: {1:.4f}".format(key, class_iou))
                log_file.write('{0}: {1:.4f}; '.format(key, class_iou))
            log_file.write('\n')

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou,
                                      args)

                log_file.write('Best model thus far, saving.\n')

    log_file.close()

    return model


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("A batch of predictions from the test set...")
        images, _ = iter(test_loader).next()
        predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = images.to(device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)


# Run only if this module is being run directly
if __name__ == '__main__':

    # program_init_start = time.time()

    # Fail fast if the dataset directory doesn't exist
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)


    # Import the requested dataset
    # INFO: uncomment for orig
    # if args.dataset.lower() == 'camvid':
    #     from data import CamVid as dataset
    # elif args.dataset.lower() == 'cityscapes':
    #     from data import Cityscapes as dataset
    # elif args.dataset.lower() == 'elements':
    #     pass# from data import Elements as dataset
    # else:
    #     # Should never happen...but just in case it does
    #     raise RuntimeError("\"{0}\" is not a supported dataset.".format(
    #         args.dataset))
    dataset = elements.Elements
    loaders, w_class, class_encoding = load_dataset(dataset)
    train_loader, val_loader, test_loader = loaders

    # INFO: mine. Empty GPU cache
    torch.cuda.empty_cache()

    # program_init_stop = time.time()
    # print('Program init time: {}'. format(program_init_stop - program_init_start))

    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader, w_class, class_encoding)

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            # Intialize a new ENet model
            num_classes = len(class_encoding)
            model = ENet(num_classes).to(device)

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the pre-trained model state to the ENet model
        # downloaded from https://github.com/davidtvs/PyTorch-ENet/tree/master/save
        model = utils.load_checkpoint(model, optimizer, args.load_dir_pretrained,
                                      args.name)[0]

        # if args.mode.lower() == 'test':
        #     print(model)

        test(model, test_loader, w_class, class_encoding)
