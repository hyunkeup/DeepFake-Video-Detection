import json
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import opts
import transforms
from datasets import dfdc
from model import FusionMultiModalCNN
from train import train_epoch
from utils import Logger, adjust_learning_rate, save_checkpoint
from validation import val_epoch


def main():
    # Command Arguments
    opt = opts.parse_opts()
    n_folds = 1
    test_accuracies = []

    # Device
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model result path
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    opt.arch = '{}'.format(opt.model)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])

    for fold in range(n_folds):
        print(opt)
        with open(os.path.join(opt.result_path, 'opts' + str(time.time()) + str(fold) + '.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)

        # Create model
        torch.manual_seed(opt.manual_seed)
        model, parameters = FusionMultiModalCNN.generate_model(device=opt.device, num_classes=opt.n_classes,
                                                               fusion=opt.fusion, num_heads=opt.num_heads,
                                                               marlin_model=opt.marlin_model)

        # Loss function
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(opt.device)

        # Training settings
        if opt.train:
            # Datasets
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])

            train_data = dfdc.DFDC(annotation_path=opt.annotation_path, subset="training",
                                   spatial_transform=video_transform,
                                   audio_transform=video_transform)
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True
            )

            # Logger
            train_logger = Logger(os.path.join(opt.result_path, 'train' + str(fold) + '.log'),
                                  ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            train_batch_logger = Logger(os.path.join(opt.result_path, 'train_batch' + str(fold) + '.log'),
                                        ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

            # Optimizer
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)

        # Validation settings
        if opt.val:
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])

            val_data = dfdc.DFDC(annotation_path=opt.annotation_path, subset="validation",
                                 spatial_transform=video_transform,
                                 audio_transform=video_transform)
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True
            )

            # Logger
            val_logger = Logger(os.path.join(opt.result_path, 'val' + str(fold) + '.log'),
                                ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            test_logger = Logger(os.path.join(opt.result_path, 'val' + str(fold) + '.log'),
                                 ['epoch', 'loss', 'prec1', 'prec5', 'lr'])

        ######################################################################
        # Load previous model if there is.
        best_prec1 = 0
        best_loss = 1e10
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.result_path))
            checkpoint = torch.load(opt.result_path)
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
            if opt.train:
                adjust_learning_rate(optimizer=optimizer, epoch=epoch, learning_rate=opt.learning_rate,
                                     lr_steps=opt.lr_steps)
                train_epoch(epoch=epoch, data_loader=train_loader, model=model, criterion=criterion,
                            optimizer=optimizer, opt=opt, epoch_logger=train_logger, batch_logger=train_batch_logger)

                state = {
                    'epoch': epoch,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                }
                save_checkpoint(state, False, opt, fold)

            if opt.val:
                validation_loss, prec1 = val_epoch(epoch=epoch, data_loader=val_loader, model=model,
                                                   criterion=criterion,
                                                   opt=opt, logger=val_logger)
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = {
                    'epoch': epoch,
                    'arch': opt.model,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                }
                save_checkpoint(state, is_best, opt, fold)

        # Test
        if opt.test:
            test_logger = Logger(
                os.path.join(opt.result_path, 'test' + str(fold) + '.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])

            test_data = dfdc.DFDC(annotation_path=opt.annotation_path, subset="testing",
                                  spatial_transform=video_transform,
                                  audio_transform=video_transform)

            # load best model
            best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name) + str(fold) + '.pth')
            model.load_state_dict(best_state['state_dict'])

            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)

            test_loss, test_prec1 = val_epoch(10000, test_loader, model, criterion, opt,
                                              test_logger)

            with open(os.path.join(opt.result_path, 'test_set_bestval' + str(fold) + '.txt'), 'a') as f:
                f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1)

    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
        f.write(
            'Prec1: ' + str(np.mean(np.array(test_accuracies))) + '+' + str(np.std(np.array(test_accuracies))) + '\n')


if __name__ == "__main__":
    # Required
    # --annotations.txt

    main()
