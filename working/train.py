import argparse
import os
import multiprocessing

from dataset import get_dataset, prepare_dataset, get_train_test, CloudDataset, get_preprocessing
from augmentations import training1, valid1
from radam import RAdam
from torch.optim import AdamW, Adam
from schedulers import NoamLR

import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from catalyst.dl import utils
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, CriterionCallback, OptimizerCallback

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='efficientnet-b0')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--pretrained', type=str, default='imagenet')
    parser.add_argument('--logdir', type=str, default='../logs/')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--data_folder', type=str, default='../input/')
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--enc_lr', type=float, default=1e-2)
    parser.add_argument('--dec_lr', type=float, default=1e-3)
    parser.add_argument('--optim', type=str, default="radam")
    parser.add_argument('--loss', type=str, default="bcedice")
    parser.add_argument('--schedule', type=str, default="rlop")
    parser.add_argument('--early_stopping', type=bool, default=True)
    

    args = parser.parse_args()

    encoder = args.encoder
    model = args.model
    pretrained = args.pretrained
    logdir = args.logdir
    name = args.exp_name
    data_folder = args.data_folder
    height = args.height
    width = args.width
    bs = args.batch_size
    accumulate = args.accumulate
    epochs = args.epochs
    enc_lr = args.enc_lr
    dec_lr = args.dec_lr
    optim = args.optim
    loss = args.loss
    schedule = args.schedule
    early_stopping = args.early_stopping

    if model == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=pretrained,
            classes=4,
            activation=None
        )
    if model == 'fpn':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=pretrained,
            classes=4,
            activation=None,
        )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, pretrained)
    log = os.path.join(logdir, name)

    ds = get_dataset(path=data_folder)
    prepared_ds = prepare_dataset(ds)

    train_set, valid_set = get_train_test(ds)

    train_ds = CloudDataset(df=prepared_ds, datatype='train', img_ids=train_set, transforms=training1(h=height, w=width), preprocessing=get_preprocessing(preprocessing_fn), folder=data_folder)
    valid_ds = CloudDataset(df=prepared_ds, datatype='train', img_ids=valid_set, transforms=valid1(h=height, w=width), preprocessing=get_preprocessing(preprocessing_fn), folder=data_folder)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=multiprocessing.cpu_count())
    valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=multiprocessing.cpu_count())

    loaders = {
        'train': train_loader,
        'valid': valid_loader,
    }

    num_epochs = epochs

    if optim == "radam":
        optimizer = RAdam([
            {'params': model.encoder.parameters(), 'lr': enc_lr},
            {'params': model.decoder.parameters(), 'lr': dec_lr},
        ])
    if optim == "adam":
        optimizer = Adam([
            {'params': model.encoder.parameters(), 'lr': enc_lr},
            {'params': model.decoder.parameters(), 'lr': dec_lr},
        ])
    if optim =="adamw":
        optimizer = AdamW([
            {'params': model.encoder.parameters(), 'lr': enc_lr},
            {'params': model.decoder.parameters(), 'lr': dec_lr},
        ])

    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    if schedule == "rlop":
        scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=3)
    if schedule == "noam":
        scheduler = NoamLR(optimizer, 10)

    if loss == "bcedice":
        criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    if loss == "dice":
        criterion = smp.utils.losses.DiceLoss(eps=1.)
    if loss == "bcejaccard":
        criterion = smp.utils.losses.BCEJaccardLoss(eps=1.)
    if loss == "jaccard":
        criterion == smp.utils.losses.JaccardLoss(eps=1.)

    callbacks = [DiceCallback(), CriterionCallback()]

    callbacks.append(OptimizerCallback(accumulation_steps=accumulate))
    if early_stopping:
        callbacks.append(EarlyStoppingCallback(patience=5, min_delta=0.001))

    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=log,
        num_epochs=num_epochs,
        verbose=True
    )


if __name__ == "__main__":
    main()