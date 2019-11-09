import argparse
import os
import multiprocessing

from dataset import get_dataset, prepare_dataset, get_train_test, CloudDataset, get_preprocessing
from augmentations import training1, valid1
from radam import RAdam

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
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    

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
    epochs = args.num_epochs
    lr = args.lr

    if model == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=pretrained,
            classes=4,
            activation=None
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

    optimizer = RAdam(
        model.parameters(), 
        lr=lr,
    )

    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    callbacks = [DiceCallback(), EarlyStoppingCallback(patience=10, min_delta=0.001), CriterionCallback(), OptimizerCallback(accumulation_steps=accumulate)]

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