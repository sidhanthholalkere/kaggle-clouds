from dataset import get_dataset, prepare_dataset, get_preprocessing, CloudDataset, post_process, mask2rle, get_train_test
from dataset import dice
from augmentations import valid1
import os
import ttach as tta
import argparse
import gc

import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import CheckpointCallback, InferCallback
from tqdm import tqdm as tqdm
import cv2
import numpy as np
import pandas as pd

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='efficientnet-b0')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--loc', type=str)
    parser.add_argument('--data_folder', type=str, default='../input/')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--optimize', type=bool, default=False)
    parser.add_argument('--tta-pre', type=bool, default=False)
    parser.add_argument('--tta-post', type=bool, default=False)
    parser.add_argument('--merge', type=str, default='mean')
    parser.add_argument('--min_size', type=int, default=10000)
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--name', type=str)
    
    args = parser.parse_args()
    encoder = args.encoder
    model = args.model
    loc = args.loc
    data_folder = args.data_folder
    bs = args.batch_size
    optimize = args.optimize
    tta_pre = args.tta_pre
    tta_post = args.tta_post
    merge = args.merge
    min_size = args.min_size
    thresh = args.thresh
    name = args.name

    if model == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights='imagenet',
            classes=4,
            activation=None
        )
    if model == 'fpn':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights='imagenet',
            classes=4,
            activation=None,
        )
    if model == 'pspnet':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights='imagenet',
            classes=4,
            activation=None,
        )
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')

    test_df = get_dataset(train=False)
    test_df = prepare_dataset(test_df)
    test_ids = test_df['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values
    test_dataset = CloudDataset(df=test_df, datatype='test', img_ids=test_ids, transforms=valid1(), preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    val_df = get_dataset(train=True)
    val_df = prepare_dataset(val_df)
    _, val_ids = get_train_test(val_df)
    valid_dataset = CloudDataset(df=val_df, datatype='train', img_ids=val_ids, transforms=valid1(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False)

    model.load_state_dict(torch.load(loc)['model_state_dict'])

    class_params = {0: (thresh, min_size), 1: (thresh, min_size), 2: (thresh, min_size), 3: (thresh, min_size)}

    if optimize:
        print("OPTIMIZING")
        if tta_pre:
            opt_model = tta.SegmentationTTAWrapper(model, tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()]), merge_mode=merge)
        else: 
            opt_model = model
        tta_runner = SupervisedRunner()
        print("INFERRING ON VALID")
        tta_runner.infer(
            model=opt_model,
            loaders={'valid': valid_loader},
            callbacks=[InferCallback()],
            verbose=True,
        )

        valid_masks = []
        probabilities = np.zeros((4*len(valid_dataset), 350, 525))
        for i, (batch, output) in enumerate(tqdm(zip(valid_dataset, tta_runner.callbacks[0].predictions["logits"]))):
            _, mask = batch
            for m in mask:
                if m.shape != (350, 525):
                    m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                valid_masks.append(m)

            for j, probability in enumerate(output):
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                probabilities[(i * 4) + j, :, :] = probability

        print("RUNNING GRID SEARCH")
        for class_id in range(4):
            print(class_id)
            attempts = []
            for t in range(30, 70, 5):
                t /= 100
                for ms in [7500, 10000, 12500, 15000, 175000]:
                    masks = []
                    for i in range(class_id, len(probabilities), 4):
                        probability = probabilities[i]
                        predict, num_predict = post_process(sigmoid(probability), t, ms)
                        masks.append(predict)

                    d = []
                    for i, j in zip(masks, valid_masks[class_id::4]):
                        if (i.sum() == 0) & (j.sum() == 0):
                            d.append(1)
                        else:
                            d.append(dice(i, j))

                    attempts.append((t, ms, np.mean(d)))

            attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])


            attempts_df = attempts_df.sort_values('dice', ascending=False)
            print(attempts_df.head())
            best_threshold = attempts_df['threshold'].values[0]
            best_size = attempts_df['size'].values[0]
            
            class_params[class_id] = (best_threshold, best_size)
        
        del opt_model
        del tta_runner
        del valid_masks
        del probabilities
    gc.collect()

    if tta_post:
        model = tta.SegmentationTTAWrapper(model, tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()]), merge_mode=merge)
    else:
        model = model
    
    runner = SupervisedRunner()
    runner.infer(
        model=model,
        loaders={'test': test_loader},
        callbacks=[InferCallback()],
        verbose=True,
    )

    encoded_pixels = []
    image_id = 0

    for i, image in enumerate(tqdm(runner.callbacks[0].predictions['logits'])):
        for i, prob in enumerate(image):
            if prob.shape != (350, 525):
                prob = cv2.resize(prob, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            predict, num_predict = post_process(sigmoid(prob), class_params[image_id % 4][0], class_params[image_id % 4][1])
            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(predict)
                encoded_pixels.append(r)
            image_id += 1

    test_df['EncodedPixels'] = encoded_pixels
    test_df.to_csv(name, columns=['Image_Label', 'EncodedPixels'], index=False)


if __name__ == "__main__":
    main()