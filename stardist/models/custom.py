import os

# from csbdeep.utils.tf import limit_gpu_memory

# # you may need to adjust this to your GPU needs and memory capacity

# # os.environ['CUDA_VISIBLE_DEVICES'] = ...
# # limit_gpu_memory(0.8, total_memory=24000)

# limit_gpu_memory(None, allow_growth=True)

import numpy as np
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

from stardist import gputools_available
from stardist.models import Config2D, StarDist2D

from .conic import get_data, oversample_classes, CLASS_NAMES

from .conic import HEStaining, HueBrightnessSaturation
from augmend import (
    Augmend,
    AdditiveNoise,
    Augmend,
    Elastic,
    FlipRot90,
    GaussianBlur,
    Identity,
)


class Custom:
    def __init__(self):
        args = SimpleNamespace()
        self.augmenter = None

        # data in
        args.datadir = "/content/drive/MyDrive/jrf/stardist/MoNuSAC/"
        # /content/drive/MyDrive/jrf/datasets/CoNSeP/Train
        # path to 'Patch-level Lizard Dataset' as provided by CoNIC organizers
        # "/content/drive/MyDrive/jrf/datasets/lizard-patch-level-modified"
        # /content/drive/MyDrive/jrf/datasets/pannuke/
        # /content/drive/MyDrive/jrf/stardist/MoNuSAC/
        args.oversample = True  # oversample training patches with rare classes
        args.frac_val = 0.1  # fraction of data used for validation during training
        args.seed = 0  # for reproducible train/val data sets
        # model out (parameters as used for our challenge submissions)
        args.modeldir = "./models"
        args.epochs = 20
        args.batchsize = 4
        args.n_depth = 4
        args.lr = 3e-4
        args.patch = 256
        args.n_rays = 64
        args.grid = (1, 1)
        args.head_blocks = 2
        args.augment = True
        args.cls_weights = True

        args.workers = 1
        args.gpu_datagen = (
            False and args.workers == 1 and gputools_available()
        )  # note: ignore potential scikit-tensor error

        vars(args)


        self.args = args

        self.conf = Config2D(
            n_rays=args.n_rays,
            grid=args.grid,
            n_channel_in=13,
            n_classes=len(CLASS_NAMES) - 1,
            use_gpu=args.gpu_datagen,
            backbone="unet",
            unet_n_filter_base=64,
            unet_n_depth=args.n_depth,
            head_blocks=args.head_blocks,
            net_conv_after_unet=256,
            train_batch_size=args.batchsize,
            train_patch_size=(args.patch, args.patch),
            train_epochs=args.epochs,
            train_steps_per_epoch=1024 // args.batchsize,
            train_learning_rate=args.lr,
            train_loss_weights=(1.0, 0.2, 1.0),
            train_class_weights=np.ones(len(CLASS_NAMES)).tolist(),  #
            train_background_reg=0.01,
            train_reduce_lr={"factor": 0.5, "patience": 80, "min_delta": 0},
        )

    def get_class_count(self, Y0):
        class_count = np.bincount(Y0[:, ::4, ::4, 1].ravel())
        print("class_count: ",class_count)
        # Desired size
        desired_size = 13

        # Calculate the amount of padding
        padding = desired_size - class_count.shape[0]

        # Pad the array
        class_count = np.pad(class_count, (0, padding), mode="constant")
        print("class_count: ",class_count)

        try:
            import pandas as pd

            df = pd.DataFrame(
                class_count, index=CLASS_NAMES.values(), columns=["counts"]
            )
            df = df.drop("BACKGROUND")
            df["%"] = (100 * (df["counts"] / df["counts"].sum())).round(2)
            # display(df)
        except ModuleNotFoundError:
            print("install 'pandas' to show class counts")
        return class_count

    def getdata(self, datadir):
        X, Y, D, Y0, idx = get_data(datadir, seed=self.args.seed)
        return X, Y, D, Y0, idx

    def traintestsplit(self, X, Y, D, Y0, idx):
        X, Xv, Y, Yv, D, Dv, Y0, Y0v, idx, idxv = train_test_split(
            X, Y, D, Y0, idx, test_size=self.args.frac_val, random_state=self.args.seed
        )
        return X, Xv, Y, Yv, D, Dv, Y0, Y0v, idx, idxv

    def oversampleclasses(self, X, Y, D, Y0, idx):
        # if args.oversample:
        X, Y, D, Y0, idx = oversample_classes(X, Y, D, Y0, idx, seed=self.args.seed)
        class_count = self.get_class_count(Y0)
        return X, Y, D, Y0, idx, class_count

    def calcweights(self, class_count):
        if self.args.cls_weights:
            inv_freq = np.where(
                class_count != 0,
                np.median(class_count[class_count != 0]) / class_count,
                0,
            )
            print(class_count)
            inv_freq = inv_freq**0.5
            class_weights = inv_freq.round(4)
        else:
            class_weights = np.ones(len(CLASS_NAMES))
        print("class_weights: ",class_weights)
        return class_weights

    def setconf(self, weights, X):
        return Config2D(
            n_rays=self.args.n_rays,
            grid=self.args.grid,
            n_channel_in=X.shape[-1],
            n_classes=len(CLASS_NAMES) - 1,
            use_gpu=self.args.gpu_datagen,
            backbone="unet",
            unet_n_filter_base=64,
            unet_n_depth=self.args.n_depth,
            head_blocks=self.args.head_blocks,
            net_conv_after_unet=256,
            train_batch_size=self.args.batchsize,
            train_patch_size=(self.args.patch, self.args.patch),
            train_epochs=self.args.epochs,
            train_steps_per_epoch=1024 // self.args.batchsize,
            train_learning_rate=self.args.lr,
            train_loss_weights=(1.0, 0.2, 1.0),
            train_class_weights=weights,  #
            train_background_reg=0.01,
            train_reduce_lr={"factor": 0.5, "patience": 80, "min_delta": 0},
        )

    def train_dataset(self, modelname, dir, add_config, rep_config):
        from stardist.models import base

        self.args.datadir = dir
        print('dataset dir: ',self.args.datadir) 
        base.set_config_values(add_config, rep_config)
        X, Y, D, Y0, idx = self.getdata(self.args.datadir)
        X, Xv, Y, Yv, D, Dv, Y0, Y0v, idx, idxv = self.traintestsplit(X, Y, D, Y0, idx)
        X, Y, D, Y0, idx, class_count = self.oversampleclasses(X, Y, D, Y0, idx)
        class_weights = self.calcweights(class_count)
        self.conf = self.setconf(class_weights.tolist(), X)
        model = StarDist2D(self.conf, name=modelname, basedir=self.args.modeldir)
        model.train(
            X,
            Y,
            classes=D,
            validation_data=(Xv, Yv, Dv),
            augmenter=self.augmenter,
            workers=self.args.workers,
        )
        return Xv, Yv, Y0v, idxv

    def configure(self, modelname, dir, add_config, rep_config):
        from stardist.models import base
        self.args.datadir = dir
        print('dataset dir: ',self.args.datadir) 
        base.set_config_values(add_config, rep_config)
        X, Y, D, Y0, idx = self.getdata(self.args.datadir)
        X, Xv, Y, Yv, D, Dv, Y0, Y0v, idx, idxv = self.traintestsplit(X, Y, D, Y0, idx)
        X, Y, D, Y0, idx, class_count = self.oversampleclasses(X, Y, D, Y0, idx)
        class_weights = self.calcweights(class_count)
        self.conf = self.setconf(class_weights.tolist(), X)

    
    def train(self, name, dir, config, epochs):
        if self.args.augment:
            aug = Augmend()
            aug.add(
                [HEStaining(amount_matrix=0.15, amount_stains=0.4), Identity()],
                probability=0.9,
            )

            aug.add([FlipRot90(axis=(0, 1)), FlipRot90(axis=(0, 1))])
            aug.add(
                [
                    Elastic(grid=5, amount=10, order=1, axis=(0, 1), use_gpu=False),
                    Elastic(grid=5, amount=10, order=0, axis=(0, 1), use_gpu=False),
                ],
                probability=0.8,
            )

            aug.add(
                [GaussianBlur(amount=(0, 2), axis=(0, 1), use_gpu=False), Identity()],
                probability=0.1,
            )
            aug.add([AdditiveNoise(0.01), Identity()], probability=0.8)

            aug.add(
                [
                    HueBrightnessSaturation(hue=0, brightness=0.1, saturation=(1, 1)),
                    Identity(),
                ],
                probability=0.9,
            )

            def augmenter(x, y):
                return aug([x, y])

            self.augmenter = augmenter

        else:
            self.augmenter = None

        self.args.epochs = epochs
        Xv, Yv, Y0v, idxv = self.train_dataset(name, dir, config[0], config[1])
        return Xv, Yv, Y0v, idxv

        # base.set_config_values([ [9, 10, 11, 12]]  ,[[2, 4, 8, 10, 11, 12]])

    def optimise_threshold(self, Xv, Yv, modelname):
        model = StarDist2D(self.conf, name=modelname, basedir=self.args.modeldir)
        model.optimize_thresholds(Xv, Yv, nms_threshs=[0.1, 0.2, 0.3])

    def predict_masks(self, images, model):
        import numpy as np
        import matplotlib.pyplot as plt

        # %matplotlib inline

        import tensorflow as tf
        from imageio import imread

        from .conic import predict
        from stardist.models import StarDist2D
        from stardist.plot import random_label_cmap, render_label

        pred_masks = []
        for image in images:
            u, count = predict(
                model,
                image,
                normalize=True,
                test_time_augment=True,
                tta_merge=dict(prob=np.median, dist=np.mean, prob_class=np.mean),
                refine_shapes=dict(),
            )

            pred_masks.append(u)
        return np.array(pred_masks)



    def get_results(self, Y, pred_masks, thresh):
        from .benchmark import matching_dataset
        return matching_dataset(
            Y,
            pred_masks,
            thresh=thresh,
            criterion="iou",
            by_image=False,
            show_progress=True,
            parallel=True,
        )
        

    def prediction(self, X, Y, name, thresh):
        import numpy as np
        import matplotlib.pyplot as plt

        # %matplotlib inline

        import tensorflow as tf
        from imageio import imread

        from .conic import predict
        from stardist.models import StarDist2D
        from stardist.plot import random_label_cmap, render_label

        np.random.seed(42)
        cmap_random = random_label_cmap()

        model = StarDist2D(None, name=name, basedir="./models")

        from .benchmark import matching_dataset

        import time

        # starting time
        start = time.time()

        pred_masks = self.predict_masks(
            X, StarDist2D(None, name=name, basedir="./models")
        )

        # starting time
        end = time.time()

        print(f"time taken: {end - start}")

        # use correct threshold
        result = matching_dataset(
            Y,
            pred_masks,
            thresh=thresh,
            criterion="iou",
            by_image=False,
            show_progress=True,
            parallel=True,
        )
        print(result)

    def get_pred(self, X, name):
        import numpy as np
        import matplotlib.pyplot as plt
        
        # %matplotlib inline
        
        import tensorflow as tf
        from imageio import imread
        
        from .conic import predict
        from stardist.models import StarDist2D
        from stardist.plot import random_label_cmap, render_label
        
        np.random.seed(42)
        cmap_random = random_label_cmap()
        
        model = StarDist2D(None, name=name, basedir="./models")
        
        from .benchmark import matching_dataset
        
        import time
        
        # starting time
        start = time.time()
        
        pred_masks = self.predict_masks(
            X, StarDist2D(None, name=name, basedir="./models")
        )
        
        # starting time
        end = time.time()
        
        print(f"time taken: {end - start}")
        return pred_masks
