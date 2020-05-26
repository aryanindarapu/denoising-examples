from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
#import glob
#import cv2

from tifffile import imread, imsave, imshow
from csbdeep.utils import plot_some, axes_dict, plot_history
from csbdeep.data import RawData, create_patches, anisotropic_distortions
from csbdeep.io import load_training_data, save_training_data
from csbdeep.models import Config, CARE

"""
You need to modify the following file:
anaconda3/envs/care/lib/python3.6/site-packages/csbdeep/utils/plot_utils.py

Change the function 'plot_history'
savefig("graph.png")

The current code is meant for the Jupyter notebook
Since we are using a remote server - there is no way to get the loss print_function

"""

raw_data = RawData.from_folder (
    basepath    = 'data/nucleo',
    source_dirs = ['in'],
    target_dir  = 'gt',
    axes        = 'CYX',
)


X, Y, XY_axes = create_patches (
    raw_data            = raw_data,
    patch_size          = (128,128),
    n_patches_per_image = 200,
    save_file           = 'data/my_training_data.npz'
)




m = np.load('data/my_training_data.npz')
m['axes']


(X,Y), (X_val,Y_val), axes = load_training_data('data/my_training_data.npz', validation_split=0.1, verbose=True)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]


config = Config(axes, n_channel_in, n_channel_out, probabilistic=True, train_steps_per_epoch=400)
print(config)
vars(config)


model = CARE(config, 'my_model', basedir='models')

history = model.train(X,Y, validation_data=(X_val,Y_val))


print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
