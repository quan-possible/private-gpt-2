from fastai.vision import *
from fastai.vision.models.wrn import wrn_22

path = untar_data(URLs.CIFAR)
ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=128).normalize(cifar_stats)
learn = Learner(data, wrn_22(), metrics=accuracy)
learn.fit_one_cycle(10, 3e-3, wd=0.4, div_factor=10, pct_start=0.5)