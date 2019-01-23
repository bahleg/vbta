import numpy as np
import matplotlib.pylab as plt
import glob
import json
import os
import scipy.misc
from scipy.spatial.distance import cdist


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    # thanks to https://github.com/carpedm20/DCGAN-tensorflow/
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def get_data_celeba():
    features = {}
    with open('./list_attr_celeba.txt') as inp:
        inp.readline()  # total count
        labels = inp.readline().split()  # header
        for line in inp:
            tokens = line.split()
            basename = tokens[0]
            features[basename] = np.array([(int(t) + 1) / 2 for t in tokens[1:]]).astype(np.bool)
            if len(features) % 5000 == 0:
                print len(features)
    X = []
    Y = []

    for f in sorted(glob.glob('./jpg/img_align_celeba/*jpg')):
        try:

            img = scipy.misc.imread(f, flatten=False).astype(np.float)
            img = center_crop(img, img.shape[1] * 5 / 6, None)
            img = img / 127.5 - 1
            X.append(img.astype(np.float32))
            try:
                feat = features[os.path.basename(f)]
                Y.append((f, feat))
            except:
                print 'bad file features', f
                break

            if len(X) % 5000 == 0:
                print len(X)

        except:
            print 'bad file', f

    FEATURES = [y[1] for y in Y]
    FEATURES = np.array(FEATURES)
    test_size = 1000
    male_idx = np.where(FEATURES[:, 20] == 1)[0]
    female_idx = np.where(FEATURES[:, 20] == 0)[0]
    male_idx_test = male_idx[:test_size]
    female_idx_test = female_idx[:test_size]
    male_idx = male_idx[test_size:]
    female_idx = female_idx[test_size:]

    np.save('test_male', np.array(X)[male_idx_test])
    np.save('test_female', np.array(X)[female_idx_test])
    malex, maley = X[male_idx], np.array(FEATURES)[male_idx]
    femalex, femaley = X[female_idx], np.array(FEATURES)[female_idx]
    new_male_idx = range(malex.shape[0])
    new_female_idx = []

    for id in new_male_idx:
        if len(new_female_idx) % 5000 == 0:
            print len(new_female_idx)
        start = 0
        end = len(femalex)
        female_batch = femaley[start:end]

        dists = cdist([maley[id]], female_batch, 'hamming')[0]
        female_id = np.argmin(dists)
        new_female_idx.append(female_id)
    np.save('malex', X[male_idx])
    np.save('femalex', np.array(femalex)[new_female_idx])
