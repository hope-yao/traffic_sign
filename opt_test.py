import numpy as np
import tensorflow as tf
import os
from trainer import Trainer
from config import get_config
from utils import prepare_dirs_and_logger, save_config

import sys

def main(*args):
    config, unparsed = get_config()

    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)


    trainer = Trainer(config)
    trained_model = "/home/exx/Documents/Hope/BEGAN-tensorflow-regressor-20170811-GED-eclipse-ptx-traffic-z3/models/GAN/GAN_2017_11_15_16_52_17/experiment_41293.ckpt"
    aa = np.load('/home/exx/Documents/Hope/BEGAN-tensorflow-regressor-20170811-GED-eclipse-ptx-traffic/attack_data_new/eps150_[1101]->[1110]_FGSM_and_feat_squeeze_data.npz')
    valid_x = np.asarray(aa['FGSM_features'], 'float32')
    valid_y = aa['orig_target'][:,1:]
    # aa = np.load(
    #     '/home/exx/Documents/Hope/BEGAN-tensorflow-regressor-20170811-GED-eclipse-ptx-traffic/traffic_sign_dataset2.npz')
    # valid_x = aa['data']
    # valid_y = (aa['label'][1:]).tolist()*len(aa['data'])
    testing_paralist = [[[0.25], [0.6], [0.7], [0.001], [7.0], [-5.0]]]

    for para_list in testing_paralist:
        para_list[2][0] = para_list[2][0] - para_list[1][0]
        log_err = trainer.test(np.expand_dims(valid_x, 1), valid_y, trained_model, para_list, 'eps-150', 1)
    print('done')


if __name__ == '__main__':
    main()
