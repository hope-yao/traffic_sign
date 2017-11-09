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
    trained_model = "/home/exx/Documents/Hope/BEGAN-tensorflow-regressor-20170811-GED-eclipse-ptx-traffic/models/GAN/GAN_2017_11_01_15_04_47/experiment_185390.ckpt"
    testing_dataset_path = ''
    testing_dataset = '/home/exx/Documents/Hope/BEGAN-tensorflow-regressor-20170811-GED-eclipse-ptx-traffic/traffic_sign_dataset2.npz'


    aa = np.load('./attack_data/TrafficSign_FGSM_and_CPPN_Datasets/eps150_FGSM_and_feat_squeeze_data.npz')
    valid_x = np.asarray( aa['FGSM_features'],'float32')
    valid_y = aa['orig_target']
    para_list = [[0.25], [0.5,0.65,0.8], [0.1,0.2,0.3], [0.001,0.01,0.02], [3.,5.,7.], [-3.,-4.,-5.,-6.,-7.]]
    para_list[1] = [para_list[1][args[1]]]
    para_list[2] = [para_list[2][args[2]]]
    para_list[3] = [para_list[3][args[0]]]
    log_err = trainer.valid(np.expand_dims(valid_x,1), valid_y, trained_model, para_list, args[0]+1)


# for loss_t in para_list[0]:
#     for y_t1 in para_list[1]:
#         for y_t_inc in para_list[2]:
#             y_t2 = y_t1 + y_t_inc
#             for c_wd in para_list[3]:
#                 for theta in para_list[4]:
#                     for bias in para_list[5]:
# self.opt_attack(x_input_opt, y_true, y0, itr=150, bias=bias,
#                     theta=theta, c_wd=0.02,
#                     loss_t=0.3, y_t1=0.7, y_t2=0.8)
if __name__ == '__main__':
    # main(0,1,0)
    main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
