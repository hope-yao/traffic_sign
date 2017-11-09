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

    trained_model = "/home/exx/Documents/Hope/BEGAN-tensorflow-regressor-20170811-GED-eclipse-ptx-traffic/models/GAN/GAN_2017_11_01_08_58_56/experiment_9600.ckpt"
    testing_dataset_path = ''
    testing_dataset = '/home/exx/Documents/Hope/BEGAN-tensorflow-regressor-20170811-GED-eclipse-ptx-traffic/traffic_sign_dataset2.npz'


    valid_x = np.asarray( np.load(testing_dataset)['data'][0:100],'float32')
    valid_y = np.load(testing_dataset)['label']
    para_list = [[0.2,0.25,0.3], [0.45,0.5,0.55], [0.6,0.7,0.8], [0.0015,0.01,0.1,1], [3.,4.,5.,6.,7.,8.], [-3.,-4.,-5.,-6.,-7.]]
    # para_list = [[0.2], [0.55], [0.6,0.7,0.8], [0.0015,0.01,0.1,1], [3.,4.,5.,6.,7.,8.], [-3.,-4.,-5.,-6.,-7.]]
    para_list[0] = [para_list[0][args[0]]]
    para_list[1] = [para_list[1][args[1]]]
    para_list[2] = [para_list[2][args[2]]]
    log_err = trainer.validation(valid_x.transpose(0,3,1,2), valid_y, trained_model, para_list, gpu_idx=args[0]+1)
    np.save('traffic_sign_{}_{}_{}.npy'.format(args[0], args[1], args[2]), log_err)
    # valid_x = np.asarray(np.load('/home/exx/Documents/Hope/rec_crs.npy').item()['rectcrs_img'][000:100],'float32')*2.-255.
    # valid_y = [1.,1.]
    # log_err_bf, log_err_mf, log_idx, log_case = trainer.validation(valid_x.reshape(valid_x.shape[0],64, 64), valid_y, trained_model, gpu_idx=1)

    # valid_x = np.asarray(np.load('/home/exx/Documents/Hope/rec_crs.npy').item()['rect_img'][000:100],'float32')*2.-255.
    # valid_y = [0.,1.]
    # log_err_bf, log_err_mf, log_idx, log_case = trainer.validation(valid_x.reshape(valid_x.shape[0],64, 64), valid_y, trained_model, gpu_idx=2)

    # valid_x = np.load('/home/exx/Documents/Hope/Datasets_For_Hope/Fixed_Datasets_For_Hope/eps-150_FGSM_and_feat_squeeze_data.npz')['FGSM_features']
    # valid_y = [1.,1.]
    # log_err_bf, log_err_mf, log_idx, log_case = trainer.validation(valid_x, valid_y, trained_model, gpu_idx=3)

    # valid_x = np.load('/home/exx/Documents/Hope/Datasets_For_Hope/CPPN/CPPNdataset_orig_scaled_bin.npz')['scaled_features_-255_to_255']
    # valid_y = [0.,0.]
    # log_err_bf, log_err_mf, log_idx, log_case = trainer.validation(valid_x, valid_y, trained_model, gpu_idx=2)

    # tt_log_y_pred = {}
    # tt_log_err_opt = {}
    # tt_log_loss_opt = {}
    # tt_log_y_pred_ref = {}
    # tt_log_err_opt_ref = {}
    # tt_log_x_input = {}
    # tt_log_y_true = {}
    # tt_log_err_bf = {}
    # tt_log_err_mf = {}
    # tt_log_idx = {}
    # tt_log_case = {}
    #
    # log_y_pred, log_err_opt, log_loss_opt, log_y_pred_ref, log_err_opt_ref, log_x_input, log_y_true, \
    # log_err_bf, log_err_mf, log_idx, log_case = trainer.test(testing_dataset,trained_model,CPPN=0,gpu_idx=0)
    # tt_log_y_pred[testing_dataset] = log_y_pred
    # tt_log_err_opt[testing_dataset] = log_err_opt
    # tt_log_loss_opt[testing_dataset] = log_loss_opt
    # tt_log_y_pred_ref[testing_dataset] = log_y_pred_ref
    # tt_log_err_opt_ref[testing_dataset] = log_err_opt_ref
    # tt_log_x_input[testing_dataset] = log_x_input
    # tt_log_y_true[testing_dataset] = log_y_true
    # tt_log_err_bf[testing_dataset] = log_err_bf
    # tt_log_err_mf[testing_dataset] = log_err_mf
    # tt_log_idx[testing_dataset] = log_idx
    # tt_log_case[testing_dataset] = log_case
    # np.savez('log14_16.npy',
    #          tt_log_y_pred=tt_log_y_pred,
    #          tt_log_err_opt=tt_log_err_opt,
    #          tt_log_loss_opt=tt_log_loss_opt,
    #          tt_log_y_pred_ref=tt_log_y_pred_ref,
    #          tt_log_err_opt_ref=tt_log_err_opt_ref,
    #          tt_log_x_input=tt_log_x_input,
    #          tt_log_y_true=tt_log_y_true,
    #          tt_log_err_bf=tt_log_err_bf,
    #          tt_log_err_mf=tt_log_err_mf,
    #          tt_log_idx=tt_log_idx,
    #          tt_log_case=tt_log_case,
    #          )

if __name__ == '__main__':
    main(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
