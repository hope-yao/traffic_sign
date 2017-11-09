from __future__ import print_function

import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque
# from style import total_style_cost, white_style

from models import GeneratorCNN, Encoder, Decoder, calc_eclipse_loss_analy, calc_pt_Euclidian
from utils import save_image, new_save_image, list2tensor, creat_dir
from load_data import *

import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque
from style import total_style_cost, white_style

from models import *
from utils import save_image, new_save_image, list2tensor, creat_dir
from load_data import *



def next(loader):
    return loader.next()[0].data.numpy()


def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image


def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image


def norm_img(image, data_format=None):
    image = image / 255.
    # image = image/127.5 - 1.
    image = (image+1)/2 #convert to (0,1)
    if data_format:
        image = to_nhwc(image, data_format)
    return image


def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc(norm * 255., data_format), 0, 255)
    # return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')
        self.g_lr_update = tf.assign(self.g_lr, self.g_lr * 0.5, name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, self.d_lr * 0.5, name='d_lr_update')

        # self.gamma = config.gamma
        self.gamma = tf.placeholder(tf.float32,())
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        self.imsize = 64
        self.channel = 1
        self.repeat_num = int(np.log2(self.imsize)) - 2 -1

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train

        self.build_model()

        self.logdir, self.modeldir = creat_dir('GAN')
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logdir)

        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # self.saver.restore(self.sess,
        #                    '/home/exx/Documents/Hope/BEGAN-tensorflow-regressor-20170811-GED-eclipse-ptx-traffic/models/GAN/GAN_2017_11_01_15_04_47/experiment_185390.ckpt')
        # self.saver.restore(self.sess,
        #                    '/home/exx/Documents/Hope/BEGAN-tensorflow-regressor-20170811-GED-eclipse-ptx-traffic/models/GAN/GAN_2017_11_01_08_58_56/experiment_9600.ckpt')
        # self.saver.restore(self.sess, "./models/GAN/GAN_2017_08_21_14_08_10/experiment_10000.ckpt")
        # self.saver.restore(self.sess, "./models/GAN/GAN_2017_08_12_23_30_24/experiment_102438.ckpt")


    def valid(self,valid_x, valid_y, trained_model, para_list, gpu_idx=0):
        with tf.device('gpu:{}'.format(gpu_idx)):
            self.saver.restore(self.sess,trained_model)

            for loss_t in para_list[0]:
                for y_t1 in para_list[1]:
                    for y_t_inc in para_list[2]:
                        y_t2 = y_t1 + y_t_inc
                        for c_wd in para_list[3]:
                            for theta in para_list[4]:
                                for bias in para_list[5]:
                                    log_erri = 0
                                    log_y_true = []
                                    log_y_pred = []
                                    log_err_opt = []
                                    log_loss_opt = []
                                    log_cur_opt = []
                                    log_y_pred_ref = []
                                    log_err_opt_ref = []
                                    log_x_input = []
                                    log_idx = []
                                    log_case = []
                                    log_err = []

                                    for opt_idx in range(0, 10, 1):
                                        y_true = valid_y[opt_idx]
                                        x_input_opt = valid_x[opt_idx]
                                        y0 = [0.5, 0.5, 0.5, 0.5]
                                        if log_erri > 3:
                                            y_pred = y0
                                            err_opt = y0
                                            loss_opt = 1
                                            cur = 1
                                            y_refine = np.asarray(0.)
                                            err_refine = np.asarray(0.)
                                            case = -1
                                        else:
                                            y_pred, err_opt, loss_opt, cur, y_refine, err_refine, case = \
                                                self.opt_attack(x_input_opt, y_true, y0, itr=150, bias=bias,
                                                                theta=theta, c_wd=c_wd,
                                                                loss_t=loss_t, y_t1=y_t1, y_t2=y_t2)
                                                # self.opt_attack(x_input_opt, y_true, y0, itr=150, bias=bias,
                                                #                     theta=theta, c_wd=0.02,
                                                #                     loss_t=0.3, y_t1=0.7, y_t2=0.8)
                                        log_y_pred += [y_pred]
                                        log_err_opt += [err_opt]
                                        log_y_pred_ref += [y_refine]
                                        log_err_opt_ref += [err_refine]
                                        log_loss_opt += [loss_opt]
                                        log_cur_opt += [cur]
                                        log_x_input += [x_input_opt]
                                        log_y_true += [y_true]
                                        log_case += [case]
                                        log_idx += [opt_idx]
                                        log_erri += int(err_refine.any())
                                    log_err += [log_erri]
                                    np.savez('./att_new_new/traffic_sign_{}_{}_{}_{}_{}_{}.npy'.format(loss_t,y_t1,y_t2,c_wd,theta,bias),
                                             log_err=log_err,log_y_true=log_y_true,log_y_pred=log_y_pred,log_y_pred_ref=log_y_pred_ref,log_case=log_case,log_idx=log_idx)
        return log_err

    def validation(self,valid_x, valid_y, trained_model, para_list, gpu_idx=0):
        with tf.device('gpu:{}'.format(gpu_idx)):
            self.saver.restore(self.sess,trained_model)

            log_y_true = []
            log_y_pred = []
            log_err_opt = []
            log_loss_opt = []
            log_cur_opt = []
            log_y_pred_ref = []
            log_err_opt_ref = []
            log_x_input = []
            log_idx = []
            log_case = []
            log_err = []

            for loss_t in para_list[0]:
                for y_t1 in para_list[1]:
                    for y_t2 in para_list[2]:
                        for c_wd in para_list[3]:
                            for theta in para_list[4]:
                                for bias in para_list[5]:
                                    log_erri = 0
                                    for opt_idx in range(0, 5, 1):
                                        y_true = valid_y
                                        x_input_opt = valid_x[opt_idx]
                                        y0 = [0.5, 0.5, 0.5, 0.5]
                                        y_pred, err_opt, loss_opt, cur, y_refine, err_refine, case = \
                                            self.opt_attack(x_input_opt, y_true, y0, itr=30, bias=bias, theta=theta, c_wd=c_wd,
                                                            loss_t=loss_t, y_t1=y_t1, y_t2=y_t2)
                                        log_y_pred += [y_pred]
                                        log_err_opt += [err_opt]
                                        log_y_pred_ref += [y_refine]
                                        log_err_opt_ref += [err_refine]
                                        log_loss_opt += [loss_opt]
                                        log_cur_opt += [cur]
                                        log_x_input += [x_input_opt]
                                        log_y_true += [y_true]
                                        log_case += [case]
                                        log_idx += [opt_idx]
                                        log_erri += np.sum(np.abs(err_refine))
                                    log_err += [log_erri]
        return log_err

    def test(self,testing_dataset, trained_model, CPPN, gpu_idx=0):
        with tf.device('gpu:{}'.format(gpu_idx)):
            self.saver.restore(self.sess,trained_model)
            raw_data = np.load(testing_dataset)

            # (self.X_train, self.y_train), (self.X_test, self.y_test) = CRS()
            # nn = np.load('/home/exx/Documents/Hope/rec_crs.npy')
            # crs = nn.item()['cross_img']
            # crs_img = np.asarray(np.reshape(crs, (crs.shape[0], 1, 64, 64)), 'float32') * 2 - 255.
            # i = 10
            # self.x_input_crs = crs_img[i * self.batch_size: (i + 1) * self.batch_size]
            # crs_label = nn.item()['cross_label']
            # self.y_input_crs = crs_label[i * self.batch_size: (i + 1) * self.batch_size]

            tt_log_y_pred = []
            tt_log_err_opt = []
            tt_log_loss_opt = []
            tt_log_y_pred_ref = []
            tt_log_err_opt_ref = []
            tt_log_x_input = []
            tt_log_y_true = []
            tt_log_err_bf = []
            tt_log_err_mf = []
            tt_log_idx = []
            tt_log_case = []

            c_wd = 0.0015
            loss_t = 0.3
            y_t1 = 0.5
            y_t2 = 0.6
            for theta in [7.]:
                for bias in [-6.,]:
                    log_y_pred = []
                    log_err_opt = []
                    log_loss_opt = []
                    log_cur_opt = []
                    log_y_pred_ref = []
                    log_err_opt_ref = []
                    log_x_input = []
                    log_y_true = []
                    log_err_bf = []
                    log_err_mf = []
                    log_idx = []
                    log_case = []
                    for opt_idx in range(0,100,1):
                        if CPPN:
                            y_true = [0,0]
                            x_input_opt = raw_data['scaled_features_-255_to_255'][opt_idx]
                        else:
                            y_true = raw_data['label']
                            x_input_opt = raw_data['data'][opt_idx].transpose(2,0,1)
                        y0 = [0.5,0.5,0.5,0.5]
                        y_pred, err_opt, loss_opt, cur, y_refine, err_refine, case =\
                            self.opt_attack(x_input_opt, y_true, y0, itr=30, bias=bias, theta=theta, c_wd=c_wd,
                                            loss_t=loss_t, y_t1 = y_t1, y_t2 = y_t2)
                        log_y_pred += [y_pred]
                        log_err_opt += [err_opt]
                        log_y_pred_ref += [y_refine]
                        log_err_opt_ref += [err_refine]
                        log_loss_opt += [loss_opt]
                        log_cur_opt += [cur]
                        log_x_input += [x_input_opt]
                        log_y_true += [y_true]
                        log_case += [case]
                        # if CPPN:
                        #     log_err_bf += [np.round(raw_data['binarized_scaled_features_classification'][opt_idx] - y_true)]
                        # else:
                        #     log_err_bf += [np.round(raw_data['bin_filt_class'][opt_idx] - y_true)]
                        #     log_err_mf += [np.round(raw_data['median_filt_class'][opt_idx] - y_true)]
                        log_idx += [opt_idx]

                    tt_log_y_pred += log_y_pred
                    tt_log_err_opt += log_err_opt
                    tt_log_loss_opt += log_loss_opt
                    tt_log_y_pred_ref += log_y_pred_ref
                    tt_log_err_opt_ref += log_err_opt_ref
                    tt_log_x_input += log_x_input
                    tt_log_y_true += log_y_true
                    tt_log_err_bf += log_err_bf
                    tt_log_err_mf += log_err_mf
                    tt_log_idx += log_idx
                    tt_log_case += log_case

        return  tt_log_y_pred, tt_log_err_opt, tt_log_loss_opt, tt_log_y_pred_ref, tt_log_err_opt_ref, \
                tt_log_x_input, tt_log_y_true, tt_log_err_bf, tt_log_err_mf, tt_log_idx, tt_log_case

    def opt_attack(self, x_input_opt, y_true, y0, itr=50, bias=0., theta=0.3, c_wd=0.001, loss_t = 0.3, y_t1 = 0.5, y_t2=0.6):
        temp = set(tf.all_variables())
        self.y_input_opt = tf.Variable(y0,name='y_input_opt')

        with tf.variable_scope("D", reuse=True):
            x_input_opt = np.tile(x_input_opt, (2 * self.batch_size, 1, 1, 1))
            self.d_z_opt_initial = Encoder(norm_img(x_input_opt), self.z_num, self.repeat_num,
                           self.conv_hidden_num, self.data_format)

            dzv  = self.d_z_opt_initial.eval(session=self.sess)
            self.z0 = tf.Variable(dzv[0], name='z_input_opt')
            self.y_input_opt_sig = tf.sigmoid(self.y_input_opt)
            self.d_z_opt = tf.tile(tf.expand_dims(self.z0,0),(self.batch_size*2,1))
            y_opt = tf.tile(tf.expand_dims(self.y_input_opt,0), (self.batch_size,1))
            self.d_out_opt, self.D_sub_norm_opt = Decoder(y_opt, self.d_z_opt, self.channel, self.z_num,
                                                  self.repeat_num, self.conv_hidden_num, self.data_format)
            # self.d_out_opt_sig = tf.zeros_like(self.d_out_opt)
            # for net_i, out_i in enumerate(self.D_sub_norm_opt):
            #     self.d_out_opt_sig += self.y_input_opt[net_i] * tf.sigmoid(out_i)
            self.bias = tf.Variable(bias)
            self.d_out_opt_sig = tf.sigmoid(theta*self.d_out_opt+self.bias)

        self.loss_opt = tf.reduce_mean(tf.abs(self.d_out_opt_sig - norm_img(x_input_opt))) + c_wd*tf.norm(self.y_input_opt)
        opt = tf.train.AdamOptimizer(0.01)
        opt_sgd = tf.train.GradientDescentOptimizer(0.01)
        grads_g = opt.compute_gradients(self.loss_opt, var_list=[self.y_input_opt]+[self.z0])
        apply_gradient_opt = opt.apply_gradients(grads_g, global_step=self.step)
        self.sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        for i in range(itr):
            self.sess.run(apply_gradient_opt)
            # x_pred = self.sess.run(self.d_out_opt_sig, {self.y_input_opt: self.y_pred, self.z0: self.z_pred})
            # x_pred = self.sess.run(self.d_out_opt-norm_img(x_input_opt), {self.y_input_opt: self.y_pred, self.z0: self.z_pred})
            # np.mean(np.abs(norm_img(x_input_opt) - x_pred))

        self.y_pred = self.y_input_opt.eval(session=self.sess)
        self.z_pred = self.z0.eval(session=self.sess)
        self.err_opt = y_true - np.round(self.y_pred)

        # np.save('test1.npy', norm_img(x_input_opt)[0, 0])
        # np.save('test2.npy', self.sess.run(self.d_out_opt_sig, {self.y_input_opt: self.y_pred, self.z0: self.z_pred})[0, 0])
        # np.save('test3.npy', self.sess.run(self.d_out_opt_sig, {self.y_input_opt: y_true, self.z0: self.z_pred})[0, 0])
        tt = np.mean(np.abs(
            self.sess.run(self.d_out_opt_sig, {self.y_input_opt: self.y_pred, self.z0: self.z_pred})[0, 0] -
            norm_img(x_input_opt)))

        case = 0
        # consider some special cases:
        self.y_refine = self.y_pred
        if tt>loss_t:
            # noise
            self.y_refine = [0,0,0,0]
            case = 1
        else:
            case = 2
            # uncertain
            self.tt_refine = 1
            for idx, yi in enumerate(self.y_pred):
                if yi < y_t2 and yi > y_t1:
                    for y_test_i in [0, 1]:
                        y_test = np.round(self.y_pred)
                        y_test[idx] = y_test_i
                        tt = np.mean(np.abs(
                            self.sess.run(self.d_out_opt_sig, {self.y_input_opt: y_test, self.z0: self.z_pred})[
                                0, 0]
                            - norm_img(x_input_opt)))
                        if tt < self.tt_refine:
                            self.y_refine = y_test.tolist()
                            self.tt_refine = tt
                        case = 3
                elif yi > y_t2:
                    continue
                else:
                    self.y_refine[idx] = 0.0

            else:
                # original prediction
                case = 4
        self.err_opt_refine = y_true - np.round(self.y_refine)

        self.curvature = opt_sgd.compute_gradients(opt_sgd.compute_gradients(self.loss_opt, self.z0)[0][1], self.z0)[0][1]
        cur = tf.reduce_mean(tf.abs(self.curvature)).eval(session=self.sess)

        return self.y_pred, self.err_opt, tt, cur, self.y_refine, self.err_opt_refine, case
    #


    def train(self):

        from tqdm import tqdm
        rawdata = np.load('traffic_sign_dataset1.npz')
        self.X_train = rawdata['data'][:-200].transpose(0,3,1,2)
        self.y_train = rawdata['label'][:-200]
        self.X_test = rawdata['data'][-200:].transpose(0,3,1,2)
        self.y_test = rawdata['label'][-200:]


        x_input_fix = self.X_test[0 * self.batch_size:(0 + 1) * self.batch_size]
        y_input_fix = self.y_test[0 * self.batch_size:(0 + 1) * self.batch_size]
        mu, sigma = 0, 1  # mean and standard deviation
        z_input_fix = np.random.normal(mu, sigma, (self.batch_size, self.z_num)).astype('float32')
        feed_dict_fix = {self.x: x_input_fix, self.y: y_input_fix, self.z: z_input_fix}

        counter = 0
        gamma = 0.5
        for epoch in range(5000):
            it_per_ep = len(self.X_train) / self.batch_size
            for i in tqdm(range(it_per_ep)):
                counter += 1
                x_input = self.X_train[i * self.batch_size:(i + 1) * self.batch_size]
                y_input = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                z_input = np.random.normal(mu, sigma, (self.batch_size, self.z_num)).astype('float32')
                feed_dict = {self.x: x_input, self.y: y_input, self.z: z_input, self.gamma:gamma}
                result = self.sess.run([self.d_loss_real, self.d_loss_fake, self.d_loss, self.g_loss, self.k_update,
                                        self.kl_mean, self.m, self.pt_x], feed_dict)
                # drv, dfv, dv, gv, = result[0:4]
                # if gv<2.:
                # # if dfv<1.:
                #     gamma = 1.
                # else:
                #     gamma = 0.5

                print(result)
                import math
                for i, val in enumerate(result):
                    if np.any(np.isnan(np.asarray(val))):
                        print('err')

                if counter in [100e3, 200e3]:
                    self.sess.run([self.g_lr_update, self.d_lr_update])

                if counter % 100 == 0:
                    x_img, x_rec, g_img, g_rec, D_sub, G_sub = \
                        self.sess.run([self.x_img, self.AE_x, self.G, self.AE_G, self.D_sub,
                                       self.G_sub], feed_dict_fix)
                    nrow = self.batch_size
                    all_G_z = np.concatenate([x_img, x_rec, g_img, g_rec, D_sub, G_sub])
                    im = save_image(all_G_z, '{}/itr{}.png'.format(self.logdir, counter), nrow=nrow)

                if counter in [10e3, 2e4, 4e4, 6e4, 2e5, 4e5]:
                    snapshot_name = "%s_%s" % ('experiment', str(counter))
                    fn = self.saver.save(self.sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                    print("Model saved in file: %s" % fn)

                if counter % 10 == 0:
                    summary = self.sess.run(self.summary_op, feed_dict)
                    self.summary_writer.add_summary(summary, counter)
                    self.summary_writer.flush()

    def build_model(self, n_net=4):

        self.x = tf.placeholder(tf.float32, [self.batch_size, 1, self.imsize, self.imsize])
        self.x_norm = x = norm_img(self.x)
        self.y = tf.placeholder(tf.float32, [self.batch_size, n_net], name='y_input')

        mask = []
        for i in range(n_net):
            mask += [tf.tile(self.y[:, i:i + 1], [1, self.z_num / n_net])]
        self.mask = list2tensor(mask, 1)

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_num], name='z_input')
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        g_optimizer, d_optimizer = tf.train.AdamOptimizer(self.g_lr), tf.train.AdamOptimizer(self.d_lr)

        tower_grads_d = []
        tower_grads_g = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in [1]:
                with tf.device('/gpu:%d' % i):
                    with tf.variable_scope('G') as vs_g:
                        self.G_norm, self.G_sub_norm = GeneratorCNN(
                            self.y, self.z, self.conv_hidden_num, self.channel,
                            self.repeat_num, self.data_format)
                        self.G_var = tf.contrib.framework.get_variables(vs_g)
                    self.pt_x_sub, self.pt_x_denom_sub =  calc_pt_Euclidian(self.G_sub_norm)
                    self.pt_x = tf.reduce_mean(list2tensor(self.pt_x_sub)/(list2tensor(self.pt_x_denom_sub)+0.01))

                    with tf.variable_scope('D') as vs_d:
                        self.all_x = tf.concat([self.G_norm, x], 0)
                        self.d_z = Encoder(self.all_x, self.z_num, self.repeat_num,
                                           self.conv_hidden_num, self.data_format)
                        L = 1
                        # self.d_zg, z_mean, z_log_var = my_sampling(self.d_z, n_net, L)
                        self.d_out, self.D_sub_norm = Decoder(self.y, self.d_z, self.channel, self.z_num,
                                                                              self.repeat_num,self.conv_hidden_num, self.data_format)
                        self.D_var = tf.contrib.framework.get_variables(vs_d)
                    tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope('G', reuse=True):
                        self.d_z_f, self.d_z_r = tf.split(self.d_z,2)
                        self.G_norm_r, self.G_sub_norm_r = GeneratorCNN(
                            self.y, self.d_z_r, self.conv_hidden_num, self.channel,
                            self.repeat_num, self.data_format)
                    self.loss_enc_gen = 80  * tf.reduce_mean(tf.square(self.G_norm_r  - x))

                    self.pt_z, self.nom, self.denom =  calc_pt_Angular(tf.split(self.d_z,2)[0],self.batch_size)
                    self.kl, self.m, self.dzf_mean_diag = calc_eclipse_loss_analy(self.d_z,self.z,n_net)
                    self.kl_f, self.kl_r = self.kl
                    self.m_f, self.m_r = self.m
                    self.mode_variance = tf.reduce_mean(tf.abs(self.dzf_mean_diag))
                    self.kl_mean = 2*tf.reduce_mean(list2tensor(self.kl))
                    self.m = 2*tf.reduce_mean(list2tensor(self.m))
                    self.kl_mean_f = 2*tf.reduce_mean(list2tensor(self.kl_f))
                    self.m_f = 2*tf.reduce_mean(list2tensor(self.m_f))
                    self.kl_mean_r = 2*tf.reduce_mean(list2tensor(self.kl_r))
                    self.m_r = 2*tf.reduce_mean(list2tensor(self.m_r))

                    self.AE_G_norm = []
                    self.AE_x_norm = []
                    for i, d_out_l in enumerate(tf.split(self.d_out, L)):
                        AE_G_norm_i, AE_x_norm_i = tf.split(d_out_l, 2)
                        self.AE_G_norm += [AE_G_norm_i]
                        self.AE_x_norm += [AE_x_norm_i]
                    self.d_loss_real = 80  * tf.reduce_mean(tf.square(list2tensor(self.AE_x_norm) - tf.tile(self.x_norm,[L,1,1,1])))
                    self.d_loss_fake = 80 * tf.reduce_mean(tf.square(list2tensor(self.AE_G_norm) - tf.tile(self.G_norm,[L,1,1,1])))

                    # self.part_AE_G_norm = []
                    # self.part_AE_x_norm = []
                    # for i, part_d_out_l in enumerate(tf.split(self.partd_out, L)):
                    #     part_AE_G_norm_i, part_AE_x_norm_i = tf.split(part_d_out_l, 2)
                    #     self.part_AE_G_norm += [part_AE_G_norm_i]
                    #     self.part_AE_x_norm += [part_AE_x_norm_i]
                    # self.part_d_loss_real = 80  * tf.reduce_mean(tf.square(list2tensor(self.part_AE_x_norm) - tf.tile(self.x_norm,[L,1,1,1])))
                    # self.part_d_loss_fake = 80 * tf.reduce_mean(tf.square(list2tensor(self.part_AE_G_norm) - tf.tile(self.G_norm,[L,1,1,1])))

                    # alpha = 0.4
                    # self.mixed_d_loss_real = (1-alpha)*self.d_loss_real + alpha*self.part_d_loss_real
                    # self.mixed_d_loss_fake = (1-alpha)*self.d_loss_fake + alpha*self.part_d_loss_fake

                    self.dgall = tf.zeros((self.batch_size, 1, self.imsize, self.imsize))
                    self.gall = tf.zeros((self.batch_size, 1, self.imsize, self.imsize))
                    for i, dsubnorm_i in enumerate(self.D_sub_norm):
                        img_i, _ = tf.split(dsubnorm_i, 2)
                        self.dgall += img_i
                    for i, img_i in enumerate(self.G_sub_norm):
                        self.gall += img_i
                    self.all_g_loss = 1 * tf.reduce_mean(tf.square(self.dgall - self.gall))
                    self.negative_penalty_g = tf.reduce_sum(tf.nn.relu(-list2tensor(self.G_sub_norm)))
                    self.negative_penalty_d = tf.reduce_sum(tf.nn.relu(-list2tensor(self.D_sub_norm)))
                    self.r_dis = 500*(self.kl_mean_r + self.m_r)
                    self.f_dis = 500*(self.kl_mean_f + self.m_f)
                    self.g_loss = self.d_loss_fake + self.f_dis + (self.pt_z) + 10*(1-self.pt_x) + 0.1*self.negative_penalty_g+ self.loss_enc_gen#+ self.all_g_loss #+ 20*self.pt_z
                    self.d_loss = self.d_loss_real - self.k_t * self.g_loss + self.r_dis + 0.1*self.negative_penalty_d+ self.loss_enc_gen#+ 5*self.mode_variance
                    # self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake + self.r_dis + self.f_dis
                    grads_g = g_optimizer.compute_gradients(self.g_loss, var_list=self.G_var)
                    tower_grads_g.append(grads_g)
                    grads_d = d_optimizer.compute_gradients(self.d_loss, var_list=self.D_var)
                    tower_grads_d.append(grads_d)

        from multi_gpu import average_gradients
        mean_grads_g = average_gradients(tower_grads_g)
        mean_grads_d = average_gradients(tower_grads_d)
        apply_gradient_g = g_optimizer.apply_gradients(mean_grads_g, global_step=self.step)
        apply_gradient_d = g_optimizer.apply_gradients(mean_grads_d)

        self.x_img = denorm_img(x, self.data_format)
        self.G = denorm_img(self.G_norm, self.data_format)
        self.AE_G = denorm_img(self.AE_G_norm[0], self.data_format)
        self.AE_x = denorm_img(self.AE_x_norm[0], self.data_format)
        self.G_sub = denorm_img(list2tensor(self.G_sub_norm), self.data_format)
        self.D_sub = denorm_img(list2tensor(self.D_sub_norm), self.data_format)

        self.balance = self.gamma * (self.d_loss_real+self.r_dis) - self.g_loss
        # self.balance = self.gamma * self.d_loss_real - self.d_loss_fake
        self.measure = self.d_loss_real + tf.abs(self.balance)
        with tf.control_dependencies([apply_gradient_d, apply_gradient_g]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),

            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
            tf.summary.scalar("misc/klf_mean", self.kl_mean),
            tf.summary.scalar("misc/mr", self.m),
            tf.summary.scalar("misc/pt_z", self.pt_z),
            tf.summary.scalar("misc/pt_x", self.pt_x),
            tf.summary.scalar("misc/all_g_loss", self.all_g_loss),
            tf.summary.scalar("misc/gamma", self.gamma),
            tf.summary.scalar("misc/mode_variance", self.mode_variance),
            tf.summary.scalar("misc/negative_penalty_d", self.negative_penalty_d),
            tf.summary.scalar("misc/negative_penalty_g", self.negative_penalty_g),
            tf.summary.scalar("misc/loss_enc_gen", self.loss_enc_gen),
        ])
