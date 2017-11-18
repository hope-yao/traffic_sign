import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from utils import *

def GeneratorCNN(y_data, z, hidden_num, output_num, repeat_num, data_format):
    bs, n_subnet = y_data.get_shape().as_list()
    z_num = z.get_shape().as_list()[1]
    ch_data = 1 # binary only
    imsize = 2**(repeat_num+2+1)
    out = tf.zeros((bs,ch_data,imsize,imsize))
    out_sub = []
    for net_i in range(n_subnet):
        z_i = tf.slice(z, (0, z_num / n_subnet * net_i), (bs, z_num / n_subnet))
        x_i = slim.fully_connected(z_i, np.prod([8, 8, hidden_num]), activation_fn=None)
        x_i = reshape(x_i, 8, 8, hidden_num, data_format)
        for idx in range(repeat_num):
            x_i = slim.conv2d_transpose(x_i, hidden_num*(idx+1), 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
            # x_i = slim.conv2d(x_i, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            # x_i = slim.conv2d(x_i, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            # if idx < repeat_num - 1:
            #     x_i = upscale(x_i, 2, data_format)
        out_i = slim.conv2d(x_i, output_num, 3, 1, activation_fn=None, data_format=data_format)
        y_i = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(y_data[:, net_i], 1), 1), 1), [1, 1, imsize, imsize])
        out += y_i * out_i
        out_sub += [out_i]
    # out = tf.clip_by_value(out, 0, 1)
    return out, out_sub

def sampling(z_mean, z_log_var,L=1):
    batch_size, latent_dim = z_mean.get_shape().as_list()
    epsilon = tf.random_normal(shape=(batch_size*L, latent_dim), mean=0.,name='epsilon')
    return tf.tile(z_mean,[L,1]) + tf.exp(tf.tile(z_log_var,[L,1]) / 2) * 0

def my_sampling(z,n,L=1):
    z_g = []
    z_mean = []
    z_log_var = []
    dim1, dim2 = z.get_shape().as_list()
    for i in range(n):
        z_mean_and_z_log_var = tf.slice(z, (0,dim2*i/n), (dim1,dim2/n))
        z_mean_i, z_log_var_i = tf.split(z_mean_and_z_log_var, 2, axis=1)
        z_g_i = sampling(z_mean_i, z_log_var_i, L)
        z_g += [z_g_i]
        z_mean += [z_mean_i]
        z_log_var += [z_log_var_i ]
    return list2tensor(z_g,dim=1), z_mean, z_log_var

def calc_kl_loss(z_mean, z_log_var,bs):
    N_Subnets = len(z_mean)
    z_mean_real = []
    z_logvar_real = []
    z_mean_fake = []
    z_logvar_fake = []
    for i in range(N_Subnets):
        z_mean_fake += [z_mean[i][0:bs, :]]
        z_logvar_fake += [z_log_var[i][0:bs, :]]
        z_mean_real += [z_mean[i][bs:2*bs, :]]
        z_logvar_real += [z_log_var[i][bs:2*bs, :]]

    kl_f = []
    kl_r = []
    for i in range(N_Subnets):
        z_mean_i = z_mean_fake[i]
        z_log_var_i = z_logvar_fake[i]
        kl_f_i = -0.5 * tf.reduce_sum(1 + z_log_var_i - tf.square(z_mean_i) - tf.exp(z_log_var_i),1)
        kl_f_i = tf.reduce_mean(kl_f_i)
        kl_f += [tf.expand_dims(kl_f_i, 0)]

        z_mean_i = z_mean_real[i]
        z_log_var_i = z_logvar_real[i]
        kl_r_i = -0.5 * tf.reduce_sum(1 + z_log_var_i - tf.square(z_mean_i) - tf.exp(z_log_var_i),1)
        kl_r_i = tf.reduce_mean(kl_r_i)
        kl_r += [tf.expand_dims(kl_r_i, 0)]
    return  kl_f, kl_r

def calc_eclipse_loss_analy(dz,z,N_Subnets):
    dzf, dzr = tf.split(dz,2)
    l_f = []
    l_r = []
    m_r = []
    m_f = []
    dim1, dim2 = dzf.get_shape().as_list()
    for i in range(N_Subnets):
        dzf_i = tf.slice(dzf,(0,dim2/N_Subnets*i),(dim1,dim2/N_Subnets))
        dzr_i = tf.slice(dzr,(0,dim2/N_Subnets*i),(dim1,dim2/N_Subnets))
        z_i = tf.slice(z,(0,dim2/N_Subnets*i),(dim1,dim2/N_Subnets))

        dzf_mean = tf.tile(tf.expand_dims(tf.reduce_mean(dzf_i, 0), 0), (dim1, 1))
        dzf_i = dzf_i - (dzf_mean+1e-8)
        dzr_mean = tf.tile(tf.expand_dims(tf.reduce_mean(dzr_i, 0), 0), (dim1, 1))
        dzr_i = dzr_i - (dzr_mean+1e-8)
        z_mean = tf.tile(tf.expand_dims(tf.reduce_mean(z_i, 0), 0), (dim1, 1))

        mi_dzf = tf.matmul(tf.transpose(dzf_i, (1, 0)), dzf_i) / dim1
        mi_dzr = tf.matmul(tf.transpose(dzr_i, (1, 0)), dzr_i) / dim1
        mi_z = tf.eye(dim2/N_Subnets,dim2/N_Subnets)


        res_f_i = tf.square((mi_dzf-mi_z))#/mi_z+1e-8)
        l_f_i = tf.reduce_mean(res_f_i) + tf.reduce_mean(tf.diag_part(res_f_i))#+res_det_f_i ##
        l_f += [tf.expand_dims(l_f_i, 0)]
        m_f_i = tf.reduce_mean(tf.square(dzf_mean - z_mean))
        m_f += [tf.expand_dims(m_f_i, 0)]

        res_r_i = tf.square((mi_dzr-mi_z))#/mi_z+1e-8)
        l_r_i = tf.reduce_mean(res_r_i) + tf.reduce_mean(tf.diag_part(res_r_i))#+res_det_f_i ##
        l_r += [tf.expand_dims(l_r_i, 0)]
        m_r_i = tf.reduce_mean(tf.square(dzr_mean - z_mean))
        m_r += [tf.expand_dims(m_r_i, 0)]

        if i==0:
            tmp = mi_dzf

    return  [l_f,l_r], [m_f,m_r], tf.reduce_mean(tmp)#, (mi_z, mi_dzf, dzf_i)

def wy_calc_eclipse_loss_analy(dz,z,y,N_Subnets):
    dzf, dzr = tf.split(dz,2)
    l_f = []
    l_r = []
    m_r = []
    m_f = []
    dim1, dim2 = dzf.get_shape().as_list()
    for i in range(N_Subnets):
        dzf_i = tf.slice(dzf,(0,dim2/N_Subnets*i),(dim1,dim2/N_Subnets))
        dzr_i = tf.slice(dzr,(0,dim2/N_Subnets*i),(dim1,dim2/N_Subnets))
        z_i = tf.slice(z,(0,dim2/N_Subnets*i),(dim1,dim2/N_Subnets))
        y_i = tf.slice(y, (0, i), (dim1, 1))

        dzf_mean = tf.tile(tf.expand_dims(tf.reduce_mean(dzf_i*y_i,0)/tf.reduce_sum(y), 0), (dim1, 1))
        dzf_i = dzf_i - (dzf_mean+1e-8)
        dzr_mean = tf.tile(tf.expand_dims(tf.reduce_mean(dzf_i*y_i,0)/tf.reduce_sum(y), 0), (dim1, 1))
        dzr_i = dzr_i - (dzr_mean+1e-8)
        z_mean = tf.tile(tf.expand_dims(tf.reduce_mean(dzf_i*y_i,0)/tf.reduce_sum(y), 0), (dim1, 1))

        mi_dzf = tf.matmul(tf.matmul(tf.transpose(dzf_i, (1, 0)),tf.diag(y_i[:,0])), dzf_i) / dim1
        mi_dzr = tf.matmul(tf.matmul(tf.transpose(dzr_i, (1, 0)),tf.diag(y_i[:,0])), dzf_i) / dim1
        mi_z = tf.eye(dim2/N_Subnets,dim2/N_Subnets)


        res_f_i = tf.square((mi_dzf-mi_z))#/mi_z+1e-8)
        l_f_i = tf.reduce_mean(res_f_i) + tf.reduce_mean(tf.diag_part(res_f_i))#+res_det_f_i ##
        l_f += [tf.expand_dims(l_f_i, 0)]
        m_f_i = tf.reduce_mean(tf.square(dzf_mean - z_mean))
        m_f += [tf.expand_dims(m_f_i, 0)]

        res_r_i = tf.square((mi_dzr-mi_z))#/mi_z+1e-8)
        l_r_i = tf.reduce_mean(res_r_i) + tf.reduce_mean(tf.diag_part(res_r_i))#+res_det_f_i ##
        l_r += [tf.expand_dims(l_r_i, 0)]
        m_r_i = tf.reduce_mean(tf.square(dzr_mean - z_mean))
        m_r += [tf.expand_dims(m_r_i, 0)]

        if i==0:
            tmp = mi_dzf

    return  [l_f,l_r], [m_f,m_r], tf.reduce_mean(tmp)#, (mi_z, mi_dzf, dzf_i)

def Encoder(x, z_num, repeat_num, hidden_num, data_format):

    # Encoder
    x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
    for idx in range(repeat_num):
        channel_num = hidden_num * (idx + 1)
        x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
        # x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        # if idx < repeat_num - 1:
        #     # x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
        #     x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID', data_format=data_format)
    x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
    z = slim.fully_connected(x, z_num, activation_fn=None) # times 2 for mean and variance

    return  z

def act_Encoder(x, z_num, repeat_num, hidden_num, data_format):

    # Encoder
    x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
    act = [x]
    for idx in range(repeat_num):
        channel_num = hidden_num * (idx + 1)
        x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
        act += [x]
        # x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        # if idx < repeat_num - 1:
        #     # x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
        #     x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID', data_format=data_format)
    x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
    z = slim.fully_connected(x, z_num, activation_fn=None) # times 2 for mean and variance

    return  z,act

def Decoder(y_data, z, input_channel, z_num, repeat_num, hidden_num, data_format):
    ch_data = 1  # binary only
    imsize = 2 ** (repeat_num + 2+1)
    bs, n_subnet = y_data.get_shape().as_list()
    dup = 2 #2 if without x_mid, which is only x_real and x_fake. Doubled to 4 if with x_mid

    # Decoder
    out = tf.zeros((bs*dup, ch_data, imsize, imsize))
    out_sub = []
    for net_i in range(n_subnet):
        z_i = tf.slice(z,(0,z_num/n_subnet*net_i),(bs*dup,z_num/n_subnet))#bs*net_i*dup
        x_i = slim.fully_connected(z_i, np.prod([8, 8, hidden_num]), activation_fn=None)
        x_i = reshape(x_i, 8, 8, hidden_num, data_format)
        for idx in range(repeat_num):
            x_i = slim.conv2d_transpose(x_i, hidden_num*(idx+1), 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
            # x_i = slim.conv2d(x_i, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            # x_i = slim.conv2d(x_i, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            # if idx < repeat_num - 1:
            #     x_i = upscale(x_i, 2, data_format)
        out_i = slim.conv2d(x_i, input_channel, 3, 1, activation_fn=None, data_format=data_format)
        y_i = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(y_data[:, net_i], 1), 1), 1), [dup, 1, imsize, imsize])
        # out = tf.logical_or(out,y_i * out_i)
        out += y_i * out_i
        out_sub += [out_i[0:bs*2]] #only output the first random sample

    return out, out_sub



def Encoder_mid(x, z_num, repeat_num, hidden_num, data_format):

    # Encoder
    x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
    for idx in range(repeat_num):
        channel_num = hidden_num * (idx + 1)
        x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        if idx < repeat_num - 1:
            # x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
            x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID', data_format=data_format)
        if idx==2:
            x_mid = x
    x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
    z = slim.fully_connected(x, z_num, activation_fn=None) # times 2 for mean and variance

    return  z, x_mid

def Decoder_mid(z, x_mid, n_subnet, input_channel, z_num, repeat_num, hidden_num, data_format):
    ch_data = 1  # binary only
    imsize = 2 ** (repeat_num + 2)
    dup = 4 #2 if without x_mid, which is only x_real and x_fake. Doubled to 4 if with x_mid
    bs = z.get_shape().as_list()[0]/2 #z is obtained from x_real and x_fake

    # Decoder
    out = tf.zeros((bs*dup, ch_data, imsize, imsize))
    out_sub = []
    for net_i in range(n_subnet):
        channel_num = hidden_num * repeat_num
        z_i = tf.slice(z,(0,z_num/n_subnet*net_i),(bs*2,z_num/n_subnet))#bs*net_i*dup
        x_i = slim.fully_connected(z_i, np.prod([8, 8, channel_num]), activation_fn=None)
        x_i = reshape(x_i, 8, 8, channel_num, data_format)
        for idx in range(repeat_num):
            if idx == 1:
                x_i = tf.concat([x_i, x_mid], 0)
            if idx >= 1:
                x_i = upscale(x_i, 2, data_format)
            if idx==repeat_num-1:
                channel_num = hidden_num
            else:
                channel_num = hidden_num * (repeat_num - idx-1)
            x_i = slim.conv2d(x_i, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x_i = slim.conv2d(x_i, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        out_i = slim.conv2d(x_i, input_channel, 3, 1, activation_fn=None, data_format=data_format)
        out += out_i
        out_sub += [out_i[0:bs*2]] #only output the first random sample

        full_rec, part_rec = tf.split(out,2)
    return full_rec, out_sub, part_rec

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)


def calc_pt_Angular(z_d_gen, bs):
    nom = tf.matmul(z_d_gen, tf.transpose(z_d_gen, perm=[1, 0]))
    denom = tf.sqrt(tf.reduce_sum(tf.square(z_d_gen), reduction_indices=[1], keep_dims=True))
    pt = tf.abs(tf.transpose((nom / denom), (1, 0)) / denom)
    pt = pt - tf.diag(tf.diag_part(pt))
    pulling_term = tf.reduce_sum(pt) / (bs * (bs - 1))
    return pulling_term, nom, denom
    #

def calc_pt_Euclidian(imgs):
    dist = []
    denom = []
    for net_i, img in enumerate(imgs):
        bs = img.get_shape()[0].value
        img_vec = tf.clip_by_value(tf.contrib.layers.flatten(img),0,1)
        img_mat = tf.tile(tf.expand_dims(img_vec,0),(bs,1,1)) - tf.tile(tf.expand_dims(img_vec,1),(1,bs,1))
        denom += [tf.reduce_mean(tf.tile(tf.expand_dims(img_vec, 0), (bs, 1, 1)),2)]
        dist += [tf.reduce_mean(tf.abs(img_mat),2)]
    return dist, denom
