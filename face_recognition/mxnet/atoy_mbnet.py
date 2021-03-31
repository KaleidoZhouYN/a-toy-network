import sys
import os
import mxnet as mx
import numpy as np
from symbol import symbol_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from default import config
from symbol.utils import spartial_attention_module,se_module,DResidual_Decode,Residual_Decode,FcResidual


def Act(data, act_type, name):
    #ignore param act_type, set it in this function
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.sym.Activation(data=data, act_type=act_type, name=name)
    return body


def Conv(data,
         num_filter=1,
         kernel=(1, 1),
         stride=(1, 1),
         pad=(0, 0),
         num_group=1,
         name=None,
         suffix=''):
    conv = mx.sym.Convolution(data=data,
                              num_filter=num_filter,
                              kernel=kernel,
                              num_group=num_group,
                              stride=stride,
                              pad=pad,
                              no_bias=True,
                              name='%s%s_conv2d' % (name, suffix))
    bn = mx.sym.BatchNorm(data=conv,
                          name='%s%s_batchnorm' % (name, suffix),
                          fix_gamma=False,
                          momentum=config.bn_mom)
    act = Act(data=bn,
              act_type=config.net_act,
              name='%s%s_relu' % (name, suffix))
    return act


def Linear(data,
           num_filter=1,
           kernel=(1, 1),
           stride=(1, 1),
           pad=(0, 0),
           num_group=1,
           name=None,
           suffix=''):
    conv = mx.sym.Convolution(data=data,
                              num_filter=num_filter,
                              kernel=kernel,
                              num_group=num_group,
                              stride=stride,
                              pad=pad,
                              no_bias=True,
                              name='%s%s_conv2d' % (name, suffix))
    bn = mx.sym.BatchNorm(data=conv,
                          name='%s%s_batchnorm' % (name, suffix),
                          fix_gamma=False,
                          momentum=config.bn_mom)
    return bn


def ConvOnly(data,
             num_filter=1,
             kernel=(1, 1),
             stride=(1, 1),
             pad=(0, 0),
             num_group=1,
             name=None,
             suffix=''):
    conv = mx.sym.Convolution(data=data,
                              num_filter=num_filter,
                              kernel=kernel,
                              num_group=num_group,
                              stride=stride,
                              pad=pad,
                              no_bias=True,
                              name='%s%s_conv2d' % (name, suffix))
    return conv


def DResidual(data,
              dim_match=0,
              num_in=1,
              num_out=1,
              kernel=(3, 3),
              stride=(2, 2),
              pad=(1, 1),
              num_group=1,
              name=None,
              suffix=''):
    suffix = str(suffix)
    use_spa = True
    use_se = True
              
              
    conv = Conv(data=data,
                num_filter=num_group,
                kernel=(1, 1),
                pad=(0, 0),
                stride=(1, 1),
                name='%s%s_conv_sep' % (name, suffix))
    conv_dw = Conv(data=conv,
                   num_filter=num_group,
                   num_group=num_group,
                   kernel=kernel,
                   pad=pad,
                   stride=stride,
                   name='%s%s_conv_dw' % (name, suffix))
    if use_se:
        conv_dw = se_module(data=conv_dw,num_filter=num_group,name=name+suffix)
    proj = Linear(data=conv_dw,
                  num_filter=num_out,
                  kernel=(1, 1),
                  pad=(0, 0),
                  stride=(1, 1),
                  name='%s%s_conv_proj' % (name, suffix))
                  
    if use_spa:
        proj = spartial_attention_module(data=proj,num_filter=num_out,name=name+suffix)
    return proj


def Residual(data,
             num_block=1,
             num_out=1,
             kernel=(3, 3),
             stride=(1, 1),
             pad=(1, 1),
             num_group=1,
             name=None,
             suffix=''):
    identity = data
    for i in range(num_block):
        shortcut = identity
        conv = DResidual(data=identity,
                         dim_match=1,
                         num_in=num_out,
                         num_out=num_out,
                         kernel=kernel,
                         stride=stride,
                         pad=pad,
                         num_group=num_group,
                         name='%s%s_block' % (name, suffix),
                         suffix='%d' % i)
        identity = conv + shortcut
    return identity
    

def get_symbol():
    num_classes = config.embedding_size
    print('in_network', config)
    fc_type = config.net_output
    data = mx.symbol.Variable(name="data")
    data = data - 127.5
    data = data * 0.0078125
    if config.fp16:
        data = mx.sym.Cast(data=data, dtype=np.float16)
    #blocks = config.net_blocks
    blocks = [1,4,6,2,2]
    channels = [32,32,32,64,64,128]
    conv_1 = Conv(data,
                  num_filter=channels[0],
                  kernel=(3, 3),
                  pad=(1, 1),
                  stride=(2, 2),
                  name="conv_1")   # 56
    if blocks[0] == 1:
        conv_2_dw = Conv(conv_1,
                         num_group=channels[1],
                         num_filter=channels[1],
                         kernel=(3, 3),
                         pad=(1, 1),
                         stride=(1, 1),
                         name="conv_2_dw")
    else:
        conv_2_dw = Residual(conv_1,
                             num_block=blocks[0],
                             num_out=channels[1],
                             kernel=(3, 3),
                             stride=(1, 1),
                             pad=(1, 1),
                             num_group=channels[1],
                             name="res_2")
    conv_23 = DResidual(conv_2_dw,
                        dim_match=0,
                        num_in=channels[1],
                        num_out=channels[2],
                        kernel=(3, 3),
                        stride=(2, 2),
                        pad=(1, 1),
                        num_group=channels[2]*2,
                        name="dconv_23")   # 28
    slice_index = [[0,16],[12,28]]
    layers = []
    for i in range(4):
        x = i // 2
        y = i % 2
        data = mx.sym.slice(conv_23,
                            begin=(None,None,slice_index[x][0],slice_index[y][0]),
                            end = (None,None,slice_index[x][1],slice_index[y][1])
                            )
        conv_3 = Residual(data,
                          num_block=blocks[1],
                          num_out=channels[2],
                          kernel=(3, 3),
                          stride=(1, 1),
                          pad=(1, 1),
                          num_group=channels[2]*2,
                          name="res_3",
                          suffix=i)
        conv_34 = DResidual(conv_3,
                            dim_match=0,
                            num_in=channels[2],
                            num_out=channels[3],
                            kernel=(3, 3),
                            stride=(2, 2),
                            pad=(1, 1),
                            num_group=channels[3]*2,
                            name="dconv_34",
                            suffix=i)   # 14
        conv_4 = Residual(conv_34,
                          num_block=blocks[2],
                          num_out=channels[3],
                          kernel=(3, 3),
                          stride=(1, 1),
                          pad=(1, 1),
                          num_group=channels[3]*2,
                          name="res_4",
                          suffix=i)   
        conv_45 = DResidual(conv_4,
                            dim_match=0,
                            num_in=channels[3],
                            num_out=channels[4],
                            kernel=(3, 3),
                            stride=(2, 2),
                            pad=(1, 1),
                            num_group=channels[4]*4,
                            name="dconv_45",
                            suffix=i)  # 7
        conv_5 = Residual(conv_45,
                          num_block=blocks[3],
                          num_out=channels[4],
                          kernel=(3, 3),
                          stride=(1, 1),
                          pad=(1, 1),
                          num_group=channels[4]*2,
                          name="res_5",
                          suffix=i)
        layers.append(conv_5)
    s1,s2,s3,s4 = layers
    s1 = mx.sym.concat(s1,s2,dim=3)
    s2 = mx.sym.concat(s3,s4,dim=3)
    conv_5 = mx.sym.concat(s1,s2,dim=2)
    
    conv_56 = DResidual(conv_5,
                            dim_match=0,
                            num_in=channels[4],
                            num_out=channels[5],
                            kernel=(3, 3),
                            stride=(2, 2),
                            pad=(1, 1),
                            num_group=channels[4]*4,
                            name="dconv_45")  # 4
    
    conv_6 = FcResidual(conv_56,
                        num_block=blocks[4],
                        num_out=channels[5],
                        H=4,
                        W=4,
                        name="fc_conv")
    fc1 = symbol_utils.get_fc1(conv_6, num_classes, fc_type)
    fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    return fc1