import sys
import os
import mxnet as mx
import numpy as np

def Conv(**kwargs):
    body = mx.sym.Convolution(**kwargs)
    return body

def Act(data, act_type, name):
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body
    
def ConvBNReLU(data,
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
                          momentum=2e-5)
    act = Act(data=bn,
              act_type='prelu',
              name='%s%s_relu' % (name, suffix))
    return act   

def ConvBN(data,
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
                          momentum=2e-5)
    return bn    

def ReLU6(data):
    return mx.sym.clip(data,0,6)

def decode_module(data,num_filter,dim_match,name,expand_dim,**kwargs):
    if dim_match:
        decode1 = Conv(data=data,
                        num_filter=num_filter*expand_dim,
                        kernel=(1,1),
                        stride=(1,1),
                        pad=(0,0),
                        no_bias=False,
                        num_group = num_filter,
                        name=name + '_decode1')
        decode_act1 = Act(data=decode1, act_type='prelu', name=name + 'decode_relu1')
        decode2 = Conv(data=decode_act1,
                        num_filter=num_filter*expand_dim,
                        kernel=(1,1),
                        stride=(1,1),
                        pad=(0,0),
                        no_bias=False,
                        num_group=num_filter*expand_dim,
                        name=name+'_docode2')
        decode_act2 = Act(data=decode2, act_type='prelu', name=name+'decode_relu2')
        decode = Conv(data=decode_act2,
                        num_filter=num_filter,
                        kernel=(1,1),
                        pad=(0,0),
                        no_bias=False,
                        num_group=num_filter,
                        name=name+'_decode3')                        
    else:
        decode = data
    return decode

def decode_expand_sub_module(data,num_filter,expand_dim,name,sub):
    decode1 = Conv(data=data,
                    num_filter=num_filter,
                    kernel=(1,1),
                    stride=(1,1),
                    pad=(0,0),
                    no_bias=False,
                    num_group=num_filter,
                    name=name+'_decode1_%d' % (sub),
                    )
    decode_act1 = ReLU6(data=decode1)
    decode_expand = Conv(data=decode_act1,
                        num_filter=num_filter*expand_dim,
                        kernel=(1,1),
                        stride=(1,1),
                        pad=(0,0),
                        no_bias=True,
                        num_group=1,
                        name=name+'_decode_expand_%d' % (sub),
                    )
    bn = mx.sym.BatchNorm(data=decode_expand,
                            fix_gamma=False,
                            eps=2e-5,
                            momentum=bn_mom,
                            name=name+'_decode_bn_%d' % (sub)
                            )
    act2 = Act(data=bn, act_type='prelu',name=name+'decode_relu_%d' % (sub))
    return act2


def decode_expand_module(data,num_filter,expand_dim,name):
    # set default sub factor to 3
    decode0 = decode_expand_sub_model(data=data,
                                        num_filter=num_filter,
                                        expand_dim=expand_dim,
                                        name=name,
                                        sub=0)
    decode1 = decode_expand_sub_module(data=data,
                                        num_filter=num_filter,
                                        expand_dim=expand_dim,
                                        name=name,
                                        sub=1)
    decode2 = decode_expand_sub_module(data=data,
                                        num_filter=num_filter,
                                        expand_dim=expand_dim,
                                        name=name,
                                        sub=2)
    decode = decode0+decode1+decode2
    return decode       


def spartial_attention_module(data,num_filter,name,**kwargs):
    body = Conv(data=data,
                num_filter=num_filter,
                kernel=(3,3),
                stride=(1,1),
                pad=(1,1),
                num_group = num_filter,
                name=name + "_sam_conv")
    body = mx.symbol.Activation(data=body,
                                act_type='sigmoid',
                                name=name+ "_sam_sigmoid")
    data = data * body
    return data
    
def se_module(data,num_filter,name,**kwargs):
    body = mx.sym.Pooling(data=data,
                            global_pool=True,
                            kernel=(7,7),
                            pool_type='avg',
                            name=name + 'se_pool1')
    body = Conv(data=body,
                num_filter=num_filter//16,
                kernel=(1,1),
                stride=(1,1),
                pad=(0,0),
                name=name+"_se_conv1")
    body = Act(data=body, act_type='prelu', name=name + '_se_relu1')
    body = Conv(data=body,
                num_filter=num_filter,
                kernel=(1, 1),
                stride=(1, 1),
                pad=(0, 0),
                name=name + "_se_conv2")
    body = mx.symbol.Activation(data=body,
                                act_type='sigmoid',
                                name=name + "_se_sigmoid")
    data = mx.symbol.broadcast_mul(data, body) 
    
    return data
    
def DResidual_Decode(data,
              dim_match=0,
              num_in=1,
              num_out=1,
              kernel=(3, 3),
              stride=(2, 2),
              pad=(1, 1),
              num_group=1,
              name=None,
              suffix=''):
    use_spa = False
    use_se = False
    use_decode=False
    use_mixconv = False
              
              
    sep = ConvBNReLU(data=data,
                num_filter=num_group,
                kernel=(1, 1),
                pad=(0, 0),
                stride=(1, 1),
                name='%s%s_conv_sep' % (name, suffix))
    if use_decode:
        sep_decode = ConvBNReLU(data=sep,
                                num_filter=num_group,
                                kernel=(1,1),
                                pad=(0,0),
                                stride=(1,1),
                                num_group=num_group//8,
                                name="%s%s_conv_sep_decode" %(name, suffix))
    else:
        sep_decode = sep
    
    if use_mixconv:
        sep1,sep2,sep3,sep4 = mx.sym.split(sep_decode,axis=1,num_outputs=4)
        sep3 = mx.sym.concat(sep3,sep4,dim=1)
        conv_dw1 = ConvBNReLU(data=sep1,
                       num_filter=num_group//4,
                       num_group=num_group//4,
                       kernel=(3,3),
                       pad=(1,1),
                       stride=stride,
                       name='%s%s_conv_dw1' % (name, suffix))
        conv_dw2 = ConvBNReLU(data=sep2,
                       num_filter=num_group//4,
                       num_group=num_group//4,
                       kernel=(7,7),
                       pad=(3,3),
                       stride=stride,
                       name='%s%s_conv_dw2' % (name, suffix))
        conv_dw3 = ConvBNReLU(data=sep2,
                       num_filter=num_group//2,
                       num_group=num_group//2,
                       kernel=(3,3),
                       pad=(1,1),
                       stride=stride,
                       name='%s%s_conv_dw2' % (name, suffix))
        conv_dw = mx.sym.concat(conv_dw1,conv_dw2,conv_dw3,dim=1)
    else:
        conv_dw = ConvBNReLU(data=sep_decode,
                            num_filter=num_group,
                            num_group=num_group,
                            kernel=kernel,
                            pad=pad,
                            stride=stride,
                            name="%s%s_conv_dw" % (name,suffix))
    if use_se:
        conv_dw = se_module(data=conv_dw,num_filter=num_group,name=name+suffix)
    proj = ConvBN(data=conv_dw,
                  num_filter=num_out,
                  kernel=(1, 1),
                  pad=(0, 0),
                  stride=(1, 1),
                  name='%s%s_conv_proj' % (name, suffix))
                  
    if use_spa:
        proj = spartial_attention_module(data=proj,num_filter=num_out,name=name+suffix)
    return proj  

def Residual_Decode(data,
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
        conv = DResidual_Decode(data=identity,
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

def FcResidual(data,
                num_block=1,
                num_out=1,
                H = 1,
                W = 1,
                name=None):
    HW = H*W
    identity = mx.sym.reshape(data,shape=(0,-1))
    for i in range(num_block):
        # try x + k*x^2
        id_r = identity.reshape_like(data)  # [N,C,H,W]
        _weight_sq = mx.symbol.Variable("%s%s_fcconv_sq" %(name,i),
                                    shape=(1,num_out,H,W),
                                    init=mx.init.Normal(0.01))
        id_r = id_r + mx.sym.broadcast_mul(id_r*id_r,_weight_sq)
        id_r = mx.sym.BatchNorm(data=id_r,
                          name='%s%s_fcconv_sq_batchnorm' % (name, i),
                          fix_gamma=False,
                          momentum=2e-5)
        id_r = id_r.reshape_like(identity)
        
        # first FC+Relu
        conv =  mx.sym.FullyConnected(data=id_r,
                                    num_hidden=num_out*HW,
                                    name='%s%s_fcconv' % (name,i))
        relu = Act(data=conv, act_type='prelu', name='%s%s_fcconv_relu' % (name,i))
        
        # group FC with group = 'HW'
        relu = relu.reshape_like(data).transpose(axes=(0,2,3,1)).reshape(shape=(0,-3,-4,1,-1))  # [N,HW,1,C]
        body = mx.sym.broadcast_to(relu,shape=(0,0,num_out,0))  # [N,HW,C,C]
        _weight = mx.symbol.Variable("%s%s_group_fc_weight" % (name,i),
                                shape=(HW,num_out,num_out),
                                init=mx.init.Normal(0.01))  # [HW,C,C]
        body = mx.sym.broadcast_mul(body,_weight.reshape(shape=(-4,1,-1,-2))) #[N,HW,C,C]
        body = mx.sym.sum(body,axis=3,keepdims=False).transpose(axes=(0,2,1)) #[N,C,HW]
        body = body.reshape_like(identity)
        
        identity = identity + body
        
    return identity