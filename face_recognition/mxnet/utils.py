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

def decode_module(data,num_filter,dim_match,name,expand_dim,**kwargs):
    workspace = kwargs.get('workspace',256)
    if dim_match:
        decode1 = Conv(data=data,
                        num_filter=num_filter*expand_dim,
                        kernel=(1,1),
                        stride=(1,1),
                        pad=(0,0),
                        no_bias=False,
                        num_group = num_filter,
                        workspace=workspace,
                        name=name + '_decode1')
        decode_act1 = Act(data=decode1, act_type=act_type, name=name + 'decode_relu1')
        decode2 = Conv(data=decode_act1,
                        num_filter=num_filter*expand_dim,
                        kernel=(1,1),
                        stride=(1,1),
                        pad=(0,0),
                        no_bias=False,
                        num_group=num_filter*expand_dim,
                        workspace=workspace,
                        name=name+'_docode2')
        decode_act2 = Act(data=decode2, act_type=act_type, name=name+'decode_relu2')
        decode = Conv(data=decode_act2,
                        num_filter=num_filter,
                        kernel=(1,1),
                        pad=(0,0),
                        no_bias=False,
                        num_group=num_filter,
                        workspace=workspace,
                        name=name+'_decode3')  
    else:
        decode = data
    return decode

def spartial_attention_module(data,num_filter,**kwargs):
    body = Conv(data=data,
                num_filter=num_filter,
                kernel=(3,3),
                stride=(1,1),
                pad=(1,1),
                num_group = num_filter,
                name=name + "_sam_conv",
                workspace=workspace)
    body = mx.symbol.Activation(data=body,
                                act_type='sigmoid',
                                name=name+ "_sam_sigmoid")
    data = data * body
    return data
    
        