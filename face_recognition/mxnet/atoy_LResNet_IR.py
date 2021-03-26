# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import mxnet as mx
import numpy as np
import symbol_utils
#import memonger
#import sklearn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from default import config

from utils import decode_module,spartial_attetion_module


def Conv(**kwargs):
    #name = kwargs.get('name')
    #_weight = mx.symbol.Variable(name+'_weight')
    #_bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
    #body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
    body = mx.sym.Convolution(**kwargs)
    return body


def Act(data, act_type, name):
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.symbol.Activation(data=data, act_type=act_type, name=name)
    return body


def residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck,
                     **kwargs):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    use_se = kwargs.get('version_se', 1)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    memonger = kwargs.get('memonger', False)
    act_type = kwargs.get('version_act', 'prelu')
    #print('in unit3')

    use_decode = True
    use_spa = True

    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name=name + '_bn1')
        conv1 = Conv(data=bn1,
                     num_filter=int(num_filter * 0.25),
                     kernel=(1, 1),
                     stride=(1, 1),
                     pad=(0, 0),
                     no_bias=True,
                     workspace=workspace,
                     name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name=name + '_bn2')
        act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')

        conv2 = Conv(data=act1,
                     num_filter=int(num_filter * 0.25),
                     kernel=(3, 3),
                     stride=(1, 1),
                     pad=(1, 1),
                     no_bias=True,
                     workspace=workspace,
                     name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name=name + '_bn3')
        act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2')
        
        if use_spa:
            act2 = spartial_attention_module(data=act2,
                                num_filter=int(num_filter*0.25),
                                name=name,
                                **kwargs,
                                )
            
        conv3 = Conv(data=act2,
                     num_filter=num_filter,
                     kernel=(1, 1),
                     stride=stride,
                     pad=(0, 0),
                     no_bias=True,
                     workspace=workspace,
                     name=name + '_conv3')
        bn4 = mx.sym.BatchNorm(data=conv3,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name=name + '_bn4')


        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data,
                           num_filter=num_filter,
                           kernel=(1, 1),
                           stride=stride,
                           no_bias=True,
                           workspace=workspace,
                           name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc,
                                        fix_gamma=False,
                                        eps=2e-5,
                                        momentum=bn_mom,
                                        name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return bn4 + shortcut
    else:
        if use_decode:
            data = decode_module(data=data,
                        num_filter=num_filter,
                        dim_match=dim_match,
                        name=name,
                        expand_dim=4,
                        **kwargs,
                    )
        bn1 = mx.sym.BatchNorm(data=data,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name=name + '_bn1')
        conv1 = Conv(data=bn1,
                     num_filter=num_filter,
                     kernel=(3, 3),
                     stride=(1, 1),
                     pad=(1, 1),
                     no_bias=True,
                     workspace=workspace,
                     name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name=name + '_bn2')
        act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
        conv2 = Conv(data=act1,
                     num_filter=num_filter,
                     kernel=(3, 3),
                     stride=stride,
                     pad=(1, 1),
                     no_bias=True,
                     workspace=workspace,
                     name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2,
                               fix_gamma=False,
                               eps=2e-5,
                               momentum=bn_mom,
                               name=name + '_bn3')
        act2 = Act(data=bn3, act_type=act_type, name=name+ '_relu2')

        if use_spa:
            act2 = spartial_attention_module(data=act2,
                        num_filter=num_filter,
                        name=name,
                        **kwargs,
                    )

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data=data,
                           num_filter=num_filter,
                           kernel=(1, 1),
                           stride=stride,
                           no_bias=True,
                           workspace=workspace,
                           name=name + '_conv1sc')
            shortcut = mx.sym.BatchNorm(data=conv1sc,
                                        fix_gamma=False,
                                        momentum=bn_mom,
                                        eps=2e-5,
                                        name=name + '_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return bn3 + shortcut


def atoy_LResNet_IR(units, num_stages, filter_list, num_classes, bottle_neck):
    bn_mom = config.bn_mom
    workspace = config.workspace
    kwargs = {
        'version_se': config.net_se,
        'version_input': config.net_input,
        'version_output': config.net_output,
        'version_unit': config.net_unit,
        'version_act': config.net_act,
        'bn_mom': bn_mom,
        'workspace': workspace,
        'memonger': config.memonger,
    }
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    version_input = kwargs.get('version_input', 1)
    assert version_input >= 0
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output
    version_unit = 3
    act_type = kwargs.get('version_act', 'prelu')
    memonger = kwargs.get('memonger', False)
    print(version_se, version_input, version_output, version_unit, act_type,
          memonger)
    num_unit = len(units)
    assert (num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    if config.fp16:
        data = mx.sym.Cast(data=data, dtype=np.float16)
    if version_input == 0:
        #data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
        data = mx.sym.identity(data=data, name='id')
        data = data - 127.5
        data = data * 0.0078125
        body = Conv(data=data,
                    num_filter=filter_list[0],
                    kernel=(7, 7),
                    stride=(2, 2),
                    pad=(3, 3),
                    no_bias=True,
                    name="conv0",
                    workspace=workspace)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn0')
        body = Act(data=body, act_type=act_type, name='relu0')
        #body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    elif version_input == 2:
        data = mx.sym.BatchNorm(data=data,
                                fix_gamma=True,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn_data')
        body = Conv(data=data,
                    num_filter=filter_list[0],
                    kernel=(3, 3),
                    stride=(1, 1),
                    pad=(1, 1),
                    no_bias=True,
                    name="conv0",
                    workspace=workspace)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn0')
        body = Act(data=body, act_type=act_type, name='relu0')
    else:
        data = mx.sym.identity(data=data, name='id')
        data = data - 127.5
        data = data * 0.0078125
        body = data
        body = Conv(data=body,
                    num_filter=filter_list[0],
                    kernel=(3, 3),
                    stride=(1, 1),
                    pad=(1, 1),
                    no_bias=True,
                    name="conv0",
                    workspace=workspace)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bn0')
        body = Act(data=body, act_type=act_type, name='relu0')

    for i in range(num_stages):
        #if version_input==0:
        #  body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
        #                       name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
        #else:
        #  body = residual_unit(body, filter_list[i+1], (2, 2), False,
        #    name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
        if i==num_stages-1 and config.fp16:
            body = mx.sym.Cast(data=body, dtype=np.float32)
        body = residual_unit(body,
                             filter_list[i + 1], (2, 2),
                             False,
                             name='stage%d_unit%d' % (i + 1, 1),
                             bottle_neck=bottle_neck,
                             **kwargs)
        for j in range(units[i] - 1):
            body = residual_unit(body,
                                 filter_list[i + 1], (1, 1),
                                 True,
                                 name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck,
                                 **kwargs)

    if bottle_neck:
        body = Conv(data=body,
                    num_filter=512,
                    kernel=(1, 1),
                    stride=(1, 1),
                    pad=(0, 0),
                    no_bias=True,
                    name="convd",
                    workspace=workspace)
        body = mx.sym.BatchNorm(data=body,
                                fix_gamma=False,
                                eps=2e-5,
                                momentum=bn_mom,
                                name='bnd')
        body = Act(data=body, act_type=act_type, name='relud')

    fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
    fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
    return fc1


def get_symbol():
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    num_classes = config.embedding_size
    num_layers = config.num_layers
    if num_layers >= 500:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stages = 4
    if num_layers == 18:
        units = [2, 2, 2, 2]
    elif num_layers == 34:
        units = [3, 4, 6, 3]
    elif num_layers == 49:
        units = [3, 4, 14, 3]
    elif num_layers == 50:
        units = [3, 4, 14, 3]
    elif num_layers == 74:
        units = [3, 6, 24, 3]
    elif num_layers == 90:
        units = [3, 8, 30, 3]
    elif num_layers == 98:
        units = [3, 4, 38, 3]
    elif num_layers == 99:
        units = [3, 8, 35, 3]
    elif num_layers == 100:
        units = [3, 13, 30, 3]
    elif num_layers == 134:
        units = [3, 10, 50, 3]
    elif num_layers == 136:
        units = [3, 13, 48, 3]
    elif num_layers == 140:
        units = [3, 15, 48, 3]
    elif num_layers == 124:
        units = [3, 13, 40, 5]
    elif num_layers == 160:
        units = [3, 24, 49, 3]
    elif num_layers == 101:
        units = [3, 4, 23, 3]
    elif num_layers == 152:
        units = [3, 8, 36, 3]
    elif num_layers == 200:
        units = [3, 24, 36, 3]
    elif num_layers == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError(
            "no experiments done on num_layers {}, you can do it yourself".
            format(num_layers))

    net = atoy_LResNet_IR(units=units,
                 num_stages=num_stages,
                 filter_list=filter_list,
                 num_classes=num_classes,
                 bottle_neck=bottle_neck)

    if config.memonger:
        dshape = (config.per_batch_size, config.image_shape[2],
                  config.image_shape[0], config.image_shape[1])
        net_mem_planned = memonger.search_plan(net, data=dshape)
        old_cost = memonger.get_cost(net, data=dshape)
        new_cost = memonger.get_cost(net_mem_planned, data=dshape)

        print('Old feature map cost=%d MB' % old_cost)
        print('New feature map cost=%d MB' % new_cost)
        net = net_mem_planned
    return net
