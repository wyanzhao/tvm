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
"""
Using External Libraries in Relay
=================================
**Author**: `Masahiro Masuda <https://github.com/masahi>`_, `Truman Tian <https://github.com/SiNZeRo>`_

This is a short tutorial on how to use external libraries such as cuDNN, or cuBLAS with Relay.

Relay uses TVM internally to generate target specific code. For example, with cuda backend TVM generates cuda kernels for all layers in the user provided network.
But sometimes it is also helpful to incorporate external libraries developed by various vendors into Relay.
Luckily, TVM has a mechanism to transparently call into these libraries.
For Relay users, all we need to do is just to set a target string appropriately.
ls
Before we can use external libraries from Relay, your TVM needs to be built with libraries you want to use.
For example, to use cuDNN, USE_CUDNN option in `cmake/config.cmake` needs to be enabled, and cuDNN include and library directories need to be specified if necessary.

To begin with, we import Relay and TVM.
"""
import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing

from tvm.nvdla_utils import nvdla_graph_info

dtype="float32"
data_shape = (1, 1, 4, 3)
weight_shape = (2, 1, 1, 1)
x = relay.var("x",
                     shape=data_shape,
                     dtype=dtype)
w = relay.var("w", shape=weight_shape, dtype=dtype)

network = relay.nn.conv2d(x, w, strides=(1, 1,), padding=(0, 0), dilation=(1, 1), groups=1, channels=2, data_layout="NCHW")
network = relay.nn.max_pool2d(network, pool_size=(1, 2))

func = relay.Function([x, w], network)
print(func)

from tvm.autotvm.graph_tuner.utils import has_multiple_inputs, get_direct_ancestor, get_in_nodes, get_out_nodes, expr2graph, bind_inputs
node_list = []
node_dict = {}
target_ops = []
expr2graph(func, target_ops, node_dict, node_list)
    
global nvdla_graph_info
nvdla_graph_info["input_op"] = [node_list[2], node_list[0], node_list[1]]
for x in node_list:
        node = {}
        if len(x['inputs']) != 0:
            node['name'] = x['op']
            node['node'] = x['node']
            node['input_shapes'] = []
            node['input_index'] = []
            node['op_shape'] = [int(x) for x in x['types'][0].shape]

            for y in x['inputs']:
                input_index = y[0]
                shape = [int(x) for x in node_list[input_index]['types'][0].shape]
                if node_list[input_index]['op'] == 'null':
                    node['is_const'] = True
                else:
                    node['is_const'] = False
                node['input_shapes'].append(shape)
                node['input_index'].append(input_index)
            
            if nvdla_graph_info['op_maps'].get(x['op']) != None:
                nvdla_graph_info['op_maps'][x['op']].append(node)
            else:
                print('Unsupport Op:{}'.format(x['op']))
        else:
            node['name'] = x['op']
            node['node'] = x['node']
            node['op_shape'] = [int(x) for x in x['types'][0].shape]
            node['input_shapes'] = []
            node['input_index'] = []
        
        nvdla_graph_info['op_infos'].append(node)
nvdla_graph_info["output_op"] = node_list[-1]

data = np.random.uniform(size=data_shape).astype(dtype)
kernel = np.array([
    [
        [
            [
                1
            ]
        ]
    ],
    [
        [
            [
                3
            ]
        ]
    ]
]
, dtype="float32")

params = {'w': tvm.nd.array(kernel)}

#target = tvm.target.nvdla(options=["-debug"])
target = tvm.target.nvdla(options=[])

with relay.build_config(opt_level=0, disabled_pass=["AlterOpLayout", "SimplifyInference"]):
    graph, lib, params = relay.build(func, target, params=params)

print(lib.get_source())

module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    # set input and parameters
x_data = np.random.uniform(size=data_shape).astype("float32")
module.set_input('x', tvm.nd.array(data))
module.set_input(**params)
# run
module.run()