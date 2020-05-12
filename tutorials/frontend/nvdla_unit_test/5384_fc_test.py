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
data_shape = (1, 1, 5, 7)
weight_shape = (1, 1, 1, 1)
x = relay.var("x",
                     shape=data_shape,
                     dtype=dtype)
w = relay.var("w", shape=weight_shape, dtype=dtype)

network = relay.nn.nvdla_conv2d(x, w, strides=(1, 1,), padding=(0, 0), dilation=(1, 1), groups=1, channels=1, data_layout="NCHW")
network = relay.nn.batch_flatten(network)

w1 = relay.var("w1", shape=(4, 35))
bias = relay.var("bias", shape=(4, ), dtype=dtype)

network = relay.nn.nvdla_fc(network, weight=w1, bias=bias, units=4, out_dtype=dtype)

func = relay.Function([x, w, w1, bias], network)

# Reference result
network_ref = relay.nn.conv2d(x, w, strides=(1, 1,), padding=(0, 0), dilation=(1, 1), groups=1, channels=1, data_layout="NCHW")
network_ref = relay.nn.batch_flatten(network_ref)
network_ref = relay.nn.dense(network, weight=w1, units=4, out_dtype=dtype)
network_ref = relay.nn.bias_add(network_ref, bias)
func_ref = relay.Function([x, w, w1, bias], network_ref)

from tvm.autotvm.graph_tuner.utils import has_multiple_inputs, get_direct_ancestor, get_in_nodes, get_out_nodes, expr2graph, bind_inputs
node_list = []
node_dict = {}
target_ops = []
expr2graph(func, target_ops, node_dict, node_list)
    
global nvdla_graph_info
nvdla_graph_info["input_op"] = [node_list[4], node_list[0], node_list[1]]
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
                if x['op'] == 'nvdla_conv2d':
                    nvdla_graph_info['op_maps']['conv2d'].append(node)
                else:
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
nvdla_graph_info['op_infos'][6]['input_index'][0] = 4

kernel = np.array([
    [
        [
            [
                1
            ]
        ]
    ]
]
, dtype="float32")
fc_kernel = np.reshape(np.array(
    [
    [
        3,
        3,
        3,
        2
    ],
    [
        3,
        1,
        3,
        2
    ],
    [
        3,
        3,
        1,
        1
    ],
    [
        3,
        2,
        1,
        3
    ],
    [
        1,
        3,
        2,
        3
    ],
    [
        3,
        4,
        4,
        2
    ],
    [
        4,
        3,
        2,
        2
    ],
    [
        3,
        2,
        1,
        4
    ],
    [
        3,
        3,
        3,
        3
    ],
    [
        1,
        4,
        2,
        1
    ],
    [
        4,
        1,
        1,
        2
    ],
    [
        3,
        2,
        1,
        2
    ],
    [
        1,
        1,
        3,
        3
    ],
    [
        4,
        3,
        1,
        2
    ],
    [
        1,
        3,
        2,
        4
    ],
    [
        2,
        1,
        3,
        1
    ],
    [
        4,
        1,
        4,
        2
    ],
    [
        4,
        3,
        3,
        1
    ],
    [
        2,
        4,
        3,
        2
    ],
    [
        4,
        4,
        3,
        3
    ],
    [
        3,
        3,
        1,
        4
    ],
    [
        4,
        2,
        1,
        2
    ],
    [
        4,
        1,
        3,
        4
    ],
    [
        4,
        1,
        4,
        4
    ],
    [
        3,
        2,
        1,
        4
    ],
    [
        2,
        2,
        2,
        3
    ],
    [
        3,
        2,
        1,
        1
    ],
    [
        2,
        3,
        3,
        2
    ],
    [
        1,
        4,
        1,
        4
    ],
    [
        1,
        1,
        2,
        4
    ],
    [
        1,
        2,
        2,
        4
    ],
    [
        3,
        4,
        2,
        2
    ],
    [
        2,
        1,
        1,
        4
    ],
    [
        4,
        1,
        2,
        4
    ],
    [
        3,
        3,
        2,
        2
    ]
], dtype=dtype
), (4, 35))
bias_ = np.array([
    3,
    2,
    3,
    2
], dtype=dtype)

#NVDLA backend
params = {'w': tvm.nd.array(kernel), "w1": tvm.nd.array(fc_kernel), "bias": tvm.nd.array(bias_)}

#target = tvm.target.nvdla(options=["-debug"])
target = tvm.target.nvdla(options=[])

with relay.build_config(opt_level=0, disabled_pass=["AlterOpLayout", "SimplifyInference"]):
    graph, lib, params = relay.build(func, target, params=params)

print(lib.get_source())

module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    # set input and parameters
x_data = np.random.uniform(size=data_shape).astype("float32")
module.set_input('x', tvm.nd.array(x_data))
module.set_input(**params)
# run
module.run()

params = {'w': tvm.nd.array(kernel), "w1": tvm.nd.array(fc_kernel), "bias": tvm.nd.array(bias_)}

target = "llvm"

with relay.build_config(opt_level=0, disabled_pass=["AlterOpLayout", "SimplifyInference"]):
    graph, lib, params = relay.build(func_ref, target, params=params)

module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    # set input and parameters
def read_pgm(filename, byteorder='>'):
    import re
    import numpy
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

x_data = read_pgm("/home/dev/nvdla-test/single_layer_test/Gemm/5384_Gemm.pgm")
x_data = np.array(x_data, dtype="float32")
x_data = np.reshape(x_data, (1, 1, 5, 7))
module.set_input('x', tvm.nd.array(x_data))
module.set_input(**params)
# run
module.run()
out = module.get_output(0)
print("Ref result:{}".format(out))

