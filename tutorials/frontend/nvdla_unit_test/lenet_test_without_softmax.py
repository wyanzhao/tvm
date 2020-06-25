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
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing

dtype="float32"
data_shape = (1, 1, 28, 28)
weight_shape = (20, 1, 5, 5)
x = relay.var("x",
                     shape=data_shape,
                     dtype=dtype)
w1 = relay.var("w1", shape=weight_shape, dtype=dtype)
b1 = relay.var("b1", shape=(20, ), dtype=dtype)

network = relay.nn.nvdla_conv2d_bias(x, w1, b1, strides=(1, 1,), padding=(0, 0), dilation=(1, 1), groups=1, channels=20, data_layout="NCHW")
network = relay.nn.max_pool2d(network, pool_size=(2, 2), strides=(2, 2))

w2 = relay.var("w2", shape=(50, 20, 5, 5), dtype=dtype)
b2 = relay.var("b2", shape=(50, ), dtype=dtype)
network = relay.nn.nvdla_conv2d_bias(network, w2, b2, strides=(1, 1,), padding=(0, 0), dilation=(1, 1), groups=1, channels=50, data_layout="NCHW")
network = relay.nn.max_pool2d(network, pool_size=(2, 2), strides=(2, 2))

network = relay.nn.batch_flatten(network)

w3 = relay.var("w3", shape=(500, 800), dtype=dtype)
b3 = relay.var("b3", shape=(500, ), dtype=dtype)
network = relay.nn.nvdla_fc(network, w3, b3, units=500)
network = relay.nn.relu(network)

w4 = relay.var("w4", shape=(10, 500), dtype=dtype)
b4 = relay.var("b4", shape=(10, ), dtype=dtype)
network = relay.nn.nvdla_fc(network, w4, b4, units=10)

func = relay.Function([x, w1, b1, w2, b2, w3, b3, w4, b4], network)

from nvdla.nvdla_utils import nvdla_analyze_compute_graph, set_nvdla_config
nvdla_analyze_compute_graph(func, 9, [0, 1])

nv_config = "nv_small"
nv_precision = "int8"

set_nvdla_config(nv_config, nv_precision)

with open("lenet-test/quantize_weight.json", "r") as f:
    import json
    scale_json = json.load(f)
from nvdla.nvdla_utils import nvdla_graph_info

nvdla_graph_info['op_infos'][14]['input_index'][0] = 12
nvdla_graph_info['scale_info'] = scale_json


w1_data = np.load("lenet-test/tensor.npy")
b1_data = np.load("lenet-test/tensor1.npy")
w2_data = np.load("lenet-test/tensor2.npy")
b2_data = np.load("lenet-test/tensor3.npy")
w3_data = np.load("lenet-test/tensor4.npy")
b3_data = np.load("lenet-test/tensor5.npy")
w4_data = np.load("lenet-test/tensor6.npy")
b4_data = np.load("lenet-test/tensor7.npy")

#NVDLA backend
params = {'w1': tvm.nd.array(w1_data), "b1": tvm.nd.array(b1_data), 
"w2": tvm.nd.array(w2_data),
"b2": tvm.nd.array(b2_data),
"w3": tvm.nd.array(w3_data),
"b3": tvm.nd.array(b3_data),
"w4": tvm.nd.array(w4_data), 
"b4": tvm.nd.array(b4_data),}

#target = tvm.target.nvdla(options=["-debug"])
target = tvm.target.nvdla(options=[])

with relay.build_config(opt_level=0, disabled_pass=["AlterOpLayout", "SimplifyInference"]):
    graph, lib, params = relay.build(func, target, params=params)

#print(lib.get_source())

module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    # set input and parameters
x_data = np.random.uniform(size=data_shape).astype("float32")
module.set_input('x', tvm.nd.array(x_data))
module.set_input(**params)
# run
module.run()