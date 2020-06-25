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

func = relay.Function([x, w1, b1], network)

from nvdla.nvdla_utils import nvdla_analyze_compute_graph, set_nvdla_config
nvdla_analyze_compute_graph(func, 3, [0, 1])

nv_config = "nv_medium_1024"
nv_precision = "int8"

set_nvdla_config(nv_config, nv_precision)

with open("/home/dev/lenet-test/quantize_weight_test.json", "r") as f:
    import json
    scale_json = json.load(f)
from nvdla.nvdla_utils import nvdla_graph_info

nvdla_graph_info['scale_info'] = scale_json

w1_data = np.load("/home/dev/lenet-test/tensor.npy")
np.savetxt("foo.txt", w1_data.reshape((500,)), delimiter=",", newline=",")

b1_data = np.load("/home/dev/lenet-test/tensor1.npy")

#NVDLA backend
params = {'w1': tvm.nd.array(w1_data), "b1": tvm.nd.array(b1_data)}

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