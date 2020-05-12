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

import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_runtime as runtime
from tvm import relay
from tvm.relay import testing

from nvdla.nvdla_utils import nvdla_graph_info

dtype="float32"
data_shape = (1, 1, 28, 28)
weight_shape = (20, 1, 5, 5)
x = relay.var("x",
                     shape=data_shape,
                     dtype=dtype)
w1 = relay.var("w1", shape=weight_shape, dtype=dtype)
b1 = relay.var("b1", shape=(20, ), dtype=dtype)

network = relay.nn.conv2d(x, w1, strides=(1, 1,), padding=(0, 0), dilation=(1, 1), groups=1, channels=20, data_layout="NCHW")
network = relay.nn.bias_add(network, b1)
network = relay.nn.max_pool2d(network, pool_size=(2, 2), strides=(2, 2))

w2 = relay.var("w2", shape=(50, 20, 5, 5), dtype=dtype)
b2 = relay.var("b2", shape=(50, ), dtype=dtype)
network = relay.nn.conv2d(network, w2, strides=(1, 1,), padding=(0, 0), dilation=(1, 1), groups=1, channels=50, data_layout="NCHW")
network = relay.nn.bias_add(network, b2)
network = relay.nn.max_pool2d(network, pool_size=(2, 2), strides=(2, 2))

network = relay.nn.batch_flatten(network)

w3 = relay.var("w3", shape=(500, 800), dtype=dtype)
b3 = relay.var("b3", shape=(500, ), dtype=dtype)
network = relay.nn.dense(network, w3, units=500)

network = relay.nn.bias_add(network, b3)
network = relay.nn.relu(network)

w4 = relay.var("w4", shape=(10, 500), dtype=dtype)
b4 = relay.var("b4", shape=(10, ), dtype=dtype)
network = relay.nn.dense(network, w4, units=10)
network = relay.nn.bias_add(network, b4)

func = relay.Function([x, w1, b1, w2, b2, w3, b3, w4, b4], network)

w1_data = np.load("/home/dev/lenet-test/tensor.npy")
b1_data = np.load("/home/dev/lenet-test/tensor1.npy")
w2_data = np.load("/home/dev/lenet-test/tensor2.npy")
b2_data = np.load("/home/dev/lenet-test/tensor3.npy")
w3_data = np.load("/home/dev/lenet-test/tensor4.npy")
b3_data = np.load("/home/dev/lenet-test/tensor5.npy")
w4_data = np.load("/home/dev/lenet-test/tensor6.npy")
b4_data = np.load("/home/dev/lenet-test/tensor7.npy")


params = {'w1': tvm.nd.array(w1_data), "b1": tvm.nd.array(b1_data), 
"w2": tvm.nd.array(w2_data),
"b2": tvm.nd.array(b2_data),
"w3": tvm.nd.array(w3_data),
"b3": tvm.nd.array(b3_data),
"w4": tvm.nd.array(w4_data),
"b4": tvm.nd.array(b4_data),}
target = "llvm"

with relay.build_config(opt_level=0, disabled_pass=["AlterOpLayout", "SimplifyInference"]):
    graph, lib, params = relay.build(func, target, params=params)

module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu())
    # set input and parameters
def read_pgm(filename, byteorder='>'):
    import re
    import numpy
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

x_data = read_pgm("/home/dev/lenet-test/input7.pgm")
x_data = np.array(x_data, dtype="float32")
x_data = np.reshape(x_data, (1, 1, 28, 28))
module.set_input('x', tvm.nd.array(x_data))
module.set_input(**params)
# run
module.run()
out = module.get_output(0)
print("Ref result:{}".format(out))

