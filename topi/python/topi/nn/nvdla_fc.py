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
"""TVM operator fully connected compute."""
from __future__ import absolute_import
import tvm
from .. import tag


def nvdla_fc_default(data, weight, bias=None, num_output=0, out_dtype=None):
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    
    if len(data.shape) == 2:
        batch, in_dim = data.shape
        out_dim, _ = weight.shape
        k = tvm.reduce_axis((0, in_dim), name='k')
        
        if bias is not None:
            matmul = tvm.compute((batch, out_dim), \
                         lambda i, j: tvm.sum(data[i, k].astype(out_dtype) * \
                                              weight[j, k].astype(out_dtype) + bias[j].astype(out_dtype), axis=k), \
                         name='T_nvdla_fc', tag='nvdla_fc', attrs={"units": num_output})
        else:
            matmul = tvm.compute((batch, out_dim), \
                         lambda i, j: tvm.sum(data[i, k].astype(out_dtype) * \
                                              weight[j, k].astype(out_dtype), axis=k), \
                         name='T_nvdla_fc', tag='nvdla_fc', attrs={"units": num_output})
        
        return matmul
    else:
        raise ValueError("Unsupport data shape:{}".format(len(data.shape)))

    
@tvm.target.override_native_generic_func("nvdla_fc")
def nvdla_fc(data, weight, bias=None, num_output=0, out_dtype=None):
    return nvdla_fc_default(data, weight, bias, num_output, out_dtype)