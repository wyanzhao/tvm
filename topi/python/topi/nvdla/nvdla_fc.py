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
# pylint: disable=invalid-name,unused-variable
"""dense schedule on ARM Mali GPU"""

from __future__ import absolute_import as _abs

import tvm

from .. import generic, nn
from ..util import traverse_inline, get_const_tuple
from .. import generic, tag, nn
from nvdla.nvdla_utils import nvdla_graph_info, find_op_info, generate_output_pass


def _intrinc_fc(op_tensor,dtype, op_info, num_output=None, has_bias = False, is_input=False, is_output = False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        global nvdla_graph_info
        data = nvdla_graph_info['op_infos'][op_info['input_index'][0]]
        weight = nvdla_graph_info['op_infos'][op_info['input_index'][1]]
        bias = nvdla_graph_info['op_infos'][op_info['input_index'][2]]

        if is_input:
            ib.emit(tvm.call_extern("handle", "nvdlaInit"))
            data_shape = ins[0].shape
            assert len(data_shape) == 4

            ib.emit(tvm.call_extern("handle", "addInputOp",
                                 data.name + "_" + str(op_info['input_index'][0]),
                                 data_shape[0], data_shape[1], 1, 1
                                 ))


        weight_shape = ins[1].shape
        assert len(weight_shape) == 2

        if has_bias:
            bias_shape = ins[2].shape
            assert len(bias_shape) == 1

            bias = tvm.call_extern("handle", "addFloatWeights",ins[2].data, 
            bias_shape[0])
        else:
            bias = tvm.call_extern("handle", "addFloatWeights", 0, 
            0)

        weight = tvm.call_extern("handle", "addFloatWeights", ins[1].data, 
            weight_shape[0] *weight_shape[1])

        if num_output != None:
            ib.emit(tvm.call_extern("handle", "addFullyConnected",
                                 data['name'] + "_" + str(op_info['input_index'][0]),
                                 #hash(data),  hash(op_info['node']), 
                                 op_info['name'] + "_" + str(op_info['current_idx']),
                                 weight, bias, num_output))
        else:
            ib.emit(tvm.call_extern("handle", "addFullyConnected",
                                 #hash(data),  hash(op_info['node']), 
                                 data['name'] + "_" + str(op_info['input_index'][0]),
                                 op_info['name'] + "_" + str(op_info['current_idx']),
                                 weight, bias, weight_shape[0]))

        if is_output:
            generate_output_pass(ib)
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op_tensor.op, intrin_func)


@generic.schedule_nvdla_fc.register(["nvdla"])
def schedule_nvdla_fc(outs):
    """Schedule for dense operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config entity for this template
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    scheduled_ops = []
    x = outs[0]
    op = x.op
    op_tensor = op.output(0)
    dtype = op_tensor.dtype
    input_tensors = op.input_tensors

    global nvdla_graph_info
    output_op = nvdla_graph_info["output_op"]['op']
    input_op = nvdla_graph_info["input_op"][0]['op']
    param1 = nvdla_graph_info["input_op"][1]["types"][0]
    param2 = nvdla_graph_info["input_op"][2]["types"][0]
    output_op = nvdla_graph_info["output_op"]['op']

    def traverse(op):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif op.name == ('T_nvdla_fc'):
            fc_op = op.output(0)
            num_output = op.attrs["units"]

            has_bias = (len(s[fc_op].op.input_tensors) == 3)
            if has_bias:
                data, kernel, bias = s[fc_op].op.input_tensors
            else:
                data, kernel = s[fc_op].op.input_tensors

            data_shape = data.shape
            kernel_shape = kernel.shape
            dtype = fc_op.dtype

            is_input = False
            is_output = False

            data_shape = [int(x) for x in data_shape]
            param1_shape = [int(x) for x in param1.shape]
            kernel_shape = [int(x) for x in kernel_shape]
            param2_shape = [int(x) for x in param2.shape]

            op_shape = [int(x) for x in fc_op.shape]
            output_shape = [int(x) for x in nvdla_graph_info['output_op']['types'][0].shape]

            if input_op == "nvdla_fc" and data_shape == param1_shape and kernel_shape == param2_shape:
                is_input = True

            if output_op == "nvdla_fc" and op_shape == output_shape:
                is_output = True

            op_info = find_op_info("nvdla_fc", op_shape, s[fc_op].op.input_tensors)

            intric = _intrinc_fc(fc_op, num_output=num_output, dtype=dtype, op_info=op_info, has_bias=has_bias,
            is_input=is_input, is_output=is_output)
            s[op].tensorize(op.axis[0], intric)
        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s