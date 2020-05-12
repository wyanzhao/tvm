
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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D schedule on x86"""

import logging
import re

import tvm
from tvm import autotvm
# from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config
from .. import generic, tag
from .. import nn
from ..util import simplify, get_const_tuple, get_const_int
from ..nn.conv2d import conv2d, conv2d_NCHWc, \
    conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..nn.pad import pad
from ..nn.util import get_pad_tuple

import operator

from nvdla.nvdla_utils import nvdla_graph_info, find_op_info, generate_output_pass

def _intrin_conv(op_tensor, data_shape, kernel_shape, strides, padding, dilation, dtype, op_info, bias = None, is_input = False, is_output = False):
    if isinstance(strides, int):
            stride_h = stride_w = strides
    else:
            stride_h, stride_w = strides

    if isinstance(dilation, int):
            dilation_h = dilation_w = dilation
    else:
            dilation_h, dilation_w = dilation

    if isinstance(padding, int):
            pad_w = pad_h = padding
    else:
        assert len(padding) == 2 or len(padding) == 4
        if len(padding) == 2:
            pad_h, pad_w = padding
        else:
            pad_h, pad_w, _, _ = padding

    batch, in_channel, in_height, in_width = data_shape
    num_filter, channel, kernel_h, kernel_w = kernel_shape
    out_channel = num_filter
    fout_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    fout_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1

    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    weight = tvm.placeholder(kernel_shape, name='weight', dtype=dtype)

    if int(stride_h) > 1:
        new_in_height = in_height - 1
        new_in_width = in_width - 1
    else:
        new_in_height = in_height
        new_in_width = in_width

    data_buffer = tvm.decl_buffer((tvm.var("e"), tvm.var("f"), tvm.var("g"), tvm.var("h")), dtype=data.dtype, strides=[tvm.var("a"), tvm.var("b"), tvm.var("c"), tvm.var("d")], name="data_buffer")
    #weight_buffer = tvm.decl_buffer(kernel_shape, dtype=weight.dtype, strides=strides, name="weight_buffer")

    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel_h), name='ry')
    rx = tvm.reduce_axis((0, kernel_w), name='rx')

    conv_op = nn.nvdla_conv2d(data, weight, bias=bias, strides=strides, padding=padding, dilation=dilation)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        global nvdla_graph_info
        data = nvdla_graph_info['op_infos'][op_info['input_index'][0]]
        kernel = nvdla_graph_info['op_infos'][op_info['input_index'][1]]

        if is_input == True:
             ib.emit(tvm.call_extern("handle", "nvdlaInit"))

             input_shape = (batch, in_channel, in_height, in_width)
             input_shape_len = len(input_shape)
             assert input_shape_len == 4
             ib.emit(tvm.call_extern("handle", "addInputOp",
                                data['name'] + "_" + str(op_info['input_index'][0]),
                                batch, in_channel, in_height, in_width
                                 ))

        weight_shape = ins[1].shape
        weight_shape_len = len(weight_shape)
        assert weight_shape_len == 4

        weight = tvm.call_extern("handle", "addFloatWeights", ins[1].data, 
            weight_shape[0] * weight_shape[1] * weight_shape[2] * weight_shape[3])

        if len(ins) == 3:
            bias = tvm.call_extern("handle", "addFloatWeights",ins[2].data, ins[2].shape[0])
        else:
            bias = tvm.call_extern("handle", "addFloatWeights",ins[1].data, 0)


        ib.emit(tvm.call_extern("handle", "addConvOp",
                                 data['name'] + "_" + str(op_info['input_index'][0]),
                                 op_info['name'] + "_" + str(op_info['current_idx']),
                                 out_channel, kernel_h, kernel_w, 
                                 pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                 weight, bias, batch))

        if is_output == True:
            generate_output_pass(ib)

        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(conv_op.op, intrin_func, binds={data: data_buffer})

@generic.schedule_nvdla_conv2d_nchw.register(["nvdla"])
def schedule_conv2d(outs):
    """Create schedule for tensors"""
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    #return s

    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    global nvdla_graph_info
    input_op = nvdla_graph_info["input_op"][0]['op']
    param1 = nvdla_graph_info["input_op"][1]["types"][0]
    param2 = nvdla_graph_info["input_op"][2]["types"][0]
    output_op = nvdla_graph_info["output_op"]['op']
    x = outs[0]

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        """
        if tag.is_broadcast(op.tag):
            op_tensor = op.output(0)
            dtype = op_tensor.dtype
            input_tensors = op.input_tensors
            if op.name == "T_multiply":
                if op not in s.outputs:
                    intric = _intrin_mul(input_tensors, op_tensor, dtype, False)
                else:
                    intric = _intrin_mul(input_tensors, op_tensor, dtype, True)
                s[op].tensorize(op.axis[0], intric)
            else:
                raise ValueError("not support this op {} yet".format(op.name))     
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        """

        if op.tag == 'conv2d_nchw':
            conv_op = op.output(0)

            if len(s[conv_op].op.input_tensors) == 2:
                data, kernel = s[conv_op].op.input_tensors
                bias = None
            else:
                data, kernel, bias = s[conv_op].op.input_tensors
                
            data_shape = data.shape
            kernel_shape = kernel.shape
            dtype = conv_op.dtype

            padding = op.attrs["padding"]
            dilation = op.attrs["dilation"]
            strides = op.attrs["strides"]

            is_input = False
            is_output = False

            data_shape = [int(x) for x in data_shape]
            param1_shape = [int(x) for x in param1.shape]
            kernel_shape = [int(x) for x in kernel_shape]
            param2_shape = [int(x) for x in param2.shape]

            op_shape = [int(x) for x in conv_op.shape]
            output_shape = [int(x) for x in nvdla_graph_info['output_op']['types'][0].shape]

            if bias is None:
                if input_op == "nvdla_conv2d" and data_shape == param1_shape and kernel_shape == param2_shape:
                    is_input = True

                if output_op == "nvdla_conv2d" and op_shape == output_shape:
                    is_output = True

                op_info = find_op_info("nvdla_conv2d", op_shape, s[conv_op].op.input_tensors)
            else:
                if input_op == "nvdla_conv2d_bias" and data_shape == param1_shape and kernel_shape == param2_shape:
                    is_input = True

                if output_op == "nvdla_conv2d_bias" and op_shape == output_shape:
                    is_output = True
                    
                op_info = find_op_info("nvdla_conv2d_bias", op_shape, s[conv_op].op.input_tensors)

            intric = _intrin_conv(conv_op, data_shape, kernel_shape, bias=bias, strides=strides, padding=padding,
                dilation=dilation, dtype=dtype, op_info=op_info, is_input=is_input, is_output=is_output)
            s[op].tensorize(op.axis[0], intric)
        else:
            raise ValueError("Unsupport Op:{}".format(op))

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s 