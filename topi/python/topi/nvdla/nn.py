"""Conv2D schedule on x86"""

import logging
import re

import tvm
from tvm import autotvm
from tvm.autotvm.task.topi_integration import deserialize_args
from tvm.autotvm.task import get_config
from .. import generic, tag
from .. import nn
from ..nn.util import get_pad_tuple
from ..util import get_const_tuple

import operator

from nvdla.nvdla_utils import nvdla_graph_info, find_op_info, generate_output_pass


def _intrin_global_average_pool(op_tensor, dtype, op_info, is_output = False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        out_shape = outs[0].shape
        out_shape_len = len(out_shape)
        assert out_shape_len == 4

        global nvdla_graph_info
        data = nvdla_graph_info['op_infos'][op_info['input_index'][0]]['node']

        op = tvm.call_extern("handle", "AddGlobalAveragePoolOp",
                                 hash(data)
                                 )
        output_tensor = tvm.call_extern("handle", "AddFloatTensor",
                                 hash(op_info['node']), out_shape_len, *(out_shape)
                                 )        
        ib.emit(tvm.call_extern("float", "AddOutput",
                                 op, output_tensor
                                 ))

        if is_output:
             ib.emit(tvm.call_extern("float", "AddOutputOp", hash(op_info['node'])))
             ib.emit(tvm.call_extern("float", "NvDlaCompile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op_tensor.op, intrin_func)


def _intrin_max_pool2d(op_tensor, dtype, op_info, kernel_shape, padding, strides, is_output = False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        if isinstance(strides, int):
            stride_h = stride_w = strides
        else:
            stride_h, stride_w = strides

        if isinstance(kernel_shape, int):
            kernel_shape_h = kernel_shape_w = kernel_shape
        else:
            kernel_shape_h, kernel_shape_w = kernel_shape

        if isinstance(padding, int):
            pad_w = pad_h = padding
        else:
            pad_w, pad_h = padding

        global nvdla_graph_info
        data = nvdla_graph_info['op_infos'][op_info['input_index'][0]]

        ib.emit(tvm.call_extern("handle", "addMaxPooling",
                                 data['name'] + "_" + str(op_info['input_index'][0]),
                                 op_info['name'] + "_" + str(op_info['current_idx']),
                                 kernel_shape_h, kernel_shape_w,
                                 pad_h, pad_w, stride_h, stride_w, 0))
                        
        if is_output:
             generate_output_pass(ib)
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op_tensor.op, intrin_func)


@generic.schedule_pool_nvdla.register(['nvdla'])
def schedule_maxpool_2d(outs, layout, attrs=None):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    global nvdla_graph_info
    output_op = nvdla_graph_info["output_op"]['op']
    def traverse(op):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif op.tag == ('pool_max'):
            maxpool_op = op.output(0)
            dtype = maxpool_op.dtype
            input_tensors = op.input_tensors
            input0 = input_tensors[0]

            is_output = False

            op_shape = [int(x) for x in maxpool_op.shape]
            output_shape = [int(x) for x in nvdla_graph_info['output_op']['types'][0].shape] 

            if output_op == "max_pool2d" and op_shape == output_shape:
                is_output = True

            op_info = find_op_info("max_pool2d", op_shape, input_tensors)

            #auto_pad = op.attrs["auto_pad"]
            kernel_shape = attrs['pool_size']
            padding = attrs['padding']
            strides = attrs["strides"]

            intric = _intrin_max_pool2d(maxpool_op, dtype, op_info, kernel_shape, padding, strides, is_output=is_output)
            s[op].tensorize(op.axis[0], intric)

        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


@generic.schedule_adaptive_pool.register(["nvdla"])
def schedule_adaptive_pool(outs):
    """Schedule for adaptive pool
    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of adaptive pool
          in the format of an array of tensors.
    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    global nvdla_graph_info
    output_op = nvdla_graph_info["output_op"]['op']

    def traverse(op):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif op.tag == ('adaptive_pool_sum'):
            global_average_pool_op = op.output(0)
            dtype = global_average_pool_op.dtype
            input_tensors = op.input_tensors

            is_output = False

            op_shape = [int(x) for x in global_average_pool_op.shape]
            output_shape = [int(x) for x in nvdla_graph_info['output_op']['types'][0].shape] 

            if output_op == "global_avg_pool2d" and op_shape == output_shape:
                is_output = True

            op_info = find_op_info("global_avg_pool2d", op_shape, input_tensors)

            intric = _intrin_global_average_pool(global_average_pool_op, dtype, op_info, is_output=is_output)
            s[op].tensorize(op.axis[0], intric)

        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s 