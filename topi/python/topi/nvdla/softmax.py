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
# pylint: disable=invalid-name, unused-variable, trailing-whitespace
"""Schedule for softmax operator"""
import tvm
from .. import generic
from nvdla.nvdla_utils import nvdla_graph_info, find_op_info


def _intrinc_softmax(op_tensor, dtype, op_info, is_output = False):
    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        global nvdla_graph_info
        data = nvdla_graph_info['op_infos'][op_info['input_index'][0]]['node']

        ib.emit(tvm.call_extern("handle", "addSoftMaxOp",
                                 hash(data), hash(op_info['node'])
                                 ))
      
        if is_output:
             ib.emit(tvm.call_extern("handle", "nvdlaCompile"))
        return ib.get()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(op_tensor.op, intrin_func)


@generic.schedule_softmax.register(["nvdla"])
def schedule_softmax(outs):
    """Schedule for softmax op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    softmax = outs[0]

    x = outs[0]
    op = x.op
    op_tensor = op.output(0)
    dtype = op_tensor.dtype
    input_tensors = op.input_tensors

    global nvdla_graph_info
    output_op = nvdla_graph_info["output_op"]['op']

    op_tag = softmax.op.tag
    if op_tag == 'softmax_output':
        softmax_op = op.output(0)
        data = s[softmax_op].op.input_tensors[0]

        data_shape = data.shape

        is_output = False
        data_shape = [int(x) for x in data_shape]
        op_shape = [int(x) for x in softmax_op.shape]
        output_shape = [int(x) for x in nvdla_graph_info['output_op']['types'][0].shape]

        if output_op == "softmax" and op_shape == output_shape:
            is_output = True
        
        op_info = find_op_info('softmax', op_shape, s[softmax_op].op.input_tensors)

        intric = _intrinc_softmax(softmax_op, dtype=dtype, op_info=op_info, is_output=is_output)
        s[op].tensorize(op.axis[0], intric)

    else:
        raise ValueError('Tag is expected to be softmax_output or log_softmax_output. \
                         Got {0}'.format(op_tag))


    return s
