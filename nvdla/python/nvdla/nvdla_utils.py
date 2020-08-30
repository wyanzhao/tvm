
def gemm_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 3
    i1, i2, i3 = input_tensors
    i1 = [int(x) for x in i1.shape]
    i2 = [int(x) for x in i2.shape]
    i3 = [int(x) for x in i3.shape]
    if op_input_shapes[0] == i1 and i2 == op_input_shapes[1] and i3 == op_input_shapes[2]:
        return True
    else:
        return False

def nvdla_fc_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 2 or len(input_tensors) == 3
    if len(input_tensors) == 2:
        i1, i2 = input_tensors
        i1 = [int(x) for x in i1.shape]
        i2 = [int(x) for x in i2.shape]
        if op_input_shapes[0] == i1 and i2 == op_input_shapes[1]:
            return True
        else:
            return False
    else:
        i1, i2, i3 = input_tensors
        i1 = [int(x) for x in i1.shape]
        i2 = [int(x) for x in i2.shape]
        i3 = [int(x) for x in i3.shape]

        sum_i1 = 1
        for i in i1:
            sum_i1 = sum_i1 * i

        sum_op_shape = 1
        for i in op_input_shapes[0]:
            sum_op_shape = sum_op_shape * i

        if sum_op_shape == sum_i1 and i2 == op_input_shapes[1] and i3 == op_input_shapes[2]:
            return True
        else:
            return False

def nvdla_conv2d_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 2 or len(input_tensors) == 3
    if len(input_tensors) == 2:
        i1, i2 = input_tensors
        i1 = [int(x) for x in i1.shape]
        i2 = [int(x) for x in i2.shape]
        if op_input_shapes[0] == i1 and i2 == op_input_shapes[1]:
            return True
        else:
            return False
    else:
        i1, i2, i3 = input_tensors
        i1 = [int(x) for x in i1.shape]
        i2 = [int(x) for x in i2.shape]
        i3 = [int(x) for x in i3.shape]

        if op_input_shapes[0] == i1 and i2 == op_input_shapes[1] and i3 == op_input_shapes[2]:
            return True
        else:
            return False

def add_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 2
    i1, i2 = input_tensors
    i1 = [int(x) for x in i1.shape]
    i2 = [int(x) for x in i2.shape]
    if op_input_shapes[0] == i1 and i2 == op_input_shapes[1]:
        return True
    else:
        return False

def reshape_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 1
    i1 = input_tensors[0]
    i1 = [int(x) for x in i1.shape]
    if op_input_shapes[0] == i1:
        return True
    else:
        return False

def softmax_input_shapes(op_input_shapes, input_tensors):
    i1 = input_tensors[0]
    i1 = [int(x) for x in i1.shape]
    if op_input_shapes[0] == i1:
        return True
    else:
        return False

def relu_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 1
    i1 = input_tensors[0]
    i1 = [int(x) for x in i1.shape]
    if op_input_shapes[0] == i1:
        return True
    else:
        return False

def conv_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 2
    i1, i2 = input_tensors
    i1 = [int(x) for x in i1.shape]
    i2 = [int(x) for x in i2.shape]
    if op_input_shapes[0] == i1 and i2 == op_input_shapes[1]:
        return True
    else:
        return False

def maxpool_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 1
    i1 = input_tensors[0]
    i1 = [int(x) for x in i1.shape]
    if op_input_shapes[0] == i1:
        return True
    else:
        return False

def global_avg_pool2d_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 1
    i1 = input_tensors[0]
    i1 = [int(x) for x in i1.shape]
    if op_input_shapes[0] == i1:
        return True
    else:
        return False

def batch_norm_input_shapes(op_input_shapes, input_tensors):
    assert len(input_tensors) == 5
    i1, i2, i3, i4, i5 = input_tensors
    i1 = [int(x) for x in i1.shape]
    i2 = [int(x) for x in i2.shape]
    i3 = [int(x) for x in i3.shape]
    i4 = [int(x) for x in i4.shape]
    i5 = [int(x) for x in i5.shape]
    if op_input_shapes[0] == i1 and i2 == op_input_shapes[1] and i3 == op_input_shapes[2] and \
    i4 == op_input_shapes[3] and i5 == op_input_shapes[4]:
        return True
    else:
        return False

support_config = [
    'nv_small', 'nv_small_128', "nv_small_256", "nv_small_512", "nv_medium_256", "nv_medium_512", "nv_medium_1024", "nv_large", "nv_full"
]

def set_nvdla_config(target_config, compute_precison):
    global nvdla_graph_info
    assert target_config in support_config
    assert compute_precison == 'int8' or compute_precison == 'fp16'

    nvdla_graph_info['nvdla_config']['target_config'] = target_config
    nvdla_graph_info['nvdla_config']['compute_precison'] = compute_precison
    return

def generate_output_pass(ib):
    global nvdla_graph_info
    import tvm

    if nvdla_graph_info.get('scale_info') != None:
        scale_info = dict(nvdla_graph_info['scale_info'])
        for i, j in scale_info.items():
                name = i
                _scale = j.get('scale')
                _min = j.get('min')
                _max = j.get('max')
                _offset = j.get('offset')
                ib.emit(tvm.call_extern("handle", "addScaleInfo", name, float(_scale),
                float(_min), float(_max), float(_offset)))
    
    target_config = nvdla_graph_info['nvdla_config']['target_config']
    compute_precision = nvdla_graph_info['nvdla_config']['compute_precison']
    ib.emit(tvm.call_extern("handle", "setNvdlaConfig", target_config, compute_precision))
    ib.emit(tvm.call_extern("handle", "nvdlaCompile"))
    



def find_op_info(op_name, op_shape, op_inputs):
    global nvdla_graph_info
    op_list = nvdla_graph_info['op_maps'].get(op_name)
    if op_list == None:
        raise ValueError("Can't find op_list in op_maps:{}".format(op_name))
    op_info = None

    if len(nvdla_graph_info['op_maps'].get(op_name)) <= 0:
        raise ValueError("Empty Op_Map List:{}".format(op_name))

    for x in range(len(nvdla_graph_info['op_maps'].get(op_name)) - 1, -1, -1):
        node = nvdla_graph_info['op_maps'][op_name][x]
        func = nvdla_graph_info['shape_functions'][op_name]
        if node['op_shape'] == op_shape and func(node['input_shapes'], op_inputs):
            op_info = node
            del nvdla_graph_info['op_maps'][op_name][x]
            break

    if op_info == None:
        raise ValueError("Can't find Op in Relay Graph")
    return op_info  

def nvdla_analyze_compute_graph(relay_func, input_op_idx = None, input_param_idx_list = []):
    from tvm.autotvm.graph_tuner.utils import has_multiple_inputs, get_direct_ancestor, get_in_nodes, get_out_nodes, expr2graph, bind_inputs
    node_list = []
    node_dict = {}
    target_ops = []
    expr2graph(relay_func, target_ops, node_dict, node_list)
    
    global   nvdla_graph_info
    if input_op_idx == None:
        raise ValueError("Needs to provide input node index info")

    param_list = []
    param_list.append(input_op_idx)
    param_list += input_param_idx_list

    nvdla_graph_info["input_op"] = [node_list[x] for x in param_list]

    for x in range(len(node_list)):
        node = {}
        if len(node_list[x]['inputs']) != 0:
            node['current_idx'] = x
            node['name'] = node_list[x]['op']
            node['node'] = node_list[x]['node']
            node['input_shapes'] = []
            node['input_index'] = []
            node['op_shape'] = [int(x) for x in node_list[x]['types'][0].shape]

            for y in node_list[x]['inputs']:
                input_index = y[0]
                shape = [int(x) for x in node_list[input_index]['types'][0].shape]
                if node_list[input_index]['op'] == 'null':
                    node['is_const'] = True
                else:
                    node['is_const'] = False
                node['input_shapes'].append(shape)
                node['input_index'].append(input_index)
            
            if nvdla_graph_info['op_maps'].get(node_list[x]['op']) != None:
                    nvdla_graph_info['op_maps'][node_list[x]['op']].append(node)
            else:
                print('Unsupport Op:{}'.format(node_list[x]['op']))
        else:
            node['name'] = node_list[x]['op']
            node['node'] = node_list[x]['node']
            node['op_shape'] = [int(x) for x in node_list[x]['types'][0].shape]
            node['input_shapes'] = []
            node['input_index'] = []
        
        nvdla_graph_info['op_infos'].append(node)
    nvdla_graph_info["output_op"] = node_list[-1]
    return

nvdla_graph_info = {
    'op_infos': [],
    "input_op": [],
    'op_maps': {
        "add":[],
        'reshape':[],
        'batch_norm':[],
        'relu':[],
        "copy":[],
        'conv2d':[],
        'max_pool2d':[],
        'global_avg_pool2d':[],
        'gemm':[],
        'dense':[],
        "nvdla_fc":[],
        'softmax':[],
        "nvdla_conv2d": [],
        "nvdla_conv2d_bias": []
    },
    "shape_functions": {
        "gemm": gemm_input_shapes,
        "add": add_input_shapes,
        "reshape": reshape_input_shapes,
        "relu": relu_input_shapes,
        "conv2d": conv_input_shapes,
        "max_pool2d": maxpool_input_shapes,
        "global_avg_pool2d": global_avg_pool2d_input_shapes,
        "batch_norm": batch_norm_input_shapes,
        'nvdla_fc': nvdla_fc_input_shapes,
        'nvdla_conv2d': nvdla_conv2d_input_shapes,
        'nvdla_conv2d_bias': nvdla_conv2d_input_shapes,
        'softmax': softmax_input_shapes
    },
    "output_op": None,
    "scale_info": None,
    "nvdla_config": {
        "target_config": "nv_full",
        "compute_precision": "fp16"
    }
}
