#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.framework import graph_pb2

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from intel_quantization.transform_graph.graph_transform_base \
    import GraphTransformBase
from intel_quantization.quantize_graph.quantize_graph_common \
    import QuantizeGraphHelper as helper

class FuseQuantizedMulRequantizeAndDequantize(GraphTransformBase):
    """
    Fuse QuantizedMatMulWithBiasAndRelu with requantize .
    """
    def __init__(self, input_pb):
        super(FuseQuantizedMulRequantizeAndDequantize, self).__init__(input_pb)

        self.output_name_index_mapping = {}
        self.input_rename = {}

    def get_fuse_index(self, input_node_map, input_name_list):
        matmul_op_list = ["_QuantizedFusedMatMulAndDequantize"]

        fuse_op_list = []
        for node_index, node_name in enumerate(input_name_list):
            node_op = input_node_map[node_name].op
            # print (node_op, node_name)
            if node_op in matmul_op_list and \
              ('key' in node_name or 'query' in node_name or 'value' in node_name):
                # import ipdb; ipdb.set_trace()
                if input_node_map[input_name_list[node_index + 6]].op == "Const" and \
                    input_node_map[input_name_list[node_index + 7]].op == "Const" and \
                    input_node_map[input_name_list[node_index + 8]].op == "QuantizeV2":
                    fuse_op_list.append(node_index)
        return fuse_op_list

    def parse_input_graph(self, input_graph_def):
        node_type_list = []
        node_name_list = []
        input_node_map = {}

        for node in input_graph_def.node:
            node_name_list.append(node.name)
            node_type_list.append(node.op)
            each_node_input = []
            if node.input:
                for _, sub_input in enumerate(node.input):
                    each_node_input.append(sub_input)

            if node.name not in input_node_map:
                input_node_map[node.name] = node
            else:
                print('Duplicate node name {}'.format(node.name))

        return input_node_map, node_type_list, node_name_list

    def check_node_existence(self, graph, node_name):
        for node in graph.node:
            if node.name == node_name:
                return node
        return None

    def generate_output_graph(self, input_graph_def, input_node_map,
                              fuse_op_list):
        output_graph_def = graph_pb2.GraphDef()
        skip_list = []
        skip_node_name = []
        float32_type = dtypes.float32.as_datatype_enum
        for index, node in enumerate(input_graph_def.node):
            if index in fuse_op_list:
                if node.op == "_QuantizedFusedMatMulAndDequantize":
                    frozen_min_node = input_graph_def.node[index + 6]
                    frozen_max_node = input_graph_def.node[index + 7]
                    quantize_node = input_graph_def.node[index + 8]

                    new_node = node_def_pb2.NodeDef()
                    new_node.name = quantize_node.name
                    new_node.op = "_QuantizedFusedMatMulAndRequantize"
                    for _, value in enumerate(node.input):
                        new_node.input.append(value)
                    new_node.input.append(frozen_min_node.name)
                    new_node.input.append(frozen_max_node.name)
                    for key, value in node.attr.items():
                        helper.copy_attr(new_node, key, value)
                    helper.set_attr_dtype(new_node, "Toutput", dtypes.qint8)
                    # import ipdb; ipdb.set_trace()
                    # skip_list.append(index + 1)
                    # skip_list.append(index + 2)
                    # skip_list.append(index + 3)
                    skip_list.append(index + 6)
                    skip_list.append(index + 7)
                    skip_list.append(index + 8)
                    output_graph_def.node.extend(
                        [new_node, frozen_min_node, frozen_max_node])
                else:
                    new_node = node_def_pb2.NodeDef()
                    new_node.CopyFrom(node)
                    output_graph_def.node.extend([new_node])

            elif index in skip_list or node.name in skip_node_name:
                continue
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                output_graph_def.node.extend([new_node])
        output_graph_def.library.CopyFrom(input_graph_def.library)
        return output_graph_def

    def do_transformation(self):
        """
        Execute the QuantizedMatMulWithBiasAndRelu And Requantize fusion transformation.
        :return: Transformed graph
        """
        input_node_map, _, node_name_list = self.parse_input_graph(
            self.input_graph)

        fuse_op_list = self.get_fuse_index(input_node_map, node_name_list)
        # print (fuse_op_list)
        # import sys
        # sys.exit(0)
        return self.generate_output_graph(self.input_graph, input_node_map,
                                          fuse_op_list)
