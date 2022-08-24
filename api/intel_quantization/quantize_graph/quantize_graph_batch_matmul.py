#  -*- coding: utf-8 -*-
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from .quantize_graph_common import QuantizeGraphHelper as helper
from .quantize_graph_base import QuantizeNodeBase

import logging


class FuseNodeStartWithBatchMatmul(QuantizeNodeBase):
    # patterns = [["BatchMatMulV2"],
    #             ["_MklFusedBatchMatMulV2"]]
    patterns = [['']]
    def __init__(self, input_graph, output_node_names, perchannel,
                 start_node_name):
        super(FuseNodeStartWithBatchMatmul,
              self).__init__(input_graph, output_node_names, perchannel,
                             start_node_name)

        self.sorted_patterns = sorted(self.patterns,
                                      key=lambda i: len(i),
                                      reverse=True)
        self.fusion_op_type = set(fusion[0] for fusion in self.patterns)
        self.fusion_mapping = {
            'BatchMatMulV2': self.apply_batch_matmul,
            '_MklFusedBatchMatMulV2': self.apply_batch_matmul,
        }

    def apply_batch_matmul(self, match_node_name):
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)

        for _, node in enumerate(self.input_graph.node):
            if node.name == match_node_name[0]:
                logging.debug("matched node {} with input {}".format(
                    node.name, node.input))

                logging.debug("apply_batch_matmul")
                if matched_node.node.op == 'BatchMatMulV2':
                    quantized_node_name = node.name + "_eightbit_quantized_batch_matmul"
                    all_input_names = self._add_eightbit_prologue_nodes(
                        matched_node.node.name)
                    quantized_node_input_names = all_input_names + control_inputs
                    quantized_matmul_node = helper.create_node(
                        "_QuantizedBatchMatMulV2AndDequantize", match_node_name[0],
                        quantized_node_input_names)

                    for key, value in node.attr.items():
                        if key == 'T':
                            continue
                        else:
                            helper.copy_attr(quantized_matmul_node, key, value)
                elif matched_node.node.op == '_MklFusedBatchMatMulV2':
                    multiplier_node_name = matched_node.node.input[2]
                    addend_node_name = matched_node.node.input[3]
                    quantized_node_name = node.name + "_eightbit_quantized_fused_batch_matmul"
                    all_input_names = self._add_eightbit_prologue_nodes(
                        matched_node.node.name)
                    quantized_node_input_names = all_input_names[0:2] + \
                        [multiplier_node_name, addend_node_name] + all_input_names[2:] + control_inputs
                    quantized_matmul_node = helper.create_node(
                        "_QuantizedFusedBatchMatMulV2AndDequantize", match_node_name[0],
                        quantized_node_input_names)

                    for key, value in node.attr.items():
                        helper.copy_attr(quantized_matmul_node, key, value)
                else:
                    raise ValueError(matched_node.node.op + " is an Unexpected op.")

                # Add additional attributes.
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.DType(matched_node.node.attr["T"].type))

                # Add quantized node to the graph.
                self.add_output_graph_node(quantized_matmul_node)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)


    def get_longest_fuse(self):
        self._get_op_list()
        matched_rule, _ = self._is_match(self.sorted_patterns)
        return matched_rule

    def apply_the_transform(self):
        self._get_op_list()
        matched_rule, matched_node_name = self._is_match(self.sorted_patterns)
        if matched_node_name:
            self.output_graph = graph_pb2.GraphDef()
            fusion_name = ''.join(matched_rule)
            if fusion_name in self.fusion_mapping:
                self.fusion_mapping[fusion_name](matched_node_name)
            else:
                print("Unknown match {}".format(fusion_name))

            self.input_graph = self.output_graph
            self._reset_output_node_maps()

            self.output_graph = self.remove_redundant_quantization(
                self.output_graph)
            return self.output_graph
        else:
            logging.debug("No more match, exit...")
            return self.input_graph
