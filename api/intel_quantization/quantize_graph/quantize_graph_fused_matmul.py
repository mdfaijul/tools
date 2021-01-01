#  -*- coding: utf-8 -*-
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes

from .quantize_graph_common import QuantizeGraphHelper as helper
from .quantize_graph_base import QuantizeNodeBase

import logging


class FuseNodeStartWithFusedMatmul(QuantizeNodeBase):
    patterns = [["_FusedMatMul", "AddV2"],
                ["_FusedMatMul"]]
    # patterns = [["_FusedMatMul"]]

    def __init__(self, input_graph, output_node_names, perchannel,
                 start_node_name):
        super(FuseNodeStartWithFusedMatmul,
              self).__init__(input_graph, output_node_names, perchannel,
                             start_node_name)

        self.sorted_patterns = sorted(self.patterns,
                                      key=lambda i: len(i),
                                      reverse=True)
        self.fusion_op_type = set(fusion[0] for fusion in self.patterns)
        self.fusion_mapping = {
            '_FusedMatMulAddV2': self.apply_fused_matmul_addv2,
            '_FusedMatMul': self.apply_fused_matmul,
        }

    def apply_fused_matmul(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)
        weight_name = normal_inputs[1]

        self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weight_name].node,
            self.per_channel)

        skip_node_name.append(weight_name)
        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[0]:
                logging.debug("matched node {} with input {}".format(
                    node.name, node.input))

                logging.debug("apply_fused_matmul")

                quantized_node_name = node.name + "_eightbit_quantized_fused_matmul"
                bias_node_name = matched_node.node.input[2]
                all_input_names = self._add_eightbit_prologue_nodes(
                    matched_node.node.name)
                fused_ops = matched_node.node.attr['fused_ops'].list.s
                if len(fused_ops) == 2 and fused_ops[1] == b'Add':
                    add_input = [matched_node.node.input[3]]
                else:
                    add_input = []
                quantized_node_input_names = all_input_names[:2] + \
                    [bias_node_name] + add_input + \
                    all_input_names[2:] + control_inputs
                quantized_matmul_node = helper.create_node(
                    "_QuantizedFusedMatMulAndDequantize", match_node_name[0],
                    quantized_node_input_names)

                for key, value in node.attr.items():
                  if key == 'T':
                    continue
                  else:
                    helper.copy_attr(quantized_matmul_node, key, value)
                # Add additional attributes.
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Targs", dtypes.DType(matched_node.node.attr["T"].type))
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.DType(matched_node.node.attr["T"].type))
                helper.set_attr_string(quantized_matmul_node, "input_quant_mode", b'MIN_FIRST')
                # Add quantized node to the graph.
                self.add_output_graph_node(quantized_matmul_node)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(node)
                self.add_output_graph_node(new_node)

    def apply_fused_matmul_addv2(self, match_node_name):
        skip_node_name = match_node_name[1:]
        matched_node = self.node_name_mapping[match_node_name[0]]
        control_inputs, normal_inputs = self._get_node_input(
            matched_node.node.name)
        weight_name = normal_inputs[1]

        self._intel_cpu_quantize_weight_eightbit(
            matched_node.node.op, self.node_name_mapping[weight_name].node,
            self.per_channel)

        skip_node_name.append(weight_name)
        for _, node in enumerate(self.input_graph.node):
            if node.name in skip_node_name:
                pass
            elif node.name == match_node_name[0]:
                logging.debug("matched node {} with input {}".format(
                    node.name, node.input))

                logging.debug("apply_fused_matmul")
                quantized_node_name = node.name + "_eightbit_quantized_fused_matmul"
                bias_node_name = matched_node.node.input[2]
                sum_index = 1 if match_node_name[0] == self.node_name_mapping[
                    match_node_name[1]].node.input[0] else 0
                summand_node_name = self.node_name_mapping[
                    match_node_name[1]].node.input[sum_index]
                all_input_names = self._add_eightbit_prologue_nodes(
                    matched_node.node.name)
                quantized_node_input_names = all_input_names[:2] + \
                    [bias_node_name] + [summand_node_name] + \
                    all_input_names[2:] + control_inputs
                quantized_matmul_node = helper.create_node(
                    "_QuantizedFusedMatMulAndDequantize", match_node_name[1],
                    quantized_node_input_names)

                for key, value in node.attr.items():
                  if key == 'T':
                    continue
                  else:
                    helper.copy_attr(quantized_matmul_node, key, value)
                # Add additional attributes.
                helper.set_attr_dtype(quantized_matmul_node, "T1", dtypes.quint8)
                helper.set_attr_dtype(quantized_matmul_node, "T2", dtypes.qint8)
                helper.set_attr_dtype(quantized_matmul_node, "Targs", dtypes.DType(matched_node.node.attr["T"].type))
                helper.set_attr_dtype(quantized_matmul_node, "Toutput", dtypes.DType(matched_node.node.attr["T"].type))
                helper.set_attr_string(quantized_matmul_node, "input_quant_mode", b'MIN_FIRST')
                helper.set_attr_string_list(quantized_matmul_node, 'fused_ops', [b'BiasAdd', b'Add'])
                helper.set_attr_int(quantized_matmul_node, 'num_args', 2)

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
