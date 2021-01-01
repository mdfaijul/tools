from argparse import ArgumentParser

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training import saver as saver_lib

import tensorflow as tf

parser = ArgumentParser()
parser.add_argument("--input_graph", type=str, required=True,
  help="Input graph in binary protobuf format.")
parser.add_argument("--output_graph", type=str, required=False,
  help="Output graph in binary protobuf format."
       " Default is grappler_output.pb in the current directory.")
parser.add_argument("--output_node_names", type=str, required=True,
  help="Output node names (comma separated).")
args = parser.parse_args()

g_def = graph_pb2.GraphDef()
with open(args.input_graph, "rb") as f:
  data = f.read()
  g_def.ParseFromString(data)

# Add CPU device to the nodes.
for node in g_def.node:
  node.device = "/device:CPU:0"

g = tf.Graph()
with g.as_default():
  importer.import_graph_def(g_def, input_map={}, name="")
  meta_graph = saver_lib.export_meta_graph(graph_def=g_def, graph=g)

  fetch_collection = meta_graph_pb2.CollectionDef()
  for fetch in args.output_node_names.split(','):
    fetch_collection.node_list.value.append(fetch)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

config = config_pb2.ConfigProto()

config.graph_options.rewrite_options.CopyFrom(
  rewriter_config_pb2.RewriterConfig(
    remapping=rewriter_config_pb2.RewriterConfig.AGGRESSIVE))
optimized_graph = tf_optimizer.OptimizeGraph(
    config, meta_graph)

output_graph_path = \
  args.output_graph if args.output_graph else "./grappler_output.pb"
with open(output_graph_path, "wb") as f:
  f.write(optimized_graph.SerializeToString())
