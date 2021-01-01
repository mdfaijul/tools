#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2020 Intel Corporation
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

import os
import intel_quantization.graph_converter as converter

_GRAPH = "/localdisk/amin/intel-quant-tools/bert-graphs/fp32_optimized_graph.pb"
_DATA_DIR = "/localdisk/amin/data-bert/"

def bert_squad_callback_cmds():
    script = "/localdisk/amin/workspace/intel-models/models/language_modeling/tensorflow/bert_large/inference/run_squad.py"
    # You can set up larger batch_size and num_batches to get better accuracy, more time is needed accordingly.
    # Leave `--frozen_graph={}` unformatted.
    flags = ' --input_graph={}' + \
            ' --vocab_file={}'.format(_DATA_DIR) + 'vocab.txt' + \
            ' --do_predict=True' + \
            ' --predict_file={}'.format(_DATA_DIR) + 'dev-v1.1.json' + \
            ' --max_seq_length=384' + \
            ' --doc_stride=128' + \
            ' --output_dir=/tmp/mini-squad-out' + \
            ' --mode=profile' + \
            ' --predict_batch_size=1' + \
            ' --bert_config_file={}'.format(_DATA_DIR) + 'bert_config.json' + \
            ' --shuffle=False'
    return script + flags


if __name__ == '__main__':
    # ResNet50 v1.0 quantization example.
    bert_squad = converter.GraphConverter(_GRAPH, None,
        inputs=['input_mask', 'segment_ids', 'input_ids'],
        outputs=['start_logits', 'end_logits'],
        excluded_nodes=['BiasAdd'],
        per_channel=True)
    # pass an inference script to `gen_calib_data_cmds` to generate calibration data.
    bert_squad.gen_calib_data_cmds = bert_squad_callback_cmds()
    bert_squad.convert()
