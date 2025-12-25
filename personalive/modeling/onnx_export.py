# adapted from https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/utilities.py
#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import onnx
import gc
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from polygraphy.backend.onnx.loader import fold_constants
import os
from onnxsim import simplify

@torch.no_grad()
def export_onnx(
    model,
    onnx_path: str,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    onnx_opset: int,
    dtype,
    device,
    auto_cast: bool = True,
):
    from contextlib import contextmanager

    @contextmanager
    def auto_cast_manager(enabled):
        if enabled:
            with torch.inference_mode(), torch.autocast("cuda"):
                yield
        else:
            yield

    # 确保父目录存在
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    with auto_cast_manager(auto_cast):
        inputs = model.get_sample_input(opt_batch_size, opt_image_height, opt_image_width, dtype, device)
        
        print(model.get_output_names())
        print(f"开始导出 ONNX 模型到: {onnx_path} ...")
        torch.onnx.utils.export(
            model,
            inputs,
            onnx_path,
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes(),
        )
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

def optimize_onnx(onnx_path, onnx_opt_path):
    model = onnx.load(onnx_path)
    name = os.path.splitext(os.path.basename(onnx_opt_path))[0]
    model_opt = model

    print(f"Saving to {onnx_opt_path}...")
    onnx.save(
        model_opt, 
        onnx_opt_path, 
        save_as_external_data=True, 
        all_tensors_to_one_file=True,
        location=f"{name}.onnx.data",
        size_threshold=1024
    )
    print("Optimization done.")

def handle_onnx_batch_norm(onnx_path: str):
    onnx_model = onnx.load(onnx_path)
    for node in onnx_model.graph.node:
        if node.op_type == "BatchNormalization":
            for attribute in node.attribute:
                if attribute.name == "training_mode":
                    if attribute.i == 1:
                        node.output.remove(node.output[1])
                        node.output.remove(node.output[1])
                    attribute.i = 0

    onnx.save_model(onnx_model, onnx_path)