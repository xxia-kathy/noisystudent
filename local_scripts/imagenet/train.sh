# coding=utf-8
# Copyright 2019 The Google NoisyStudent Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash

model_dir=./ckpt/yt8m
label_data_dir=/home/kathy/mnt/data/live_action_vs_animated_tfrecord
unlabel_data_dir=/home/kathy/data/yt8m/2/frame/train
model_name=efficientnet-b0 
teacher_model_path=./ckpt/teachter_yt8m
rm -r ${model_dir}

# Training command base:
python main.py \
    --task_name=yt8m \
    --input_image_size=128 \
    --use_tpu=False \
    --use_bfloat16=False \
    --mode=train \
    --model_name=${model_name} \
    --label_data_dir=${label_data_dir} \
    --model_dir=${model_dir} \
    --unlabel_data_dir=${unlabel_data_dir} \
    --teacher_model_name=${model_name} \
    --teacher_model_path=${teacher_model_path} \
    --teacher_softmax_temp=1 \
    --unlabel_ratio=1 \
    --train_ratio=2 \

# Additional config for EfficientNet-B1
#     --unlabel_ratio=3 \
#     --train_ratio=2

# Additional config for EfficientNet-B2
#     --unlabel_ratio=3 \
#     --train_ratio=2 \
#     --augment_name=v1 \
#     --randaug_mag=6

# Additional config for EfficientNet-B3
#     --unlabel_ratio=3 \
#     --train_ratio=2 \
#     --augment_name=v1
#     --randaug_mag=12 \

# Additional config for EfficientNet-B4
#     --unlabel_ratio=3 \
#     --train_ratio=2 \
#     --augment_name=v1 \
#     --randaug_mag=15 \

# Additional config for EfficientNet-B5
#     --unlabel_ratio=3 \
#     --augment_name=v1 \
#     --randaug_mag=18 \

# Additional config for EfficientNet-B6
#     --unlabel_ratio=3 \
#     --augment_name=v1 \
#     --randaug_mag=24 \

# Additional config for EfficientNet-B7
#     --unlabel_ratio=3 \
#     --augment_name=v1 \
#     --randaug_mag=27 \
