#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v3_on_flowers.sh
set -e

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/wuwei/TF_classification/pretrain_models

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/wuwei/TF_classification/logs/stand_inception_v3

# Where the dataset is saved to.
DATASET_DIR=/wuwei/TF_classification/data/stand

# where Test dir
TESTSET_DIR=/wuwei/TF_classification/data/chair/chair
TESTSET_DIR=/wuwei/GroudPlane_Estimation/Data/SUNCG

# whera label infor dir
LABEL_DIR=/wuwei/TF_classification/data/label_without_content_bt2


#CUDA_VISIBLE_DEVICES="5" python3 image_eval.py \
#    --checkpoint_path=${TRAIN_DIR}/all \
#    --test_list=/wuwei/TF_classification/data/test_list.txt \
#    --test_dir=${TESTSET_DIR}/ \
#    --label_dir=${LABEL_DIR}/ \
#    --batch_size=1 \
#    --num_classes=285 \
#    --model_name=inception_v3

## Download the dataset we set this step manually
#python download_and_convert_data.py \
#  --dataset_name=flowers \
#  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 10000 steps.
python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=stand \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=5000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python3 eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=stand \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3

# Fine-tune all the new layers for 100000 steps.
python3 train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=stand \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=20000 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python3 eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=stand \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3


