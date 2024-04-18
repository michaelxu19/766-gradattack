# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

TASK="Task01_BrainTumour"
DATA_PATH="data/" + TASK + "/"
SAVED_MODEL_NAME="3d_unet_decathlon"

#Lightning
NUM_WORKERS = 32
####

EPOCHS=40
BATCH_SIZE=8
BZ_TRAIN=8
BZ_VAL=4
BZ_TEST=1
TILE_HEIGHT=144
TILE_WIDTH=144
TILE_DEPTH=144
NUMBER_INPUT_CHANNELS=1
NUMBER_OUTPUT_CLASSES=3
CROP_DIM = (128,128,128,1)
TRAIN_VAL_SPLIT=.8
TRAIN_TEST_SPLIT=0.80
TEST_VAL_SPLIT=0.50

PRINT_MODEL=False
FILTERS=8
USE_UPSAMPLING=False

RANDOM_SEED=816

