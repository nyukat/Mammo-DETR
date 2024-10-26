#!/bin/bash

#SBATCH -p gpu4_medium,radiology,a100_dev,a100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100GB
#SBATCH --time=01:00:00
#SBATCH --job-name=detr
#SBATCH --output=/gpfs/data/geraslab/Yanqi/Mammo-DETR/logs/gmic-%J.log

source ~/.bashrc
conda activate deformable_detr_a100
module load gcc/6.1.0
cd Mammo-DETR

NUM_PROCESSES=10
DEVICE_TYPE='gpu'
GPU_NUMBER=0
MODEL_INDEX='1'

MODEL_PATH='models/'
DATA_FOLDER='sample_data/images'
INITIAL_EXAM_LIST_PATH='sample_data/exam_list_before_cropping.pkl'
CROPPED_IMAGE_PATH='sample_output/cropped_images'
CROPPED_EXAM_LIST_PATH='sample_output/cropped_images/cropped_exam_list.pkl'
SEG_PATH='sample_data/segmentation'
EXAM_LIST_PATH='sample_output/data.pkl'
OUTPUT_PATH='sample_output'
export PYTHONPATH=$(pwd):$PYTHONPATH


# echo 'Stage 1: Crop Mammograms'
# python3 src/cropping/crop_mammogram.py \
#     --input-data-folder $DATA_FOLDER \
#     --output-data-folder $CROPPED_IMAGE_PATH \
#     --exam-list-path $INITIAL_EXAM_LIST_PATH  \
#     --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH  \
#     --num-processes $NUM_PROCESSES

# echo 'Stage 2: Extract Centers'
# python3 src/optimal_centers/get_optimal_centers.py \
#     --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
#     --data-prefix $CROPPED_IMAGE_PATH \
#     --output-exam-list-path $EXAM_LIST_PATH \
#     --num-processes $NUM_PROCESSES

# echo 'Stage 3: Run Classifier'
python3 src/scripts/run_model.py \
    --model-path $MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --segmentation-path $SEG_PATH \
    --output-path $OUTPUT_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER \
    --model-index $MODEL_INDEX \
    #--visualization-flag

