########
#Python script to run main.py with all required arguments
########

#the following paths need to be adapted
INCEPTION_GRAPH_ROOT="/home/james/Documents/Seagrass-Repository/Inception-v4"
DATASET_ROOT="/home/james/Documents/Seagrass-Repository/Images"
RESULTS_ROOT="/home/james/Documents/Seagrass-Repository/Results"
DATA_CONFIG_FILE="data_config.txt"
LOGGING_FILE="/output_log.txt"
CHECKPOINTS="/checkpoints"
MODELS="/models"
USING_SMALL=0
#set to 1 if model is trained and just needs loading
SKIP_TRAINING=0

MAIN_PROGRAM="../main.py"

python3 $MAIN_PROGRAM \
--graph $INCEPTION_GRAPH_ROOT \
--root_img_dir $DATASET_ROOT \
--image_data_file "$DATASET_ROOT/$DATA_CONFIG_FILE" \
--results_dir $RESULTS_ROOT \
--logging_file $LOGGING_FILE \
--checkpoint_dir $CHECKPOINTS \
--model_dir $MODELS \
--using_small $USING_SMALL \
