

# Set Environment Variables
VIRTUAL_ENV_NAME=cloud_ml
DATASET_TO_SCORE="nfldata2.csv"  # This is a CSV file that is located in ../data/
PATH_TO_MODEL="/tmp/model.joblib"



# Activate Conda Env
echo "[ INFO ] Activating conda virtual env ($VIRTUAL_ENV_NAME)"
source activate $VIRTUAL_ENV_NAME



# Train Sklearn Model
python sklearn_nfl_model_batch.py \
    --path_to_data=$DATASET_TO_SCORE \
    --path_to_model=$PATH_TO_MODEL



# Deactivate Conda Virtual Env
echo "[ INFO ] Deactivating conda virtual env..."
source deactivate



#ZEND
