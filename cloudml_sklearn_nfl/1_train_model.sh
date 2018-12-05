

# Set Environment Variables
TRAINING_DATA="nfldata2.csv"  # This is a CSV file that is located in ../data/
TARGET_VARIABLE_NAME="Yards_Gained"
VIRTUAL_ENV_NAME=cloud_ml
GCS_BUCKET_PATH="gs://sklearn_nfl_model"



# Activate Conda Env
echo "[ INFO ] Activating conda virtual env ($VIRTUAL_ENV_NAME)"
source activate $VIRTUAL_ENV_NAME



# Train Sklearn Model
python sklearn_nfl_model.py \
    --training_data=$TRAINING_DATA \
    --target_variable_name=$TARGET_VARIABLE_NAME \
#    --gcs_bucket_path=$GCS_BUCKET_PATH



# Deactivate Conda Virtual Env
echo "[ INFO ] Deactivating conda virtual env..."
source deactivate



#ZEND
