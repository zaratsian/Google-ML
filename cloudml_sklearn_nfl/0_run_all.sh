

##################################################################################################
#
#   Google Cloud ML Demo - Sklearn NFL Play Prediction
#
#   This script will call the Sklearn model training script, create and save the ML model,
#   load to Google Cloud Storage, deploy to Google CLoud ML, then make a few predictions.
#
#   Reference:
#       Runtime Versions: https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list
#
##################################################################################################


##################################################################################################
# Set ENV variables
##################################################################################################
echo "[ INFO ] Setting env variables"

VIRTUAL_ENV_NAME=cloud_ml

BUCKET_NAME=sklearn_nfl_model
REGION=us-east1
#PROJECT_ID=$(gcloud config list project --format "value(core.project)")

MODEL_NAME=$BUCKET_NAME
MODEL_DIR="gs://$BUCKET_NAME/"
VERSION_NAME="v1"
FRAMEWORK="SCIKIT_LEARN"
INPUT_VARIABLES_FILE="input.json"



##################################################################################################
# Create model and save to cloud storage
##################################################################################################


# Create Conda Virtual Env (NOTE: This only needs to be ran one time to get the vir env setup)
#echo "[ INFO ] Creating conda virtual env ($VIRTUAL_ENV_NAME)"
#conda create --name $VIRTUAL_ENV_NAME python=3.5 \
#    numpy=1.14.5 \
#    pandas=0.23.3 \
#    scipy=1.1.0 \
#    scikit-learn=0.19.2

echo "[ INFO ] Activating conda virtual env ($VIRTUAL_ENV_NAME)"
source activate $VIRTUAL_ENV_NAME

#conda env list
#conda remove --name ato_sklearn_basic --all
#source deactivate


# Build sklearn model pipeline object
echo "[ INFO ] Training and saving Sklearn Model"
python ./sklearn_nfl_model.py


# Exit Virtual Env
echo "[ INFO ] Deactivating conda virtual env..."
source deactivate


# Setup Cloud Storage Bucket
echo "[ INFO ] Creating Cloud Storage Bucket at gs://$BUCKET_NAME"
gsutil mb -l $REGION gs://$BUCKET_NAME


# Upload model.joblib to your GCS Bucket
echo "[ INFO ] Uploading model.joblib to gs://$BUCKET_NAME"
gsutil cp ./model.joblib gs://$BUCKET_NAME/model.joblib



##################################################################################################
# Deploy Model to Google Cloud ML
##################################################################################################


# Create model resource
echo "[ INFO ] Creating Cloud ML model called $MODEL_NAME"
gcloud ml-engine models create "$MODEL_NAME" --regions $REGION

# Create model version on Cloud ML
echo "[ INFO ] Creating Cloud ML model version ($VERSION_NAME)"
gcloud ml-engine versions create $VERSION_NAME \
    --model $MODEL_NAME --origin $MODEL_DIR \
    --runtime-version=1.10 --framework $FRAMEWORK \
    --python-version=3.5

# Verify / check model version info
gcloud ml-engine versions describe $VERSION_NAME --model $MODEL_NAME

# List Model versions
gcloud ml-engine versions list --model=$MODEL_NAME

# List Cloud ML Models
gcloud ml-engine models list



##################################################################################################
# Online Predictions
##################################################################################################

# List input variables/data for testing
echo "[ INFO ] Creating file to send to model for scoring..."
echo "['Drive', 'qtr', 'down', 'TimeSecs', 'PlayTimeDiff', 'yrdline100', 'ydstogo', 'ydsnet', 'PosTeamScore', 'DefTeamScore', 'FirstDown', 'posteam', 'DefensiveTeam', 'PlayType_lag', 'PlayType', 'year', 'month', 'day']"
echo "[1, 1, 1, 3600, 0,  10, 10, 18, 0,  0,  1, 24, 18, 0, 1, 2015, 9, 10]" >  input.json
echo "[1, 1, 1, 11,   12, 13, 13, 45, 13, 13, 1, 24, 18, 0, 1, 2015, 9, 10]" >> input.json
echo "[ INFO ] Displaying a few records for the model that will be scored:"
sleep 5
head input.json
sleep 5

# Get predictions
echo "[ INFO ] Scoring data against deployed Sklearn Model"
gcloud ml-engine predict --model $MODEL_NAME --version \
	    $VERSION_NAME --json-instances $INPUT_VARIABLES_FILE



##################################################################################################
# Test (with local predictions)
##################################################################################################

#gcloud ml-engine local predict --model-dir=$MODEL_DIR \
	#    --json-instances $INPUT_VARIABLES_FILE \
	#    --framework $FRAMEWORK


##################################################################################################
# Delete Cloud ML Model (NOTE: Model versions must be deleted first before model can be deleted)
##################################################################################################

# List Model Versions
#gcloud ml-engine versions list --model=$MODEL_NAME

# Delete model versions
#gcloud ml-engine versions delete $VERSION_NAME --model=$MODEL_NAME

# List models
#gcloud ml-engine models list

# Delete model
#gcloud ml-engine models delete $MODEL_NAME



#ZEND
