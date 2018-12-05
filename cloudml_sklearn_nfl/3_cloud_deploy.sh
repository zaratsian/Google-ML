

##################################################################################################
#
#   Google Cloud ML Demo - Sklearn NFL Play Prediction
#
#
#   Reference:
#       Runtime Versions: https://cloud.google.com/ml-engine/docs/tensorflow/runtime-version-list
#
##################################################################################################


##################################################################################################
# Set ENV variables
##################################################################################################
echo "[ INFO ] Setting env variables"



BUCKET_NAME=sklearn_nfl_model
REGION=us-east1
#PROJECT_ID=$(gcloud config list project --format "value(core.project)")

MODEL_NAME=$BUCKET_NAME
MODEL_LOCAL_DIR=/tmp/model.joblib
MODEL_DIR="gs://$BUCKET_NAME/"
VERSION_NAME="v1"
FRAMEWORK="SCIKIT_LEARN"
INPUT_VARIABLES_FILE="input.json"



##################################################################################################
# Save model.joblib to cloud storage
##################################################################################################


# Setup Cloud Storage Bucket
echo "[ INFO ] Creating Cloud Storage Bucket at gs://$BUCKET_NAME"
gsutil mb -l $REGION gs://$BUCKET_NAME


# Upload model.joblib to your GCS Bucket
echo "[ INFO ] Uploading model.joblib to gs://$BUCKET_NAME"
gsutil cp $MODEL_LOCAL_DIR gs://$BUCKET_NAME/model.joblib



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
