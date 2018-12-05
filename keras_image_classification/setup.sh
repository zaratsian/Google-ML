echo "Creating Keras Virtual Env with Python 3.6..."
sleep 5
conda create --name keras_venv python=3.6 -y
echo ""
echo ""
echo "Activating Virtual Env..."
sleep 3
source activate keras_venv
#source deactivate                   # Deactivate venv
#conda env list                      # List Virtual Environments
#conda remove --name myenv --all     # Remove / Delete venv 
echo ""
echo ""
echo "Installing libraries from requirements.txt"
pip install --upgrade -r requirements.txt
echo ""
echo ""
echo "Setup Complete!"
echo ""
