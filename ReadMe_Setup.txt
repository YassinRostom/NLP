# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Note: I have to add the spaCy small model manually using the .tar file
# I have downloaded the .tar file from the internet
pip install en_core_web_sm-3.8.0.tar.gz