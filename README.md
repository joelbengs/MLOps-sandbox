# Below is commands for how to create and manage a python virtual environment.
mkdir sandbox
cd sandbox

# Create venv
python -m venv sandboxvenv # tells python to create a venv named sandboxvenv

# To activate venv (mac, standing in dir)
source sandboxvenv/bin/activate

# To deactivate venv (mac, and not conda envs)
deactivate

# To activate CONDA base venv in mac
source activate base

# To deactivate CONDA
conda deactivate

# run requirements file, located in dir.
pip install -r requirements.txt

# VS CODE
cmd + shift + p = search command palette
cmd + p = searh files
cmd + t = search functions

# Copilot chat 
"New chat in sidebar" is available from the command palettec