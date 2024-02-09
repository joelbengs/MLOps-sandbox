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

# Too many loose objects warning
You can clean up these loose objects by running the following commands:

The git gc command runs a number of housekeeping tasks within the repository, such as compressing file revisions (to reduce disk space and increase performance) and removing unreachable objects which may have been created from prior invocations of git add.

The git repack command is used to combine all objects that do not currently reside in a "pack", into a pack. It can greatly reduce the disk space of your repository.

After running these commands, you should remove the .git/gc.log file as suggested by the warning message:

Please make sure to backup your work before running these commands