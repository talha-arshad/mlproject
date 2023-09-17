# Intro
Starter code for a generic machine learning project.

# Setup
    git clone mlproject

    EITHER create a new environment:
    conda create --name mlprojectenv --file mlproject/requirements.txt
    conda activate mlprojectenv

    OR use an existing environment:
    conda activate myenv
    conda install --file mlproject/requirements.txt

    ln -s mlproject/setup.py 
    pip install -e .

# Use
    cd mlproject
    python training/train.py runs/config.yaml
