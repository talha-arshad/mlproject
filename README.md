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

THEN

    ln -s mlproject/setup.py 
    pip install -e .

# Use

    cd mlproject

### Train
Edit runs/config.yaml to define data, model, optimizer and training settings.

Run training:
    
    python training/train.py runs/config.yaml

### Evaluate

Plot the metrics logged during training.

    python evaluate/plot_training.py [--save_file runs/metrics.png]
