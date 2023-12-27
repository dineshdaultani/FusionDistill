import os
import sys
# Adding the parent directory to the sys path, fix so that this file can be run from utils dir.
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(1, parent_dir)

import argparse
import torch
import copy
from parse_config import ConfigParser
from utils import read_yaml, write_yaml
from datetime import datetime

def model_fusion(dataset='CIFAR10'):
    """
    This function is used to perform the fusion of several fine-tuned individual degradation models, 
    i.e., Step-3 of our proposed method.

    Example usage: python utils/model_soups.py --dataset CIFAR10
    It will generate a combined model in the saved/combined_deg/SLTrainer/ResNet56-56_CIFAR10_soups/train/ directory.
    """
    # Load the configuration file
    config_file =  'configs/deg_all/{}/ResNet56_soups.yaml'.format(dataset.lower())
    config = ConfigParser(read_yaml(config_file), dry_run = True)

    # Get the model from the configuration
    model = config.get_class('model')

    # Extract the pretrained model paths from the configuration
    model_paths = []
    for key, value in config['model'].items():
        if key.startswith('pretrained_path'):
            model_paths.append(value)

    # Load the checkpoints of the pretrained individual degradation models
    checkpoints = [torch.load(path) for path in model_paths]

    # Initialize the models and load their state
    models_all = [copy.deepcopy(model) for _ in range(len(checkpoints))]
    for i, model in enumerate(models_all):
        model.load_state_dict(checkpoints[i]['state_dict']) 
        
    # Combine the weights of the model
    combined_model = None
    global_count = 0
    for model in models_all:
        if combined_model is None:
            combined_model = copy.deepcopy(model)
        else:
            for param_q, param_k in zip(model.parameters(), combined_model.parameters()):
                param_k.data = (param_k.data * global_count + param_q.data) / (1. + global_count)
        global_count += 1
        
    # Prepare the checkpoint directory and model state
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    checkpoint_dir = 'saved/combined_deg/SLTrainer/ResNet56-56_{}_soups/train/{}/'.format(dataset, run_id)
    model_name = type(combined_model).__name__
    config.config['name'] = config.config['name'] + '_soups'
    state = {
        'model': model_name,
        'state_dict': combined_model.state_dict(),
        'config': config
    }

    # Save the combined model and the configuration
    model_path = checkpoint_dir + 'model_best.pth'
    os.makedirs(checkpoint_dir, exist_ok=True)
    write_yaml(config.config, checkpoint_dir + 'config.yaml')
    torch.save(state, model_path)
    print('saved combined model:', model_path)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combine Models')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset to use such as CIFAR100 or TinyImagenet')
    args = parser.parse_args()

    # Call the model fusion function
    model_fusion(args.dataset)
