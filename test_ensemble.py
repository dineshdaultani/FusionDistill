import argparse
import collections
import torch
import data_loader.data_loaders as module_data
import model.metric as module_metric
from parse_config import ConfigParser
from utils.data import degradedimagedata as deg_data
from logger import TensorboardWriter
from utils.util import set_seeds
from utils import prepare_device
import copy

# fix random seeds for reproducibility
set_seeds()

def main(config):
    logger = config.get_logger('test')
    logger.info(config)
    device, device_ids = prepare_device(config['n_gpu'])
    
    writer = TensorboardWriter(config.log_dir, logger, 
                               config['trainer']['args']['tensorboard'])
    deg_range = deg_data.get_type_range(config['data_loader']['args']['deg_type'])

    # build model architecture
    if 'model' in config:
        model = config.get_class('model')
    else:
        model = config.get_class('student_model', _class = 'model')
    logger.info(model)

    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))
    model = model.to(device)

    # Loading model paths for all deg models
    logger.info('Loading checkpoints of below models:')
    model_paths = []
    for key, value in config['model'].items():
        if key.startswith('pretrained_path'):
            model_paths.append(value)
            logger.info(value)
    
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    checkpoints = [torch.load(path) for path in model_paths]
    models_all = [copy.deepcopy(model) for _ in range(len(checkpoints))]
    # Loading all models given the model paths for all degradations
    for i, model in enumerate(models_all):
        model.load_state_dict(checkpoints[i]['state_dict']) 
        model = model.to(device)
        model.eval()

    for lev in range(deg_range[0],deg_range[1]+1):
        # setup data_loader instances
        data_loader = getattr(module_data, config['data_loader']['type'])(
            config['data_loader']['args']['data_dir'],
            batch_size=100,
            shuffle=False,
            validation_split=0.0,
            num_workers=2,
            train=False,
            deg_type = config['data_loader']['args']['deg_type'], 
            deg_range = [lev, lev]
        )
        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))

        with torch.no_grad():
            for i, (images, targets) in enumerate(data_loader):
                (image_clean, image_deg) = images
                (labels, _) = targets
                image_clean = image_clean.to(device)
                image_deg = image_deg.to(device)
                target = labels.to(device)
                
                outputs_all, pred_labels_all = [], []
                for i, model in enumerate(models_all):
                    _, _, _, _, feat, output = model(image_deg, image_deg)
                    outputs_all.append(output)    
                
                for output in outputs_all:
                    pred_labels_all.append(torch.argmax(output, dim=1))
                
                # Stack all lists together as tensor
                outputs_all = torch.stack(outputs_all)
                pred_labels_all = torch.stack(pred_labels_all)
                
                # Transpose the tensors to apply single image-wise operations
                outputs_all = torch.permute(outputs_all, (1, 0, 2))
                pred_labels_all = pred_labels_all.T
                # Take the sum of prob and then max of all predictions
                outputs_all_sum_max = torch.argmax(outputs_all.sum(dim=1), dim=1)
                
                ensemble_outputs = []
                # Iterate over each sub-tensor along the first dimension
                for i, sub_tensor in enumerate(pred_labels_all):
                    values, counts = torch.unique(sub_tensor, return_counts=True)
                    max_count = counts.max()
                    mode_values = values[counts == max_count]
                    
                    # Breaking the pluraity ensemble tie here
                    if len(mode_values) > 1:
                        ensemble_outputs.append(outputs_all_sum_max[i])
                    else:
                        ensemble_outputs.append(mode_values[0])
                ensemble_outputs = torch.stack(ensemble_outputs)

                batch_size = image_clean.shape[0]
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(ensemble_outputs, target) * batch_size

        n_samples = len(data_loader.sampler)
        log = {'deg_level': lev}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        writer.set_step(lev, mode = 'eval')
        for met, val in log.items():
            writer.add_scalar(met, val)
        logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Degraded Image Classification - KD')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-m', '--mode', default='eval', type=str,
                      help='Activate eval mode for config')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--dt', '--deg_type'], type=str, target='data_loader;args;deg_type')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
