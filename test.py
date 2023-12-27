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

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
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
                
                _, _, _, _, feat, output = model(image_deg, image_deg)

                batch_size = image_clean.shape[0]
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size

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
