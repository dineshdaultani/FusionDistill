import argparse
import collections
import data_loader.data_loaders as module_data
import model.metric as module_metric
from parse_config import ConfigParser
from utils.util import set_seeds, set_seeds_prev

def main(config):
    logger = config.get_logger('train')
    logger.info(config)

    # setup data_loader instances
    train_data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        validation_split=0.0,
        num_workers=2,
        train=False,
        deg_type = config['data_loader']['args']['deg_type']
    )

    Trainer = config.get_class('trainer', init = False)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    trainer = Trainer(metrics, config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Degraded Image Classification - KD')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--dt', '--deg_type'], type=str, target='data_loader;args;deg_type'),
        CustomArgs(['--rs', '--random_seed'], type=int, target='random_seed')
    ]
    config = ConfigParser.from_args(args, options)
    
    # fix random seeds for reproducibility
    if 'random_seed' in config:
        set_seeds(config['random_seed'])
    else:
        # Provides backward compability for previous experiments
        set_seeds_prev()
    main(config)
