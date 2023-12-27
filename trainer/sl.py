import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import model.loss as module_loss
import torch.nn as nn
import torch.nn.functional as F

class SLTrainer(BaseTrainer):
    """
    Trainer class for fine-tuning the base model trained on clean images for specific degradation, i.e., step-2
    """
    def __init__(self, metric_ftns, config, train_data_loader, valid_data_loader=None, 
                 len_epoch=None):
        super().__init__(metric_ftns, config, train_data_loader, valid_data_loader, len_epoch)
        self.teacher, self.student = self._build_model(config)
        self.criterion = self._load_loss(config)
        self.optimizer = self._load_optimizer(self.student, config)
        self.lr_scheduler = self._load_scheduler(self.optimizer, config)
        self.config = config
        self.loss_names = {type: loss for losses in config['loss'] for type, loss in losses.items()}
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        train_misc_metrics = ['loss', 'sup_loss', 'inh_loss', 'lr']
        valid_misc_metrics = ['loss', 'sup_loss', 'inh_loss']
        self.train_metrics = MetricTracker(*train_misc_metrics, 
                                           *[m.__name__ + '_' + self.deg_flag for m in self.metric_ftns], 
                                           writer=self.writer)
        self.valid_metrics = MetricTracker(*valid_misc_metrics, 
                                           *[m.__name__ + '_' + self.deg_flag for m in self.metric_ftns], 
                                           writer=self.writer)

    def _build_model(self, config):
        """
        Building model from the configuration file

        :param config: config file
        :return: model with loaded state dict
        """
        # build model architecture, then print to console
        teacher = config.get_class('teacher_model', _class = 'model')
        student = config.get_class('student_model', _class = 'model')
        self.logger.info('Teacher Network: {} \n Student Network: {}'.format(teacher, student))
        self.logger.info("Loading checkpoint for teacher: {} ...".format(
            config['teacher_model']['pretrained_path']))
        
        teacher = teacher.to(self.device)
        student = student.to(self.device)
        if len(self.device_ids) > 1:
            teacher = torch.nn.DataParallel(teacher, device_ids=self.device_ids)
            student = torch.nn.DataParallel(student, device_ids=self.device_ids)
        
        checkpoint = torch.load(config['teacher_model']['pretrained_path'])
        teacher.load_state_dict(checkpoint['state_dict'])
        if 'pretrained_path' in config['student_model']:
            checkpoint = torch.load(config['student_model']['pretrained_path'])
            student.load_state_dict(checkpoint['state_dict'])
        # Feezing parameters of teacher
        if self.config['student_model']['type'].startswith('ShakePyramidNet'):
            for param in teacher.parameters():
                param.requires_grad = False
        
        return teacher, student
    
    def _load_loss(self, config):
        """
        Build model from the configuration file

        :param config: config file
        :return: criterion dictionary in the format: {loss_type: loss}
        """
        criterion = {type: getattr(module_loss, type)(loss) for losses in config['loss'] \
                        for type, loss in losses.items()}
        return criterion

    def _load_optimizer(self, model, config):
        """
        Load optimizer from the configuration file

        :param model: model for which optimizer is to be initialized
        :param config: config file
        :return: initialized optimizer
        """
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        return optimizer

    def _load_scheduler(self, optimizer, config):
        """
        Load scheduler from the configuration file

        :param optimizer: optimizer for which scheduler is to be initialized
        :param config: config file
        :return: initialized scheduler
        """
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        return lr_scheduler

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.teacher.eval()
        self.student.train()
        self.train_metrics.reset()
        for batch_idx, (images, targets) in enumerate(self.train_data_loader):
            (image_clean, image_deg) = images
            (labels, _) = targets
            image_clean = image_clean.to(self.device)
            image_deg = image_deg.to(self.device)
            target = labels.to(self.device)
            if self.loss_names['inheritance_loss'] == 'COS':
                dum = torch.ones((image_clean.size(0),))
                dum = dum.to(self.device)
            
            self.optimizer.zero_grad()
            stem_t, rb1_t, rb2_t, rb3_t, t_feat, t_out = self.teacher(image_clean, image_deg)
            stem_s, rb1_s, rb2_s, rb3_s, s_feat, s_out = self.student(image_clean, image_deg)

            sup_loss = self.criterion['supervised_loss'](s_out, target) * \
                    self.config['loss_weights'][0]
            inh_loss = 0
            if self.loss_names['inheritance_loss'] == 'COS':
                inh_loss += self.criterion['inheritance_loss'](rb1_s[1], rb1_t[1].detach(), dum) * \
                            self.config['loss_weights'][1][0]
                inh_loss += self.criterion['inheritance_loss'](rb2_s[1], rb2_t[1].detach(), dum) * \
                            self.config['loss_weights'][1][1]
                inh_loss += self.criterion['inheritance_loss'](rb3_s[1], rb3_t[1].detach(), dum) * \
                            self.config['loss_weights'][1][2]
            elif self.loss_names['inheritance_loss'] == 'AT':
                inh_loss += self.criterion['inheritance_loss'](rb1_s[1], rb1_t[1].detach()) * \
                            self.config['loss_weights'][1][0]
                inh_loss += self.criterion['inheritance_loss'](rb2_s[1], rb2_t[1].detach()) * \
                            self.config['loss_weights'][1][1]
                inh_loss += self.criterion['inheritance_loss'](rb3_s[1], rb3_t[1].detach()) * \
                            self.config['loss_weights'][1][2]
            elif self.loss_names['inheritance_loss'] == 'KLD':
                inh_loss += self.criterion['inheritance_loss'](torch.log(torch.clamp(F.softmax(rb3_s[1],dim=1),1e-10,1.0)), 
                                                            torch.clamp(F.softmax(rb3_t[1], dim=1),1e-10,1.0)) * \
                            self.config['loss_weights'][1][2]
            else:
                raise NotImplementedError
            # inh_loss = inh_loss / (self.config['loss_weights'][1][0] + 
            #                        self.config['loss_weights'][1][1] + 
            #                        self.config['loss_weights'][1][2])
            loss = sup_loss + inh_loss
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('inh_loss', inh_loss.item())
            self.train_metrics.update('sup_loss', sup_loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__ + '_' + self.deg_flag, 
                                          met(s_out, target))
            self.train_metrics.update('lr', self.lr_scheduler.get_last_lr()[0])
            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    'Train Epoch: {} {} Loss: {:.6f} Sup Loss: {:.6f} Inh Loss: {:.6f}'.format(
                        epoch, self._progress(batch_idx), loss.item(), 
                        sup_loss.item(), inh_loss.item()))
                # self.writer.add_image('input_clean', make_grid(image_clean.cpu(), nrow=8, normalize=True))
                # self.writer.add_image('input_deg', make_grid(image_deg.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            self.logger.info('Testing on validation data')
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.teacher.eval()
        self.student.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.valid_data_loader):
                (image_clean, image_deg) = images
                (labels, _) = targets
                image_clean = image_clean.to(self.device)
                image_deg = image_deg.to(self.device)
                target = labels.to(self.device)
                if self.loss_names['inheritance_loss'] == 'COS':
                    dum = torch.ones((image_clean.size(0),))
                    dum = dum.to(self.device)
                
                self.optimizer.zero_grad()
                stem_t, rb1_t, rb2_t, rb3_t, t_feat, t_out = self.teacher(image_clean, image_deg)
                stem_s, rb1_s, rb2_s, rb3_s, s_feat, s_out = self.student(image_clean, image_deg)

                sup_loss = self.criterion['supervised_loss'](s_out, target) * \
                self.config['loss_weights'][0]
                inh_loss = 0
                if self.loss_names['inheritance_loss'] == 'COS':
                    inh_loss += self.criterion['inheritance_loss'](rb1_s[1], rb1_t[1].detach(), dum) * \
                                self.config['loss_weights'][1][0]
                    inh_loss += self.criterion['inheritance_loss'](rb2_s[1], rb2_t[1].detach(), dum) * \
                                self.config['loss_weights'][1][1]
                    inh_loss += self.criterion['inheritance_loss'](rb3_s[1], rb3_t[1].detach(), dum) * \
                                self.config['loss_weights'][1][2]
                elif self.loss_names['inheritance_loss'] == 'AT':
                    inh_loss += self.criterion['inheritance_loss'](rb1_s[1], rb1_t[1].detach()) * \
                                self.config['loss_weights'][1][0]
                    inh_loss += self.criterion['inheritance_loss'](rb2_s[1], rb2_t[1].detach()) * \
                                self.config['loss_weights'][1][1]
                    inh_loss += self.criterion['inheritance_loss'](rb3_s[1], rb3_t[1].detach()) * \
                                self.config['loss_weights'][1][2]
                elif self.loss_names['inheritance_loss'] == 'KLD':
                    inh_loss += self.criterion['inheritance_loss'](torch.log(torch.clamp(F.softmax(rb3_s[1],dim=1),1e-10,1.0)), 
                                                                torch.clamp(F.softmax(rb3_t[1], dim=1),1e-10,1.0)) * \
                                self.config['loss_weights'][1][2]
                else:
                    raise NotImplementedError
                # inh_loss = inh_loss / (self.config['loss_weights'][1][0] + 
                #                     self.config['loss_weights'][1][1] + 
                #                     self.config['loss_weights'][1][2])
                loss = sup_loss + inh_loss

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                self.valid_metrics.update('inh_loss', inh_loss.item())
                self.valid_metrics.update('sup_loss', sup_loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__ + '_' + self.deg_flag, 
                                              met(s_out, target))
                # self.writer.add_image('input_clean', make_grid(image_clean.cpu(), nrow=8, normalize=True))
                # self.writer.add_image('input_deg', make_grid(image_deg.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.student.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        model_name = type(self.student).__name__
        state = {
            'model': model_name,
            'epoch': epoch,
            'state_dict': self.student.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        # torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, config):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(config.resume)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['student_model'] != self.config['student_model']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.student.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
