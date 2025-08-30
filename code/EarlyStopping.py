import copy
import os

import torch


class EarlyStopping:
    def __init__(self, threshold=0.5, patience=20, best_score=-float('inf'), exp='default', meta=None):
        self.threshold = threshold
        self.patience = patience
        self.best_score = best_score
        self.counter = 0

        self.early_stop = False
        self.best_model = None
        self.best_optimizer = None
        self.best_scheduler = None
        self.best_epoch = 0
        self.current_model = None
        self.new_best_score = False
        self.exp = exp
        self.meta = meta
        os.makedirs(self.exp, exist_ok=True)
        if self.meta is not None:
            torch.save(self.meta, os.path.join(self.exp, '{}'.format('meta.txt')))

    def composite_metric(self, val_loss, val_f1, val_mcc, val_auc, training_loss, train_f1, train_mcc):
        return (val_f1 + val_auc) / 2

    def save_stage(self):
        if not self.early_stop:
            os.makedirs('{}/{}'.format(self.exp, self.best_epoch), exist_ok=True)
            output_dir = os.path.join(self.exp, '{}/{}'.format(self.best_epoch, 'model.bin'))
        else:
            output_dir = os.path.join(self.exp, '{}'.format('model.bin'))

        torch.save({
            'epoch': self.best_epoch,
            'patience': self.patience,
            'model_state_dict': self.best_model,
            'optimizer_state_dict': self.best_optimizer,
            'scheduler': self.best_scheduler}, output_dir)

    def __call__(self, model, optimizer, scheduler, epoch, val_loss=None, val_f1=None, val_mcc=None, training_loss=None, train_f1=None, train_mcc=None, val_auc=None):
        composite_score = self.composite_metric(val_loss, val_f1, val_mcc,val_auc, training_loss, train_f1, train_mcc)
        print(
            f"Epoch {epoch}: Val Loss {val_loss} , F1 {val_f1} , MCC {val_mcc} , AUC {val_auc}, Training Loss {training_loss} , Composite Score {composite_score} , Current Best Score {self.best_score}",
            flush=True)
        if epoch >= 10 and composite_score >= self.best_score:
            self.best_score = composite_score
            self.counter = 0
            print(f"new best score f1 + mcc / 2:  {self.best_score}", flush=True)
            self.new_best_score = True
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_optimizer = copy.deepcopy(optimizer.state_dict())
            self.best_scheduler = copy.deepcopy(scheduler.state_dict())
            self.best_epoch = epoch
            self.save_stage()
        elif epoch >= 10:
            self.new_best_score = False
            self.counter += 1
            print(f"patient {self.counter} of {self.patience}", flush=True)
            if self.counter >= self.patience:
                self.early_stop = True
                self.save_stage()
                return True
        return False
