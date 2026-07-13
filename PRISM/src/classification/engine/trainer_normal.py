import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config.default import cfg
from numpy import tile
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms
from utils.create_dir import create_dir
from utils.dataset_n import Garbage_Dataset
from utils.log import get_logger


class BASE:

    def __init__(self, cfg):

        self.gpu_id = cfg.SYSTEM.GPU_ID
        self.num_workers = cfg.SYSTEM.NUM_WORKERS
        self.train_dir = cfg.DATASET.TRAIN_DIR
        self.val_dir = cfg.DATASET.VAL_DIR
        self.test_dir = cfg.DATASET.TEST_DIR
        self.sub_dir = cfg.OUTPUT_DIR.SUB_DIR
        self.log_dir = cfg.OUTPUT_DIR.LOG_DIR
        self.out_dir = cfg.OUTPUT_DIR.OUT_DIR
        self.model_name = cfg.MODEL.MODEL_NAME
        self.img_size = cfg.TRAIN_PARAM.IMAGE_SIZE
        self.train_batch_size = cfg.TRAIN_PARAM.TRAIN_BATCH_SIZE
        self.val_batch_size = cfg.TRAIN_PARAM.VAL_BATCH_SIZE
        self.test_batch_size = cfg.TRAIN_PARAM.TEST_BATCH_SIZE
        self.momentum = cfg.TRAIN_PARAM.MOMENTUM
        self.weight_decay = cfg.TRAIN_PARAM.WEIGHT_DECAY
        self.num_epochs = cfg.TRAIN_PARAM.NUM_EPOCHS
        self.lr = cfg.TRAIN_PARAM.LR
        self.val_interval = cfg.TRAIN_PARAM.VAL_INTERVAl
        self.print_interval = cfg.TRAIN_PARAM.PRINT_INTERVAL
        self.min_save_epoch = cfg.TRAIN_PARAM.MIN_SAVE_EPOCH

        create_dir(self.out_dir)
        create_dir(self.log_dir)
        create_dir(os.path.join(self.out_dir, self.model_name))
        self.logger = get_logger(os.path.join(self.log_dir, self.model_name + ".log"))

    def loaddata(self, train_dir, batch_size, shuffle, is_train=True):

        image_datasets = Garbage_Dataset(
            train_dir, is_train=is_train, img_size=self.img_size
        )
        dataset_loaders = torch.utils.data.DataLoader(
            image_datasets,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        data_set_sizes = len(image_datasets)
        return dataset_loaders, data_set_sizes

    def train_model(self, model, criterion, optimizer, lr_scheduler):

        self.logger.info("Using: {}".format(self.model_name))
        self.logger.info("Using the GPU: {}".format(self.gpu_id))
        self.logger.info("img size {}".format(self.img_size))
        self.logger.info("start training...")
        train_loss = []
        since = time.time()
        best_acc = 0.0

        # Modification 1: correctly initialize the EMA model with config parameters
        ema_model = ModelEmaV3(
            model,
            decay=0.999,
            update_after_step=100,
            use_warmup=True,
            device=torch.device("cuda"),
        )

        # Modification 2: add a global step counter
        global_step = 0

        for epoch in range(self.num_epochs):

            begin_time = time.time()
            data_loaders, dset_sizes = self.loaddata(
                train_dir=self.train_dir,
                batch_size=self.train_batch_size,
                shuffle=True,
                is_train=True,
            )
            self.logger.info("-" * 10)
            self.logger.info("Epoch {}/{}".format(epoch, self.num_epochs - 1))
            self.logger.info(
                "learning rate:{}".format(optimizer.param_groups[-1]["lr"])
            )
            self.logger.info("-" * 10)
            running_loss = 0.0
            running_corrects = 0
            count = 0
            for i, data in enumerate(data_loaders):
                model.train()
                count += 1
                # Modification 2: increment the global step
                global_step += 1

                inputs, labels = data
                labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Modification 2: pass the global_step parameter
                ema_model.update(model, step=global_step)

                if (
                    i % self.print_interval == 0
                    or outputs.size()[0] < self.train_batch_size
                ):
                    spend_time = time.time() - begin_time
                    self.logger.info(
                        " Epoch:{}({}/{}) loss:{:.3f} learning rate:{}".format(
                            epoch,
                            count,
                            dset_sizes // self.train_batch_size,
                            loss.item(),
                            optimizer.param_groups[-1]["lr"],
                        )
                    )
                    train_loss.append(loss.item())
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            val_acc, val_loss = self.test_model(ema_model.module, criterion)
            print(val_acc, val_loss)
            val_acc, val_loss = self.test_model(model, criterion)
            lr_scheduler.step()
            epoch_loss = running_loss / dset_sizes
            epoch_acc = running_corrects.double() / dset_sizes

            self.logger.info(
                "Epoch:[{}/{}] Loss={:.5f}  Acc={:.3f} Epoch_Time:{} min: ETA: {} hours".format(
                    epoch,
                    self.num_epochs - 1,
                    epoch_loss,
                    epoch_acc,
                    spend_time / 60,
                    (self.num_epochs - epoch) * spend_time / 3600,
                )
            )
            if val_acc > best_acc and epoch > self.min_save_epoch:
                best_acc = val_acc
                best_model_wts = model.module.state_dict()
            # if val_acc > 0.999:
            #     break
            save_dir = os.path.join(self.out_dir, self.model_name)
            model_out_path = (
                save_dir + "/" + "{}_".format(self.model_name) + str(epoch) + ".pth"
            )
            torch.save(model.module.state_dict(), model_out_path)

            model_out_path = (
                save_dir + "/" + "{}_".format(self.model_name) + str(epoch) + "_ema.pth"
            )
            torch.save(ema_model.module.module.state_dict(), model_out_path)
        # save best model
        self.logger.info("Best Accuracy: {}".format(best_acc))
        # model.load_state_dict(best_model_wts)
        best_model_out_path = save_dir + "/" + "{}_best.pth".format(self.model_name)
        torch.save(best_model_wts, best_model_out_path)
        time_elapsed = time.time() - since
        self.logger.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    def test_model(self, model, criterion):

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        cont = 0
        outPre = []
        outLabel = []
        pres_list = []
        labels_list = []
        data_loaders, dset_sizes = self.loaddata(
            train_dir=self.val_dir,
            batch_size=self.val_batch_size,
            shuffle=False,
            is_train=False,
        )
        for data in data_loaders:
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            if cont == 0:
                outPre = outputs.data.cpu()
                outLabel = labels.data.cpu()
            else:
                outPre = torch.cat((outPre, outputs.data.cpu()), 0)
                outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
            pres_list += preds.cpu().numpy().tolist()
            labels_list += labels.data.cpu().numpy().tolist()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            cont += 1

        self.logger.info(
            "val_size: {}  valLoss: {:.4f} valAcc: {:.4f}".format(
                dset_sizes,
                running_loss / dset_sizes,
                running_corrects.double() / dset_sizes,
            )
        )
        return running_corrects.double() / dset_sizes, running_loss / dset_sizes
