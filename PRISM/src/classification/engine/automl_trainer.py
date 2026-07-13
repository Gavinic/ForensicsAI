import glob
import json
import os
import time

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from config.default import cfg
from numpy import tile
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, models, transforms
from utils.compare import count
from utils.create_dir import create_dir
from utils.log import get_logger
from utils.lr_scheduler import cos_lr_scheduler, exp_lr_scheduler
from utils.split_train_val import split_train_val


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

        self.train_set, self.val_set = split_train_val(
            self.train_dir, ratio=(0.99, 0.01)
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_model(self, model, criterion, optimizer, lr_scheduler):

        self.logger.info("Using: {}".format(self.model_name))
        self.logger.info("Using the GPU: {}".format(self.gpu_id))
        self.logger.info("start training...")
        train_loss = []
        since = time.time()
        best_acc = 0.0

        # Cosine annealing strategy

        if lr_scheduler is cos_lr_scheduler or lr_scheduler is cos_lr_scheduler_normal:
            return_lr_scheduler = lr_scheduler(optimizer)
        if lr_scheduler is exp_lr_scheduler:
            return_lr_scheduler = lr_scheduler(optimizer)

        for epoch in range(self.num_epochs):

            begin_time = time.time()
            self.logger.info("-" * 10)
            self.logger.info("Epoch {}/{}".format(epoch, self.num_epochs - 1))
            self.logger.info("train_set size:{}".format(len(self.train_set)))
            # self.logger.info('learning rate:{}'.format(return_lr_scheduler.get_last_lr()[0]))
            self.logger.info("-" * 10)
            running_loss = 0.0
            running_corrects = 0
            iteration = 0
            for i, data in enumerate(self.train_loader):
                model.train()
                iteration += 1
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)
                loss.backward()
                optimizer.step()

                if (
                    i % self.print_interval == 0
                    or outputs.size()[0] < self.train_batch_size
                ):
                    spend_time = time.time() - begin_time
                    self.logger.info(
                        " Epoch:{}({}/{}) loss:{:.3f} learning rate:{}".format(
                            epoch,
                            iteration,
                            len(self.train_set) // self.train_batch_size,
                            loss.item(),
                            return_lr_scheduler.get_last_lr()[0],
                        )
                    )
                    train_loss.append(loss.item())
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # self.test_infer(model, epoch)
            val_acc = self.test_model(model, criterion)
            epoch_loss = running_loss / len(self.train_set)
            epoch_acc = running_corrects.double() / len(self.train_set)
            return_lr_scheduler.step(epoch + i / len(self.train_loader))
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
            if epoch_acc > val_acc and epoch > self.min_save_epoch:
                best_acc = val_acc
            save_dir = os.path.join(self.out_dir, self.model_name)
            model_out_path = (
                save_dir + "/" + "{}_".format(self.model_name) + str(epoch) + ".pth"
            )

            torch.save(model.module.state_dict(), model_out_path)
        time_elapsed = time.time() - since
        self.logger.info("Best val f1: {}".format(best_acc))
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
        for data in self.val_loader:
            inputs, labels = data
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
            running_corrects += torch.sum(preds == labels.data).to(torch.float32)
            cont += 1
        _, _, f_class, _ = precision_recall_fscore_support(
            y_true=labels_list, y_pred=pres_list, labels=[0, 1, 2, 3], average=None
        )
        fper_class = {
            "smooth": f_class[0],
            "slow": f_class[1],
            "congested": f_class[2],
            "closed": f_class[3],
        }
        submit_score = (
            0.1 * f_class[0] + 0.2 * f_class[1] + 0.3 * f_class[2] + 0.4 * f_class[3]
        )
        self.logger.info(
            "Per-class F1:{}  Weighted F-score:{}".format(fper_class, submit_score)
        )
        self.logger.info(
            "val_size: {}  valLoss: {:.4f} valAcc: {:.4f}".format(
                len(self.val_set),
                running_loss / len(self.val_set),
                running_corrects.double() / len(self.val_set),
            )
        )
        return submit_score

    @staticmethod
    def get_test_transforms():
        return A.Compose(
            [
                A.Resize(height=500, width=900),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=1
                ),
                A.Normalize(
                    mean=(0.446, 0.469, 0.472),
                    std=(0.326, 0.330, 0.338),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )

    def test_infer(self, model, epoch):
        pre_result = []
        pre_name = []
        pre_dict = {}
        test_json = "<BASE_PATH>/GD/test.json"
        image_paths = sorted(glob.glob(self.test_dir + "/*/*"))
        for index in range(len(image_paths)):
            sample_path = image_paths[index]
            img = Image.open(sample_path)
            img = img.convert("RGB")
            img = np.array(img)

            transforms = self.get_test_transforms()
            input = transforms(image=img)["image"]
            input = input.unsqueeze(0)
            input = input.float()
            input = input.cuda()

            with torch.no_grad():
                output = model(input)

            _, preds = torch.max(output.data, 1)
            pre_result += preds.cpu().numpy().tolist()
            pre_name.append(
                sample_path.split("/")[-2] + "_" + sample_path.split("/")[-1]
            )

        for idx in range(len(pre_result)):
            pre_dict[pre_name[idx]] = pre_result[idx]

        with open(test_json) as f:
            submit = json.load(f)
        submit_annos = submit["annotations"]
        submit_result = []
        for i in range(len(submit_annos)):
            submit_anno = submit_annos[i]
            imgId = submit_anno["id"]
            frame_name = [imgId + "_" + i["frame_name"] for i in submit_anno["frames"]]
            status_all = [pre_dict[i] for i in frame_name]
            status = max(status_all, key=status_all.count)
            submit["annotations"][i]["status"] = status

        submit_json = "<BASE_PATH>/GD/result_b.json"
        json_data = json.dumps(submit)
        with open(submit_json, "w") as w:
            w.write(json_data)
        count_result = count(submit_json)
        self.logger.info(
            "{} epoch {} prediction result:{}".format(
                self.model_name, epoch, count_result
            )
        )
