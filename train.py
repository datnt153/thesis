import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path
import wandb 
from time import strftime


from data import MyDataset
from sklearn.metrics import confusion_matrix
from model import Model
from utils import *
import argparse


def main(args):
    modelname = args.modelname
    img_size = args.img_size
    data_path = args.data_path
    use_pose = args.use_pose
    use_amp = args.use_amp
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    num_workers = args.num_workers
    COSINE = args.COSINE
    init_lr = args.init_lr
    mixup = args.mixup
    DEBUG = args.DEBUG
    device = args.device

    criterion = nn.CrossEntropyLoss().cuda()
    #criterion = nn.BCEWithLogitsLoss().cuda()



    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    def train_epoch(loader, model, optimizer, epoch, mixup=True):
        losses = AverageMeter()

        n_sum = 0 
        correct_predictions = 0 
        model.train()

        train_loss = []
        bar = tqdm(loader)
        for data in bar:
            if use_pose:
                img, pose, target = data 
                img, pose, target = img.to(device), pose.to(device) ,target.long().to(device)

            else:
                img, target = data 
                img, target = img.to(device) ,target.long().to(device)
                pose = None

            loss_func = criterion
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, target, target2, lam = model(img, pose, target)
                if mixup:
                    loss = mixup_criterion(criterion, logits, target, target2, lam).mean()
                else:
                    loss = loss_func(logits, target)

                # Convert model output to predicted classes
                _, predicted = torch.max(logits, 1)

                # Update the correct predictions counter
                # print(f"predicted: {predicted} label: {label}")
                correct_predictions += (predicted == target).sum().item()
                n_sum += len(img)
            losses.update(loss.item(), img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description('loss: %.5f, smth: %.5f, len: %.2d' % (loss_np, smooth_loss, len(img)))

            wandb.log({
                "train_loss": loss_np,
                "train_loss_smooth": smooth_loss,
                "epoch": epoch

            })


        acc = correct_predictions / n_sum

        wandb.log({
            "train_acc": acc
        })
        return np.mean(train_loss)


    def val_epoch(loader, log_file, epoch):
        model.eval()
        val_loss = []
        n_sum = 0 
        correct_predictions = 0 

        preds = []
        targets = []
        bar = tqdm(loader)
        with torch.no_grad():
            for data in bar:
                if use_pose:
                    img, pose, target = data 
                    img, pose, target = img.to(device), pose.to(device) ,target.long().to(device)

                else:
                    img, target = data 
                    img, target = img.to(device) ,target.long().to(device)
                    pose = None
                logits = model(img, pose)
                logits = logits.squeeze(1)
                loss_func = criterion
                loss = loss_func(logits, target)

                targets.append(target)
                # Convert model output to predicted classes
                _, predicted = torch.max(logits, 1)
                preds.append(predicted)

                # Update the correct predictions counter
                # print(f"predicted: {predicted} label: {label}")
                correct_predictions += (predicted == target).sum().item()
                n_sum += len(img)

                val_loss.append(loss.detach().cpu().numpy())
                bar.set_description('val_loss: %.5f'% (val_loss[-1]))


            val_loss = np.mean(val_loss)


        acc = correct_predictions / n_sum
        wandb.log({"val_acc": acc})

        preds = torch.cat(preds).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        cm = confusion_matrix(targets, preds)
        print(cm)
        # analyze
        log_file.write(f"\n--------------------------epoch {epoch} -------------------\n")
        np.savetxt(f, cm, fmt='%d')
        log_file.write('\n')
        for i, val in enumerate(cm):
            print(f"for class {i}: accuracy: {val[i] / sum(val) * 100}")
            log_file.write(f"for class {i}: accuracy: {val[i] / sum(val) * 100} \n")
        log_file.write(f"accuracy: {acc}\n")

        return val_loss, acc


    i = 0 
    timeline = strftime("%Y_%m_%d_%H.%M.%S")
    views = [  'Dashboard', 'Rear_view', 'Right_side_window']

    for view in views :
        if use_pose:
            folder_name = f"pose-{modelname}"
        else:
            folder_name = f"image-{modelname}"
        
        print(f"folder name: {folder_name}")

        Path(f"logs/{timeline}/{folder_name}").mkdir(parents=True, exist_ok=True)
        log_file = f"logs/{timeline}/{folder_name}/log_fold_{i}_{view}_{modelname}.txt"
        f = open(log_file, 'a')

        dir_model_path = f'models/{timeline}/fold{i}/{folder_name}/{view}/'
        model_path = f'{dir_model_path}/best_{modelname}_fold_{i}.pth'

        Path(dir_model_path).mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        wandb_name = f"train {folder_name} {img_size} with {view}"
        run = wandb.init(project="Test", name=wandb_name, group=f"{view}")  

        # Log hyperparameters
        wandb.config.update({
            "model_name": modelname,
            "use_amp": use_amp,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "num_workers": num_workers,
            "COSINE": COSINE,
            "init_lr": init_lr,
            "mixup": mixup,
            "device": device,
            "img_size": img_size 
        })


        print(f"Fold {i} view: {view}")

        # setup dataset
        df_train = pd.read_csv(f"folds/fold_{i}/train_{i}.csv")
        # df_train = df_train.iloc[:100]
        df_val = pd.read_csv(f"folds/fold_{i}/val_{i}.csv")

        df_train = df_train[df_train["view"] == view]
        df_val = df_val[df_val["view"] == view]

        len_train = len(df_train)
        len_val = len(df_val)
        print(f"Train: {len_train} Val: {len_val}")
        # Need to make sure batch size is a factor of 2 to execute mixed precision training 
        if len_train % batch_size == 1:
            df_train = df_train[:(batch_size * (len_train // batch_size))]
        
    #   df_train = df_train[:100]
    #   df_val = df_val[:100]

        print(f"Train: {len(df_train)} Val: {len(df_val)}")

        # Setup dataloader
        dataset_train = MyDataset(df = df_train, view=view, img_size=img_size, use_pose=use_pose, data_path=data_path)
        dataset_valid = MyDataset(df = df_val, view=view, img_size=img_size, mode="valid", use_pose =use_pose, data_path=data_path)

        print(len(dataset_train), len(dataset_valid))
        # Setup dataloader
        train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train),
                                                num_workers=num_workers)
        valid_loader = DataLoader(dataset_valid, batch_size=batch_size,
                                                sampler=SequentialSampler(dataset_valid), num_workers=num_workers)

        # Initialize model
        model = Model(model_name = modelname, use_pose=use_pose)
        model = model.to(device)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # We use Cosine annealing LR scheduling
        fold = 1
        optimizer = optim.Adam(model.parameters(), lr=init_lr)
        scheduler = CosineAnnealingLR(optimizer, n_epochs)

        best = 0
        epoch_best = 1
        for epoch in range(1, n_epochs + 1):
            # start = time.time()
            wandb.log({
                "epoch": epoch
            })
            torch.cuda.empty_cache()
            print(time.ctime(), 'Epoch:', epoch)

            

            train_loss = train_epoch(train_loader, model, optimizer, epoch=epoch)
            val_loss, acc = val_epoch(valid_loader, log_file=f, epoch=epoch)
            print(f"accuracy in epoch {epoch}: {acc}")
    #        torch.save(model.state_dict(), os.path.join(f'models/last_{model_name}_fold_{i}.pth'))
            # create folde models/fold if not exist
            
            if acc > best:
                f.write(f"save for best model with acc: {acc}\n")
                print(f"save for best model with acc: {acc}")
                torch.save(model.state_dict(), os.path.join(model_path))
                best = acc
                epoch_best = epoch

            f.write(f"Best model with acc: {best} in epoch {epoch_best}\n")
            print(f"Best model with acc: {best} in epoch {epoch_best}")

            scheduler.step(epoch - 1)
        f.close()
        wandb.finish()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    
    parser.add_argument("--modelname", type=str, default="tf_efficientnetv2_m_in21k", help="Model name")
    parser.add_argument("--img_size", type=int, default=512, help="Image size")
    parser.add_argument("--data_path", type=str, default="/home/datnt114/thesis/aicity2023/code/tmp", help="Data path")
    parser.add_argument("--use_pose", action="store_true", help="Use pose dataset")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use AMP")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=40, help="Number of workers")
    parser.add_argument("--COSINE", action="store_true", help="Use cosine")
    parser.add_argument("--init_lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--mixup", action="store_true", help="Use mixup")
    parser.add_argument("--DEBUG", action="store_true", help="Debug mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    main(args)