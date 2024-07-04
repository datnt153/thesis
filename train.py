import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
import argparse


def main(args):
    print(f"args: {args}")
    modelname = args.modelname
    project_name = args.project_name
    img_size = args.img_size
    data_path = args.data_path
    use_pose = args.use_pose
    use_amp = args.use_amp
    use_wandb = args.use_wandb
    use_log = args.use_log
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    num_workers = args.num_workers
    COSINE = args.COSINE
    init_lr = args.init_lr
    mixup = args.mixup
    device = args.device
    print(repr(args))

    # Define loss criterion
    criterion = nn.CrossEntropyLoss().to(device=device)

    # Helper function to log with WandB
    def wandb_log(key, value):
        if use_wandb:
            wandb.log({key: value})

    # Helper function for logging to a file
    def log_to_file(log_file, message):
        if use_log:
            log_file.write(message + '\n')

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def train_epoch(loader, model, optimizer, epoch, mixup=True):

        n_sum = 0
        correct_predictions = 0
        model.train()

        train_loss = []
        bar = tqdm(loader)
        for data in bar:
            if use_pose:
                img, pose, target = data
                img, pose, target = img.to(device), pose.to(device), target.long().to(device)
            else:
                img, target = data
                img, target = img.to(device), target.long().to(device)
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
                correct_predictions += (predicted == target).sum().item()
                n_sum += len(img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description('loss: %.5f, smth: %.5f, len: %.2d' % (loss_np, smooth_loss, len(img)))

            wandb_log("train_loss", loss_np)
            wandb_log("train_loss_smooth", smooth_loss)

        acc = correct_predictions / n_sum
        wandb_log("train_acc", acc)
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
                    img, pose, target = img.to(device), pose.to(device), target.long().to(device)
                else:
                    img, target = data
                    img, target = img.to(device), target.long().to(device)
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
                correct_predictions += (predicted == target).sum().item()
                n_sum += len(img)

                val_loss.append(loss.detach().cpu().numpy())
                bar.set_description('val_loss: %.5f' % (val_loss[-1]))

            val_loss = np.mean(val_loss)

        acc = correct_predictions / n_sum
        wandb_log("val_acc", acc)

        preds = torch.cat(preds).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        cm = confusion_matrix(targets, preds)
        print(cm)

        log_to_file(log_file, f"\n--------------------------epoch {epoch} -------------------\n")
        if use_log:
            np.savetxt(log_file, cm, fmt='%d')
        log_to_file(log_file, '\n')
        for i, val in enumerate(cm):
            print(f"for class {i}: accuracy: {val[i] / sum(val) * 100}")
            log_to_file(log_file, f"for class {i}: accuracy: {val[i] / sum(val) * 100}")
        log_to_file(log_file, f"accuracy: {acc}\n")

        return val_loss, acc


    timeline = strftime("%Y_%m_%d_%H.%M")
    views = ['Dashboard', 'Rear_view', 'Right_side_window']

    for view in views:
        if use_pose:
            folder_name = f"pose-aagcn-{modelname}"
        else:
            folder_name = f"image-{modelname}"

        print(f"folder name: {folder_name}")

        configs = {
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
            }
        logs = None
        if use_log:
            Path(f"logs/{timeline}").mkdir(parents=True, exist_ok=True)
            log_file = f"logs/{timeline}/log_{view}_{folder_name}.txt"
            logs = open(log_file, 'a')

            print("use logs")

            logs.write("configs\n")
            # Iterate over the dictionary items and write them to the file
            for key, value in configs.items():
                print(f"{key}: {value}\n")
                logs.write(f"{key}: {value}\n")
            print("end use logs")

        dir_model_path = f'models/{timeline}/{folder_name}/{view}'
        model_path = f'{dir_model_path}/best_{modelname}.pth'

        Path(dir_model_path).mkdir(parents=True, exist_ok=True)

        if use_wandb:
            # Initialize wandb
            wandb_name = f"train {folder_name} imgs {img_size} bs {batch_size} {view}"
            run = wandb.init(project=project_name, name=wandb_name, group=f"{view}")

            # Log hyperparameters
            wandb.config.update(configs)

        print(f"Fold 0 view: {view}")

        # setup dataset
        df_train = pd.read_csv(f"folds/fold_0/train_0.csv")
        df_val = pd.read_csv(f"folds/fold_0/val_0.csv")

        df_train = df_train[df_train["view"] == view]
        df_val = df_val[df_val["view"] == view]

        len_train = len(df_train)
        len_val = len(df_val)
        print(f"Train: {len_train} Val: {len_val}")

        # Need to make sure batch size is a factor of 2 to execute mixed precision training
        if len_train % batch_size == 1:
            df_train = df_train[:(batch_size * (len_train // batch_size))]

        print(f"Train: {len(df_train)} Val: {len(df_val)}")

        # Setup dataloader
        dataset_train = MyDataset(df=df_train, view=view, img_size=img_size, use_pose=use_pose, data_path=data_path)
        dataset_valid = MyDataset(df=df_val, view=view, img_size=img_size, mode="valid", use_pose=use_pose, data_path=data_path)

        print(len(dataset_train), len(dataset_valid))

        # Setup dataloader
        train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=RandomSampler(dataset_train),
                                  num_workers=num_workers)
        valid_loader = DataLoader(dataset_valid, batch_size=batch_size,
                                  sampler=SequentialSampler(dataset_valid), num_workers=num_workers)

        # Initialize model
        model = Model(model_name=modelname, use_pose=use_pose)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs.")
            model = nn.DataParallel(model)

        model = model.to(device)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # We use Cosine annealing LR scheduling
        optimizer = optim.Adam(model.parameters(), lr=init_lr)
        scheduler = CosineAnnealingLR(optimizer, n_epochs)

        best_acc = 0
        epoch_best = 1
        for epoch in range(1, n_epochs + 1):

            wandb_log("epoch", epoch)

            torch.cuda.empty_cache()
            print(time.ctime(), 'Epoch:', epoch)

            train_loss = train_epoch(train_loader, model, optimizer, epoch=epoch)
            val_loss, acc = val_epoch(valid_loader, log_file=logs, epoch=epoch)
            print(f"accuracy in epoch {epoch}: {acc}")

            if acc > best_acc:
                log_to_file(logs, f"save for best model with acc: {acc}\n")
                print(f"save for best model with acc: {acc}")
                save_model = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(save_model, os.path.join(model_path))
                # torch.save(model.module.state_dict(), os.path.join(model_path))
                best_acc = acc
                epoch_best = epoch

            
            log_to_file(logs, f"Best model with acc: {best_acc} in epoch {epoch_best}\n")
            print(f"Best model with acc: {best_acc} in epoch {epoch_best}")

            wandb_log("best acc", best_acc)
            wandb_log("best epoch", epoch_best)
            scheduler.step(epoch - 1)

        if use_log:
            logs.close()
        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--modelname", type=str, default="tf_efficientnetv2_m_in21k", help="Model name")
    parser.add_argument("--project_name", type=str, default="Thesis master 2023", help="Project name")
    parser.add_argument("--img_size", type=int, default=512, help="Image size")
    parser.add_argument("--data_path", type=str, default="/home/datnt114/thesis/aicity2023/code/tmp", help="Data path")
    parser.add_argument("--use_pose", action="store_true", help="Use pose dataset")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use AMP")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use wandb")
    parser.add_argument("--use_log", action="store_true", default=False, help="Use log")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=40, help="Number of workers")
    parser.add_argument("--COSINE", action="store_true", help="Use cosine")
    parser.add_argument("--init_lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--mixup", action="store_true", help="Use mixup")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    main(args)
