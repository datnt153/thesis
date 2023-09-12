import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from sklearn.metrics import confusion_matrix
from model import Model
from data import MyDataset
import numpy as np
from tqdm import tqdm

import argparse

def main(args):
    print(f"args: {args}")
    modelname = args.modelname
    img_size = args.img_size
    data_path = args.data_path
    use_pose = args.use_pose
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = args.device
    timeline = args.timeline
    print(repr(args))



    def val_epoch(loader, model):
        model.eval()
        val_loss = []
        LOGITS = []
        PREDS = []
        TARGETS = []

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

                pred = logits.sigmoid().detach()
                LOGITS.append(logits)
                PREDS.append(pred)
                TARGETS.append(target)

                val_loss.append(loss.detach().cpu().numpy())
                bar.set_description('val_loss: %.5f'% (val_loss[-1]))
            val_loss = np.mean(val_loss)

        LOGITS = torch.cat(LOGITS).cpu().numpy()
        PREDS = torch.cat(PREDS).cpu().numpy()
        TARGETS = torch.cat(TARGETS).cpu().numpy()
        acc = np.sum(PREDS.argmax(1) == TARGETS) / len(PREDS.argmax(1)) * 100

        cm = confusion_matrix(TARGETS, PREDS.argmax(1))
        print(cm)

        return val_loss, acc

    # Define loss criterion
    criterion = torch.nn.CrossEntropyLoss().to(device=device)

    views = ['Dashboard', 'Rear_view', 'Right_side_window']

    for view in views:
        if use_pose:
            folder_name = f"pose-{modelname}"
        else:
            folder_name = f"image-{modelname}"
        
        # Set the path to the saved model
        dir_model_path = f'models/{timeline}/{folder_name}/{view}'
        model_path = f'{dir_model_path}/best_{modelname}.pth'

        # Set the path to the validation data CSV file
        val_data_path = "folds/fold_0/val_0.csv"

        # Load the saved model
        model = Model(model_name=modelname, use_pose=use_pose)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs.")
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path))

        # Load the validation dataset
        df_val = pd.read_csv(val_data_path)
        df_val = df_val[df_val["view"] == view]

        len_val = len(df_val)
        print(f"Val: {len_val}")

        dataset_valid = MyDataset(df=df_val, view=view, img_size=img_size, mode="valid", use_pose=use_pose, data_path=data_path)
        valid_loader = DataLoader(dataset_valid, batch_size=batch_size,
                                  sampler=SequentialSampler(dataset_valid), num_workers=num_workers)


        # Validate the model
        val_loss, accuracy = val_epoch(valid_loader, model)

        print("Validation Loss: {:.5f}".format(val_loss))
        print("Accuracy: {:.2f}%".format(accuracy))

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference model")
    
    parser.add_argument("--modelname", type=str, default="tf_efficientnetv2_m_in21k", help="Model name")
    parser.add_argument("--img_size", type=int, default=512, help="Image size")
    parser.add_argument("--data_path", type=str, default="/home/datnt114/thesis/aicity2023/code/tmp", help="Data path")
    parser.add_argument("--use_pose", action="store_true", help="Use pose dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=40, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--timeline", type=str, default="2023_09_11_15.50", help="Timeline train")

    
    args = parser.parse_args()
    main(args)