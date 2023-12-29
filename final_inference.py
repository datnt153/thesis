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
import torch.nn.functional as F


def val_epoch(loader, model, device, use_pose, criterion):
    model.eval()
    val_loss = []
    LOGITS = []
    PREDS = []
    TARGETS = []
    fold_names = []

    bar = tqdm(loader)
    with torch.no_grad():
        for data in bar:
            if use_pose:
                img, pose, target, fold_name = data
                img, pose, target = (
                    img.to(device),
                    pose.to(device),
                    target.long().to(device),
                )
            else:
                img, target, fold_name = data
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
            fold_names = fold_names + list(fold_name)
            print(f"fold name: {fold_name}")

            val_loss.append(loss.detach().cpu().numpy())
            bar.set_description('val_loss: %.5f'% (val_loss[-1]))
        val_loss = np.mean(val_loss)

    LOGITS = torch.cat(LOGITS).cpu().numpy()
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = np.sum(PREDS.argmax(1) == TARGETS) / len(PREDS.argmax(1)) * 100

    print(f"acc")

    cm = confusion_matrix(TARGETS, PREDS.argmax(1))
    print(cm)

    return val_loss, acc, PREDS.argmax(1) , TARGETS, LOGITS, fold_names

def main(args):
    print(f"args: {args}")
    img_size = args.img_size
    data_path = args.data_path
    use_pose = args.use_pose
    batch_size = args.batch_size
    num_workers = args.num_workers
    device = args.device
    timeline = args.timeline
    print(repr(args))

    # Define loss criterion
    criterion = torch.nn.CrossEntropyLoss().to(device=device)

    views = ["Dashboard", "Rear_view", "Right_side_window"]

    # Define all the scenarios you want to run
    img_size = 256
    configs = [
        {
            "model_name": "tf_efficientnetv2_l_in21k",
            "bs": 16,
            "path": [
                {"time_line": "2023_09_14_09.48", "use_pose": True},
                {"time_line": "2023_09_14_17.18", "use_pose": False},
            ],
            "gpus": 1,
        },

        # {
        #     "model_name": "tf_efficientnetv2_m_in21k",
        #     "bs": 48,
        #     "path": [{"time_line": "2023_09_09_16.11", "use_pose": True}, {"time_line": "2023_09_09_19.32", "use_pose": False} ],
        #     "gpus": 2
        # },

        # {

        #     "model_name": "tf_efficientnetv2_s_in21k",
        #     "bs": 48,
        #     "path": [{"time_line": "2023_09_13_17.46", "use_pose": True}, {"time_line": "2023_09_13_20.17", "use_pose": False} ],
        #     "gpus": 2
        # }
    ]

    for config in configs:
        modelname = config["model_name"]
        batch_size = config["bs"]

        for path in config["path"]:
            timeline = path["time_line"]
            use_pose = path["use_pose"]

            folder_name = f"image-{modelname}"

            if use_pose:
                folder_name = f"pose-{modelname}"

            all_logits = {folder_name: []}
            all_targets = None  # Assuming targets are same for all views

            for view in views:
                # Modify the model_path based on your directory structure
                dir_model_path = f"final_models/{modelname} imgs {img_size} bs {batch_size}/{timeline}/{folder_name}/{view}"
                model_path = f"{dir_model_path}/best_{modelname}.pth"

                # Load the saved model
                model = Model(model_name=modelname, use_pose=use_pose)

                if torch.cuda.device_count() > 1:
                    print(f"Using {torch.cuda.device_count()} GPUs.")
                    model = torch.nn.DataParallel(model)
                model = model.to(device)
                model.load_state_dict(torch.load(model_path))

                # Load the validation dataset
                val_data_path = "folds/fold_0/val_0.csv"
                df_val = pd.read_csv(val_data_path)
                df_val = df_val[df_val["view"] == view]

                len_val = len(df_val)
                print(f"Val: {len_val}")

                dataset_valid = MyDataset(
                    df=df_val,
                    view=view,
                    img_size=img_size,
                    mode="valid",
                    use_pose=use_pose,
                    data_path=data_path,
                )
                valid_loader = DataLoader(
                    dataset_valid,
                    batch_size=batch_size,
                    sampler=SequentialSampler(dataset_valid),
                    num_workers=num_workers,
                )

                # Validate the model
                val_loss, accuracy, PREDS, TARGETS, logits, list_file = val_epoch(
                    valid_loader, model, device, use_pose, criterion
                )
                print(f"len list file: {len(list_file)}")
                # print(PREDS)
                # print(TARGETS)
                # print(type(PREDS), type(TARGETS))
                # print(logits)
                print(f"{folder_name}_view_{view}: {accuracy}")
                data = {"file": list_file, 'predict': PREDS, "targets": TARGETS}
                new = pd.DataFrame(data)
                new.to_csv(f"preds/{folder_name}_view_{view}.csv")

                all_logits[folder_name].append(logits)
                all_targets = (
                    TARGETS  # Update targets - assumed to be same for all views
                )
                
                # print(logtis.shape)

            avg_logits = np.mean(all_logits[folder_name], axis=0)
            print(avg_logits.shape)
            avg_predict = avg_logits.argmax(1)
            # print(avg_predict)
            acc = np.mean(avg_predict == all_targets) * 100  # Calculate accuracy
            print(f"Accuracy for {folder_name}: {acc}%")

            # Save to CSV
            data = {"fold name": list_file, "target": all_targets, "avg_predict": avg_predict}
            df = pd.DataFrame(data)
            df.to_csv(f"{folder_name}_embedded.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference model")

    parser.add_argument("--img_size", type=int, default=512, help="Image size")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/datnt114/thesis/aicity2023/code/tmp",
        help="Data path",
    )
    parser.add_argument("--use_pose", action="store_true", help="Use pose dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=40, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--timeline", type=str, default="2023_09_11_15.50", help="Timeline train"
    )

    args = parser.parse_args()
    main(args)
