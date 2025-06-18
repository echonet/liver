import sys
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import glob
import cv2
import argparse

from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve

import torch
from lightning_utilities.core.imports import compare_version
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.models import densenet121

from utils import sensivity_specifity_cutoff, sigmoid, EchoDataset,get_frame_count

with torch.no_grad():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Predict script for Liver disease Prediction From Echocardiography.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--label", type=str, required=True, choices = ['cirrhosis', 'SLD'] ,help="Label for prediction from echo dataset.")
    
    args = parser.parse_args()   
    
    #Weight label setting    
    if args.label == "cirrhosis":
        weights_path = "EchoNet-Liver(YOUR_WORKING_DIR)/pretrained_models/pretrained_model_weight_cirrhosis.pt"    
    elif args.label == "SLD":
        weights_path = "EchoNet-Liver(YOUR_WORKING_DIR)/pretrained_models/pretrained_model_weight_fattyliver.pt"
    
    data_path = args.dataset    #update the manifest file when needed
    video_files = glob.glob(os.path.join(data_path, "*.avi"))
    manifest = pd.DataFrame({"filename": video_files})
    manifest["split"] = "test"
    manifest['filename']= manifest['filename'].apply(lambda x: os.path.basename(x))
    
    #If your dataset have a video with less than 32 frames, please remove it from the manifest file.
    manifest['frames']=  manifest["filename"].apply(lambda x: get_frame_count(os.path.join(args.dataset, f"{x}")))
    manifest = manifest[manifest['frames'] > 31]
    
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.csv")
    print(f"Manifest file was updated and saved to {manifest_path}")
    manifest.to_csv(manifest_path, index = False)
    
    #--------------------------------------------------
    #Step: Opportunistic Liver Disease Screening
    print('--- Step: AI-guided Liver disease Opportunistic Screening---')
    print("Prediction LABEL: ", args.label)
   
    test_ds = EchoDataset(
        split="test",data_path=data_path,manifest_path=manifest_path,
        n_frames =1, #From Video Avi image, we have only 1 frame (random pick if random_start=True)
        resize_res = (480, 640), #EchoNet-Liver : Disease Prediction model Input size 480x640
        random_start = True
        )
    
    test_dl = DataLoader(
        test_ds, num_workers=8,  batch_size=10, drop_last=False, shuffle=False,
        )
    
    #Load the model for view-classifier 
    pretrained_weights = torch.load(weights_path)
    new_state_dict = {}
    for k, v in pretrained_weights.items():
        new_key = k[2:] if k.startswith('m.') else k
        new_state_dict[new_key] = v
        
    backbone = densenet121(pretrained=False)
    num_ftrs = backbone.classifier.in_features
    backbone.classifier = torch.nn.Linear(num_ftrs, 1)
    backbone.load_state_dict(new_state_dict, strict=True)
    backbone = backbone.to(device).eval()

    filenames = []
    predictions = []
    
    for batch in tqdm(test_dl):
        preds = backbone(batch["primary_input"].to(device))
        filenames.extend(batch["filename"])
        predictions.extend(preds.detach().cpu().squeeze(dim = 1))
    
    df_preds = pd.DataFrame({'filename': filenames, 'preds': predictions})
    manifest_v1 = manifest.merge(df_preds, on="filename", how="inner").drop_duplicates('filename')
    manifest_v1.preds = manifest_v1.preds.apply(sigmoid)
    
    manifest_v1.to_csv(
        Path(os.path.dirname(os.path.abspath(__file__)))
        / Path(f"disease_detection_{args.label}.csv"),
        index=False,
    )
    
    print(f"Predict LIVER DISEASE DETECTION -{args.label}- was done. See Output csv and Calculate AUC")
    
#SAMPLE SCRIPT
#python predict.py  --dataset YOUR HIGH-QUALITY-SUBCOSTAL-480640resolution  --label SLD OR cirrhosis