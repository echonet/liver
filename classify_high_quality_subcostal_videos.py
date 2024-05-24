import sys
import numpy as np
import pandas as pd
import os
import shutil
import torch
from lightning_utilities.core.imports import compare_version
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18

import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix,roc_curve
from tqdm import tqdm
import cv2
from utils import sensivity_specifity_cutoff, sigmoid, EchoDataset, get_frame_count


#RUN CHECK ON Yuki's environment (as of 2024-05-30) Using SHC external dataset


with torch.no_grad():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Predict script for doing View-classification / Quality-control.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to the manifest file that includes echo-file-uid and labels (Subcostal or non-subcostal).")
    args = parser.parse_args()   
    
    #Path to the model weights (View-classifier (VC) and Quality control model (QC))
    view_classifier_weights_path = "/workspace/yuki/EchoNet-Liver/pretrained_models/subcostal_view_classifier_model.pt"
    quality_control_model_weights_path = "/workspace/yuki/EchoNet-Liver/pretrained_models/quality_control_model.pt"
    
    data_path = args.dataset
    manifest_path = args.manifest_path
    #Plese see sample csv columns -- sample_manifest_step1_and_2.csv
    
    #update the manifest file when needed
    manifest = pd.read_csv(manifest_path)
    manifest["split"] = "test"
    if 'file_uid' in manifest.columns:
        manifest = manifest.rename(columns={'file_uid': 'filename'})
    
    if 'filename' in manifest.columns and manifest['filename'].str.contains('.avi').all() == False:
        manifest['filename'] = manifest['filename'].apply(lambda x: x + '.avi') 
        
    manifest['frames']=  manifest["filename"].apply(lambda x: get_frame_count(os.path.join(args.dataset, f"{x}")))
    manifest = manifest[manifest['frames'] > 31]
    manifest.to_csv(manifest_path, index = False)
    
    #-----------------------------------------------------------------------------------------
    #Step 1: Predict Subcostal View-Classifier
    print("---Step 1: Start Predict View-Classifier Model")    
    #load the dataset for view-classifier
    
    print("Note: Please make sure that our dataset (for VC and QC)is in the following format: 112x112x3, avi format. (In our dataset, we have preprocess the videos including de-identified, removing ECG, Respiratory signals)")
    print("Note: Please make sure that the video frame count is more than 16 frames")
    
    test_ds = EchoDataset(split="test",data_path=data_path, manifest_path=manifest_path)
    test_dl = DataLoader(test_ds,num_workers=8, batch_size=10,drop_last=False, shuffle=False)
    
    #Load the model for view-classifier 
    pretrained_weights = torch.load(view_classifier_weights_path)
    new_state_dict = {}
    for k, v in pretrained_weights.items():
        new_key = k[2:] if k.startswith('m.') else k
        new_state_dict[new_key] = v 
    backbone = r2plus1d_18(num_classes=1)
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
    
    manifest_v2 = manifest_v1[manifest_v1.preds > 0.8414]
    manifest_v2.to_csv(
        Path(os.path.dirname(os.path.abspath(__file__)))
        / Path("view_classification_predictions_above_threshold.csv"),
        index=False,
    )

    #-----------------------------------------------------------------------------------------
    #Step 2: Quality Control
    print("---Step 2: Start Predict Quality Control Model")
    manifest_step2 = pd.read_csv('view_classification_predictions_above_threshold.csv')
    manifest_step2.drop(columns = ['preds'], inplace = True) #You need Drop predict value on Step 1
    #load the dataset for Quality Control Model
    test_ds_step2 = EchoDataset(
        split="test",
        data_path=data_path,
        manifest_path=Path(os.path.dirname(os.path.abspath(__file__)))
        / Path("view_classification_predictions_above_threshold.csv")
    )
    
    test_dl = DataLoader(
        test_ds_step2, num_workers=8, batch_size=10, drop_last=False, shuffle=False)

    #load the model for Quality Control Model (QC)
    pretrained_weights_qc = torch.load(quality_control_model_weights_path)
    new_state_dict_qc = {}
    for k, v in pretrained_weights_qc.items():
        new_key = k[2:] if k.startswith('m.') else k
        new_state_dict_qc[new_key] = v 
    backbone = r2plus1d_18(num_classes=1)
    backbone.load_state_dict(new_state_dict_qc, strict=True)
    backbone = backbone.to(device).eval()
    
    filenames_qc = []
    predictions_qc = []
    
    for batch in tqdm(test_dl):
        preds = backbone(batch["primary_input"].to(device))
        filenames_qc.extend(batch["filename"])
        predictions_qc.extend(preds.detach().cpu().squeeze(dim = 1))
    
    df_preds = pd.DataFrame({'filename': filenames_qc, 'preds': predictions_qc})
    manifest_step2_v1 = manifest_step2.merge(df_preds, on="filename", how="inner").drop_duplicates('filename')
    manifest_step2_v1.preds = manifest_step2_v1.preds.apply(sigmoid)
    
    manifest_step2_v2 = manifest_step2_v1[manifest_step2_v1.preds > 0.925]
    manifest_step2_v2 = manifest_step2_v2.drop_duplicates('filename').reset_index(drop=True)
    manifest_step2_v2.to_csv(
        Path(os.path.dirname(os.path.abspath(__file__)))
        / Path("quality_control_predictions_above_threshold.csv"),
        index=False,
    )
    
    print("★Predict Subcostal View-Classifier was done. See Output csv")
    print("You can get high-quality subcostal videos file_path. In our experiment using external some blurred videos were remained. Please change cut-off value depending on your dataset.")