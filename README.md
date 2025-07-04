# EchoNet-Liver : High throughput approach for detecting chronic liver disease using echocardiography

Chronic liver disease affects 1.5 billion people worldwide, often leading to severe health outcomes. Despite increasing prevalence, most patients remain undiagnosed. Various screening methods (like CT, MRI and liver biopsy) exist, but barriers like cost and availability limit their use.

Echocardiography, widely used in the clinic and tertialy center, can provide valuable information about liver tissue through subcostal views as well as cardiac structures. Deep learning applied to these echocardiographic images have been developed to detect cardiovascular diseases and predict disease progression. 
**Echo-Net-Liver**, a deep-learning algorithm pipeline, is developed to identify chronic liver disease (particularly steatotic liver disease (SLD) and cirrhosis), using subcostal echocardiographic images. Opportunistic liver disease screening using AI-guided echocardiography may contribute to early detection and patient care by utilizing existing procedures.

![EchoNet-Liver Pipeline](https://github.com/echonet/liver/blob/main/EchoNet_Liver_Figure_github.png)


**Presentation:**: 2024 AHA Chicago. Yuki Sahashi. Opportunistic Screening of Chronic Liver Disease With Deep Learning Enhanced Echocardiography 
Co-author: Milos Vukadinovic, Fatemeh Amrollahi, Hirsh Trivedi, Justin Rhee, Jonathan Chen, Susan Cheng, David Ouyang*, Alan C. Kwan* (Equal contribution)

**Preprint:** Sahashi Y, Vukadinovic M, Amrollahi F, Trivedi H, Rhee J, Chen J, Cheng S, Ouyang D, Kwan AC. Opportunistic Screening of Chronic Liver Disease with Deep Learning Enhanced Echocardiography. medRxiv. 2024 Jun 14:2024.06.13.24308898. doi: 10.1101/2024.06.13.24308898.

**Paper:** [Sahashi Y, Vukadinovic M, Amrollahi F, Trivedi H, Rhee J, Chen J, Cheng S, Ouyang D, Kwan AC. Opportunistic Screening of Chronic Liver Disease with Deep Learning Enhanced Echocardiography. NEJM AI 2025;2(3)
DOI: 10.1056/AIoa2400948](https://ai.nejm.org/doi/full/10.1056/AIoa2400948?query=ai_toc)

### Prerequisites

1. Python: we used 3.10.12
2. PyTorch we used pytorch==2.2.0
3. Other dependencies listed in `requirements.txt`

### Installation
First, clone the repository and install the required packages:

## Quickstart for inference

```sh
git clone https://github.com/echonet/liver.git
cd EchoNet-Liver
pip install -r requirements.txt
```

This repository uses Git Large File Storage (Git LFS) to manage large files (model weights). A simple git clone will only download placeholder files (around 4KB each) instead of the actual ckpt large files.

```sh
(sudo) apt update
(sudo) apt install git-lfs
git lfs install
git lfs pull
```

There are three models for Echo-Net Liver pipeline
1. View-classification model
2. Quality-control model
3. Disease-Detection-Model (cirrhosis model and SLD model) 

you can get all four pretrained model in this repo (See prettrained_models)

All you need to prepare is 
- Dataset (112*112 avi video, all echocardiography views)  

- Manifest file (that include filename, label etc)   
See sample_manifest_step1_and_2.csv (classify_high_quality_subcostal_videos.py)  and sample_manifest_step_liver_disease.csv (for predict.py)

- 480-640 subcostal echocardiography videos. (corresponding file name should be in output csv from Step 1and2 ) 
For disease detection model (DenseNet), we used 480-640, not 112-112.

In your environment, run this to classify high-quality subcostal. 
please modify sample_manifest_step1_and_2.csv to your dataset manifest

```sh
python classify_high_quality_subcostal_videos.py --dataset YOUR DATASET PATH --manifest_path YOURMANIFEST PATH.csv
```

Then, you need run below to Predict Cirrhosis or SLD (predict.py). Then you can calculate AUC in your dataset.

```sh
python predict.py --dataset YOUR HIGH-QUALITY-SUBCOSTAL-480640 DATASET --label (SLD OR cirrhosis)
```

Fin.
