# EchoNet-Liver : Highthroughput approach for detecting chronic liver disease using echocardiography

Chronic liver disease affects 1.5 billion people worldwide, often leading to severe health outcomes. Despite increasing prevalence, most patients remain undiagnosed. Various screening methods (like CT, MRI and liver biopsy) exist, but barriers like cost and availability limit their use.

Echocardiography, widely used in the clinic and tertialy center, can provide valuable information about liver tissue through subcostal views as well as cardiac structures. Deep learning applied to these echocardiographic images have been developed to detect cardiovascular diseases and predict disease progression. 
**Echo-Net-Liver**, a deep-learning algorithm pipeline, is developed to identify chronic liver disease (particularly steatotic liver disease (SLD) and cirrhosis), using subcostal echocardiographic images. Opportunistic liver disease screening using AI-guided echocardiography may contribute to early detection and patient care by utilizing existing procedures.

#![EchoNet-Liver Pipeline](/workspace/yuki/EchoNet-Liver/Echonet-liver.png)

**Presentation:** Conference information will be updated.
**Preprint:** link will be added once preprint is released

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


