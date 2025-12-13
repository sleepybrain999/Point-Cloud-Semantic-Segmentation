📦 ML Pipeline for Point Cloud Semantic Segmentation

This project implements the initial version of a scalable ML pipeline for semantic segmentation on outdoor LiDAR point clouds.
It includes:

✔ Commercial-use-permitted dataset (PandaSet)

✔ Python-based dataset downloader

✔ Exploratory Data Analysis (EDA)

✔ Preprocessing (voxel downsampling, sampling, augmentation)

✔ PyTorch Dataset for PointNext

✔ Model selection (PointNeXt)

✔ Training stub (no-op)


🔧 Project Structure
.
├── src/

│   ├── download.py          # Dataset downloader

│   ├── preprocess.py        # Filtering, voxelization, sampling, augmentation

│   ├── datasets.py          # PointNeXt-compatible PyTorch Dataset

│   ├── utils.py             # Helper functions

│   ├── EDA.py               # Visualization & stats

├── pipeline.ipynb           # Full pipeline execution 

├── MODEL_CHOICE.md          # Explain model choice

├── Illustrations            # Stores visualisations of pcds

│     

├── class.json               # Class mapping from dataset 

├── requirements.txt

├── README.md

└── Licensing.md             # Explain the license of the DataSet/Libraries used


🚀 1. Clone this Repository

git clone https://github.com/sleepybrain999/Point-Cloud-Semantic-Segmentation.git

cd Point-Cloud-Semantic-Segmentation

📥 2. Create a clean conda environment on python 3.11

conda create -n Your_env python=3.11

conda activate Your_env

📥 3. Install Python Requirements

pip install -r requirements.txt

📦 4. Install PandaSet Devkit 

git clone https://github.com/scaleapi/pandaset-devkit.git

cd pandaset-devkit/python

pip install .

📦 5. Modify PandaSet Devkit 
    
    In pandaset-devkit/python/pandaset/sensors, Change:

    class Lidar(Sensor):
    @property
    def _data_file_extension(self) -> str:
        return 'pkl.gz'
    
    to:
    class Lidar(Sensor):
    @property
    def _data_file_extension(self) -> str:
        return 'pkl'

    In pandaset-devkit/python/pandaset/annotations, Change:

    class SemanticSegmentation(Annotation):
    @property
    def _data_file_extension(self) -> str:
        return 'pkl.gz'

    to:
    class SemanticSegmentation(Annotation):

    @property
    def _data_file_extension(self) -> str:
        return 'pkl'

  
📡 6. Download PandaSet Dataset and run the code

Open:

pipeline.ipynb

Replace the kaggle username and api key with your own credentials

The notebook walks through:

Dataset download

EDA

Preprocessing

Dataset construction

Model selection (PointNeXt)

Training stub

functions used in this notebook can be found in either panda-devkit or src folder


📝 Attribution

PandaSet Dataset Attribution (CC-BY-ND 4.0)

This project uses the PandaSet dataset provided by Scale AI & Hesai.
PandaSet is licensed under Creative Commons Attribution–NoDerivatives 4.0 International (CC-BY-ND 4.0).


PandaSet was created in collaboration between Scale AI and Hesai.
© 2020 Scale AI. Licensed under the Creative Commons Attribution–NoDerivatives 4.0 International License (CC BY 4.0). https://creativecommons.org/licenses/by/4.0/#ref-appropriate-credit

Dataset source: https://pandaset-git-master.scaleai1.vercel.app/

PandaSet was preprocessed and used for training machine learning models


PointNeXt Model Attribution

This project references the PointNeXt architecture for model selection and configuration examples.

@InProceedings{qian2022pointnext,
  title   = {PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies},
  author  = {Qian, Guocheng and Li, Yuchen and Peng, Houwen and Mai, Jinjie and Hammoud, Hasan and Elhoseiny, Mohamed and Ghanem, Bernard},
  booktitle=Advances in Neural Information Processing Systems (NeurIPS),
  year    = {2022},
}

