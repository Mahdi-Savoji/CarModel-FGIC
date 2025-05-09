### 📦 CarModel-FGIC

**Fine-Grained Car Model Classification using Stanford Cars + Iran Cars Datasets**

------

#### 🚀 Overview

This project implements a **Fine-Grained Image Classification (FGIC)** model to detect car **make and model** using both the [Stanford Cars Dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset) and an additional [Iran Cars Dataset](https://www.kaggle.com/datasets/usefashrfi/iran-used-cars-dataset). The main focus is on distinguishing between visually similar car types using advanced deep learning techniques.

------

#### 📁 Project Structure

| File                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `CarModel-FGIC.ipynb`        | Main notebook for training and evaluating the FGIC model. Includes data loading, model definition, training loop, and evaluation. |
| `CreateCroppedDataset.ipynb` | Preprocessing script to crop and organize images from the original Stanford Cars dataset for cleaner training. |



------

#### 🧠 Model Highlights

- Based on transfer learning using models like **ResNet**, **EfficientNet**, or a hybrid (ResNet + VGG + CBAM + Cross-Attention).
- Supports **fine-grained classification** across combined classes from both datasets.
- Incorporates **bilinear pooling**, **channel & spatial attention (CBAM)**, and **cross-attention** to improve feature interaction.
- Uses data augmentation and balanced sampling for better generalization.
- Evaluation includes Top-1 and Top-5 accuracy metrics.

------

#### 🗃 Dataset

- **Sources**:
  - [Stanford Cars Dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)
  - [Cropped Stanford Cars Dataset](https://www.kaggle.com/datasets/mahdisavoji/croppedstanfordcardataset)
  - [Iran Cars Dataset](https://www.kaggle.com/datasets/usefashrfi/iran-used-cars-dataset)
- **Images**: Combined total of car images from both datasets
- **Labels**: Unified set of car make and model classes
- **Format**: `.mat` annotations and `.jpg` images for Stanford, standard image files and metadata for Iran Cars

> Preprocessing crops each image based on bounding box annotations (Stanford) or custom bounding boxes (Iran), and organizes them by class.

------

#### ⚙️ How to Run

1. Clone the repository:

   ```
   bashCopyEditgit clone https://github.com/Mahdi-Savoji/CarModel-FGIC.git
   cd CarModel-FGIC
   ```

2. Prepare the dataset:

   - Download the Stanford dataset from [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
   - Collect/preprocess the Iran Cars dataset (images + bounding boxes)
   - Run `CreateCroppedDataset.ipynb` to crop and organize both datasets

3. Train the model:

   - Open and run `CarModel-FGIC(StanfordCars+IranCars).ipynb` in Jupyter or Colab

------

#### 📊 Results

| Metric              | Accuracy |
| ------------------- | -------- |
| **Top-1 Val Acc**   | 92.63%   |
| **Top-5 Val Acc**   | 98.87%   |
| **Val Recall**      | 88%      |
| **Val Precision**   | 89%      |
| **Top-1 Train Acc** | 97.26%   |
| **Top-5 Train Acc** | 99.95%   |
| **Train Recall**    | 95%      |
| **Train Precisoin** | 95%      |



> ✅ Model trained for 10 epochs with early stopping, using a mix of Stanford and Iran car images.

------

#### 🛠️ Dependencies

```
torchvision
numpy
matplotlib
scipy
opencv-python
scikit-learn
```

Install all dependencies with:

```
pip install -r requirements.txt
```
