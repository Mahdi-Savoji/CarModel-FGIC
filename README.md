### 📦 CarModel-FGIC

**Fine-Grained Car Model Classification using the Stanford Cars Dataset**

------

#### 🚀 Overview

This project implements a **Fine-Grained Image Classification (FGIC)** model to detect car **make and model** using the [Stanford Cars Dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset). The main focus is on distinguishing between visually similar car types using advanced deep learning techniques.

------

#### 📁 Project Structure

| File                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `CarModel-FGIC.ipynb`        | Main notebook for training and evaluating the FGIC model. Includes data loading, model definition, training loop, and evaluation. |
| `CreateCroppedDataset.ipynb` | Preprocessing script to crop and organize images from the original Stanford Cars dataset for cleaner training. |



------

#### 🧠 Model Highlights

- Based on transfer learning using models like **ResNet**, **EfficientNet**, or a hybrid (ResNet + VGG + CBAM + Cross-Attention).
- Supports **fine-grained classification** of 196 car classes.
- Incorporates **bilinear pooling**, **channel & spatial attention (CBAM)**, and **cross-attention** to improve feature interaction.
- Uses data augmentation and balanced sampling for better generalization.
- Evaluation includes Top-1 and Top-5 accuracy metrics.

------

#### 🗃 Dataset

- **Source**:
  - [Stanford Cars Dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)
  - [Cropped Stanford Cars Dataset](https://www.kaggle.com/datasets/mahdisavoji/croppedstanfordcardataset)
- **Images**: 16,185 car images
- **Labels**: 196 car makes and models
- **Format**: `.mat` annotations and `.jpg` images

> Preprocessing crops each image based on bounding box annotations and organizes them by class folder.

------

#### ⚙️ How to Run

1. Clone the repository:

   ```
   bashCopyEditgit clone https://github.com/Mahdi-Savoji/CarModel-FGIC.git
   cd CarModel-FGIC
   ```

2. Prepare the dataset:

   - Download the dataset from [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
   - Run `CreateCroppedDataset.ipynb` to preprocess images

3. Train the model:

   - Open and run `CarModel-FGIC.ipynb` in Jupyter or Colab

------

#### 📊 Results

| Metric              | Accuracy |
| ------------------- | -------- |
| **Top-1 Val Acc**   | 87.54%   |
| **Top-5 Val Acc**   | 97.85%   |
| **Top-1 Train Acc** | 98.04%   |
| **Top-5 Train Acc** | 100.00%  |



> ✅ Model trained for 16 epochs with early stopping.
>  ⚠️ The test accuracy may need verification due to its unexpectedly low value.

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

