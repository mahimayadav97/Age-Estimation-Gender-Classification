# Age Estimation and Gender Classification


This project involves training two Convolutional Neural Network (CNN) models to estimate a person's age and predict their gender from a given face image. The models are evaluated using the UTKFace dataset and are designed for two tasks:


1. **Age Estimation** (Mean Absolute Error - MAE)
2. **Gender Classification** (Accuracy)


The project includes both a custom CNN model and a fine-tuned pre-trained model.


## Dataset


The models are trained on the `train_val/` directory, which contains 5,000 labeled face images (size: 128x128) derived from the UTKFace dataset. The images are labeled with age and gender.


## Project Structure


- `data/`: Contains the training and validation images.
- `model/`: Contains the saved models (`age_gender_A.h5` and `age_gender_B.h5`).
- `notebooks/`: Jupyter notebooks with detailed steps for training the models.
- `scripts/`: Python scripts for data preprocessing, defining model architecture, and training the models.
- `requirements.txt`: The necessary Python dependencies for setting up the environment.


## Setup


1. Clone the repository to your local machine:
    ```bash
    git clone <repository_url>
    cd age-gender-estimation
    ```


2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


3. Download the dataset and place it in the `data/train_val/` folder.


4. Run the training process in the `notebooks/age_gender_classification.ipynb` or execute the `scripts/train.py` for training the models.


## Training Process


### Step 1: Data Preprocessing
The dataset is preprocessed, which includes:
- Rescaling pixel values to the [0, 1] range.
- Data augmentation (rotation, width/height shifts, shear, zoom, brightness variation, and horizontal flip).


### Step 2: Model Architecture


1. **Custom CNN (age_gender_A.h5)**: A custom-built CNN model with two output layers (one for age prediction and one for gender classification).
2. **Pretrained Model (age_gender_B.h5)**: A fine-tuned model based on a pre-trained CNN architecture to improve performance.


### Step 3: Training
The models are trained using the training data with the following steps:
- Compile the models using an appropriate optimizer and loss function.
- Fit the models on the training data and validate on the validation set.


### Step 4: Evaluation
The models are evaluated using:
- **MAE** for age prediction.
- **Accuracy** for gender classification.


## Usage


Once trained, the models can be used to make predictions on new face images. The models will predict both the age and gender of a given face image.


## Requirements


The following Python libraries are required to run this project:


- `tensorflow` or `keras`
- `numpy`
- `opencv-python`
- `matplotlib`
- `seaborn`
- `pandas`
- `scikit-learn`


You can install these dependencies by running:
```bash
pip install -r requirements.txt

