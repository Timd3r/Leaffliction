# Leaffliction

An innovative computer vision project utilizing leaf image analysis for disease recognition using machine learning and image processing techniques.

## Installation

### Prerequisites
- Python 3.8+
- pip or another package manager

### Setup

1. Clone or navigate to the repository
2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Leaffliction/
├── Augmentation.py          # Image augmentation utilities
├── Distribution.py          # Dataset distribution analysis
├── Transformation.py        # Core image transformation pipeline
├── train.py                 # Model training script
├── predict.py              # Disease prediction script
├── clean.py                # Dataset cleaning utility
├── feature_config.json      # Feature configuration
├── Classification/
│   ├── prepare.py          # Data preparation and train/test split
│   ├── create_dataset.py   # Feature extraction and dataset creation
│   └── _path_setup.py      # Path configuration
├── leaves/                 # Original leaf image dataset
├── data/                   # Processed data (train/test splits)
├── output/                 # Model outputs and learnings
└── color_histograms/       # Saved color histogram visualizations
```

## Usage

### 1. Data Preparation
Prepare and split your leaf image dataset:
```bash
cd Classification
python3 prepare.py
cd ..
```

### 2. Feature Extraction
Create training and test datasets with extracted color features:
```bash
python3 Classification/create_dataset.py
```

### 3. Model Training
Train the Random Forest classifier on extracted features:
```bash
python3 train.py Apple
python3 train.py Grape
```
This generates model learnings in `output/{Plant}_learnings.zip`

### 4. Disease Prediction
Predict disease on a leaf image:
```bash
python3 predict.py "path/to/leaf/image.jpg"
```

### 5. Image Augmentation
Augment a single image with various transformations:
```bash
python3 Augmentation.py "path/to/image.jpg"
```

### 6. Dataset Distribution Analysis
Analyze the distribution of classes in your dataset:
```bash
python3 Distribution.py "./data/train/Apple"
python3 Distribution.py "./data/train/Grape"
```

## Dependencies

- **numpy** (2.4.4) - Numerical computing
- **pandas** (3.0.2) - Data manipulation
- **scikit-learn** (1.8.0) - Machine learning
- **opencv-python** (4.13.0.92) - Computer vision
- **Pillow** (12.2.0) - Image processing
- **matplotlib** (3.10.8) - Visualization
- **joblib** (1.5.3) - Model serialization
- **tqdm** (4.67.3) - Progress bars
- **plantcv** (4.10.2) - Plant phenotyping image analysis

## Data Analysis & Visualizations

The project includes several visualization outputs showing feature relationships and data distributions:

### feature_correlations.png
Correlation matrix showing relationships between the 45 extracted color histogram features across all disease categories. Helps identify redundant features and feature dependencies.

### top_features.png
Bar chart ranking the most important features for disease classification as determined by the Random Forest model. Shows which color statistics have the highest predictive power.

### top_feature_distributions.png
Distribution plots of the top predictive features across different disease categories. Illustrates how well features separate healthy leaves from diseased ones.

### pca_clusters.png
PCA (Principal Component Analysis) visualization reducing 45 dimensions to 2D. Shows how different disease categories cluster in lower-dimensional space.

### tsne_clusters.png
t-SNE (t-Distributed Stochastic Neighbor Embedding) visualization providing another 2D projection of the feature space. Often reveals more natural groupings than PCA.

These visualizations help validate that extracted features effectively distinguish between disease categories.

## Key Scripts

### Transformation.py
Core image transformation pipeline that:
- Loads and preprocesses leaf images
- Detects and isolates leaf from background
- Identifies diseased regions
- Performs color analysis
- Extracts morphological features

### train.py
Trains a Random Forest classifier:
- Loads feature-extracted CSV datasets (train/test)
- Trains classifier on color histogram features
- Validates on test set with accuracy and classification reports
- Packages model, config, and learnings into a ZIP file for delivery

### predict.py
Loads a trained model and predicts leaf disease with:
- Color histogram feature extraction
- Classification confidence scores
- Visual comparison (original vs. transformed)

### Classification/prepare.py
Prepares raw image dataset by:
- Organizing images into train/test splits (80/20)
- Balancing dataset classes through augmentation
- Analyzing class distribution

### Classification/create_dataset.py
Extracts features from images:
- Processes each image through Transformation pipeline
- Extracts 45 color histogram statistics per image
- Generates CSV datasets with features and labels for training
