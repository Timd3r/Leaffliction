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
├── predict.py              # Disease prediction script
├── clean.py                # Dataset cleaning utility
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
```

### 2. Feature Extraction
Create training and test datasets with extracted color features:
```bash
python3 create_dataset.py
```

### 3. Disease Prediction
Predict disease on a leaf image:
```bash
python3 ../predict.py "path/to/leaf/image.jpg"
```

### 4. Image Augmentation
Augment a single image with various transformations:
```bash
python3 ../Augmentation.py "path/to/image.jpg"
```

### 5. Dataset Distribution Analysis
Analyze the distribution of classes in your dataset:
```bash
python3 ../Distribution.py "./data/train/Apple"
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

## Key Scripts

### Transformation.py
Core image transformation pipeline that:
- Loads and preprocesses leaf images
- Detects and isolates leaf from background
- Identifies diseased regions
- Performs color analysis
- Extracts morphological features

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
