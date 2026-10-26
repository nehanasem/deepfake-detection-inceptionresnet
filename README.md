# Deepfake Image Detection Using InceptionResNetV1

## Project Overview

This project explores deepfake image detection using **InceptionResNetV1**, a deep neural network pretrained on the VGGFace2 dataset and adapted for binary image classification.

The model classifies input images into two categories: **Fake** and **Real**. The project uses transfer learning followed by partial fine-tuning to adapt pretrained image representations to the deepfake detection task.

## Objectives

- Develop a deep learning pipeline for real-versus-fake image classification.
- Apply transfer learning using a pretrained InceptionResNetV1 backbone.
- Fine-tune selected network layers for the target classification task.
- Evaluate model performance using classification metrics and a confusion matrix.

## Model Architecture

- **Backbone:** InceptionResNetV1
- **Pretraining:** VGGFace2
- **Input image size:** 160 × 160 pixels
- **Task:** Binary classification
- **Class mapping:** `Fake = 0`, `Real = 1`
- **Training strategy:** Classifier-head training followed by partial fine-tuning

## Dataset Organization

The dataset is organized into training, validation, and test splits:

```text
Dataset/
├── Train/
│   ├── Fake/
│   └── Real/
├── Validation/
│   ├── Fake/
│   └── Real/
└── Test/
    ├── Fake/
    └── Real/
```

The dataset contains real and fake images in each split. Dataset files are not included in this repository.

## Training Pipeline

1. Load the pretrained InceptionResNetV1 model.
2. Train the binary classification head while the backbone is frozen.
3. Unfreeze selected layers and fine-tune the network.
4. Monitor validation loss and classification metrics to select a checkpoint.
5. Evaluate the selected model on the held-out test set.

## Recorded Results

One recorded experiment achieved **91.60% test accuracy** on 10,905 test images.

| Metric | Fake | Real |
|---|---:|---:|
| Precision | 89.10% | 94.50% |
| Recall | 94.94% | 88.21% |
| F1-score | 91.93% | 91.25% |

**Overall test accuracy:** 91.60%

These results describe one recorded experiment. Performance may vary depending on the training environment, checkpoint, and experimental configuration.

## Repository Contents

- `deepfake_detection_inceptionresnet_github_ready.ipynb` — notebook containing the training and evaluation workflow.
- `README.md` — project documentation.
- `.gitignore` — rules for excluding datasets, credentials, model weights, and temporary files.

## Running the Notebook

1. Open the notebook in Google Colab.
2. Select a GPU runtime if available.
3. Install the required Python dependencies.
4. Make the dataset available using the directory structure described above.
5. Run the notebook cells in order to train and evaluate the model.

A Kaggle API credential file may be required if the notebook downloads the dataset from Kaggle. Credentials must be kept private and must never be committed to the repository.

## Limitations

The reported accuracy applies to the dataset and experimental setup used for this run. It does not establish generalization to every deepfake generation method, image source, or real-world scenario.

## Future Improvements

- Evaluate performance on additional unseen datasets.
- Analyze robustness to image compression and resizing.
- Compare alternative model architectures.
- Investigate class-specific errors and improve generalization.

## Disclaimer

This project is intended for academic and research purposes. Predictions should not be treated as definitive proof that an image is authentic or manipulated.
