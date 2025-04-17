# Leaffliction 🥬🩺  
_Image classification &amp; disease recognition on plant leaves_

A complete computer‑vision pipeline that balances your dataset, augments images, extracts key features (PlantCV), trains a CNN with TensorFlow / Keras, signs the resulting artefacts, and predicts diseases from new leaf photos.

---

## ✨ Features
| Stage | Script | What it does |
|-------|--------|--------------|
| **Dataset insight** | `src/Distribution.py` | Draw pie / bar charts to visualise class imbalance |
| **Augmentation** | `src/Augmentation.py` | Flip, rotate, crop, blur, contrast &amp; projective transforms |
| **Pre‑processing** | `src/train_preprocessing.py` | Balances each class, calls augmentation, removes background |
| **Image transforms** | `src/Transformation.py` | Masking, landmarks, histograms &amp; other PlantCV goodies |
| **Training** | `src/train.py` | Splits data, trains a CNN, logs metrics, zips model + data + SHA‑256 signature |
| **Inference** | `src/predict.py` | Predict a single image or a whole directory and display results |

---

## 🚀 Quick start

```bash
# 1 – create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2 – install dependencies
pip install -r requirements.txt   # opencv-python, tensorflow, plantcv, matplotlib, tqdm …

# 3 – inspect your raw dataset
python src/Distribution.py data/raw

# 4 – balance & augment (optional standalone run)
python src/train_preprocessing.py data/raw --augmented_dir data/augmented

# 5 – train the model
python src/train.py data/raw \
     --augmented_dir data/augmented \
     --mask_directory data/augmented_mask \
     --model_dir data/model

# 6 – predict
python src/predict.py path/to/leaf.jpg --model_dir data/model
# or an entire folder
python src/predict.py data/test_images --model_dir data/model
