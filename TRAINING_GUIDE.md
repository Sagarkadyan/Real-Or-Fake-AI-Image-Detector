# 🚀 Complete Training Guide - Enhanced AI vs Human Image Detector

## 📋 Step-by-Step Commands to Train Your Amazing Model

Follow these commands in order to train a highly accurate model that can distinguish between AI-generated and human-prepared images.

---

## Step 1: Prepare the Dataset (Download from Kaggle)

```bash
python prepare_dataset.py
```

**What this does:**

- Downloads 300 images per class from Kaggle dataset
- Organizes them into `dataset/real/` and `dataset/fake/` folders
- Takes about 2-5 minutes depending on your internet speed

**Expected output:**

```
✅ Copied 300 images to dataset\real
✅ Copied 300 images to dataset\fake
🎉 Dataset prepared successfully!
```

---

## Step 2: Verify Dataset

```bash
python images_check.py
```

**Expected output:**

```
REAL images: 300
FAKE images: 300
```

---

## Step 3: Train the Advanced Model

```bash
python train.py
```

**What this does:**

- **Phase 1**: Trains with frozen MobileNetV2 base (25 epochs)
- **Phase 2**: Fine-tunes the base model layers (25 epochs)
- Uses advanced data augmentation
- Applies class weights to handle imbalance
- Uses learning rate scheduling
- Saves the best model automatically

**Training time:**

- Approximately 30-60 minutes depending on your GPU/CPU
- On CPU: ~1-2 hours
- On GPU: ~15-30 minutes

**Expected output:**

```
📊 Preparing data with advanced augmentation...
✅ Training samples: 480
✅ Validation samples: 120
✅ Class indices: {'fake': 0, 'real': 1}
✅ Class weights: {0: 1.0, 1: 1.0}

🏗️ Building advanced model with fine-tuning capability...
✅ Model compiled successfully!

🚀 Phase 1: Training with frozen base model...
[Training progress...]

🔄 Phase 2: Fine-tuning base model...
[Fine-tuning progress...]

✅ Model trained and saved as real_vs_ai_model.h5
✅ Final Training Accuracy: 0.XXXX
✅ Final Validation Accuracy: 0.XXXX
✅ Best Validation Accuracy: 0.XXXX
```

---

## Step 4: Test the Model

```bash
python test.py
```

**What this does:**

- Tests the model on 20 real and 20 fake images
- Shows accuracy, precision, recall, and confusion matrix

**Expected output:**

```
📊 TEST RESULTS
✅ Overall Accuracy: XX.XX%

📋 Classification Report:
[Detailed metrics...]

📊 Confusion Matrix:
                    Predicted
                  Human    AI
Actual Human      XX     XX
Actual AI         XX     XX
```

---

## Step 5: Predict on Your Images

```bash
python predict.py path/to/your/image.jpg
```

**Example:**

```bash
python predict.py test.jpg
```

**Expected output:**

```
🧠 Loading AI model...
✅ Model loaded successfully!
👀 Receiving image...
🔍 Analyzing lighting and shadows...
🧬 Examining facial and texture patterns...
🖼️ Inspecting background consistency...
🔎 Detecting color gradients and artifacts...
🤔 Processing deep learning analysis...

==================================================
🤖 FINAL RESULT: AI Generated Image
📊 Confidence: 87.45%
📸 Human Probability: 12.55%
🔢 Raw Prediction Value: 0.1255
==================================================

🎉 Analysis complete!
```

---

## 🎯 Model Features

### Advanced Architecture:

- ✅ MobileNetV2 base (pre-trained on ImageNet)
- ✅ Batch Normalization layers
- ✅ Multiple dense layers (512 → 256 → 128 neurons)
- ✅ Dropout regularization (0.6 → 0.5 → 0.4)
- ✅ Fine-tuning capability

### Training Improvements:

- ✅ **300 images per class** (increased from 100)
- ✅ **Advanced data augmentation** (rotation, zoom, shifts, flips, brightness, shear, color shifts)
- ✅ **Class weights** to handle dataset imbalance
- ✅ **Two-phase training** (frozen base + fine-tuning)
- ✅ **Learning rate scheduling** (reduces LR when stuck)
- ✅ **Early stopping** (prevents overfitting)
- ✅ **Model checkpointing** (saves best model)

### Expected Performance:

- **Training Accuracy:** 75-85%
- **Validation Accuracy:** 70-80%
- **Test Accuracy:** 65-75%
- **Better AI detection** compared to previous model

---

## 🔧 Troubleshooting

### If dataset download fails:

```bash
# Make sure kagglehub is installed
pip install kagglehub
```

### If training is too slow:

- Reduce `IMAGES_PER_CLASS` in `prepare_dataset.py` (try 200)
- Reduce `EPOCHS` in `train.py` (try 30)
- Reduce `BATCH_SIZE` if you get memory errors

### If you get memory errors:

- Reduce `BATCH_SIZE` to 16 or 8 in `train.py`
- Close other applications
- Use fewer images per class

### If model accuracy is low:

- Train for more epochs
- Increase dataset size
- Check if dataset is balanced

---

## 📊 Model Output Format

The model will output one of two results:

1. **"AI Generated Image"** - When prediction ≤ 0.5

   - Shows AI confidence percentage
   - Shows human probability

2. **"Prepared by human"** - When prediction > 0.5
   - Shows human confidence percentage
   - Shows AI probability

---

## 🎉 Success Indicators

Your model is working well if:

- ✅ Validation accuracy > 70%
- ✅ Test accuracy > 65%
- ✅ Both classes (AI and Human) have good recall (>60%)
- ✅ Confusion matrix shows balanced predictions

---

## 💡 Tips for Best Results

1. **More data = Better model**: Use 300+ images per class
2. **Patience**: Let training complete fully (don't stop early)
3. **Test thoroughly**: Test on various image types
4. **Retrain if needed**: If accuracy is low, try training again

---

## 🚀 Quick Start (All Commands)

Run these commands in sequence:

```bash
# 1. Prepare dataset
python prepare_dataset.py

# 2. Verify dataset
python images_check.py

# 3. Train model (this takes time!)
python train.py

# 4. Test model
python test.py

# 5. Predict on your image
python predict.py test.jpg
```

---

**Good luck training your amazing AI vs Human image detector! 🎯**
