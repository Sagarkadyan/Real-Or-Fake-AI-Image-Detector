"""
Comprehensive model evaluation script
This script evaluates the model with various metrics and provides detailed analysis
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "real_vs_ai_model.h5"
DATASET_PATH = "dataset"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    print("Please train the model first by running: python train.py")
    exit(1)

# -----------------------
# LOAD MODEL
# -----------------------
print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!\n")

# -----------------------
# PREPARE TEST DATA
# -----------------------
print("📊 Preparing test data...")
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False  # Keep order consistent
)

print(f"✅ Test samples: {test_generator.samples}")
print(f"✅ Class indices: {test_generator.class_indices}")

# -----------------------
# PREDICTIONS
# -----------------------
print("\n🔍 Making predictions...")
predictions = model.predict(test_generator, verbose=1)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = test_generator.classes

print("✅ Predictions completed!\n")

# -----------------------
# EVALUATION METRICS
# -----------------------
print("=" * 60)
print("📊 COMPREHENSIVE MODEL EVALUATION")
print("=" * 60)

# Basic accuracy
accuracy = np.mean(y_pred == y_true)
print(f"✅ Overall Accuracy: {accuracy * 100:.2f}%")

# Detailed classification report
print(f"\n📋 Detailed Classification Report:")
class_names = ['AI Generated', 'Human Prepared']
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"\n🧮 Confusion Matrix:")
print("                 Predicted")
print("               AI    Human")
print(f"Actual AI     {cm[0][0]:5d} {cm[0][1]:5d}")
print(f"Actual Human  {cm[1][0]:5d} {cm[1][1]:5d}")

# Calculate specific metrics
tn, fp, fn, tp = cm.ravel()
precision_ai = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_ai = tp / (tp + fn) if (tp + fn) > 0 else 0
precision_human = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_human = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n🎯 Per-Class Metrics:")
print(
    f"   AI Generated - Precision: {precision_ai:.3f}, Recall: {recall_ai:.3f}")
print(
    f"   Human Prepared - Precision: {precision_human:.3f}, Recall: {recall_human:.3f}")

# AUC-ROC
fpr, tpr, thresholds = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)
print(f"\n📈 AUC-ROC Score: {roc_auc:.3f}")

# Confidence analysis
high_confidence_predictions = predictions[(
    predictions >= 0.9) | (predictions <= 0.1)]
high_confidence_accuracy = np.mean(
    (high_confidence_predictions > 0.5).astype(int).flatten() ==
    y_true[(predictions >= 0.9) | (predictions <= 0.1)]
)
print(
    f"🎯 High Confidence Accuracy (predictions > 0.9 or < 0.1): {high_confidence_accuracy * 100:.2f}%")

# Confidence distribution
low_confidence_predictions = predictions[(
    predictions > 0.4) & (predictions < 0.6)]
low_confidence_accuracy = np.mean(
    (low_confidence_predictions > 0.5).astype(int).flatten() ==
    y_true[(predictions > 0.4) & (predictions < 0.6)]
)
print(
    f"🎯 Low Confidence Accuracy (predictions 0.4-0.6): {low_confidence_accuracy * 100:.2f}%")

print("\n" + "=" * 60)

# -----------------------
# VISUALIZATION
# -----------------------
plt.figure(figsize=(15, 5))

# Plot 1: Confusion Matrix Heatmap
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['AI Generated', 'Human Prepared'],
            yticklabels=['AI Generated', 'Human Prepared'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Plot 2: Prediction Distribution
plt.subplot(1, 3, 2)
plt.hist(predictions[y_true == 0], bins=50,
         alpha=0.7, label='AI Generated', color='red')
plt.hist(predictions[y_true == 1], bins=50, alpha=0.7,
         label='Human Prepared', color='blue')
plt.xlabel('Prediction Score')
plt.ylabel('Frequency')
plt.title('Prediction Distribution by Class')
plt.legend()

# Plot 3: ROC Curve
plt.subplot(1, 3, 3)
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2,
         linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("📊 Evaluation plots saved as 'model_evaluation.png'")
plt.show()

print("\n🎉 Model evaluation complete!")
print("💡 Run 'python predict.py <image_path>' to test on individual images.")
