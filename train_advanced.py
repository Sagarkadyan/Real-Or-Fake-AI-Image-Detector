"""
Advanced AI vs Human Image Detection Model
Improved training script with enhanced architecture and techniques
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# -----------------------
# CONFIG
# -----------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50  # Increased epochs for better training
DATASET_PATH = "dataset"
MODEL_PATH = "real_vs_ai_model.h5"

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print("❌ Dataset folder not found!")
    print("Please run 'python prepare_dataset.py' first to download and prepare the dataset.")
    exit(1)

# -----------------------
# DATA PREPARATION
# -----------------------
print("📊 Preparing data with optimized augmentation...")
print("⚠️  Note: Reduced augmentation to preserve key distinguishing features")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,  # Reduced rotation to preserve artifacts
    zoom_range=0.2,  # Reduced zoom
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,  # Removed vertical flip to preserve artifacts
    brightness_range=[0.7, 1.3],  # Reduced brightness range
    shear_range=0.1,
    channel_shift_range=0.1,  # Reduced color channel shifts
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Note: Classes are sorted alphabetically in flow_from_directory:
# "fake" = class 0 (AI Generated)
# "real" = class 1 (Human Prepared)
# Model output > 0.5 = class 1 (real/human), <= 0.5 = class 0 (fake/AI)

print(f"✅ Training samples: {train_data.samples}")
print(f"✅ Validation samples: {val_data.samples}")
print(f"✅ Class indices: {train_data.class_indices}")

# Calculate class weights to handle imbalance
class_counts = {}
for class_name, class_idx in train_data.class_indices.items():
    # Count samples in training data
    class_dir = os.path.join(DATASET_PATH, class_name)
    if os.path.exists(class_dir):
        count = len([f for f in os.listdir(class_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_idx] = count

# Calculate class weights (handle division by zero)
if len(class_counts) > 0 and sum(class_counts.values()) > 0:
    total_samples = sum(class_counts.values())
    class_weights = {}
    for class_idx, count in class_counts.items():
        if count > 0:
            class_weights[class_idx] = total_samples / \
                (len(class_counts) * count)
        else:
            class_weights[class_idx] = 1.0
    print(f"✅ Class weights: {class_weights}")
else:
    class_weights = None
    print("⚠️ Could not calculate class weights, using default")

# -----------------------
# ENHANCED MODEL ARCHITECTURE
# -----------------------
print("\n🏗️ Building enhanced model with advanced feature extraction...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    alpha=1.0  # Full MobileNetV2
)

# Freeze base model initially
base_model.trainable = False

# Build model with enhanced architecture
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),  # Added batch normalization
    layers.Dense(1024, activation="relu"),  # Increased neurons
    layers.Dropout(0.5),  # Increased dropout
    layers.BatchNormalization(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.4),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

# Compile with enhanced metrics
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Lower initial LR for stability
    loss="binary_crossentropy",
    metrics=["accuracy", "precision", "recall", "AUC"]
)

print("✅ Enhanced model compiled successfully!")
model.summary()

# -----------------------
# ADVANCED CALLBACKS
# -----------------------
checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_auc",  # Monitor AUC instead of accuracy
    save_best_only=True,
    verbose=1,
    mode='max'
)

early_stopping = EarlyStopping(
    monitor="val_auc",  # Monitor AUC instead of accuracy
    patience=12,  # Increased patience
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,  # Reduced factor for more gradual decrease
    patience=7,
    min_lr=1e-8,
    verbose=1
)

# -----------------------
# ADVANCED TRAINING PHASE 1: Train with frozen base
# -----------------------
print("\n🚀 Phase 1: Training with frozen base model...")
print("=" * 50)

# Prepare fit arguments
fit_kwargs = {
    'epochs': EPOCHS // 2,  # First half with frozen base
    'callbacks': [checkpoint, early_stopping, reduce_lr],
    'verbose': 1
}

# Add class weights only if calculated
if class_weights is not None:
    fit_kwargs['class_weight'] = class_weights

history1 = model.fit(
    train_data,
    validation_data=val_data,
    **fit_kwargs
)

# -----------------------
# ADVANCED TRAINING PHASE 2: Fine-tune base model
# -----------------------
print("\n🔄 Phase 2: Fine-tuning base model...")
print("=" * 50)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
# Freeze early layers, fine-tune later layers
for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.00005),  # Lower LR for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy", "precision", "recall", "AUC"]
)

print(
    f"✅ Trainable layers: {sum([1 for layer in base_model.layers if layer.trainable])}")

# Calculate starting epoch for phase 2
initial_epoch_phase2 = len(history1.history['accuracy'])
total_epochs_phase2 = initial_epoch_phase2 + (EPOCHS // 2)

# Prepare fit arguments for phase 2
fit_kwargs2 = {
    'epochs': total_epochs_phase2,  # Total epochs including phase 1
    'callbacks': [checkpoint, early_stopping, reduce_lr],
    'verbose': 1,
    'initial_epoch': initial_epoch_phase2
}

# Add class weights only if calculated
if class_weights is not None:
    fit_kwargs2['class_weight'] = class_weights

history2 = model.fit(
    train_data,
    validation_data=val_data,
    **fit_kwargs2
)

# Combine histories
history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
    'auc': history1.history['auc'] + history2.history['auc'] if 'auc' in history1.history else [],
    'val_auc': history1.history['val_auc'] + history2.history['val_auc'] if 'val_auc' in history1.history else []
}

# -----------------------
# SAVE FINAL MODEL
# -----------------------
model.save(MODEL_PATH)
print("\n" + "=" * 50)
print("✅ Enhanced model trained and saved as", MODEL_PATH)
print(f"✅ Final Training Accuracy: {history['accuracy'][-1]:.4f}")
print(f"✅ Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
print(f"✅ Best Validation Accuracy: {max(history['val_accuracy']):.4f}")
if history['val_auc']:
    print(f"✅ Best Validation AUC: {max(history['val_auc']):.4f}")
print("=" * 50)
print("\n🎉 Training complete! Your enhanced model is ready to analyze images.")
print("💡 Run 'python predict.py <image_path>' to test on images.")
print("💡 Run 'python evaluate.py' for comprehensive model evaluation.")
print("💡 Run 'python test.py' to evaluate model performance.")
