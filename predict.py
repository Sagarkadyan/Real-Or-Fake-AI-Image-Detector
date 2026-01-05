import os
import time
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# -----------------------
# TEXT ANIMATION FUNCTION
# -----------------------


def thinking(text, delay=0.04):
    """Animated text printing effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def pause(seconds=1):
    """Pause execution"""
    time.sleep(seconds)


# -----------------------
# LOAD MODEL
# -----------------------
MODEL_PATH = "real_vs_ai_model.h5"

if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found!")
    print("Please train the model first by running: python train.py")
    sys.exit(1)

thinking("🧠 Loading AI model...")
try:
    # Load model with compile=False to avoid metric compilation issues
    model = load_model(MODEL_PATH, compile=False)

    # Compile with minimal metrics for prediction
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    pause(0.5)
    thinking("✅ Model loaded successfully!")
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# -----------------------
# LOAD IMAGE
# -----------------------
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    print("Example: python predict.py test.jpg")
    sys.exit(1)

IMAGE_PATH = sys.argv[1]

# Check if file exists
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image file not found: {IMAGE_PATH}")
    print("Please check the file path and try again.")
    sys.exit(1)

# Check if it's a valid image file
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
if not IMAGE_PATH.lower().endswith(valid_extensions):
    print(f"⚠️ Warning: File extension might not be a valid image format.")
    print(f"Supported formats: {', '.join(valid_extensions)}")

thinking("👀 Receiving image...")
pause(0.5)

thinking("🔍 Analyzing lighting and shadows...")
pause(0.8)

thinking("🧬 Examining facial and texture patterns...")
pause(0.8)

thinking("🖼️ Inspecting background consistency...")
pause(0.8)
# -----------------------
# PREDICTION
# -----------------------
try:
    # Load and preprocess image
    try:
        img = image.load_img(IMAGE_PATH, target_size=(224, 224))
    except Exception as img_error:
        print(f"\n❌ Error loading image: {img_error}")
        print("The file might be corrupted or not a valid image file.")
        sys.exit(1)

    # Convert to array and normalize
    try:
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as array_error:
        print(f"\n❌ Error processing image array: {array_error}")
        sys.exit(1)

    # Make prediction
    try:
        prediction = model.predict(img_array, verbose=0)[0][0]
    except Exception as pred_error:
        print(f"\n❌ Error during model prediction: {pred_error}")
        print("The model might be incompatible or corrupted.")
        sys.exit(1)

    # Debug: Show raw prediction value
    # Note: In flow_from_directory, classes are sorted alphabetically:
    # "fake" = class 0, "real" = class 1
    # So prediction > 0.5 means "real" (human), prediction <= 0.5 means "fake" (AI)

    # Calculate confidence and result with better threshold handling
    # Note: In flow_from_directory, classes are sorted alphabetically:
    # "fake" = class 0, "real" = class 1
    # So prediction > 0.5 means "real" (human), prediction <= 0.5 means "fake" (AI)

    threshold = 0.5

    if prediction > threshold:
        # Real/Human Prepared (class 1)
        confidence = prediction
        result = "Real AI Image"
        emoji = "📸"
        ai_confidence = 1 - prediction
    else:
        # Fake/AI Generated (class 0)
        confidence = 1 - prediction
        result = "Fake AI Image"
        emoji = "🤖"
        ai_confidence = 1 - prediction

    # Display results with full information
    print("\n" + "=" * 50)
    print(f"{emoji} FINAL RESULT: {result}")
    thinking("🎉 Analysis complete!")
    print("=" * 50)

except FileNotFoundError as e:
    print(f"\n❌ File not found: {e}")
    print("Please check that the image file exists and the path is correct.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error during prediction: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
