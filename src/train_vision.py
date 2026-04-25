import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = str(PROJECT_ROOT / 'data' / 'processed' / 'train')
VAL_DIR = str(PROJECT_ROOT / 'data' / 'processed' / 'val')
MODEL_DIR = PROJECT_ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = str(MODEL_DIR / 'efficientnet_damage_type.h5')
CLASS_MAP_PATH = str(MODEL_DIR / 'class_mappings.json')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def build_and_train():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # 1. Load Data Directly from Directories
    print("Loading Training Data...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='categorical'
    )

    print("Loading Validation Data...")
    # If you haven't processed validation data yet, we can split it from train for now
    if os.path.exists(VAL_DIR) and len(os.listdir(VAL_DIR)) > 0:
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            VAL_DIR,
            shuffle=True,
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE,
            label_mode='categorical'
        )
    else:
        print("Validation folder empty or missing. Splitting from training data...")
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR, validation_split=0.2, subset="training", seed=42,
            image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
        )
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR, validation_split=0.2, subset="validation", seed=42,
            image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
        )

    # Save class names for the Streamlit app later
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    with open(CLASS_MAP_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"Saved class mappings: {class_names}")

    # Prefetch for GPU optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    # 2. Data Augmentation (Runs on GPU)
    # UPGRADE 1: Advanced Shadow/Texture Augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2) # Forces the AI to find dents in dark shadows
    ])

    # 3. Build the EfficientNet Model
    # Note: EfficientNet automatically normalizes pixel values, no Rescaling needed!
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model for Phase 1
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x) # Increased dropout to prevent overfitting
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)

    # 4. Phase 1: Train Classification Head
    print("\n--- PHASE 1: Training Classification Head ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.F1Score(average='macro', name='f1_score')]
    )

    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=8
    )

    # 5. Phase 2: Fine-Tuning (Unfreeze top layers)
    print("\n--- PHASE 2: Fine-Tuning ---")
    base_model.trainable = True
    
    # UPGRADE 2: Unfreeze Top 40 layers for deeper texture learning
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    # Compile with a much lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.F1Score(average='macro', name='f1_score')]
    )

    # UPGRADE 3: Advanced Callbacks (Learning Rate Scheduler)
    early_stop = callbacks.EarlyStopping(monitor='val_f1_score', patience=5, restore_best_weights=True, mode='max')
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)
    checkpoint = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_f1_score', mode='max')
    
    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=[early_stop, reduce_lr, checkpoint]
    )

    print(f"\nTraining Complete! Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    build_and_train()