import os
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, efficientnet
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
    TerminateOnNaN,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils import class_weight
from datetime import datetime
import time

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ---------------- CONFIGURATION ----------------
class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ❗️❗️ DATASET PATH (directly contains class folders) ❗️❗️
    # Use raw string (r"...") or forward slashes for Windows paths
    DATASET_PATH = r"C:\TensorFlowProject\dataset"
    # DATASET_PATH = "C:/TensorFlowProject/dataset" # Alternative

    # --- Output directories relative to where the script is run ---
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    TENSORBOARD_LOG_DIR = os.path.join(LOGS_DIR, 'tensorboard_logs')

    for dir_path in [LOGS_DIR, MODELS_DIR, TENSORBOARD_LOG_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # --- Training Hyperparameters ---
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    FINE_TUNE_EPOCHS = 20
    SEED = 42
    INITIAL_LR = 1e-4
    FINE_TUNE_LR = 1e-5
    # ❗️❗️ VALIDATION SPLIT RATIO (e.g., 20% for validation) ❗️❗️
    VALIDATION_SPLIT = 0.2

    # --- Model Configuration ---
    L2_REG = 0.0001
    DROPOUT_RATE = 0.4
    FINE_TUNE_UNFREEZE_LAYERS = 50

    # --- Performance (CPU Specific) ---
    USE_MIXED_PRECISION = False  # CPU training
    BUFFER_SIZE = 1000           # Set to a positive integer for shuffling
    PREFETCH_BUFFER = tf.data.AUTOTUNE

# ---------------- CPU Setup Confirmation ----------------
def setup_cpu(config):
    print("ℹ️ Checking for devices...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"⚠️ Found GPU(s): {gpus}, but configured for CPU training.")
    else:
        print("✅ No GPU found. Proceeding with CPU training.")

    config.USE_MIXED_PRECISION = False
    tf.keras.mixed_precision.set_global_policy('float32')
    print("✅ Mixed precision disabled (using float32).")
    print("⏳ NOTE: Training on CPU will be significantly slower than on GPU.")

# ---------------- DATA PIPELINE (tf.data with validation_split) ----------------
def create_data_pipeline(config, subset_type):
    """Creates a tf.data pipeline, splitting data if needed."""
    if subset_type not in ['training', 'validation']:
        raise ValueError("subset_type must be 'training' or 'validation'")

    print(f"Creating {subset_type} pipeline from: {config.DATASET_PATH}")
    print(f"Using {config.VALIDATION_SPLIT*100:.1f}% of data for validation.")

    try:
        dataset = tf.keras.utils.image_dataset_from_directory(
            config.DATASET_PATH,  # Point to the main dataset directory
            labels='inferred',
            label_mode='categorical',
            image_size=config.IMG_SIZE,
            interpolation='nearest',
            batch_size=None,  # Load individually for mapping
            shuffle=True,     # Shuffle before splitting
            seed=config.SEED, # MUST use the same seed for train/val split
            validation_split=config.VALIDATION_SPLIT,  # Specify the split ratio
            subset=subset_type  # Specify 'training' or 'validation'
        )
    except Exception as e:
        raise ValueError(f"❌ ERROR creating dataset from '{config.DATASET_PATH}' for subset '{subset_type}'. "
                         f"Check path and directory structure (should contain class folders directly). Original error: {e}")

    class_names = dataset.class_names
    num_classes = len(class_names)

    # Count samples
    initial_sample_count = tf.data.experimental.cardinality(dataset).numpy()
    if initial_sample_count == tf.data.experimental.UNKNOWN_CARDINALITY:
        print(f"   Could not determine exact sample count for {subset_type} subset beforehand.")
    else:
        print(f"   Estimated samples in this subset: {initial_sample_count}")
        if initial_sample_count == 0:
            raise ValueError(f"❌ ERROR: The '{subset_type}' subset created from '{config.DATASET_PATH}' is empty! "
                             f"Check if the dataset path is correct and contains images, or adjust VALIDATION_SPLIT.")

    augmentation_layers = tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=config.SEED),
        layers.RandomRotation(0.15, seed=config.SEED),
        layers.RandomZoom(0.2, seed=config.SEED),
        layers.RandomTranslation(height_factor=0.15, width_factor=0.15, seed=config.SEED),
    ], name='augmentation')

    def preprocess_image(image, label):
        if subset_type == 'training':
            image = augmentation_layers(image, training=True)
        image = efficientnet.preprocess_input(image)
        return image, label

    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if subset_type == 'training':
        dataset = dataset.shuffle(config.BUFFER_SIZE, seed=config.SEED)

    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=config.PREFETCH_BUFFER)

    print(f"✅ {subset_type.capitalize()} pipeline created.")
    print(f"   Classes found: {num_classes} -> {class_names}")

    return dataset, class_names, num_classes

# ---------------- MODEL BUILDING ----------------
def build_model(num_classes, config, base_model_trainable=False):
    """Builds the model with EfficientNetB0 base."""
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*config.IMG_SIZE, 3),
        pooling=None
    )

    base_model.trainable = base_model_trainable

    if base_model_trainable and config.FINE_TUNE_UNFREEZE_LAYERS > 0:
        print(f"Unfreezing the top {config.FINE_TUNE_UNFREEZE_LAYERS} layers of the base model.")
        for layer in base_model.layers[-config.FINE_TUNE_UNFREEZE_LAYERS:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False

    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 3))
    x = base_model(inputs, training=base_model_trainable)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="head_dropout1")(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(config.L2_REG), name="head_dense1")(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="head_dropout2")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output_predictions")(x)

    model = models.Model(inputs, outputs)

    return model

# ---------------- TRAINING FUNCTION ----------------
def train_model():
    start_time = time.time()
    set_random_seeds()
    config = Config()
    setup_cpu(config)  # Setup for CPU execution

    # --- Create Data Pipelines using validation_split ---
    try:
        train_dataset, class_names, num_classes = create_data_pipeline(
            config, subset_type='training'
        )
        val_dataset, _, _ = create_data_pipeline(
            config, subset_type='validation'
        )
    except ValueError as e:
        print(f"❌ Failed to create data pipelines: {e}")
        return None, None

    # --- Calculate Class Weights (from the training subset only) ---
    print("\nCalculating class weights from the training subset...")
    try:
        labels = np.concatenate([y for x, y in train_dataset], axis=0)
        class_indices = np.argmax(labels, axis=1)
        unique_classes_in_train = np.unique(class_indices)

        print(f"   Unique class indices found in training data: {unique_classes_in_train}")
        if len(unique_classes_in_train) == 0:
            raise ValueError("No labels found in the training dataset after splitting.")
        if len(unique_classes_in_train) < num_classes:
            print(f"⚠️ Warning: Only {len(unique_classes_in_train)} out of {num_classes} classes are present in the training subset.")
            print(f"   Missing class indices: {set(range(num_classes)) - set(unique_classes_in_train)}")
            print(f"   This might happen if VALIDATION_SPLIT is large or some classes have very few samples.")

        class_weights_array = class_weight.compute_class_weight(
            'balanced',
            classes=unique_classes_in_train,  # Use only classes present in train set
            y=class_indices
        )
        class_weights = dict(zip(unique_classes_in_train, class_weights_array))

        for i in range(num_classes):
            if i not in class_weights:
                print(f"   Assigning default weight 1.0 to class index {i} (missing from training split).")
                class_weights[i] = 1.0

        print(f"Class Weights computed: {class_weights}")

    except Exception as e:
        print(f"⚠️ Warning: Could not compute class weights ({e}). Using None.")
        class_weights = None

    # --- Callbacks ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(config.MODELS_DIR, f'best_model_{timestamp}.keras')
    csv_logger_path = os.path.join(config.LOGS_DIR, f'training_log_{timestamp}.csv')
    tensorboard_logdir = os.path.join(config.TENSORBOARD_LOG_DIR, f'run_{timestamp}')

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                        monitor='val_loss', mode='min', verbose=1),
        CSVLogger(csv_logger_path),
        TensorBoard(log_dir=tensorboard_logdir, histogram_freq=1),
        TerminateOnNaN()
    ]

    # === Phase 1: Initial Training (Frozen Base Model) ===
    print("\n===== PHASE 1: Initial Training (Base Model Frozen) =====")
    model = build_model(num_classes, config, base_model_trainable=False)

    model.compile(
        optimizer=Adam(learning_rate=config.INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    print("Initial Model Summary:")
    model.summary(line_length=120)

    print("Starting Initial Training Phase (on CPU)... Be Patient!")
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # --- Load Best Weights from Phase 1 ---
    print(f"\nLoading best weights from Phase 1 saved at: {checkpoint_path}")
    try:
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)
            print("✅ Best weights loaded successfully.")
        else:
            print(f"⚠️ Checkpoint file '{checkpoint_path}' not found. Proceeding with current weights.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load best weights from {checkpoint_path}. Proceeding with current weights. Error: {e}")

    # === Phase 2: Fine-Tuning (Unfreeze Base Model Layers) ===
    print("\n===== PHASE 2: Fine-Tuning (Unfreezing Base Model Layers) =====")
    base_model = model.get_layer('efficientnetb0')
    base_model.trainable = True
    unfreeze_count = 0
    num_base_layers = len(base_model.layers)
    layers_to_unfreeze = min(config.FINE_TUNE_UNFREEZE_LAYERS, num_base_layers)

    print(f"Attempting to unfreeze the top {layers_to_unfreeze} layers of the base model.")
    for layer in base_model.layers[-layers_to_unfreeze:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
            unfreeze_count += 1
        else:
            layer.trainable = False
    print(f"Successfully unfroze {unfreeze_count} layers in the base model for fine-tuning.")

    model.compile(
        optimizer=Adam(learning_rate=config.FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    print("\nFine-Tuning Model Summary:")
    model.summary(line_length=120)

    total_epochs = config.EPOCHS + config.FINE_TUNE_EPOCHS
    callbacks[3] = CSVLogger(csv_logger_path, append=True)

    print("Starting Fine-Tuning Phase (on CPU)... Be Patient!")
    history_fine_tune = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=config.EPOCHS,
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks
    )

    final_model_path = os.path.join(config.MODELS_DIR, f'final_model_epoch_{total_epochs}_{timestamp}.keras')
    model.save(final_model_path)
    print(f"\nTraining completed successfully! 🎯")
    print(f"Best model during training saved to: {checkpoint_path}")
    print(f"Final model state (end of training) saved to: {final_model_path}")
    print(f"Training logs saved to: {csv_logger_path}")
    print(f"TensorBoard logs: Use 'tensorboard --logdir \"{config.TENSORBOARD_LOG_DIR}\"'")
    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    total_time_hours = total_time_minutes / 60
    print(f"Total Training Time: {total_time_minutes:.2f} minutes ({total_time_hours:.2f} hours)")

    return model, history_fine_tune

if __name__ == '__main__':
    try:
        model, history = train_model()
    except Exception as e:
        print(f"\n❌❌❌ Training failed: {str(e)} ❌❌❌")
        import traceback
        traceback.print_exc()
