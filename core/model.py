# -*- coding: utf-8 -*-
"""
Model Building and Training
Xây dựng và huấn luyện emotion detection model
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import (EMOTIONS, MODEL_CONFIG, TRAINING_CONFIG, 
                     AUGMENTATION_CONFIG, PATHS)


def build_mobilenet_model():
    """
    Xây dựng MobileNetV2 model cho emotion detection
    
    Returns:
        Compiled Keras model
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=MODEL_CONFIG['input_shape'],
        include_top=False,
        weights=MODEL_CONFIG['weights']
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Add dropout and dense layers
    x = Dropout(MODEL_CONFIG['dropout_rates'][0])(x)
    x = Dense(MODEL_CONFIG['dense_units'][0], activation='relu', 
             kernel_regularizer=tf.keras.regularizers.l2(MODEL_CONFIG['l2_regularization']))(x)
    
    x = Dropout(MODEL_CONFIG['dropout_rates'][1])(x)
    x = Dense(MODEL_CONFIG['dense_units'][1], activation='relu',
             kernel_regularizer=tf.keras.regularizers.l2(MODEL_CONFIG['l2_regularization']))(x)
    
    x = Dropout(MODEL_CONFIG['dropout_rates'][2])(x)
    predictions = Dense(len(EMOTIONS), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[:MODEL_CONFIG['unfreeze_layers']]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_data_generator():
    """
    Tạo ImageDataGenerator cho data augmentation
    
    Returns:
        ImageDataGenerator instance
    """
    return ImageDataGenerator(**AUGMENTATION_CONFIG)


def create_callbacks(progress, epoch_label, status_label, time_label, total_epochs, root):
    """
    Tạo training callbacks
    
    Args:
        progress: progress bar widget
        epoch_label: epoch label widget
        status_label: status label widget
        time_label: time label widget
        total_epochs: total number of epochs
        root: tkinter root window
    
    Returns:
        list of callbacks
    """
    from ui.training_window import TrainingProgressCallback
    
    # Progress callback
    progress_callback = TrainingProgressCallback(
        progress, epoch_label, status_label, time_label, total_epochs, root
    )
    
    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=TRAINING_CONFIG['early_stop_patience'],
        restore_best_weights=True
    )
    
    # Reduce learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=TRAINING_CONFIG['reduce_lr_factor'],
        patience=TRAINING_CONFIG['reduce_lr_patience'],
        min_lr=TRAINING_CONFIG['min_lr']
    )
    
    return [progress_callback, early_stop, reduce_lr]


def train_model(model, X_train, y_train, X_test, y_test, callbacks):
    """
    Train the emotion detection model
    
    Args:
        model: Keras model
        X_train: training images
        y_train: training labels
        X_test: test images
        y_test: test labels
        callbacks: list of callbacks
    
    Returns:
        training history
    """
    datagen = create_data_generator()
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=TRAINING_CONFIG['batch_size']),
        epochs=TRAINING_CONFIG['epochs'],
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def save_model(model, path=None):
    """
    Lưu model
    
    Args:
        model: Keras model
        path: save path (optional)
    """
    if path is None:
        path = PATHS['model']
    model.save(path)


def load_model(path=None):
    """
    Load model
    
    Args:
        path: model path (optional)
    
    Returns:
        Keras model
    """
    if path is None:
        path = PATHS['model']
    return tf.keras.models.load_model(path)
