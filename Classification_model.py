import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Опис моделі
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(768, 768, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Компіляція моделі
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Визначення генераторів для навчання та валідації
train_data_dir = 'data/train_data'
valid_data_dir = 'data/valid_data'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(768, 768),
    batch_size=32,
    class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(768, 768),
    batch_size=32,
    class_mode='binary')

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("ship_detection_model.keras", save_best_only=True, verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                         patience=3, verbose=1, mode='max',
                                         epsilon=0.0001, cooldown=2, min_lr=1e-6)
]

# Навчання моделі
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=40,
    callbacks=callbacks,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples / valid_generator.batch_size)
