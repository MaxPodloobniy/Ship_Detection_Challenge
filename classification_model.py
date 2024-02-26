"""

This is ship detection model. It follows a typical CNN (Convolutional Neural Network) architecture commonly
used for classification tasks. In this architecture, there are two classes: ship and no ship, indicated by
the single neuron in the output layer with a sigmoid activation function. The choise to use this model is
influenced by the dataset's characteristics, specifically the fact that around 65% of the images do not feature ships.

"""

from tensorflow.keras.models import Sequential
from tensorflow import keras


# Model architecture
model = Sequential([
    keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(768, 768, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(16, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Model compilation
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Create train and validation ImageDataGenerator
train_data_dir = 'data/train_data'
valid_data_dir = 'data/valid_data'

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(768, 768),
    batch_size=16,
    class_mode='binary')

valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(768, 768),
    batch_size=32,
    class_mode='binary')

# Model callbacks
callbacks = [
    # Save the best model.
    keras.callbacks.ModelCheckpoint("ship_detection_model.keras", save_best_only=True, verbose=1),
    # Write logs to TensorBoard
    keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=40,
    callbacks=callbacks,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples / valid_generator.batch_size)'''
