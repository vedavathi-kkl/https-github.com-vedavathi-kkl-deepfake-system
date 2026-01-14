from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=1,
    class_mode='binary'
)

print(train_gen.class_indices)