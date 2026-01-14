from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
gen = datagen.flow_from_directory("dataset", target_size=(224, 224))

print("\nDetected classes:", gen.class_indices)