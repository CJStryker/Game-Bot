# Arda Mavi
import numpy as np
from PIL import Image


def resize_image(image, size=(150, 150)):
    image = Image.fromarray(np.asarray(image).astype('uint8')).convert('RGB')
    return np.asarray(image.resize(size, Image.Resampling.LANCZOS))


def predict(model, X):
    X = resize_image(X).astype('float32') / 255.
    return model.predict(X.reshape(1, 150, 150, 3))
