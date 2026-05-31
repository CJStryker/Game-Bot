# Arda Mavi
import os
from uuid import uuid4
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (150, 150)


def get_img(data_path):
    # Getting image array from path:
    with Image.open(data_path) as img:
        return np.asarray(img.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.LANCZOS))


def save_img(path, img):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    image = Image.fromarray((np.asarray(img) * 255).astype('uint8') if np.asarray(img).max() <= 1 else np.asarray(img).astype('uint8'))
    image.convert('RGB').save(f'{path}__{uuid4().hex}.jpg')
    return


def get_dataset(dataset_path='Data/Train_Data'):
    # Getting all data from data path:
    try:
        X = np.load('Data/npy_train_data/X.npy')
        Y = np.load('Data/npy_train_data/Y.npy')
    except (FileNotFoundError, OSError):
        labels = os.listdir(dataset_path) # Geting labels
        X = []
        Y = []
        categories = {}
        for label in labels:
            datas_path = dataset_path+'/'+label
            for data in os.listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X.append(img)
                action_label = os.path.splitext(data)[0].split('__', 1)[0]
                if action_label not in categories:
                    categories[action_label] = len(categories)
                Y.append(categories[action_label])
        # Create dateset:
        X = np.array(X).astype('float32')/255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, len(categories))
        if not os.path.exists('Data/npy_train_data/'):
            os.makedirs('Data/npy_train_data/')
        np.save('Data/npy_train_data/X.npy', X)
        np.save('Data/npy_train_data/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X, X_test, Y, Y_test
