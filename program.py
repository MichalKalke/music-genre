import json
import math
import random as rand

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# path to json
DATA_PATH = "C:/Users/Dom/Desktop/biali/data.json"

# music file to process
example_file = "sample1.wav"
#example_file = "sample2.wav"
#example_file = "sample3.wav"
#example_file = "sample4.wav"
#example_file = "sample5.wav"




def loadData():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    # convert lists to numpy arrays
    X = np.array(data["mfcc"])
    Y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, Y


def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")

    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    plt.show()
    #create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")




def cnn():

    # load data
    X, Y = loadData()

    # create train, validation and test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2)

    # add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    # build the CNN
    model_cnn = tf.keras.Sequential()

    # 1st conv layer
    model_cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model_cnn.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model_cnn.add(tf.keras.layers.BatchNormalization())

    # 2nd conv layer
    model_cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model_cnn.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model_cnn.add(tf.keras.layers.BatchNormalization())

    # 3rd conv layer
    model_cnn.add(tf.keras.layers.Conv2D(64, (2, 2), activation='relu', input_shape=input_shape))
    model_cnn.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model_cnn.add(tf.keras.layers.BatchNormalization())


    # flatten output and feed it into dense layer
    model_cnn.add(tf.keras.layers.Flatten())
    model_cnn.add(tf.keras.layers.Dense(64, activation='relu'))
    model_cnn.add(tf.keras.layers.Dropout(0.3))

    # output layer
    model_cnn.add(tf.keras.layers.Dense(10, activation='softmax'))

    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model_cnn.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model_cnn.summary()

    # train model
    history = model_cnn.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), batch_size=32, epochs=500)

    # plot accuracy and error as a function of the epochs
    plot_history(history)

    # evaluate model on Test Set
    test_loss, test_acc = model_cnn.evaluate(X_test, Y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    model_cnn.save("Music_Genre_CNN")
    model_cnn.save("Music_Genre_CNN.h5")

    # it can be used to reconstruct the model identically
    # reconstructed_model = tf.keras.models.load_model("Music_Genre_CNN.h5")


# audio files pre-processing
def process_input(audio_file, track_duration):
    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    N_FTT = 2048
    HOP_LENGTH = 512
    TRACK_DURATION = track_duration
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    NUM_SEGMENTS = 10

    samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

    signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)

    for d in range(10):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
        mfcc = mfcc.T

        return mfcc


def program():

    genre_dict = {0: "blues", 1: "classical", 2: "country", 3: "disco", 4: "hiphop", 5: "jazz", 6: "metal", 7: "pop", 8: "reggae", 9: "rock"}
    new_input_mfcc = process_input(example_file, 30)

    X_to_predict = new_input_mfcc[np.newaxis, ..., np.newaxis]

    model_cnn = tf.keras.models.load_model("Music_Genre_CNN.h5")
    prediction = model_cnn.predict(X_to_predict)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Predicted Genre: ", genre_dict[int(predicted_index)])