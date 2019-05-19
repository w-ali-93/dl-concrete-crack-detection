from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pickle
import matplotlib.pyplot as plt

perform_training = 0
plot_learning_curve = 1
evaluate_best_trained_model = 0

################## SPECIFY THE DIMENSIONS (in px.) OF INPUT IMAGES (Relevant for Training, Validation, Testing)
TARGET_SIZE = (224, 224)

################## SPECIFY THE BATCH SIZE FOR GRADIENT DESCENT (Only relevant to Training phase)
BATCH_SIZE = 16

################## SPECIFY THE NUMBER OF TRAINING EPOCHS FOR GRADIENT DESCENT (Only relevant to Training phase)
N_TRAINING_EPOCHS = 150

################## SPECIFY THE NUMBER OF OPTIMIZER AND LEARNING RATE FOR GRADIENT DESCENT (Only relevant to Training phase)
LEARNING_RATE = 1e-4
OPTIMIZER = optimizers.adam(lr=LEARNING_RATE)

################## SPECIFY THE NUMBER OF TRAINING, TEST AND VALIDATION SETS FOR GRADIENT DESCENT (Only relevant to Training phase)
N_TRAIN = 10180
N_VALID = 3394
N_TEST = 3394


# Create Generators
train_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(
        directory=r"SDNET2018\train",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
        )

valid_datagen=ImageDataGenerator(rescale=1./255)
valid_generator=valid_datagen.flow_from_directory(
        directory=r"SDNET2018\valid",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
        )

test_datagen=ImageDataGenerator(rescale=1./255)
test_generator=test_datagen.flow_from_directory(
        directory=r"SDNET2018\test",
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True
        )


if perform_training:
        # Train model
        # Create Model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        # checkpoint
        cp_filepath="simple_cnn_history/weights.best.hdf5"
        checkpoint = ModelCheckpoint(cp_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        model.compile(loss='binary_crossentropy',
                optimizer=OPTIMIZER,
                metrics=['accuracy'])

        # Train Model
        train_hist = model.fit_generator(
                train_generator,
                steps_per_epoch=N_TRAIN // BATCH_SIZE,
                epochs=N_TRAINING_EPOCHS,
                validation_data=valid_generator,
                validation_steps=N_VALID // BATCH_SIZE,
                callbacks = callbacks_list)

        # Dump train history to disk
        with open('simple_cnn_history/trainHistoryDict.pickle', 'wb') as file_pi:
                pickle.dump(train_hist.history, file_pi)

if plot_learning_curve:
        with open('simple_cnn_history/trainHistoryDict.pickle', 'rb') as handle:
                training_history = pickle.load(handle)

                accuracy = training_history['acc']
                val_acc = training_history['val_acc']
                loss = training_history['loss']
                val_loss = [round(item,2) for item in training_history['val_loss']]
                epochs = list(range(len(val_loss)))

                plt.plot(epochs, accuracy, '-', color="g", label='Training accuracy')
                plt.plot(epochs, val_acc, '-', color="r", label='Validation accuracy')
                plt.plot(epochs, loss, '--', color="blue", label='Training loss')
                plt.plot(epochs, val_loss, '--', color="orange", label='Validation loss')
                plt.plot(epochs, [0.82 for epoch in epochs], color="black", label='Validation accuracy threshold (0.82)' )

                plt.legend()
                plt.xlabel("Epochs")
                plt.ylabel("Score")
                plt.show()

if evaluate_best_trained_model:
        # Estimate accuracy on test set using loaded weights for the best trained model
        # Create Model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        cp_filepath="simple_cnn_history/weights.best.hdf5"
        model.load_weights(cp_filepath)

        model.compile(loss='binary_crossentropy',
                optimizer=optimizers.adam(lr=1e-4),
                metrics=['accuracy'])

        scores = model.evaluate_generator(test_generator, N_TEST/BATCH_SIZE, workers=12)
        print('Score')
        print("%s: %.2f%%" % (model.metrics_names[0], scores[0]))
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))

        y_pred_prob = model.predict_generator(test_generator, N_TEST/BATCH_SIZE, workers=12)
        y_pred_classified = [1 * (x[0]>=0.5) for x in y_pred_prob]
        print('Confusion Matrix')
        print(confusion_matrix(test_generator.classes, y_pred_classified))
        print('Classification Report')
        print(classification_report(test_generator.classes, y_pred_classified, target_names=['non-cracked', 'cracked']))

        print(model.summary())