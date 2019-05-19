from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import pickle

################## SPECIFY THE DIMENSIONS (in px.) OF INPUT IMAGES (Relevant for Training, Validation, Testing)
TARGET_SIZE = (224, 224)

################## SPECIFY THE BATCH SIZE FOR GRADIENT DESCENT (Only relevant to Training phase)
BATCH_SIZE = 16


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

# Create Model
base_model = applications.InceptionV3(weights='imagenet', 
                                include_top=False, 
                                input_shape=(224, 224,3))




# Number of non-trainable layers in the base model is the sole hyper-parameter:
n_non_trainable_layers = range(100,101)

perf_summary = {}

# Perform grid search over this range:
for non_trainable_layers in n_non_trainable_layers:

        for layer in base_model.layers[:non_trainable_layers]:
                layer.trainable=False
        for layer in base_model.layers[non_trainable_layers:]:
                layer.trainable=True

        add_model = Sequential()
        add_model.add(base_model)
        add_model.add(GlobalAveragePooling2D())
        add_model.add(Dropout(0.1))
        add_model.add(Dense(1, activation='sigmoid'))

        model = add_model
        model.compile(loss='binary_crossentropy', 
                optimizer=optimizers.adam(lr=1e-4),
                metrics=['accuracy'])

        # Train Model
        train_hist = model.fit_generator(
                train_generator,
                steps_per_epoch=2000 // BATCH_SIZE,
                epochs=50,
                validation_data=valid_generator,
                validation_steps=800 // BATCH_SIZE)

        # Record last epoch's history in grid search performance summary dict
        perf_summary[non_trainable_layers] = {'loss': train_hist.history['loss'][-1], 'acc': train_hist.history['acc'][-1], 'val_loss': train_hist.history['val_loss'][-1], 'val_acc': train_hist.history['val_acc'][-1]}

        # Dump train history to disk
        with open('inception_v3_history/trainHistoryDict_' + str(non_trainable_layers) + '.pickle', 'wb') as file_pi:
                pickle.dump(train_hist.history, file_pi)

# Dump grid search summary to disk
with open('inception_v3_history/gridSearchSummaryDict.pickle', 'wb') as file_pi:
        pickle.dump(perf_summary, file_pi)

# Print grid search summary
print(perf_summary)