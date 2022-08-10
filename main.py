import collections
import datetime
import itertools
from re import X
import fluidsynth
import glob
import numpy as np
import pathlib
import pandas as pd
import pretty_midi
import seaborn as sns
import tensorflow as tf

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

from read_composer import get_composer_midi
from config import Config
from config_params import settings
from midi_processing import midi_to_notes
from midi_processing import notes_to_midi
config = Config()

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Sampling rate for audio playback
_SAMPLING_RATE = 16000

# print(datetime.datetime.now().strftime("%d_%m_%y_%H:%M"))
# exit()

# Download the dataset
data_dir = pathlib.Path('data/maestro-v2.0.0') 
if not data_dir.exists():
    tf.keras.utils.get_file(
        'maestro-v2.0.0-midi.zip',
        origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
        extract=True,
        cache_dir='.', cache_subdir='data',
    )

filenames = glob.glob(str(data_dir/'2004/*.mid*'))   # **/*.mid*
filenames_alex = glob.glob(str(data_dir/'alex/*.midi'))
filenames_alex = glob.glob(str(data_dir/settings.data.source))
print('Number of files:', len(filenames))

sample_file = filenames_alex[1]

# get a specific sample file
# for file in filenames:
#     if file == "data\\maestro-v2.0.0\\" + "alex\\AI_Training_NotePad.midi":
#         sample_file = file
#         print("FOUND THE FILE")
#         break

print(sample_file)

pm = pretty_midi.PrettyMIDI(sample_file)

# def display_audio(pm: pretty_midi.PrettyMIDI, seconds=30):
#     waveform = pm.fluidsynth(fs=_SAMPLING_RATE)
#     # Take a sample of the generated waveform to mitigate kernel resets
#     waveform_short = waveform[:seconds*_SAMPLING_RATE]
#     return display.Audio(waveform_short, rate=_SAMPLING_RATE)

# display_audio(pm)

# print('Number of instruments:', len(pm.instruments))
instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
# print('Instrument name:', instrument_name)

# for i, note in enumerate(instrument.notes[:10]):
#   note_name = pretty_midi.note_number_to_name(note.pitch)
#   duration = note.end - note.start
#   print(f'{i}: pitch={note.pitch}, note_name={note_name},'
#         f' duration={duration:.4f}')


raw_notes = midi_to_notes(sample_file)
raw_notes.head()

get_note_names = np.vectorize(pretty_midi.note_number_to_name)
sample_note_names = get_note_names(raw_notes['pitch'])
sample_note_names[:10]


# Create the training dataset
num_files = 640
all_notes = []
# get the first num_files files
for f in filenames_alex:# [:num_files]:
    notes = midi_to_notes(f)                                                                                                                                                              
    all_notes.append(notes)

# get all files from a composer
# for f in get_composer_midi("Franz Liszt")[:num_files]:
#     notes = midi_to_notes(f)                                                                                                                                                              
#     all_notes.append(notes)

all_notes = pd.concat(all_notes)
n_notes = len(all_notes)
print('Number of notes parsed:', n_notes)

key_order = ['pitch', 'step', 'duration']
train_notes = np.stack([all_notes[key] for key in key_order], axis=1)

notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
notes_ds.element_spec

def create_sequences(
    dataset: tf.data.Dataset, 
    seq_length: int,
    vocab_size = 128,
) -> tf.data.Dataset:
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length+1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Normalize note pitch
  def scale_pitch(x):
    x = x/[vocab_size,1.0,1.0]
    return x

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

    return scale_pitch(inputs), labels

  return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


seq_length = [32, 32, 32, 32, 32, 32]
batch_size = [4, 4, 4, 4, 4, 4]
learning_rate = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
neurons = [32, 64, 128, 254, 512, 1024]

for idx,parameterset in enumerate(list(itertools.product(seq_length, batch_size,learning_rate,neurons))):
    config.train_pram["seq_length"] = parameterset[0]
    config.train_pram["batch_size"] = parameterset[1]
    config.train_pram["learning_rate"] = parameterset[2]
    config.train_pram["neurons"] = parameterset[3]

for idx in range(len(config.grid_params["seq_length"])):

    seq_length = config.grid_params["seq_length"][idx]   # 50
    vocab_size = 128
    seq_ds = create_sequences(notes_ds, seq_length, vocab_size)
    seq_ds.element_spec

    # for seq, target in seq_ds.take(1):
    #     print('sequence shape:', seq.shape)
    #     print('sequence elements (first 10):', seq[0: 10])
    #     print()
    #     print('target:', target)

    batch_size = config.grid_params["batch_size"][idx]      # 64
    buffer_size = n_notes - seq_length  # the number of items in the dataset
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    train_ds.element_spec

    def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
        mse = (y_true - y_pred) ** 2
        positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
        return tf.reduce_mean(mse + positive_pressure)


    input_shape = (seq_length, 3)

    learning_rate = config.grid_params["learning_rate"][idx]    # 0.004

    inputs = tf.keras.Input(input_shape)

    x = tf.keras.layers.LSTM(config.grid_params["neurons"][idx], return_sequences=True, stateful=False)(inputs)   # 128

    for neurons in config.grid_params["layers"][idx]:
        x = tf.keras.layers.LSTM(neurons, return_sequences=True)(x)
 
    x = tf.keras.layers.LSTM(128)(x)    #8



    outputs = {
    'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs, name="music_model_" + str(idx))

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss=loss,
        loss_weights={
            'pitch': 0.05,
            'step': 1.0,
            'duration':1.0,
        },
        optimizer=optimizer,
    )

    model.compile(loss=loss, optimizer=optimizer)

    model.summary()

    # losses = model.evaluate(train_ds, return_dict=True)
    # print(losses)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=100,
            verbose=1,
            restore_best_weights=True),
    ]

    try:
        history = model.fit(
            train_ds,
            epochs=settings.training.epochs,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        pass

    # Plot the train loss
    # plt.plot(history.epoch, history.history['loss'], label='total loss')
    # plt.show()

    def predict_next_note(
        notes: np.ndarray, 
        keras_model: tf.keras.Model, 
        temperature: float = 1.0) -> int:
        """Generates a note IDs using a trained sequence model."""

        assert temperature > 0

        # Add batch dimension
        inputs = tf.expand_dims(notes, 0)

        predictions = model.predict(inputs)
        pitch_logits = predictions['pitch']
        step = predictions['step']
        duration = predictions['duration']

        pitch_logits /= temperature
        pitch = tf.random.categorical(pitch_logits, num_samples=1)
        pitch = tf.squeeze(pitch, axis=-1)
        duration = tf.squeeze(duration, axis=-1)
        step = tf.squeeze(step, axis=-1)

        # `step` and `duration` values should be non-negative
        step = tf.maximum(0, step)
        duration = tf.maximum(0, duration)

        return int(pitch), float(step), float(duration)

    # temperature = 1.5
    # num_predictions = 2*seq_length
    temperature = settings.composition.temperature
    num_predictions = settings.composition.num_predictions

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    out_file = 'raw_notes.mid'
    out_pm = notes_to_midi(raw_notes, out_file=out_file, instrument_name=instrument_name)

    # The initial sequence of notes; pitch is normalized similar to training
    # sequences
    input_notes = (sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    out_file_input_notes = 'sequence_notes.mid'
    input_pm = notes_to_midi(raw_notes[:seq_length], out_file=out_file_input_notes, instrument_name=instrument_name)

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(generated_notes, columns=(*key_order, 'start', 'end'))


    out_file = 'output.mid'
    out_pm = notes_to_midi(generated_notes, out_file=out_file, instrument_name=instrument_name)

    # plot_piano_roll(generated_notes)
    # plot_distributions(generated_notes)


    # "Checkpoints_" + str(idx)
    config.save_training("Checkpoints/" + datetime.datetime.now().strftime("%d_%m_%y_%H-%M"), input_pm, out_pm, model, history, generated_notes, raw_notes)