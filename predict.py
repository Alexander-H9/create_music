import glob
import pathlib
import tensorflow as tf
import numpy as np
import pandas as pd

from midi_processing import midi_to_notes, notes_to_midi

def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
	mse = (y_true - y_pred) ** 2
	positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
	return tf.reduce_mean(mse + positive_pressure)

loss = {
      'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True),
      'step': mse_with_positive_pressure,
      'duration': mse_with_positive_pressure,
}

data_dir = pathlib.Path('data/maestro-v2.0.0')
filenames = glob.glob(str(data_dir/'2017/MIDI-Unprocessed_082_PIANO082_MID--AUDIO-split_07-09-17_Piano-e_2_-04_wav--1.midi'))
filenames = glob.glob(str(data_dir/"2015/MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_20_R1_2015_wav--1.midi"))

data_dir = pathlib.Path('data/maestro-v2.0.0/alex')
filenames = glob.glob(str(data_dir/'the_begining.midi'))

filenames_alex = glob.glob(str(data_dir/'*.midi'))

model = tf.keras.models.load_model('Saved_Checkpopints\\Checkpoints_1\\model', custom_objects={"mse_with_positive_pressure": mse_with_positive_pressure})
key_order = ['pitch', 'step', 'duration']
sample_file = filenames[0]

for idx,sample_file in enumerate(filenames_alex):

	raw_notes = midi_to_notes(sample_file)
	seq_length = 50
	vocab_size = 128
	instrument_name="Acoustic Grand Piano"

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

	temperature = 1
	num_predictions = 200

	sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

	# The initial sequence of notes; pitch is normalized similar to training
	# sequences
	input_notes = (sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

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


	# save the input file
	out_file_sequence = str(idx) + '_input_prediction.mid'
	out_pm = notes_to_midi(raw_notes[:seq_length], out_file=out_file_sequence, instrument_name=instrument_name, save=True)

	out_file = str(idx) + '_output_prediction.mid'
	out_pm = notes_to_midi(generated_notes, out_file=out_file, instrument_name=instrument_name, save=True)