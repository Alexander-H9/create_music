import json
import os
from midi_processing import notes_to_midi, plot_distributions, plot_piano_roll, plot_loss
from matplotlib import pyplot as plt
import pandas as pd


class Config:

    def __init__(self):
        self.grid_params = {
        "seq_length":[500], #15, 25, 50, 100, 
        "batch_size":[128], #32, 64, 128, 128, 
        "learning_rate":[ 0.006], #0.002, 0.002, 0.004, 0.006,
        "neurons":[ 512], #64, 128, 254, 512,
        "composer":["Wolfgang Amadeus Mozart", "Sergei Rachmaninoff", "Ludwig van Beethoven", "Johann Sebastian Bach", "Johannes Brahms"]
        }


    def save_training(self, path, pm_input=None, pm_output=None, model=None, history=None, generated_notes=None, input_notes=None):

        # Make a directory
        os.makedirs(path, exist_ok=True)
        json_object = json.dumps(self.grid_params, indent = 4)

        # Save the training parameter set
        with open(path+"/config.json", "w") as outfile:
            outfile.write(json_object)

        # Save the trained model
        if model!=None: model.save(path+'/model')

        if history != None:
            plot_loss(history, path)

        # Save the midi input sequence and the prediction
        if pm_input!=None and pm_output!=None:
            pm_input.write(path+"/input.midi")
            pm_output.write(path+"/output.midi")

        # Save the midi plots for the input sequence and the prediction
        if type(input_notes) == pd.DataFrame and type(generated_notes) == pd.DataFrame:
            plot_distributions(input_notes).savefig(path+"/distributions_input.png")
            plot_distributions(generated_notes).savefig(path+"/distributions_output.png")
            plot_piano_roll(input_notes).savefig(path+"/piano_roll_input.png")
            plot_piano_roll(generated_notes).savefig(path+"/piano_roll_output.png")

    def __str__(self):
        return str(self.grid_params)