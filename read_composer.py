import json
import pathlib

def get_composer_midi(composer="Ludwig van Beethoven"):

    data_dir = pathlib.Path('data/maestro-v2.0.0/')

    json_file = pathlib.Path('data/maestro-v2.0.0/maestro-v2.0.0.json')

    midi_files = []

    with open(json_file, 'r') as f:
        data = json.load(f)
        for element in data:
            if element["canonical_composer"] == composer:
                midi_files.append('data/maestro-v2.0.0/' + element["midi_filename"])

        return midi_files