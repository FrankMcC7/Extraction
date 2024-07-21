pip install allennlp allennlp-models torch transformers
from allennlp.modules.elmo import Elmo, batch_to_ids
import os

# Define the options and weights files
options_file = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_options.json'
weight_file = 'https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_weights.hdf5'

# Specify a directory to save the model
model_dir = 'elmo_model'
os.makedirs(model_dir, exist_ok=True)

# Download the files
import requests
options_response = requests.get(options_file)
with open(os.path.join(model_dir, 'options.json'), 'wb') as f:
    f.write(options_response.content)

weights_response = requests.get(weight_file)
with open(os.path.join(model_dir, 'weights.hdf5'), 'wb') as f:
    f.write(weights_response.content)
