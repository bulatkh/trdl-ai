import torch
import torch.nn as nn

from src.dataset.sensor_readings import SensorReadings
from src.models.conv_ae import Autoencoder
from src.utils.utils import load_yaml_to_dict


# TODO: check if we need to use enum from main.py
SENSORS = {
    'ACCELEROMETER': 0,
    'GYROSCOPE': 1,
    'MAGNETOMETER': 2
}


def init_autoencoder_model(config, model_path):
    exp_configs = config['experiment']
    # TODO: Add try except case for correct model load
    model = Autoencoder(
        in_channels=len(exp_configs['sensors'] * 3),
        max_len=int(exp_configs['desired_freq'] * exp_configs['len_ts']), 
        **config['model']['cae']['kwargs']
        )
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def preprocess(config, input_chunk):
    align = config['experiment']['align']
    desired_freq = config['experiment']['desired_freq']
    len_ts = config['experiment']['len_ts']
    use_sensors =  [SENSORS[s] for s in config['experiment']['sensors']]
    norm = config['experiment']['norm_apply']

    # TODO: design a separate efficient class to process chunks (currently, part of SensorReadings)
    # curr_sr = SensorChunk(input_chunk)
    curr_sr = SensorReadings(data=input_chunk, use_sensors=use_sensors, chunk=True)

    if align:
        curr_sr.align()
    curr_sr.resample(desired_freq=desired_freq)
    frame = curr_sr.stack()
    if norm:
        frame = (frame - frame.mean(axis=0)) / frame.std(axis=0)
    return torch.Tensor(frame).unsqueeze(dim=0).permute(0, 2, 1)


def get_reconstruction_loss(autoencoder, data):
    output = autoencoder(data)
    assert data.shape == output.shape
    return nn.MSELoss()(data, output).mean()


def make_prediction(config_path, input_chunk, model_path, threshold_path):
    config = load_yaml_to_dict(config_path)
    data = preprocess(config, input_chunk)
    cae = init_autoencoder_model(config, model_path)
    threshold = load_yaml_to_dict(threshold_path)['threshold']
    loss = get_reconstruction_loss(cae, data).detach().numpy()
    return loss < threshold


def read_test_chunk(txt_path):
    # TODO: move to the tests and refactor
    import json
    with open(txt_path) as f:
        chunk = json.load(f)
    return chunk


def test(txt_path, config_path, model_path, threshold_path):
    # TODO: move to the tests and separate into multiple tests for each function
    chunk = read_test_chunk(txt_path)
    # TODO: check that yml reads correctly and has the needed fields from preprocess and init_autoencoder_model
    config = load_yaml_to_dict(config_path)
    # TODO: check that model is correctly read
    model = init_autoencoder_model(config, model_path)
    # TODO: check that data is pre-processed correctly, shape is [1, num_of_used_sensors, expected_frequency * len_sec]
    input_data = preprocess(config, chunk)
    # TODO check that loss is one number greater than 0
    loss = get_reconstruction_loss(model, input_data)
    # Make prediction uses all functions above
    # TODO: this function can be tested on multiple inputs to see if it returns True or False
    print(make_prediction(config_path, chunk, model_path, threshold_path))


if __name__ == '__main__':
    # TODO: move to the tests and refactor
    import sys
    txt_path = sys.argv[1]
    config_path = sys.argv[2]
    model_path = sys.argv[3]
    threshold_path = sys.argv[4]
    test(txt_path, config_path, model_path, threshold_path)
