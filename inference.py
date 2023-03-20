import torch
import torch.nn as nn

from src.dataset.sensor_readings import SensorReadings
from src.models.conv_ae import Autoencoder
from src.utils.utils import load_yaml_to_dict


def init_autoencoder_model(config, model_path):
    exp_configs = config['experiments']
    model = Autoencoder(
        in_channels=len(exp_configs['sensors'] * 3),
        max_len=int(exp_configs['desired_freq'] * exp_configs['len_ts']), 
        **config['model']['cae']['kwargs']
        )
    model.load_state_dict(torch.load(model_path))
    return model


def preprocess(config, input_file):
    align = config['experiment']['align']
    desired_freq = config['experiment']['desired_freq']
    len_ts = config['experiment']['len_ts']
    use_sensors = config['experiment']['sensors']
    norm = config['experiment']['norm_apply']

    # curr_sr = SensorReadings(row['session_id'], row['data'], row['activity'], row['position'], row['model'], use_sensors=use_sensors)
    # curr_sr = SensorChunk(input_file)

    if align:
        curr_sr.align()
    curr_sr.resample(desired_freq=desired_freq)
    frame = curr_sr.stack()
    if norm:
        frame = (frame - frame.mean(axis=0)) / frame.std(axis=0)
    return torch.Tensor(frame).unsqueeze()


def get_reconstruction_loss(autoencoder, data):
    output = autoencoder(data)
    assert data.shape == output.shape
    return nn.MSELoss(data, autoencoder(data))


def make_prediction(config, input_file, model_path, threshold_path):
    data = preprocess(config, input_file)
    cae = init_autoencoder_model(config, model_path)
    threshold = load_yaml_to_dict(threshold_path)['threshold']
    loss = get_reconstruction_loss(cae, data).detach().numpy()
    return int(loss < threshold)


