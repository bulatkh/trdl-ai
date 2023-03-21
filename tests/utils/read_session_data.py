import os
import pandas as pd

SENSOR_VALUES = {
    "ACCELEROMETER": 0,
    "GYROSCOPE": 1,
    "MAGNETOMETER": 2
}


def get_parent_dir(yourpath: str):
    return os.path.abspath(os.path.join(yourpath, os.pardir))

def sensors_readings_map_function(sensor_reading: list):
    timestamp, sensor, *readings = sensor_reading
    return [timestamp, SENSOR_VALUES[sensor], *readings]


def read_session_data(file_name: str):
    project_dir = get_parent_dir(os.path.dirname(__file__))
    data = pd.read_csv(project_dir + "/assets/sample_session/" + file_name)
    data_list = data.values.tolist()
    return list(map(sensors_readings_map_function, data_list))


