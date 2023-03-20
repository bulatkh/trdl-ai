import json
import os
from abc import abstractmethod

import dateutil
import numpy as np
import pandas as pd


class SessionDataManager:
    def __init__(self, data_path):
        self.data_path = data_path
        self.session_dict = self._parse_data()
        
    def _parse_data(self):
        files = os.listdir(self.data_path)
        sessions = {}
        for file_ in files:
            user = file_.split('-')[0]
            session = file_.split('-')[1]
            if session not in sessions:
                sessions[session] = {'user': user}
            if 'sensor-readings' in file_:
                sessions[session]['data'] = os.path.join(self.data_path, file_)
                sensors = pd.read_csv(os.path.join(self.data_path, file_))['sensor']
                devices = sensors.unique()
                num_devices = len(devices)
                sessions[session]['devices'] = devices
                sessions[session]['num_devices'] = num_devices
            if 'session' in file_:
                metadata = SessionMetadata(os.path.join(self.data_path, file_))
                start, end, duration, duration_sec = metadata.get_dt_duration_sec()
                readings = metadata.get_total_sensor_readings()
                sessions[session].update({
                    'session_id': metadata.get_sessionId(),
                    'activity': metadata.get_activity(),
                    'position': metadata.get_position(),
                    'os': metadata.get_operating_system(),
                    'model': metadata.get_model(),
                    'duration_dt': duration,
                    'duration_sec': duration_sec,
                    'readings': readings,
                    'expected_freq': (1000 / metadata.get_update_interval()),
                    'metadata': os.path.join(self.data_path, file_),
                    'start_dt': start,
                    'end_dt': end,
                })
            if 'readings' in sessions[session] and 'num_devices' in sessions[session]:
                sessions[session]['approx_freq'] = round(sessions[session]['readings'] / sessions[session]['num_devices'] / duration_sec, 2) if readings != 0 else np.nan
        return sessions
    
    def get_summary(self, drop_extra=False):
        df = pd.DataFrame.from_dict(self.session_dict).T.reset_index(drop=True)
        if drop_extra:
            df = df.drop(['data', 'metadata', 'start_dt', 'end_dt'], axis=1)
        return df
    
                
class SessionMetadata:
    def __init__(self, metadata_path):
        self.path = metadata_path
        self.meta = self._read_meta()

    def _read_meta(self):
        with open(self.path, 'r') as f:
            dict_ = json.load(f)
        return dict_
    
    def get_sessionId(self):
        return self.meta['sessionId']

    def get_position(self):
        return self.meta['sessionMetadata']['smartphonePosition']
    
    def get_activity(self):
        return self.meta['sessionMetadata']['activity']
    
    def get_operating_system(self):
        return self.meta['sessionMetadata']['operatingSystem']
    
    def get_model(self):
        return self.meta['sessionMetadata']['smartphoneModel']
    
    def get_total_sensor_readings(self):
        return self.meta['totalSensorReadings']
    
    def get_update_interval(self):
        return self.meta['sessionMetadata']['sensorUpdateInterval']
    
    def get_dt_duration_sec(self):
        start = self.meta['createdAt']
        end = self.meta['destroyedAt']
        duration, duration_sec = SessionMetadata.get_duration(start, end)
        return start, end, duration, duration_sec
    
    @abstractmethod
    def get_duration(start, end):
        start_dt = dateutil.parser.isoparse(start)
        end_dt = dateutil.parser.isoparse(end)
        duration = end_dt - start_dt
        duration_sec = duration.total_seconds()
        return duration, duration_sec
