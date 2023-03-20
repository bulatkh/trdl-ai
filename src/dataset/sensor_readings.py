from abc import abstractmethod
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.spatial.distance import euclidean
from tslearn.metrics import dtw_path


class SensorReadings:
    def __init__(self, session_id, data_path, activity=None, phone_position=None, model=None, avg_duplicate_ts=False, use_sensors=['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']):
        self.session_id = session_id
        self.data_path = data_path
        self.activity = activity
        self.phone_position = phone_position
        self.model = model
        self.avg_duplicate_ts = avg_duplicate_ts
        self.device_data = self.read_data(use_sensors)
    
    def read_data(self, use_sensors=['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER']):
        data = pd.read_csv(self.data_path)
        self.sensors = []
        all_sensors = list(sorted(data.sensor.unique()))
        for s in all_sensors:
            if s in use_sensors:
                self.sensors.append(s)
        device_data = {}
        for sensor in self.sensors:
            device_data[sensor] = {}
            sensor_data_ts = data[data['sensor'] == sensor].reset_index(drop=True)
            if self.avg_duplicate_ts:
                sensor_data_ts = sensor_data_ts.groupby('timestamp').mean().reset_index()
                sensor_data = sensor_data_ts.drop(['timestamp'], axis=1)
            else:
                sensor_data = sensor_data_ts.drop(['sensor', 'timestamp'], axis=1)
            ts = sensor_data_ts['timestamp']
            device_data[sensor]['ts'] = ts
            device_data[sensor]['dt'] = ts.apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.0))
            device_data[sensor]['data'] = sensor_data
        return device_data
    
    def get_ts_summary(self):
        summary = {}
        for sensor in self.sensors:
            summary[sensor] = {}
            num_ts = len(self.device_data[sensor]['ts'])
            eff_dict = mean_effective_frequency(self.device_data[sensor]['ts'])
            summary[sensor]['Repeating timestamps (%)'] = \
                round(100 * eff_dict['Repeating timestamps'] / num_ts, 2)
            summary[sensor]['Overall timestamps'] = num_ts
            summary[sensor].update(eff_dict)
        return summary
    
    def get_random_ts_vis(self, length_sec=5, size=3):
        if len(self.device_data['ACCELEROMETER']['ts']) <= 5000:
            return
        random_start_frames = sorted(np.random.choice(len(self.device_data['ACCELEROMETER']['ts']) - 5000, size=size, replace=False))
        fig, ax = plt.subplots(len(self.sensors), size, figsize=(25, 10) if len(self.sensors) == 3 else (25,6))
        for k, sensor in enumerate(self.sensors):
            eff_freq = mean_effective_frequency(self.device_data[sensor]['ts'])['Mean effective frequency']
            num_frames = int(length_sec * eff_freq)      
            for i, s in enumerate(random_start_frames):
                frame = self.device_data[sensor]['data'][s: s+num_frames]
                if len(self.sensors) == 1:
                    ax[i].plot(frame)
                else:
                    ax[k, i].plot(frame)
#                 plt.legend()
                    ax[k, i].set_title(f'{sensor}')
        fig.suptitle(f'{self.activity} - {self.phone_position} - {self.model}', fontsize=16)
        plt.show()
        
    def draw_delays(self):
        timestamps = {}
        if len(self.sensors) > 1:
            fig, ax = plt.subplots(1, 3 if len(self.sensors) == 3 else 1, figsize=(25,3))
            for sensor in self.sensors:
                timestamps[sensor] = self.device_data[sensor]['ts']
            ts_df = pd.DataFrame.from_dict(timestamps)
            
            delay_acc_gyro = ts_df[self.sensors[1]] - ts_df[self.sensors[0]]
            if len(self.sensors) == 2:
                ax.plot(delay_acc_gyro)
            if len(self.sensors) == 3:
                ax[0].plot(delay_acc_gyro)
                delay_acc_mg = ts_df[self.sensors[2]] - ts_df[self.sensors[0]]
                ax[1].plot(delay_acc_mg)
                delay_gyro_mg = ts_df[self.sensors[2]] - ts_df[self.sensors[1]]
                ax[2].plot(delay_gyro_mg)
            plt.show()
        else:
            print('Only one sensor in the recording.')
        return timestamps
    
    def resample(self, desired_freq=33.3):
        for i, s in enumerate(self.sensors):
            ts = np.array(self.device_data[s]['ts'])
            len_sec = np.floor((ts[-1] - ts[0]) / 1000)
            if i == 0:
                num_ts = int(len_sec * desired_freq)
            self.device_data[s]['data'] = resample(self.device_data[s]['data'], num_ts)  

    def align(self, method='fast-dtw'):
        for i in range(len(self.sensors) - 1):
            # create alignment
            if method == 'fast-dtw':
                path ,_ = dtw_path(self.device_data[self.sensors[i]]['ts'], self.device_data[self.sensors[i + 1]]['ts'])
            else:
                raise ValueError('Provide the available alighning algorithm')
            # get first sensor data and ts
            aligned_s1_ts = self.device_data[self.sensors[i]]['ts'][[x[0] for x in path]].reset_index(drop=True)
            aligned_s2_ts = self.device_data[self.sensors[i + 1]]['ts'][[x[1] for x in path]].reset_index(drop=True)
            # get second sensor data and ts
            aligned_s1_data = self.device_data[self.sensors[i]]['data'].iloc[[x[0] for x in path]].reset_index(drop=True)
            aligned_s2_data = self.device_data[self.sensors[i + 1]]['data'].iloc[[x[1] for x in path]].reset_index(drop=True)
            # change data to aligned version
            self.device_data[self.sensors[i]]['ts'] = aligned_s1_ts
            self.device_data[self.sensors[i]]['data'] = aligned_s1_data
            self.device_data[self.sensors[i + 1]]['ts'] = aligned_s2_ts
            self.device_data[self.sensors[i + 1]]['data'] = aligned_s2_data
        if len(self.sensors) == 3:
            # get first sensor data and ts
            aligned_s0_ts = self.device_data[self.sensors[0]]['ts'][[x[0] for x in path]].reset_index(drop=True)
            aligned_s0_data = self.device_data[self.sensors[0]]['data'].iloc[[x[0] for x in path]].reset_index(drop=True)
            self.device_data[self.sensors[0]]['ts'] = aligned_s0_ts
            self.device_data[self.sensors[0]]['data'] = aligned_s0_data
    
    def stack(self):
        data = []
        for s in self.sensors:
            data.append(self.device_data[s]['data'].copy())
        return np.concatenate(data, axis=1)
    
    @abstractmethod
    def sample(self, stacked, len_ts=5, desired_freq=33.3):
        frame_size = int(len_ts * desired_freq)
        sampled = []
        for i in range(frame_size, stacked.shape[0] - frame_size, frame_size):
            sampled.append(stacked[i: i + frame_size])
        return sampled
            

    

def mean_effective_frequency(ts):
    intervals = np.array(ts)[1:] - np.array(ts)[:-1]
    mean_upd = intervals.mean()
    return {
        'Repeating timestamps': (intervals == 0).sum(),
        'Mean effective frequency': round(1000 / mean_upd, 3),
        'Mean update interval': round(mean_upd, 3)
    }
