import os
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from dataset.session_data_manager import SessionDataManager
from dataset.sensor_readings import SensorReadings


class SensorDataset(Dataset):
    def __init__(
            self, 
            data_path, 
            subjects=None, 
            activities=None, 
            transforms=None, 
            use_sensors=['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER'], 
            align=True, 
            desired_freq=33.3,
            norm_apply=True,
            act_to_lab=None,
            len_ts=5) -> None:
        super().__init__()
        self.data_path = data_path

        self.subjects = subjects
        self.activities = activities

        self.align = align
        self.use_sensors = use_sensors
        self.desired_freq = desired_freq
        self.len_ts = len_ts
        self.transforms = transforms

        self.norm_apply = norm_apply

        self.act_to_lab = act_to_lab
        
   
        self.summary = self._filter_dataset()
        self.data, self.labels = self._process_files()
        if self.norm_apply:
            self.data = self._norm_apply()
    
    def _filter_dataset(self):
        full_summary = SessionDataManager(self.data_path).get_summary()
        filtered_sub = full_summary.copy()
        # filter subjects
        if self.subjects is not None:
            filtered_sub = filtered_sub[filtered_sub['user'].isin(self.subjects)]
            filtered = filtered_sub.copy()
        # filter activity - position combinations  
        rows_filtered = []
        for act in self.activities.keys():
            if self.activities[act] == 'all':
                rows_filtered.append(filtered_sub[filtered_sub['activity'] == act])
            elif self.activities[act] is not None:
                for pos in self.activities[act]:
                    rows_filtered.append(filtered_sub[(filtered_sub['activity'] == act) & (filtered_sub['position'] == pos)])
        filtered = pd.concat(rows_filtered)
        return filtered

    def _process_files(self):
        all_windows = []
        labels = []
        print('Reading data...')
        for i, row in tqdm(self.summary.iterrows()):
            print(row['session_id'], row['user'], row['position'], row['activity'])
            curr_sr = SensorReadings(row['session_id'], row['data'], row['activity'], row['position'], row['model'], use_sensors=self.use_sensors)
            if self.align:
                curr_sr.align()
            curr_sr.resample(desired_freq=self.desired_freq)
            stacked = curr_sr.stack()
            curr_sampled = curr_sr.sample(stacked, len_ts=self.len_ts, desired_freq=self.desired_freq)
            all_windows.extend(curr_sampled)
            if self.act_to_lab is not None:
                label = self.act_to_lab[row['activity']]
                labels.extend([label for _ in range(len(curr_sampled))])
        return all_windows, labels
    
    def _norm_apply(self):
        norm_applied = []

        print('Normalizing data...')    
        for frame in self.data:
            norm_param = (frame.mean(axis=0), frame.std(axis=0))
            norm_frame = (frame - norm_param[0]) / norm_param[1]
            norm_applied.append(norm_frame)
        return norm_applied

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        

class SensorDataModule(LightningDataModule):
    def __init__(self,
            data_path,
            batch_size,
            train_transforms = {},
            test_transforms = {},
            subjects=None, 
            activities=None,  
            use_sensors=['ACCELEROMETER', 'GYROSCOPE', 'MAGNETOMETER'], 
            align=True, 
            desired_freq=33.3, 
            len_ts=5,
            norm_apply=True,
            use_validation=False,
            act_to_lab=None,
            num_workers = 1):
        super().__init__()
        # paths
        self.data_path = data_path
        # batch and transforms
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        # experiment related
        self.train_subjects, self.val_subjects, self.test_subjects = subjects
        self.train_activities, self.val_activities, self.test_activities = activities
        self.use_sensors = use_sensors
        self.align = align
        self.desired_freq = desired_freq
        self.len_ts = len_ts

        self.norm_apply = norm_apply

        self.act_to_lab = act_to_lab

        self.use_validation = use_validation
        # other
        self.num_workers = num_workers

        self._init_dataloaders()
        self.save_hyperparameters("batch_size")

    def _init_dataloaders(self):
        train_dataset = self._create_train_dataset()
        test_dataset = self._create_test_dataset()
        self.train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)

        if self.use_validation:
            val_dataset = self._create_val_dataset()
            self.val = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
        else:
            self.val = None

    def _create_train_dataset(self):
        return SensorDataset(
            data_path=self.data_path,
            transforms=self.train_transforms,
            subjects=self.train_subjects, 
            activities=self.train_activities, 
            use_sensors=self.use_sensors, 
            align=self.align, 
            desired_freq=self.desired_freq, 
            len_ts=self.len_ts,
            norm_apply=self.norm_apply,
            act_to_lab=self.act_to_lab
        )
        
    def _create_val_dataset(self):
        return SensorDataset(
            data_path=self.data_path,
            transforms=self.test_transforms,
            subjects=self.val_subjects, 
            activities=self.val_activities, 
            use_sensors=self.use_sensors, 
            align=self.align, 
            desired_freq=self.desired_freq, 
            len_ts=self.len_ts,
            norm_apply=self.norm_apply,
            act_to_lab=self.act_to_lab
        )

    def _create_test_dataset(self):
        return SensorDataset(
            data_path=self.data_path,
            transforms=self.test_transforms,
            subjects=self.test_subjects, 
            activities=self.test_activities, 
            use_sensors=self.use_sensors, 
            align=self.align, 
            desired_freq=self.desired_freq, 
            len_ts=self.len_ts,
            norm_apply=self.norm_apply,
            act_to_lab=self.act_to_lab
        )

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test