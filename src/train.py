import argparse
import os
from email.policy import default

import numpy as np
import seaborn as sns
import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
# from utils.augmentations import init_transforms
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
# from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision import transforms

from callbacks.recontruction_vis import ReconstructionVis
from models.conv_ae import Autoencoder
from sensor_datamodule import SensorDataModule
from utils.utils import load_yaml_to_dict

ACT_TO_LABEL = {
    'CYCLING': 0,
    'RUNNING': 1,
    'WALKING': 2,
    'SCOOTERING': 3,
    'DRIVING': 4
}

ACT_TO_LABEL_OUTLIER = {
    'CYCLING': 1,
    'RUNNING': -1,
    'WALKING': -1,
    'SCOOTERING': -1,
    'DRIVING': -1
}

LABEL_TO_ACT = {
    0: 'CYCLING',
    1: 'RUNNING',
    2: 'WALKING',
    3: 'SCOOTERING',
    4: 'DRIVING'
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    # paths
    parser.add_argument('--data_path', required=True, help='path to sampled data')
    parser.add_argument('--experiment_config', required=True, help='path to experiment config')
    parser.add_argument('--model_weights_path', required=True, help='path for model parameters to be saved')

    # model
    parser.add_argument('--model', default='cae')
    parser.add_argument('--export_onnx', action='store_true', default=False)
    parser.add_argument('--as_ptl', action='store_true', default=False)

    # misc
    parser.add_argument('--num_workers', default=1, type=int)

    return parser.parse_args()


def train_test_autoencoder(args, config):
    exp_configs = config['experiment']
    batch_size = exp_configs['batch_size']
    num_epochs = exp_configs['num_epochs']
    

    train_transforms = None
    test_transforms = None
    
    train_subjects = exp_configs['train_subjects']
    val_subjects = exp_configs['val_subjects']
    test_subjects = exp_configs['test_subjects']

    train_activities = exp_configs['train_activities']
    val_activities = exp_configs['val_activities']
    test_activities = exp_configs['test_activities']
    
    print(train_activities)

    subjects = (train_subjects, val_subjects, test_subjects)
    activities = (train_activities, val_activities, test_activities)
    
    cae = Autoencoder(
        in_channels=len(exp_configs['sensors'] * 3), 
        max_len=int(exp_configs['desired_freq'] * exp_configs['len_ts']), 
        **config['model'][args.model]['kwargs'])

    datamodule = SensorDataModule(
        data_path=args.data_path,
        batch_size=batch_size,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        subjects=subjects,
        activities=activities,
        use_sensors=exp_configs['sensors'],
        align=exp_configs['align'],
        desired_freq=exp_configs['desired_freq'],
        len_ts=exp_configs['len_ts'],
        norm_apply=exp_configs['norm_apply'],
        use_validation=exp_configs['use_validation'],
        act_to_lab=ACT_TO_LABEL,
        num_workers=args.num_workers
    )

    recvis_callback = ReconstructionVis(id_to_act=LABEL_TO_ACT)
    checkpoint_callback = ModelCheckpoint(dirpath=args.model_weights_path, save_top_k=1, monitor="train_loss", mode='min')

    trainer = Trainer.from_argparse_args(args=args, gpus=1, deterministic=True, max_epochs=num_epochs, default_root_dir='train_logs', callbacks=[recvis_callback, checkpoint_callback])
    # trainer = Trainer.from_argparse_args(args=args, gpus=1, deterministic=True, max_epochs=num_epochs, default_root_dir='logs', callbacks=[checkpoint_callback])

    trainer.fit(cae, datamodule)
    trainer.test(cae, datamodule)

    return cae


def extract_features(args, config, cae):
    exp_configs = config['experiment']
    batch_size = exp_configs['batch_size']

    train_subjects = exp_configs['train_subjects']
    val_subjects = exp_configs['val_subjects']
    test_subjects = exp_configs['test_subjects']

    train_activities = exp_configs['train_activities']
    val_activities = exp_configs['val_activities']
    test_activities = exp_configs['test_activities']

    subjects = (train_subjects, val_subjects, test_subjects)
    activities = (train_activities, val_activities, test_activities)

    datamodule = SensorDataModule(
        data_path=args.data_path,
        batch_size=batch_size,
        train_transforms=None,
        test_transforms=None,
        subjects=subjects,
        activities=activities,
        use_sensors=exp_configs['sensors'],
        align=exp_configs['align'],
        desired_freq=exp_configs['desired_freq'],
        len_ts=exp_configs['len_ts'],
        norm_apply=exp_configs['norm_apply'],
        use_validation=exp_configs['use_validation'],
        act_to_lab=ACT_TO_LABEL,
        num_workers=args.num_workers
    )

    if exp_configs['prediction_type'] != 'reconstruction_error':
        extract_encoder_part(cae)

    train_inp, train_feat, train_label = generate_outputs(cae, datamodule.train_dataloader())
    test_inp, test_feat, test_label = generate_outputs(cae, datamodule.test_dataloader())
    return train_inp, test_inp, train_feat, test_feat, train_label, test_label


def generate_outputs(model, dataloader):
    inputs = []
    outputs = []
    labels = []
    for batch_idx, batch in enumerate(dataloader):
        x, out, y = model.predict_step(batch, batch_idx)
        inputs.append(x)
        outputs.append(out)
        labels.extend(list(y))
    inputs = torch.vstack(inputs)
    outputs = torch.vstack(outputs)
    return inputs, outputs, labels


def train_test_one_class_svm(args, config, X_train, X_test, y_train, y_test):
    clf = svm.OneClassSVM(kernel="rbf", gamma='scale', nu=0.6)
    clf.fit(X_train)

    y_train_binary = [1 if y == 0 else -1 for y in y_train]
    y_test_binary = [1 if y == 0 else -1 for y in y_test]

    # tsne = TSNE(n_components=2, verbose=1, random_state=123)
    tsne_vis(X_train, y_train_binary, './results/tsne_train.png')
    tsne_vis(X_test, y_test_binary, './results/tsne_test.png')

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print('Train classification report:')
    print(classification_report(y_train_binary, y_pred_train, target_names=['Non-CYCLING', 'CYCLING']))
    print('Test classification report:')
    print(classification_report(y_test_binary, y_pred_test, target_names=['Non-CYCLING', 'CYCLING']))


def extract_encoder_part(cae_model):
    cae_model.bottleneck.linear2 = torch.nn.Identity()
    cae_model.decoder = torch.nn.Identity()


def tsne_vis(X, y_binary, save_path):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(X)

    plt.clf() 
    df = pd.DataFrame()
    df["y"] = y_binary
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 10),
                data=df).set(title="T-SNE projection")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    plt.clf() 


def reconstruction_thresholding(X_test, y_test, X_test_rec, threshold=0.5):
    pairwise_loss = nn.MSELoss(reduce=False)(X_test, X_test_rec).view(X_test.shape[0], -1).mean(dim=1)
    y_pred = (pairwise_loss < threshold).double().detach().numpy()
    y_pred_binary = [1 if y == 1 else 0 for y in y_pred]
    y_test_binary = [1 if y == 0 else 0 for y in y_test]
    print(classification_report(y_test_binary, y_pred_binary, target_names=['Non-CYCLING', 'CYCLING']))
    print()


def main():
    seed_everything(28)
    args = parse_arguments()
    config = load_yaml_to_dict(args.experiment_config)
    cae = train_test_autoencoder(args, config)
    cae.eval()
    train_inp, test_inp, train_feat, test_feat, train_label, test_label = extract_features(args, config, cae)
    if config['experiment']['prediction_type'] == 'reconstruction_error':
        reconstruction_thresholding(test_inp, test_label, test_feat)
    elif config['experiment']['prediction_type'] == 'oc-svm':
        train_test_one_class_svm(args, config, train_feat.detach().numpy(), test_feat.detach().numpy(), train_label, test_label)
    

if __name__ == '__main__':
    main()

    # if args.as_ptl:  
    #     scripted_model = torch.jit.script(cae)
    #     optimized_model = optimize_for_mobile(scripted_model)
    #     optimized_model._save_for_lite_interpreter("model_weights.ptl")
    # if args.export_onnx:
    #     cae.eval()
    #     data_placeholder = torch.randn(1, config['model'][args.model]['kwargs']['in_channels'], config['model'][args.model]['kwargs']['max_len'], requires_grad=True)
    #     output = cae(data_placeholder)
    #     print(output.shape)
    #     torch.onnx.export(
    #         cae,
    #         data_placeholder,
    #         os.path.join(args.model_weights_path, f'{args.model}_model_19ch.onnx'),
    #         export_params=True,
    #         opset_version=10,
    #         do_constant_folding=True,
    #         input_names=['input'],
    #         output_names=['output'],
    #         dynamic_axes={'input' : {0 : 'batch_size'},  'output' : {0 : 'batch_size'}}
    #     )