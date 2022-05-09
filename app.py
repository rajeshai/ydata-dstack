import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ydata_synthetic.synthesizers.regular import DRAGAN
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

df = pd.read_csv('train.csv')
df.dropna(inplace=True)

model = DRAGAN

#Load data and define the data processor parameters
num_cols = ['id','Age', 'Region_Code','Annual_Premium','Policy_Sales_Channel', 'Vintage']
cat_cols = ['Gender','Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Response']

# DRAGAN training
#Defining the training parameters of DRAGAN

noise_dim = 128
dim = 128
batch_size = 500

log_step = 100
epochs = 50
learning_rate = 1e-5
beta_1 = 0.5
beta_2 = 0.9
models_dir = './cache'

gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,
                             sample_interval=log_step)

synthesizer = model(gan_args, n_discriminator=3)
synthesizer.train(df, train_args, num_cols, cat_cols)
synthesizer.save('ins_synth.pkl')
synthesizer = model.load('ins_synth.pkl')
df1 = synthesizer.sample(380000)
filepath = './folder'
df1.to_csv(filepath)
