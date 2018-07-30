import numpy as np
from hyperopt import hp

params = dict()
param_space_bilstm_att_g = {
    'task': 'BILSTM_ATT_G',
    'batch_size': hp.choice('batch_size',[32,64,128]),
    'n_hidden': hp.choice('n_hidden', [200,300]),
    'n_hidden_ae': hp.choice('n_hidden_ae', [100,200]),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.1, 0.01),
    'l2_reg': hp.quniform('l2_reg', 8e-4, 0.001, 0.001),
    'keep_rate': hp.quniform('keep_rate', 0.3, 0.8, 0.1),
    'n_iter': hp.quniform('n_iter', 1, 8, 1),
    'grad_clip': hp.quniform('grad_clip', 20.0, 100.0, 10),
    'dim_z': hp.quniform('dim_z', 50, 100, 10),
    'n_unlabel': hp.quniform('n_unlabel', 1000, 10000, 1000),
    'alpha': hp.quniform('alpha', 5.0, 10.0, 1.0), 
    'position_enc': hp.choice('position_enc', ['binary','distance','']),
    'position_dec': hp.choice('position_dec', ['binary','distance','']),
    'bidirection_enc': hp.choice('bidirection_enc', [True, False]),
    'bidirection_dec': hp.choice('bidirection_dec', [True,False]),
    'decoder_type': hp.choice('decoder_type', ['sclstm', 'lstm'])
}

params['BILSTM_ATT_G'] = param_space_bilstm_att_g 
