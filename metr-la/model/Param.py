DATANAME = 'METR-LA'
TIMESTEP_IN = 12
TIMESTEP_OUT = 12
N_NODE = 207
CHANNEL = 1
BATCHSIZE = 64
LEARN = 0.001
EPOCH = 200
PATIENCE = 10
OPTIMIZER = 'Adam'
# OPTIMIZER = 'RMSprop'
# LOSS = 'MSE'
LOSS = 'MAE'
TRAINRATIO = 0.8 # TRAIN + VAL
TRAINVALSPLIT = 0.125 # val_ratio = 0.8 * 0.125 = 0.1
FLOWPATH = './metr-la/metr-la.h5'
C_2_PATH = './metr-la/all.npy'
ADJPATH = './metr-la/W_metrla.csv'
ADJTYPE = 'doubletransition'