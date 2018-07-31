"""
#############################################################################################################

Functions for training, saving and loading weights of a model

    Alice   2018

#############################################################################################################
"""

import  os
import  datetime

from    math        import ceil
from    keras       import utils, models, losses, optimizers, preprocessing, callbacks

import  msg         as ms


SHUFFLE         = True          # shuffle dataset

RGB             = None          # set by nn_main.py
img_size        = None          # set by nn_main.py
dir_current     = None          # set by nn_main.py
dir_check       = 'chkpnt'

cnfg            = {
    'dir_dset'      : None,
    'n_gpus'        : None,     # [int] number of GPUs (0 if using CPU)
    'n_epochs'      : None,     # [int] number of epochs
    'batch_size'    : None,     # [int] batch size
    'lrate'         : None,     # [float] learning rate
    'optimizer'     : None,     # [str] code for optimizer algorithm
    'loss'          : None,     # [str] code for objective function
    'n_check'       : None      # [int] number of checkpoints to save during training
}


# ===========================================================================================================


def gen_dataset( dir_dset ):
    """ -----------------------------------------------------------------------------------------------------
    Take the path to a directory and generate batches of data.

    The three sub-dataset must have these paths:
        * dir_dset/train/frames/
        * dir_dset/valid/frames/
        * dir_dset/test/frames/

    https://keras.io/preprocessing/image/#imagedatagenerator-methods

    dir_dset:       [str]

    return:         [list] of keras_preprocessing.image.DirectoryIterator
    ----------------------------------------------------------------------------------------------------- """

    # 'rescale' to normalize pixels in [0..1]
    train_idg       = preprocessing.image.ImageDataGenerator( rescale=1./255 )
    valid_idg       = preprocessing.image.ImageDataGenerator( rescale=1./255 )

    train_datagen   = train_idg.flow_from_directory(
            directory   = os.path.join( cnfg[ 'dir_dset' ], 'train' ),
            target_size = img_size if RGB else img_size[ :-1 ],
            color_mode  = 'rgb' if RGB else 'grayscale',
            class_mode  = 'input',          # use this when model output is supposed to be equal to input
            batch_size  = cnfg[ 'batch_size' ],
            shuffle     = SHUFFLE
    )

    valid_datagen   = valid_idg.flow_from_directory(
            directory   = os.path.join( cnfg[ 'dir_dset' ], 'valid' ),
            target_size = img_size if RGB else img_size[ :-1 ],
            color_mode  = 'rgb' if RGB else 'grayscale',
            class_mode  = 'input',          # use this when model output is supposed to be equal to input
            batch_size  = cnfg[ 'batch_size' ],
            shuffle     = SHUFFLE
    )

    return train_datagen, valid_datagen



def set_callback( n_check ):
    """ -----------------------------------------------------------------------------------------------------
    Save checkpoints of the model during training

    n_check:        [int] number of total checkpoints

    return:         [keras.callbacks.ModelCheckpoint]
    ----------------------------------------------------------------------------------------------------- """
    p       = os.path.join( dir_current, dir_check )
    fname   = os.path.join( p, "check_{epoch:04d}.h5" )
    os.makedirs( p )

    period  = ceil( cnfg[ 'n_epochs' ] / n_check )
    return callbacks.ModelCheckpoint( fname, save_weights_only=True, period=period )



def train_model( model ):
    """ -----------------------------------------------------------------------------------------------------
    Train the network

    https://keras.io/models/model/
    https://keras.io/optimizers/
    https://keras.io/losses/
    https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus

    model:          [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    op                      = None
    ls                      = None
    cb                      = [ set_callback( cnfg[ 'n_check' ] ) ] if cnfg[ 'n_check' ] > 0 else None
    train_feed, valid_feed  = gen_dataset( cnfg[ 'dir_dset' ] ) # train and valid dataset generators

    # optimizer
    if cnfg[ 'optimizer' ] == 'ADAGRAD':
        op      = optimizers.Adagrad( lr=cnfg[ 'lrate' ] )      # default lr=0.010
    elif cnfg[ 'optimizer' ] == 'SDG':
        op      = optimizers.SGD( lr=cnfg[ 'lrate' ] )          # default lr=0.010
    elif cnfg[ 'optimizer' ] == 'RMS':
        op      = optimizers.RMSprop( lr=cnfg[ 'lrate' ] )      # default lr=0.001
    elif cnfg[ 'optimizer' ] == 'ADAM':
        op      = optimizers.Adam( lr=cnfg[ 'lrate' ] )         # default lr=0.001
    else:
        ms.print_err( "Optimizer {} not valid".format( cnfg[ 'optimizer' ] ) )
    
    # loss
    if cnfg[ 'loss' ] == 'MSE':
        ls      = losses.mean_squared_error
    elif cnfg[ 'loss' ] == 'CXE':
        ls      = losses.categorical_crossentropy
    else:
        ms.print_err( "Loss {} not valid".format( cnfg[ 'loss' ] ) )
    
    # train using multiple GPUs
    if cnfg[ 'n_gpus' ] > 0:
        model                   = utils.multi_gpu_model( model, gpus=cnfg[ 'n_gpus' ] )
        cnfg[ 'batch_size' ]    *= cnfg[ 'n_gpus' ]
                                # in this way, each GPU has a batch of the size passed as argument

    model.compile(
            optimizer       = op,
            loss            = ls
    )

    t_start                 = datetime.datetime.now()           # starting time of execution

    model.fit_generator(
            train_feed,
            epochs          = cnfg[ 'n_epochs' ],
            validation_data = valid_feed,
            steps_per_epoch = train_feed.samples / cnfg[ 'batch_size' ],
            callbacks       = cb,
            shuffle         = SHUFFLE
    )

    t_end   = datetime.datetime.now()                           # end time of execution

    fname   = os.path.join( dir_current, "train.time" )
    with open( fname, 'w' ) as f:
        f.write( str( t_end - t_start ) )                       # save total time of execution



def save_model( model, together=False, name="nn" ):
    """ -----------------------------------------------------------------------------------------------------
    Save a trained model in a single HDF5 file or in two different files,
    one for the architecture (JSON) and one for the weight (HDF5)

    model:          [keras.engine.training.Model]
    together:       [bool] if True save a single file
    name:           [str] name of the model
    ----------------------------------------------------------------------------------------------------- """
    name    = os.path.join( dir_current, name )

    if together:
        model.save( name + '.h5' )

    else:                                               # save separately architecture and weights
        model.save_weights( name + '_wght.h5' )
        with open( name + '_arch.json', 'w' ) as f:
            f.write( model.to_json() )



def load_model( *fname ):
    """ -----------------------------------------------------------------------------------------------------
    Load a trained model for file. The argument can be a single HDF5 file (with architecture and weights)
    or two separate files
    
    In the second case, the first file passed is the model architecture, the second the model weights

    fname:          [str] name(s) of file(s)
    ----------------------------------------------------------------------------------------------------- """
    if len( fname ) == 0:
        ms.print_wrn( "no file passed" )
        return None

    if len( fname ) == 1:                               # if a single file is passed
        return models.load_model( fname )                   

    model   = models.model_from_json( fname[ 0 ] )
    model.load_weights( fname[ 1 ] )

    return model
