"""
#############################################################################################################

Functions for training a model

    Alice   2018

#############################################################################################################
"""

import  os
import  datetime

from    math                import ceil, sqrt
from    keras               import utils, models, losses, optimizers, callbacks, preprocessing
from    keras               import backend  as K

import  matplotlib
matplotlib.use( 'agg' )     # to use matplotlib with unknown 'DISPLAY' var (when using remote display)
from    matplotlib          import pyplot   as plt

import  msg                                 as ms
import  nn_dset                             as nd


SHUFFLE         = True
dir_check       = 'chkpnt'
name_best       = 'nn_best.h5'

cnfg            = {
    'img_size'      : None,
    'dir_current'   : None,
    'dir_dset'      : None,
    'data_class'    : None,
    'patience'      : None,
    'n_gpus'        : None,     # [int] number of GPUs (0 if using CPU)
    'n_epochs'      : None,     # [int] number of epochs
    'batch_size'    : None,     # [int] batch size
    'lrate'         : None,     # [float] learning rate
    'optimizer'     : None,     # [str] code for optimizer algorithm
    'loss'          : None,     # [str] code for objective function
    'n_check'       : None      # [int] number of checkpoints to save during training
}



def get_unbalanced_loss():
    """ -----------------------------------------------------------------------------------------------------
    Setup a loss function for unbalanced loss

    return:         the symbol to a function in y_true, y_pred
    ----------------------------------------------------------------------------------------------------- """
    dr      = os.path.join( cnfg[ 'dir_dset' ], 'valid', cnfg[ 'data_class' ].lower(), 'img' )

    if not os.path.isdir( dr ):
        raise ValueError( "{} directory does not exist".format( dr ) )

    imgs    = [ f for f in os.listdir( dr ) if f.lower().endswith( ( '.png', '.jpg', '.jpeg' ) ) ]
    pix_img = cnfg[ 'img_size' ][ 0 ] * cnfg[ 'img_size' ][ 1 ]
    n_tot   = 0
    n_ones  = 0

    for i in imgs:
        f       = os.path.join( dr, i )
        im      = preprocessing.image.load_img( f, grayscale=True, target_size = cnfg[ 'img_size' ][ :-1 ] )
        im      = preprocessing.image.img_to_array( im )
        n_ones  += ( im == 255 ).sum()
        n_tot   +=  pix_img

    p_ones  = sqrt( float( n_ones ) ) / n_tot
    p_zeros = 1 - p_ones

    def unbalanced_loss( y_true, y_pred ):
        w   = p_zeros * y_true + p_ones * ( 1. - y_true )   # compute weights from inverse probability
        l   = K.binary_crossentropy( y_true, y_pred )       # ordinary binary crossentropy
        u   = w * l                                         # the umbalanced loss
        return K.mean( u )

    return unbalanced_loss                                  # return the symbol of the funciont yet defined



def set_callback():
    """ -----------------------------------------------------------------------------------------------------
    Save checkpoints of the model during training.
    Stop training when val_loss stop decreasing

    return:         [keras.callbacks.ModelCheckpoint]
    ----------------------------------------------------------------------------------------------------- """
    calls   = []
    period  = ceil( cnfg[ 'n_epochs' ] / cnfg[ 'n_check' ] )

    if cnfg[ 'n_check' ] > 0:
        calls.append( callbacks.ModelCheckpoint(
                os.path.join( cnfg[ 'dir_current' ], name_best ),
                save_best_only      = True,
                save_weights_only   = False,
                period              = 1
        ) )

    if cnfg[ 'n_check' ] > 1:
        p       = os.path.join( cnfg[ 'dir_current' ], dir_check )
        fname   = os.path.join( p, "check_{epoch:04d}.h5" )
        os.makedirs( p )
        calls.append( callbacks.ModelCheckpoint( fname, save_weights_only=True, period=period ) )

    if cnfg[ 'patience' ] > 0:
        calls.append( callbacks.EarlyStopping( monitor='val_loss', patience=cnfg[ 'patience' ] ) )

    return calls



def train_model( model, tlog='train.time' ):
    """ -----------------------------------------------------------------------------------------------------
    Train the network

    https://keras.io/models/model/
    https://keras.io/optimizers/
    https://keras.io/losses/
    https://keras.io/getting-started/faq/#how-can-i-run-a-keras-model-on-multiple-gpus

    model:          [keras.engine.training.Model]
    tlog:           [str] file name

    return:         [keras.callbacks.History]
    ----------------------------------------------------------------------------------------------------- """
    op                      = None
    ls                      = None
    cb                      = set_callback()

    # train and valid dataset generators
    train_feed, valid_feed                      = nd.gen_dataset( cnfg[ 'dir_dset' ] )
    train_samples, valid_samples, test_samples  = nd.len_dataset( cnfg[ 'dir_dset' ] )

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
    elif cnfg[ 'loss' ] == 'BXE':
        ls      = losses.binary_crossentropy
    elif cnfg[ 'loss' ] == 'UNB':
        ls      = get_unbalanced_loss()
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

    history = model.fit_generator(
            train_feed,
            epochs              = cnfg[ 'n_epochs' ],
            validation_data     = valid_feed,
            steps_per_epoch     = train_samples / cnfg[ 'batch_size' ],
            validation_steps    = valid_samples / cnfg[ 'batch_size' ],
            callbacks           = cb,
            verbose             = 2,
            shuffle             = SHUFFLE
    )

    t_end   = datetime.datetime.now()                           # end time of execution

    fname   = os.path.join( cnfg[ 'dir_current' ], tlog )
    with open( fname, 'w' ) as f:
        f.write( str( t_end - t_start ) + '\n' )                # save total time of execution

    return history



def plot_history( history, fname='loss.pdf' ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the loss performance during training

    history:        [keras.callbacks.History]
    ----------------------------------------------------------------------------------------------------- """
    train_loss  = history.history[ 'loss' ]
    valid_loss  = history.history[ 'val_loss' ]
    epochs      = range( 1, len( train_loss ) + 1 )

    plt.plot( epochs, train_loss, 'r--' )
    plt.plot( epochs, valid_loss, 'b-' )
    plt.legend( [ 'Training Loss', 'Validation Loss' ] )
    plt.xlabel( 'Epoch' )
    plt.ylabel( 'Loss' )
    plt.savefig( os.path.join( cnfg[ 'dir_current' ], fname ) )
    plt.close()
