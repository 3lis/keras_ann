"""
#############################################################################################################

Definition of the architecture of the neural network

    Alice   2018

#############################################################################################################
"""

import  os
import  numpy       as np

from    keras       import models, layers, utils, initializers, regularizers
from    keras       import Model, Input

import  msg         as ms


name_wght       = 'nn_wght.h5'
name_arch       = 'nn_arch.json'

cnfg            = {
    'seed'              : None,
    'arch'              : None,
    'img_size'          : None,
    'ref_model'         : None,
    'dir_current'       : None,
    'k_initializer'     : None,
    'k_regularizer'     : None,
    'k_regularizer_w'   : None,

    'n_conv'            : None,
    'conv_filters'      : None,
    'conv_kernel_size'  : None,
    'conv_strides'	: None,
    'conv_padding'	: None,
    'conv_activation'	: None,
    'conv_pool_size'    : None,
    'conv_train'        : None,

    'n_core'            : None,
    'core_dense'        : None,         # last value of list is computed in create_ae_simple()
    'core_reshape'      : None,         # computed in create_ae_simple()
    'core_activation'	: None,
    'core_train'	: None,

    'n_decn'            : None,
    'decn_filters'	: None,
    'decn_kernel_size'  : None,
    'decn_strides'	: None,
    'decn_padding'	: None,
    'decn_activation'	: None,
    'decn_train'	: None
}


# ===========================================================================================================


def create_ae_simple():
    """ -----------------------------------------------------------------------------------------------------
    Create a standard autoencoder

    return:         [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    init        = None 
    kreg        = None 

    # initializer
    if cnfg[ 'k_initializer' ] == 'RUNIF':
        init    = initializers.RandomUniform( minval=-0.05, maxval=0.05, seed=cnfg[ 'seed' ] )
    elif cnfg[ 'k_initializer' ] == 'GLOROT':
        init    = initializers.glorot_normal( seed=cnfg[ 'seed' ] )
    elif cnfg[ 'k_initializer' ] == 'HE':
        init    = initializers.he_normal( seed=cnfg[ 'seed' ] )
    else:
        ms.print_err( "Initializer {} not valid".format( cnfg[ 'k_initializer' ] ) )
    
    # regularizer
    if cnfg[ 'k_regularizer' ] == 'L2':
        kreg    = regularizers.l2( cnfg[ 'k_regularizer_w' ] )
    elif cnfg[ 'k_regularizer' ] == 'NONE':
        kreg    = None
    else:
        ms.print_err( "Regularizer {} not valid".format( cnfg[ 'k_regularizer' ] ) )

    xs          = Input(                                        # INPUT LAYER
            shape           = cnfg[ 'img_size' ]                    # height, width, channels
    )
    x           = xs

    for i in range( cnfg[ 'n_conv' ] ):
        x       = layers.Conv2D(                                # CONVOLUTIONAL LAYERs
            cnfg[ 'conv_filters' ][ i ],                            # number of filters
            kernel_size         = cnfg[ 'conv_kernel_size' ][ i ],  # size of window
            strides             = cnfg[ 'conv_strides' ][ i ],      # stride (window shift)
            padding             = cnfg[ 'conv_padding' ][ i ],      # zero-padding around the image
            activation          = cnfg[ 'conv_activation' ][ i ],   # activation function
            kernel_initializer  = init,
            kernel_regularizer  = kreg,         # NOTE check also activity_regularizer
            use_bias            = False,        # NOTE watch out for the biases!
            trainable           = cnfg[ 'conv_train' ][ i ]
        )( x )

        x       = layers.MaxPooling2D(                          # MAX POOLING LAYERs
            pool_size       = cnfg[ 'conv_pool_size' ][ i ],        # pooling size
            padding         = cnfg[ 'conv_padding' ][ i ]           # zero-padding around the image
        )( x )

    # save output shape of last convolution...
    cnfg[ 'core_reshape' ]  = list( map( int, x.shape[ 1: ] ) )

    # ...and use it as flat shape of last dense layer
    cnfg[ 'core_dense' ].append( np.prod( cnfg[ 'core_reshape' ] ) )

    x       = layers.Flatten()( x )                             # FLATTEN LAYER

    for i in range( cnfg[ 'n_core' ] ):
        x           = layers.Dense(                             # DENSE LAYERs
            cnfg[ 'core_dense' ][ i ],                              # dimensionality of the output
            activation      = cnfg[ 'core_activation' ][ i ],       # activation function
            trainable       = cnfg[ 'core_train' ][ i ]
        )( x )

    x           = layers.Reshape(                               # RESHAPE LAYER
            target_shape    = cnfg[ 'core_reshape' ]                # new shape (height, width, channels)
    )( x )

    # NOTE consider initializer & regularizer also in deconv
    for i in range( cnfg[ 'n_decn' ] ):
        x       = layers.Conv2DTranspose(                       # DECONVOLUTIONAL LAYERs
            cnfg[ 'decn_filters' ][ i ],                            # number of filters
            kernel_size     = cnfg[ 'decn_kernel_size' ][ i ],      # size of window
            strides         = cnfg[ 'decn_strides' ][ i ],          # stride
            padding         = cnfg[ 'decn_padding' ][ i ],          # zero-padding
            activation      = cnfg[ 'decn_activation' ][ i ],       # activation function
            use_bias        = False,
            trainable       = cnfg[ 'decn_train' ][ i ]
        )( x )

    return Model( inputs=xs, outputs=x, name='AE' )



def create_ae_segm( ref_model ):
    """ -----------------------------------------------------------------------------------------------------
    Create an autoencoder which loads some layers from a reference model

    ref_model:      [str] folder containing the saved model

    return:         [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    cond    = lambda x: True                # NOTE load weights of all possible layers
    #cond    = lambda x: not x.trainable    # load weights of untrainable layers

    nn      = create_ae_simple()
    mm      = load_model( ref_model )
    indx    = [ i for i in range( len( nn.layers ) ) if cond( nn.layers[ i ] ) ]

    for i in indx:
        try:
            nn.layers[ i ].set_weights( mm.layers[ i ].get_weights() )
        except:
            ms.print_msg( "Unable to load layer {}".format( i ) )

    return nn

    
    
def create_model():
    """ -----------------------------------------------------------------------------------------------------
    Create the model of the neural network

    return:         [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    if cnfg[ 'arch' ] == 'AE_SIMPLE':
        return create_ae_simple()

    if cnfg[ 'arch' ] == 'AE_SEGM':
        return create_ae_segm( ref_model=cnfg[ 'ref_model' ] )

    ms.print_err( "Architecture {} not valid".format( cnfg[ 'arch' ] ) )


# ===========================================================================================================


def model_summary( model ):
    """ -----------------------------------------------------------------------------------------------------
    Print a summary of the model

    model:          [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    utils.print_summary( model )



def model_graph( model, fname="graph" ):
    """ -----------------------------------------------------------------------------------------------------
    Plot a graph of the model and save it to a file

    model:          [keras.engine.training.Model]
    fname:          [str] name of the output image
    ----------------------------------------------------------------------------------------------------- """

    # JPEG rendering works badly for some reason...
    f   = os.path.join( cnfg[ 'dir_current' ], fname + '.png' )

    utils.plot_model( model, to_file=f, show_shapes=True, show_layer_names=True )



def save_model( model ):
    """ -----------------------------------------------------------------------------------------------------
    Save a trained model in one file for the architecture (JSON) and one for the weights (HDF5)

    model:          [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    model.save_weights( os.path.join( cnfg[ 'dir_current' ], name_wght ) )

    with open( os.path.join( cnfg[ 'dir_current' ], name_arch ), 'w' ) as f:
        f.write( model.to_json() )



def load_model( arg ):
    """ -----------------------------------------------------------------------------------------------------
    Load a trained model from file.
    If a folder is passed, it should contain the two files HDF5 and JSON.
    If a single file is passed, it is considered as model+weights HDF5 file

    arg:            [str] folder or filename

    return:         [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    if arg.endswith( '.h5' ):
        return models.load_model( arg )

    js  = os.path.join( arg, name_arch )
    h5  = os.path.join( arg, name_wght )

    with open( js, 'r' ) as f:
        json    = f.read()

    model   = models.model_from_json( json )
    model.load_weights( h5 )

    return model
