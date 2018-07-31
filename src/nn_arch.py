"""
#############################################################################################################

Definition of the architecture of the neural network

    Alice   2018

#############################################################################################################
"""

import  os
import  numpy       as np

from    keras       import layers, utils, initializers, regularizers
from    keras       import Model, Input


RGB             = None                  # set by nn_main.py
dir_current     = None                  # set by nn_main.py

cnfg            = {

    'SEED'              : None,
    'model_name'        : 'ae',
    'img_size'          : None,
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

    'n_core'            : None,
    'core_dense'        : None,         # last value of list is computed in create_ae_simple()
    'core_reshape'      : None,         # computed in create_ae_simple()
    'core_activation'	: None,

    'n_decn'            : None,
    'decn_filters'	: None,
    'decn_kernel_size'  : None,
    'decn_strides'	: None,
    'decn_padding'	: None,
    'decn_activation'	: None
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
        init    = initializers.RandomUniform( minval=-0.05, maxval=0.05, seed=cnfg[ 'SEED' ] )
    elif cnfg[ 'k_initializer' ] == 'GLOROT':
        init    = initializers.glorot_normal( seed=cnfg[ 'SEED' ] )
    elif cnfg[ 'k_initializer' ] == 'HE':
        init    = initializers.he_normal( seed=cnfg[ 'SEED' ] )
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
            # NOTE maybe just 'shape' and not 'batch_shape' ?
            batch_shape     = ( None, *cnfg[ 'img_size' ] )         # batch_size, height, width, channels
    )
    x           = xs

    for i in range( cnfg[ 'n_conv' ] ):
        x       = layers.Conv2D(                                # CONVOLUTIONAL LAYERs
            cnfg[ 'conv_filters' ][ i ],                            # number of filters
            kernel_size         = cnfg[ 'conv_kernel_size' ][ i ],  # size of window
            strides             = cnfg[ 'conv_strides' ][ i ],      # stride (window shift)
            padding             = cnfg[ 'conv_padding' ][ i ],      # zero-padding around the image
            activation          = cnfg[ 'conv_activation' ][ i ],   # activation function
            # NOTE check also activity_regularizer
            kernel_initializer  = init,
            kernel_regularizer  = kreg,
            # NOTE watch out for the biases!
            use_bias            = False
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
            activation      = cnfg[ 'core_activation' ][ i ]        # activation function
        )( x )

    x           = layers.Reshape(                               # RESHAPE LAYER
            target_shape    = cnfg[ 'core_reshape' ]                # new shape (height, width, channels)
    )( x )

    for i in range( cnfg[ 'n_decn' ] ):
        x       = layers.Conv2DTranspose(                       # DECONVOLUTIONAL LAYERs
            cnfg[ 'decn_filters' ][ i ],                            # number of filters
            kernel_size     = cnfg[ 'decn_kernel_size' ][ i ],      # size of window
            strides         = cnfg[ 'decn_strides' ][ i ],          # stride
            padding         = cnfg[ 'decn_padding' ][ i ],          # zero-padding
            activation      = cnfg[ 'decn_activation' ][ i ],       # activation function
            # NOTE consider initializer & regularizer also in deconv
            use_bias        = False
        )( x )

    return Model( inputs=xs, outputs=x, name=cnfg[ 'model_name' ] )



def create_model():
    """ -----------------------------------------------------------------------------------------------------
    Create the model of the neural network

    return:         [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    # TODO for now, it is available only the simple AE
    return create_ae_simple()



def model_summary( model ):
    """ -----------------------------------------------------------------------------------------------------
    Print a summary of the model

    model:          [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    utils.print_summary( model )



def model_graph( model, fname="nn_graph" ):
    """ -----------------------------------------------------------------------------------------------------
    Plot a graph of the model and save it to a file

    model:          [keras.engine.training.Model]
    fname:          [str] name of the output image
    ----------------------------------------------------------------------------------------------------- """
    f   = os.path.join( dir_current, fname + '.png' )   # JPEG rendering works badly for some reason...

    utils.plot_model( model, to_file=f, show_shapes=True, show_layer_names=True )
