"""
#############################################################################################################

Test routines to evaluate performance of model

    Alice   2018

#############################################################################################################
"""

import  os
import  numpy       as np

from    keras       import Model, preprocessing
from    math        import sqrt, ceil, inf
from    PIL         import Image

import  matplotlib
matplotlib.use( 'agg' )         # to use matplot with unknown 'DISPLAY' var
from    matplotlib  import pyplot, cm, ticker

import  msg         as ms


DEBUG0              = False

RGB                 = None      # set by nn_main.py
img_size            = None      # set by nn_main.py
dir_test            = None      # set by nn_main.py
dir_current         = None      # set by nn_main.py

dir_plot            = 'plot'



def create_test_model( model, weights=None ):
    """ -----------------------------------------------------------------------------------------------------
    Create a new version of an existing model, producing an output after every layer,
    useful for testing.

    model:          [keras.engine.training.Model] original model
    weights:        [str] name of the file with saved weights

    return:         [keras.engine.training.Model] new model
    ----------------------------------------------------------------------------------------------------- """
    if weights is not None:
        try:
            model.load_weights( weights )
        except:
            ms.print_err( "while opening file " + weights )
            raise

    layers_out   = [ l.output for l in model.layers[ 1: ] ]

    #return Model( inputs=model.input, outputs=layers_out )

    # NOTE get_input_at() is required when using multiple GPUs
    return Model( inputs=model.get_input_at( 0 ), outputs=layers_out )



def predict_image( model, img ):
    """ -----------------------------------------------------------------------------------------------------
    Return the model prediction on an image

    model:          [keras.engine.training.Model]
    img:            [str] image file

    return:         [numpy.ndarray]
    ----------------------------------------------------------------------------------------------------- """
    try:
        i   = preprocessing.image.load_img( img, grayscale=( not RGB ), target_size=img_size[ :-1 ] )
    except:
        ms.print_err( "while opening file " + img )
        raise

    i   = preprocessing.image.img_to_array( i )
    i   = np.expand_dims( i, axis=0 )
    i  /= 255.

    return model.predict( i )



def img_collage( imgs, w, h, pad_size=5, pad_color="#ff5555" ):
    """ -----------------------------------------------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------- """
    n_imgs  = len( imgs )
    n_cols  = ceil( sqrt( n_imgs ) )
    n_rows  = ceil( n_imgs / n_cols )
    width   = n_cols * w + ( n_cols - 1 ) * pad_size
    height  = n_rows * h + ( n_rows - 1 ) * pad_size

    i       = 0
    img     = Image.new( 'RGB', ( width, height ), color=pad_color )
    
    for r in range( n_rows ):
        y   = r * ( h + pad_size )

        for c in range( n_cols ):
            x   = c * ( w + pad_size )

            img.paste( imgs[ i ].resize( ( w, h ) ), ( x, y ) )
            i   += 1

            if i >= n_imgs: 
                break

        if i >= n_imgs: 
            break

    return img



def layer_outputs( layer, output, pth, normalize=True, prfx="out_" ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the output of a convolutional layer

    layer:          [str] name of layer
    output:         [numpy.ndarray]
    pth:            [str] where to store results
    normalize:      [bool] if True, normalize each sub-plot individually
    prfx:           [str] prefix of output image files

    return:         [tuple] number of dead filters and total number of filters
    ----------------------------------------------------------------------------------------------------- """
    n_feat      = output.shape[ -1 ]
    h, w        = img_size[ :2 ]
    feat_list   = []
    cnt         = 0

    for f in range( n_feat ):
        pixels  = output[ 0, :, :, f ]              # activation values of current feature
        ptp     = pixels.ptp()                      # difference between min and max activation value

        if ptp == 0:
            if DEBUG0:
                ms.print_msg( "Dead filter in layer {}, feature {}, activation {}".format(
                            layer, feat, ptp ) )
            cnt += 1

        elif normalize:                             # pixel normalization (only when ptp is not zero)
            pixels      = 255. * ( pixels - pixels.min() ) / ptp    # NOTE check this formula

        feat_list.append( Image.fromarray( pixels ) )

    fname   = prfx + layer + '.jpg'
    i       = img_collage( feat_list, w, h )
    i.save( os.path.join( pth, fname ) )

    return cnt, n_feat



def layer_weights( layer, wght, pth, dpi=10, normalize=False, prfx="wght_" ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the kernel weights of a convolutional layer

    layer:          [str] name of layer
    wght:           [numpy.ndarray]
    pth:            [str] where to store results
    dpi:            [int] resolution of kernel images
    normalize:      [bool] if True, normalize each sub-plot individually
    prfx:           [str] prefix of output image files
    ----------------------------------------------------------------------------------------------------- """
    mx, mn      = -inf, inf
    wght        = wght.reshape( *wght.shape[ 0:2 ], -1 )    # from 4D to 3D shape
    n_wght      = wght.shape[ -1 ]
    h, w        = dpi * np.array( wght.shape[ :2 ] )
    wght_list   = []

    for v in range( n_wght ):
        pixels  = wght[ :, :, v ]

        if np.isnan( pixels.sum() ):
            ms.print_msg( "Overflowed weights in layer {}, kernel {}".format( layer, v ) )
            continue

        ptp     = pixels.ptp()

        if ptp == 0:
            ms.print_msg( "No active weights in layer {}, kernel {}".format( layer, v ) )

        elif normalize:
            pixels  = 255. * ( pixels - pixels.min() ) / ptp

        else:
            mn      = min( pixels.min(), mn )
            mx      = max( pixels.max(), mx )

        wght_list.append( pixels )

    if not normalize:
        wght_list   = [ 255. * ( pixels - mn ) / ( mx - mn ) for pixels in wght_list ]

    wght_list   = [ Image.fromarray( pixels ) for pixels in wght_list ]

    fname   = prfx + layer + '.jpg'
    i       = img_collage( wght_list, w, h )
    i.save( os.path.join( pth, fname ) )



def model_outputs( model, img, dir_test=None, normalize=True ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the output of all convolutional layers in the model.
    The argument 'img' can be the path of image file, or and integer index of a frame in the test set

    model:          [keras.engine.training.Model] the result of create_test_model()
    img:            [str or int] image file
    dir_test:       [str] dataset directory (only when 'img' is an int)
    normalize:      [bool] if True, normalize each sub-plot individually
    ----------------------------------------------------------------------------------------------------- """
    pth     = os.path.join( dir_current, dir_plot )
    if not os.path.exists( pth ):
        os.makedirs( pth )

    if isinstance( img, int ):
        try:
            img =  os.path.join( dir_test, os.listdir( dir_test )[ img ] )
        except:
            ms.print_err( "while opening file " + img )
            raise

    if not os.path.isfile( img ):
        ms.print_err( "while opening file " + img )

    cnt     = 0
    tot     = 0
    lay     = [ l.name for l in model.layers[ 1: ] ]
    out     = predict_image( model, img )

    for l, o in zip( lay, out ):
        if 'conv' in l:             # plot output only for convolutional layers
            c, t    = layer_outputs( l, o, pth, normalize=normalize )
            cnt     += c
            tot     += t

    ms.print_msg( "Total dead filters percentage: {:.1f}%".format( 100 * cnt / tot ) )



def model_weights( model, normalize=False ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the kernel weights of all convolutional layers in the model

    model:          [keras.engine.training.Model]
    normalize:      [bool] if True, normalize each sub-plot individually
    ----------------------------------------------------------------------------------------------------- """
    pth     = os.path.join( dir_current, dir_plot )
    if not os.path.exists( pth ):
        os.makedirs( pth )

    # NOTE get_weights returns a list of (weights,biases)
    wght    = [ l.get_weights() for l in model.layers[ 1 : ] ]
    lay     = [ l.name for l in model.layers[ 1: ] ]

    for l, w in zip( lay, wght ):
        if 'conv' in l:             # plot weights only for convolutional layers
            layer_weights( l, w[ 0 ], pth, normalize=normalize )



def model_dead( model, layer, n_img, dir_test ):
    """ -----------------------------------------------------------------------------------------------------
    Count how many features of a layer are inactive, testing the prediction on serevar input images

    model:          [keras.engine.training.Model] the result of create_test_model()
    layer:          [int] index of layer
    n_img:          [int] number of input images
    dir_test:       [str] dataset directory
    ----------------------------------------------------------------------------------------------------- """

    # sample 'n_img' images, evenly spaced, from the test set
    dir_list    = [ f for f in os.listdir( dir_test ) if f.lower().endswith( ( '.png', '.jpg', '.jpeg' ) ) ]
    indx        = list( map( int, np.linspace( 0, len( dir_list ), n_img, endpoint=False ) ) )
    img_list    = [ os.path.join( dir_test, dir_list[ i ] ) for i in indx ]

    # outputs of layer 'layer' on the 'n_img' images
    out_list    = [ predict_image( model, i )[ layer ] for i in img_list ]

    n_feat      = out_list[ 0 ].shape[ -1 ]
    dead_feat   = np.zeros( n_feat, dtype=int )

    for out in out_list:
        for feat in range( n_feat ):
            activation  = out[ 0, :, :, feat ].ptp()
            #activation  = out[ 0, :, :, feat ].sum()
            if activation == 0:
                dead_feat[ feat ]   += 1                # count the inactive features (dead filters)

    # number of filters which result dead in at least 'threshold' images
    threshold   = n_img // 2
    dead_cnt    = sum( i >= threshold for i in dead_feat )

    ms.print_msg( "{:.1f}% of filters in layer {} are inactive on {}/{} different images".format(
                100 * dead_cnt / n_feat, layer, threshold, n_img ) )
