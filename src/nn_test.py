"""
#############################################################################################################

Test routines to evaluate performance of model

    Alice   2018

#############################################################################################################
"""

import  os
import  numpy       as np

from    keras       import Model, preprocessing, losses, optimizers
from    math        import sqrt, ceil, inf
from    PIL         import Image

import  msg         as ms
from    nn_train    import get_unbalanced_loss


DEBUG0      = False
dir_plot    = 'plot'
dir_test    = 'test'

cnfg        = {
    'loss'          : None,     # [str] used in evaluate_tset()
    'dir_current'   : None
}

samples     = {                 # TODO list of manually selected representative samples to be fed to a model
    'frame'     : [  12, 102, 320, 682, 685, 752, 805, 812, 866, 904, 931, 941, 993, 1027, 1081, 1090 ],
    'cars'      : [  12, 102, 320, 682, 685, 752, 805, 812, 866, 904, 931, 941, 993, 1027, 1081, 1090 ],
    'lane'      : [  12, 102, 320, 682, 685, 752, 805, 812, 866, 904, 931, 941, 993, 1027, 1081, 1090 ]
}



# ===========================================================================================================
#
#   - in_rgb
#   - out_rgb
#   - imgsize
#
#   - array_to_image
#   - save_image
#   - save_collage
#
#   - create_test_model
#   - attach_loss
#
# ===========================================================================================================

def imgsize( model ):
    """ -----------------------------------------------------------------------------------------------------
    Return height and width of input

    model:          [keras.engine.training.Model]

    return:         [list]
    ----------------------------------------------------------------------------------------------------- """
    s   = model.input_shape

    if len( s ) == 3:
        return model.input_shape[ :2 ]

    if len( s ) == 4:
        return model.input_shape[ 1:3 ]



def in_rgb( model ):
    """ -----------------------------------------------------------------------------------------------------
    Return True if output is RGB, False if graylevel

    model:          [keras.engine.training.Model]

    return:         [bool]
    ----------------------------------------------------------------------------------------------------- """
    return model.input_shape[ -1 ] != 1



def out_rgb( model ):
    """ -----------------------------------------------------------------------------------------------------
    Return True if output is RGB, False if graylevel

    model:          [keras.engine.training.Model]

    return:         [bool]
    ----------------------------------------------------------------------------------------------------- """
    return model.output_shape[ -1 ] != 1



def array_to_image( array, rgb ):
    """ -----------------------------------------------------------------------------------------------------
    Convert numpy.ndarray to PIL.Image

    array:          [numpy.ndarray] pixel values
    rgb:            [bool] True if RGB, false if grayscale

    return:         [PIL.Image.Image]
    ----------------------------------------------------------------------------------------------------- """
    if len( array.shape ) == 4:
        array   = array[ 0, :, :, : ]                           # remove batch axis

    pixels  = array if rgb else array[ :, :, 0 ]
    pixels  = 255. * ( pixels - pixels.min() ) / pixels.ptp()   # normalization
    pixels  = np.uint8( pixels )

    if rgb:
        img     = Image.fromarray( pixels, 'RGB' )

    else:
        img     = Image.fromarray( pixels )
        img     = img.convert( 'RGB' )

    return img



def save_image( array, rgb, fname ):
    """ -----------------------------------------------------------------------------------------------------
    Save a pixel matrix in image file

    array:          [numpy.ndarray] pixel values
    rgb:            [bool] True if RGB, false if grayscale
    fname:          [str] path of output file
    ----------------------------------------------------------------------------------------------------- """
    img = array_to_image( array, rgb )
    img.save( fname )



def save_collage( imgs, w, h, fname, pad_size=5, pad_color="#aa0000" ):
    """ -----------------------------------------------------------------------------------------------------
    Combine a set of images into a collage

    imgs:           [list of PIL.Image.Image]
    w:              [int] desired width of single image tile inside the collage
    h:              [int] desired height of single image tile inside the collage
    pad_size:       [int] pixels between image tiles
    pad_color:      [str] padding color

    return:         [PIL.Image.Image]
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

    img.save( fname )
    return img



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
        except Exception as e:
            raise e

    # TODO when loading a model entirely from HDF5, it gives the error:
    # "the notion of "layer output" is ill-defined. Use `get_output_at(node_index)` instead."
    layers_out  = [ l.output for l in model.layers[ 1: ] ]

    return Model( inputs=model.get_input_at( 0 ), outputs=layers_out )



def attach_loss( model ):
    """ -----------------------------------------------------------------------------------------------------
    Attach a loss to an un-compiled model

    model:          [keras.engine.training.Model] original model, not returned by create_test_model()
    ----------------------------------------------------------------------------------------------------- """
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
    
    # optimizer is not relevant, but required by compile()
    model.compile( optimizer=optimizers.SGD(), loss=ls )



# ===========================================================================================================
#
#   - predict_image
#   - evaluate_image
#   - evaluate_tset
#   - test_samples
#
# ===========================================================================================================

def predict_image( model, img, save=False ):
    """ -----------------------------------------------------------------------------------------------------
    Return the model prediction, given an input image

    model:          [keras.engine.training.Model]
    img:            [str] path of image file
    save:           [bool] if True save input and output images

    return:         [numpy.ndarray] prediction
    ----------------------------------------------------------------------------------------------------- """
    try:
        i   = preprocessing.image.load_img(
                img,
                grayscale   = ( not in_rgb( model ) ),
                target_size = imgsize( model )
        )
    except Exception as e:
        raise e

    i       = preprocessing.image.img_to_array( i )

    if save:
        save_image( i, in_rgb( model ), os.path.join( cnfg[ 'dir_current' ], 'p_input.jpg' ) )

    i       = np.expand_dims( i, axis=0 )
    i       /= 255.
    pred    = model.predict( i )

    if save:
        save_image(
                pred,
                out_rgb( model ),
                os.path.join( cnfg[ 'dir_current' ], 'p_output.jpg' )
        )

    return pred



def evaluate_image( model, img_x, img_y ):
    """ -----------------------------------------------------------------------------------------------------
    Return the model loss, given an input image and its target image (ground truth)

    model:          [keras.engine.training.Model]
    img_x:          [str] path of input image file
    img_y:          [str] path of target image file

    return:         [float] loss
    ----------------------------------------------------------------------------------------------------- """
    try:
        i1  = preprocessing.image.load_img(
                img_x,
                grayscale   = ( not in_rgb( model ) ),
                target_size = imgsize( model )
        )
    except Exception as e:
        raise e

    i1      = preprocessing.image.img_to_array( i1 )
    i1      = np.expand_dims( i1, axis=0 )
    i1      /= 255.

    try:
        i2  = preprocessing.image.load_img(
                img_y,
                grayscale   = ( not out_rgb( model ) ),
                target_size = imgsize( model )
        )
    except Exception as e:
        raise e

    i2      = preprocessing.image.img_to_array( i2 )
    i2      = np.expand_dims( i2, axis=0 )
    i2      /= 255.

    loss    = model.evaluate( i1, i2, batch_size=1, verbose=0 )

    return loss



def evaluate_tset( model, input_dir, output_dir, fname="eval_test.txt"  ):
    """ -----------------------------------------------------------------------------------------------------
    Evaluate the model on a test set

    model:          [keras.engine.training.Model] original model
    input_dir:      [str] folder of input images
    output_dir:     [str] folder of target images
    fname:          [str] path of output file (if asked)

    return:         [dir] values:   [float] losses
                          keys:     [str] frame number
    ----------------------------------------------------------------------------------------------------- """
    if not os.path.isdir( input_dir ):
        raise ValueError( "{} directory does not exist".format( input_dir ) )

    if not os.path.isdir( output_dir ):
        raise ValueError( "{} directory does not exist".format( output_dir ) )

    pth         = os.path.join( cnfg[ 'dir_current' ], dir_test )
    if not os.path.exists( pth ):
        os.makedirs( pth )

    # if the model is not compiled yet
    if not hasattr( model, 'loss' ):
        attach_loss( model )

    d               = {}
    input_imgs      = sorted( [ f for f in os.listdir( input_dir )
            if f.lower().endswith( ( '.png', '.jpg', '.jpeg' ) ) ] )
    output_imgs     = sorted( [ f for f in os.listdir( output_dir )
            if f.lower().endswith( ( '.png', '.jpg', '.jpeg' ) ) ] )

    for ix, iy in zip( input_imgs, output_imgs ):
        frame       = ix.split( '_' )[ 0 ]
        img_x       = os.path.join( input_dir, ix )
        img_y       = os.path.join( output_dir, iy )
        loss        = evaluate_image( model, img_x, img_y )
        d[ frame ]  = loss

    if fname is not None:
        f   = open( os.path.join( pth, fname ), 'w' )
        for kv in sorted( d.items(), key=lambda x: x[ 1 ] ):
            f.write( '{}\t{}\n'.format( *kv ) )
        f.close()

    mean        = np.array( list( d.values() ) ).mean()
    d[ 'mean' ] = mean
        
    return d

    

def test_samples( model, data_class=None, dtest="dataset/dset_rgb" ):
    """ -----------------------------------------------------------------------------------------------------
    Test the model on a set of samples, and plot the predicted output

    model:          [keras.engine.training.Model]
    data_class:     [str] if None consider the standard autoencoder output
    dtest:          [str] dataset directory
    ----------------------------------------------------------------------------------------------------- """
    pth         = os.path.join( cnfg[ 'dir_current' ], dir_test )
    if not os.path.exists( pth ):
        os.makedirs( pth )

    # get names of all image files of the samples
    if data_class is not None:
        i_fmt       = "{:06d}_frame.jpg"
        t_fmt       = "{:06d}_{}.jpg"
        i_names     = [ os.path.join( dtest, i_fmt.format( f ) ) for f in samples[ data_class ] ]
        t_names     = [ os.path.join( dtest, t_fmt.format( f, data_class ) ) for f in samples[ data_class ] ]

    else:
        fmt         = "{:06d}_frame.jpg"
        i_names     = [ os.path.join( dtest, fmt.format( f ) ) for f in samples[ 'frame' ] ]
        t_names     = i_names

    targets     = [ Image.open( img ) for img in t_names ]                          # get ground truths
    outputs     = [ predict_image( model, img ) for img in i_names ]                # get predictions
    outputs     = [ array_to_image( o, out_rgb( model ) ) for o in outputs ]

    h, w        = imgsize( model )
    save_collage( targets, w, h, os.path.join( pth, "smpl_target.jpg" ) )
    save_collage( outputs, w, h, os.path.join( pth, "smpl_predict.jpg" ) )



# ===========================================================================================================
#
#   - layer_outputs
#   - layer_weights
#   - model_outputs
#   - model_weights
#   - model_dead
#
# ===========================================================================================================

def layer_outputs( model, layer, output, pth, normalize=True, prfx="out_" ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the output of a convolutional layer

    model:          [keras.engine.training.Model]
    layer:          [str] name of layer
    output:         [numpy.ndarray]
    pth:            [str] where to store results
    normalize:      [bool] if True, normalize each sub-plot individually
    prfx:           [str] prefix of output image files

    return:         [tuple] number of dead filters and total number of filters
    ----------------------------------------------------------------------------------------------------- """
    n_feat      = output.shape[ -1 ]
    h, w        = imgsize( model )
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
            pixels      = 255. * ( pixels - pixels.min() ) / ptp

        feat_list.append( Image.fromarray( pixels ) )

    fname   = prfx + layer + '.jpg'
    i       = save_collage( feat_list, w, h, os.path.join( pth, fname ) )

    return cnt, n_feat



def layer_weights( model, layer, wght, pth, dpi=10, normalize=False, prfx="wght_" ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the kernel weights of a convolutional layer

    model:          [keras.engine.training.Model]
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

        else:                                       # for normalizing the entire image collage
            mn      = min( pixels.min(), mn )
            mx      = max( pixels.max(), mx )

        wght_list.append( pixels )

    if not normalize:
        wght_list   = [ 255. * ( pixels - mn ) / ( mx - mn ) for pixels in wght_list ]

    wght_list   = [ Image.fromarray( pixels ) for pixels in wght_list ]

    fname   = prfx + layer + '.jpg'
    i       = save_collage( wght_list, w, h, os.path.join( pth, fname ) )



def model_outputs( model, img, dtest=None, normalize=True ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the output of all convolutional layers in the model.
    The argument 'img' can be the path of image file, or and integer index of a frame in the test set

    model:          [keras.engine.training.Model] the result of create_test_model()
    img:            [str or int] image file
    dtest:          [str] dataset directory (only when 'img' is an int)
    normalize:      [bool] if True, normalize each sub-plot individually
    ----------------------------------------------------------------------------------------------------- """
    pth     = os.path.join( cnfg[ 'dir_current' ], dir_plot )
    if not os.path.exists( pth ):
        os.makedirs( pth )

    if isinstance( img, int ):
        try:
            img =  os.path.join( dtest, os.listdir( dtest )[ img ] )
        except Exception as e:
            raise e

    if not os.path.isfile( img ):
        ms.print_err( "while opening file " + img )

    cnt     = 0
    tot     = 0
    lay     = [ l.name for l in model.layers[ 1: ] ]
    out     = predict_image( model, img )

    for l, o in zip( lay, out ):
        if 'conv' in l:             # plot output only for convolutional layers
            c, t    = layer_outputs( model, l, o, pth, normalize=normalize )
            cnt     += c
            tot     += t

    if DEBUG0:
        ms.print_msg( "Total dead filters percentage: {:.1f}%".format( 100 * cnt / tot ) )



def model_weights( model, normalize=False ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the kernel weights of all convolutional layers in the model

    model:          [keras.engine.training.Model]
    normalize:      [bool] if True, normalize each sub-plot individually
    ----------------------------------------------------------------------------------------------------- """
    pth     = os.path.join( cnfg[ 'dir_current' ], dir_plot )
    if not os.path.exists( pth ):
        os.makedirs( pth )

    # NOTE get_weights returns a list of (weights,biases)
    wght    = [ l.get_weights() for l in model.layers[ 1 : ] ]
    lay     = [ l.name for l in model.layers[ 1: ] ]

    for l, w in zip( lay, wght ):
        if 'conv' in l:             # plot weights only for convolutional layers
            layer_weights( model, l, w[ 0 ], pth, normalize=normalize )



def model_dead( model, layer, n_img, dtest ):
    """ -----------------------------------------------------------------------------------------------------
    Count how many features of a layer are inactive, testing the prediction on serevar input images

    model:          [keras.engine.training.Model] the result of create_test_model()
    layer:          [int] index of layer
    n_img:          [int] number of input images
    dtest:          [str] dataset directory
    ----------------------------------------------------------------------------------------------------- """

    # sample 'n_img' images, evenly spaced, from the test set
    dir_list    = [ f for f in os.listdir( dtest ) if f.lower().endswith( ( '.png', '.jpg', '.jpeg' ) ) ]
    indx        = list( map( int, np.linspace( 0, len( dir_list ), n_img, endpoint=False ) ) )
    img_list    = [ os.path.join( dtest, dir_list[ i ] ) for i in indx ]

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
