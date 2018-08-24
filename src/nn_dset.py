"""
#############################################################################################################

Functions for reading the dataset

    The sub-dataset must have a structure like these:
        * dir_dset/train/frames/
        * dir_dset/valid/frames/
        * dir_dset/test/frames/

    Alice   2018

#############################################################################################################
"""

import  os
from    keras       import preprocessing

import  msg         as ms



cnfg            = {
    'img_size'      : None,     # [list] 3 dimensions, even in the case of grayscale
    'batch_size'    : None,     # [int]
    'data_class'    : None      # [str]
}


# ===========================================================================================================
#
#   - iter_simple
#   - iter_double
#
# ===========================================================================================================

def iter_simple( dr, mode, color, shuffle=True, seed=None ):
    """ -----------------------------------------------------------------------------------------------------
    Simple generic dataset iterator

    https://keras.io/preprocessing/image/#imagedatagenerator-methods

    dr:             [str] folder of dataset (it must contain a subfolder for each class)
    mode:           [str] type of target outputs (labels) in the dataset
    color:          [str] 'rgb' or 'grayscale'
    shuffle:        [bool] whether to shuffle the data
    seed:           [int]

    return:         [keras_preprocessing.image.DirectoryIterator]
    ----------------------------------------------------------------------------------------------------- """
    
    # 'rescale' to normalize pixels in [0..1]
    idg     = preprocessing.image.ImageDataGenerator( rescale=1./255 )

    flow    = idg.flow_from_directory(
            directory   = dr,
            target_size = cnfg[ 'img_size' ][ :-1 ],
            color_mode  = color,
            class_mode  = mode,
            batch_size  = cnfg[ 'batch_size' ],
            shuffle     = shuffle,
            seed        = seed
    )

    return flow



def iter_double( dir_in, dir_out, color_in, color_out ):
    """ -----------------------------------------------------------------------------------------------------
    Custom generic dataset iterator, combining two Iterators

    dir_in:         [str] folder of inputs
    dir_out:        [str] folder of target outputs (labels)
    color_in:       [str] 'rgb' or 'grayscale'
    color_out       [str] 'rgb' or 'grayscale'

    return:         [Generator]
    ----------------------------------------------------------------------------------------------------- """

    # use mode=None when creating custom DirectoryIterator
    flow_in     = iter_simple( dir_in, None, color_in, shuffle=True, seed=1 )
    flow_out    = iter_simple( dir_out, None, color_out, shuffle=True, seed=1 )
    
    while True:
        inp     = flow_in.next()
        out     = flow_out.next()

        yield [ inp, out ]



# ===========================================================================================================
#
#   - dset_same
#   - dset_class
#
# ===========================================================================================================

def dset_same( dir_dset ):
    """ -----------------------------------------------------------------------------------------------------
    Dataset where the target image is equal to the input image

    dir_dset:       [str] folder of dataset (it must contain subfolders for train/valid)

    return:         [list] of keras_preprocessing.image.DirectoryIterator
    ----------------------------------------------------------------------------------------------------- """
    color       = 'grayscale' if cnfg[ 'img_size' ][ -1 ] == 1 else 'rgb'

    train_dir   = os.path.join( dir_dset, 'train' )
    valid_dir   = os.path.join( dir_dset, 'valid' )

    # use mode=input when target is equal to input
    train_flow  = iter_simple( train_dir, 'input', color, shuffle=True )
    valid_flow  = iter_simple( valid_dir, 'input', color, shuffle=True )

    return train_flow, valid_flow



def dset_class( dir_dset, data_class ):
    """ -----------------------------------------------------------------------------------------------------
    Dataset where the target image is the segmented image of the class specified

    dir_dset:       [str] folder of dataset (it must contain subfolders for train/valid)
    data_class:     [str] class of segmentation (cars, lane)

    return:         [list] of Generator
    ----------------------------------------------------------------------------------------------------- """
    color_in    = 'grayscale' if cnfg[ 'img_size' ][ -1 ] == 1 else 'rgb'
    color_out   = 'grayscale'

    train_dir_in    = os.path.join( dir_dset, 'train', 'frame' )
    train_dir_out   = os.path.join( dir_dset, 'train', data_class )
    valid_dir_in    = os.path.join( dir_dset, 'valid', 'frame' )
    valid_dir_out   = os.path.join( dir_dset, 'valid', data_class )

    train_flow      = iter_double( train_dir_in, train_dir_out, color_in, color_out )
    valid_flow      = iter_double( valid_dir_in, valid_dir_out, color_in, color_out )

    return train_flow, valid_flow



# ===========================================================================================================
#
#   - len_dataset
#   - gen_dataset
#
# ===========================================================================================================

def len_dataset( dir_dset ):
    """ -----------------------------------------------------------------------------------------------------
    Return the number of samples in each subset

    dir_dset:       [str] folder of dataset (it must contain subfolders for train/valid)

    return:         [list]
    ----------------------------------------------------------------------------------------------------- """
    train   = None
    valid   = None
    test    = None

    for dirpath, dirname, filename in os.walk( dir_dset ):
        if not dirname:
            if "train" in dirpath:
                train   = dirpath
            elif "valid" in dirpath:
                valid   = dirpath
            elif "test" in dirpath:
                test    = dirpath

    tr  = len( [ f for f in os.listdir( train ) if f.lower().endswith( ( '.png', '.jpg', '.jpeg' ) ) ] )
    vl  = len( [ f for f in os.listdir( valid ) if f.lower().endswith( ( '.png', '.jpg', '.jpeg' ) ) ] )
    ts  = len( [ f for f in os.listdir( test ) if f.lower().endswith( ( '.png', '.jpg', '.jpeg' ) ) ] )

    return tr, vl, ts



def gen_dataset( dir_dset ):
    """ -----------------------------------------------------------------------------------------------------
    Take the path to a directory and generate batches of data

    dir_dset:       [str] folder of dataset (it must contain subfolders for train/valid)

    return:         [list]
    ----------------------------------------------------------------------------------------------------- """
    if cnfg[ 'data_class' ] == 'FRAME':
        return dset_same( dir_dset )

    if cnfg[ 'data_class' ] in ( 'LANE', 'CARS' ):
        return dset_class( dir_dset, cnfg[ 'data_class' ].lower() )

    ms.print_err( "Data class {} not valid".format( cnfg[ 'data_class' ] ) )
