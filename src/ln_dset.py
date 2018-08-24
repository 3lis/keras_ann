"""
#############################################################################################################

Utilities for handling the dataset structures, by creating symbolic links

    Alice   2018

#############################################################################################################
"""

import  os
import  sys
import  shutil
import  random
import  numpy   as np

from    PIL     import Image

frac_train      = lambda x: int( 0.8 * x )
frac_valid      = lambda x: int( 0.1 * x )
frac_test       = lambda x: int( 0.1 * x )

indx_train      = lambda x: slice( 0, frac_train( x ) )
indx_valid      = lambda x: slice( frac_train( x ), frac_train( x ) + frac_valid( x ) )
indx_test       = lambda x: slice( frac_train( x ) + frac_valid( x ), None )

root_dir        = 'dataset'
dset_rgb        = os.path.join( root_dir, 'dset_rgb' )
dset_gray       = os.path.join( root_dir, 'dset_gray' )


def make_symlink( src_dir, dest_dir, files ):
    """ -----------------------------------------------------------------------------------------------------
    Make symbolic links of files from a folder to a second one

    src_dir:        [str] source folder
    dest_dir:       [str] destination folder
    files:          [list of str] name of files to be linked
    ----------------------------------------------------------------------------------------------------- """
    src_rel     = os.path.relpath( src_dir, dest_dir )      # src path relative to dest

    for f in files:
        os.symlink( os.path.join( src_rel, f ), os.path.join( dest_dir, f ) )



def gray_same( dest_dir, size=None ):
    """ -----------------------------------------------------------------------------------------------------
    Create dataset with graylevel frames, with no categories (input equal to output)

    dest_dir:       [str] destination folder
    size:           [int] amount of file to link (if None consider all files)
    ----------------------------------------------------------------------------------------------------- """
    src_dir     = dset_gray
    dest_dir    = os.path.join( root_dir, dest_dir )

    dest_train  = os.path.join( dest_dir, 'train/img' )
    dest_valid  = os.path.join( dest_dir, 'valid/img' )
    dest_test   = os.path.join( dest_dir, 'test/img' )
    
    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )
    os.makedirs( dest_train )
    os.makedirs( dest_valid )
    os.makedirs( dest_test )

    cond        = lambda x: x.endswith( '.png' )
    files       = [ f for f in sorted( os.listdir( src_dir ) ) if cond( f ) ]

    if size is None:
        size    = len( files )

    random.seed( 1 )
    files       = random.sample( files, size )              # random permutation

    files_train = files[ indx_train( size ) ]
    files_valid = files[ indx_valid( size ) ]
    files_test  = files[ indx_test( size ) ]

    make_symlink( src_dir, dest_train, files_train )
    make_symlink( src_dir, dest_valid, files_valid )
    make_symlink( src_dir, dest_test, files_test )



def rgb_same( dest_dir, size=None ):
    """ -----------------------------------------------------------------------------------------------------
    Create dataset with RGB frames, with no categories (input equal to output)

    dest_dir:       [str] destination folder
    size:           [int] amount of file to link (if None consider all files)
    ----------------------------------------------------------------------------------------------------- """
    src_dir     = dset_rgb
    dest_dir    = os.path.join( root_dir, dest_dir )

    dest_train  = os.path.join( dest_dir, 'train/img' )
    dest_valid  = os.path.join( dest_dir, 'valid/img' )
    dest_test   = os.path.join( dest_dir, 'test/img' )
    
    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )
    os.makedirs( dest_train )
    os.makedirs( dest_valid )
    os.makedirs( dest_test )

    cond        = lambda x: x.endswith( '_frame.jpg' )
    files       = [ f for f in sorted( os.listdir( src_dir ) ) if cond( f ) ]

    if size is None:
        size    = len( files )

    random.seed( 1 )
    files       = random.sample( files, size )              # random permutation

    files_train = files[ indx_train( size ) ]
    files_valid = files[ indx_valid( size ) ]
    files_test  = files[ indx_test( size ) ]

    make_symlink( src_dir, dest_train, files_train )
    make_symlink( src_dir, dest_valid, files_valid )
    make_symlink( src_dir, dest_test, files_test )



def rgb_segm( dest_dir, size=None ):
    """ -----------------------------------------------------------------------------------------------------
    Create dataset with RGB frames, with corresponding 'cars' and 'lane' segmented frames

    dest_dir:       [str] destination folder
    size:           [int] amount of file to link (if None consider all files)
    ----------------------------------------------------------------------------------------------------- """
    src_dir         = dset_rgb
    dest_dir        = os.path.join( root_dir, dest_dir )

    dest_train_f    = os.path.join( dest_dir, 'train/frame/img' )
    dest_train_c    = os.path.join( dest_dir, 'train/cars/img' )
    dest_train_l    = os.path.join( dest_dir, 'train/lane/img' )

    dest_valid_f    = os.path.join( dest_dir, 'valid/frame/img' )
    dest_valid_c    = os.path.join( dest_dir, 'valid/cars/img' )
    dest_valid_l    = os.path.join( dest_dir, 'valid/lane/img' )

    dest_test_f     = os.path.join( dest_dir, 'test/frame/img' )
    dest_test_c     = os.path.join( dest_dir, 'test/cars/img' )
    dest_test_l     = os.path.join( dest_dir, 'test/lane/img' )
    
    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )
    os.makedirs( dest_train_f )
    os.makedirs( dest_train_c )
    os.makedirs( dest_train_l )
    os.makedirs( dest_valid_f )
    os.makedirs( dest_valid_c )
    os.makedirs( dest_valid_l )
    os.makedirs( dest_test_f )
    os.makedirs( dest_test_c )
    os.makedirs( dest_test_l )

    cond_f      = lambda x: x.endswith( '_frame.jpg' )
    cond_c      = lambda x: x.endswith( '_cars.jpg' )
    cond_l      = lambda x: x.endswith( '_lane.jpg' )

    files_f     = []
    files_c     = []
    files_l     = []

    for f in sorted( os.listdir( src_dir ) ):
        if cond_f( f ):
            files_f.append( f )
        elif cond_c( f ):
            files_c.append( f )
        elif cond_l( f ):
            files_l.append( f )

    files_f         = np.array( files_f )
    files_c         = np.array( files_c )
    files_l         = np.array( files_l )

    if size is None:
        size    = len( files_f )

    random.seed( 1 )
    indx            = random.sample( range( size ), size )      # random permutation
    files_f         = files_f[ indx ]
    files_c         = files_c[ indx ]
    files_l         = files_l[ indx ]

    files_train_f   = files_f[ indx_train( size ) ]
    files_valid_f   = files_f[ indx_valid( size ) ]
    files_test_f    = files_f[ indx_test( size ) ]

    files_train_c   = files_c[ indx_train( size ) ]
    files_valid_c   = files_c[ indx_valid( size ) ]
    files_test_c    = files_c[ indx_test( size ) ]

    files_train_l   = files_l[ indx_train( size ) ]
    files_valid_l   = files_l[ indx_valid( size ) ]
    files_test_l    = files_l[ indx_test( size ) ]

    make_symlink( src_dir, dest_train_f, files_train_f )
    make_symlink( src_dir, dest_valid_f, files_valid_f )
    make_symlink( src_dir, dest_test_f, files_test_f )

    make_symlink( src_dir, dest_train_c, files_train_c )
    make_symlink( src_dir, dest_valid_c, files_valid_c )
    make_symlink( src_dir, dest_test_c, files_test_c )

    make_symlink( src_dir, dest_train_l, files_train_l )
    make_symlink( src_dir, dest_valid_l, files_valid_l )
    make_symlink( src_dir, dest_test_l, files_test_l )



def check_valid( img, threshold ):
    """ -----------------------------------------------------------------------------------------------------
    Check if an image has at least 'threshold' white pixels

    img:            [str] path of image file
    threshold:      [int] required number of white pixels
    ----------------------------------------------------------------------------------------------------- """
    f       = Image.open( img )
    i       = np.array( f )
    f.close()
    return ( i == 255 ).sum() > threshold



def rgb_segm_valid( dest_dir, data_class, threshold=5, batch_size=100, size=None ):
    """ -----------------------------------------------------------------------------------------------------
    Create dataset with RGB frames and corrisponding segmented frames of one class.
    Only frames with valid (non-black) ground truth will be considered.

    dest_dir:       [str] destination folder
    data_class:     [str] class of segmentation (cars, lane)
    threshold:      [int] required number of white pixels
    batch_size:     [int] batch size
    size:           [int] amount of files to link (if None consider all files)
    ----------------------------------------------------------------------------------------------------- """
    src_dir         = dset_rgb
    dest_dir        = os.path.join( root_dir, dest_dir )

    dest_train_f    = os.path.join( dest_dir, 'train/frame/img' )
    dest_train_c    = os.path.join( dest_dir, 'train/' + data_class + '/img' )

    dest_valid_f    = os.path.join( dest_dir, 'valid/frame/img' )
    dest_valid_c    = os.path.join( dest_dir, 'valid/' + data_class + '/img' )

    dest_test_f     = os.path.join( dest_dir, 'test/frame/img' )
    dest_test_c     = os.path.join( dest_dir, 'test/' + data_class + '/img' )
    
    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )
    os.makedirs( dest_train_f )
    os.makedirs( dest_train_c )
    os.makedirs( dest_valid_f )
    os.makedirs( dest_valid_c )
    os.makedirs( dest_test_f )
    os.makedirs( dest_test_c )

    cond_c      = lambda x: x.endswith( '_' + data_class + '.jpg' )

    files_f     = []
    files_c     = []

    for f in sorted( os.listdir( src_dir ) ):
        if cond_c( f ):
            if check_valid( os.path.join( src_dir, f ), threshold ):
                files_c.append( f )
                files_f.append( f.replace( data_class, "frame" ) )

    files_f         = np.array( files_f )
    files_c         = np.array( files_c )

    if size is None:
        size    = batch_size * ( len( files_f ) // batch_size )

    random.seed( 1 )
    indx            = random.sample( range( size ), size )      # random permutation
    files_f         = files_f[ indx ]
    files_c         = files_c[ indx ]

    files_train_f   = files_f[ indx_train( size ) ]
    files_valid_f   = files_f[ indx_valid( size ) ]
    files_test_f    = files_f[ indx_test( size ) ]

    files_train_c   = files_c[ indx_train( size ) ]
    files_valid_c   = files_c[ indx_valid( size ) ]
    files_test_c    = files_c[ indx_test( size ) ]

    make_symlink( src_dir, dest_train_f, files_train_f )
    make_symlink( src_dir, dest_valid_f, files_valid_f )
    make_symlink( src_dir, dest_test_f, files_test_f )

    make_symlink( src_dir, dest_train_c, files_train_c )
    make_symlink( src_dir, dest_valid_c, files_valid_c )
    make_symlink( src_dir, dest_test_c, files_test_c )


# ===========================================================================================================


if __name__ == '__main__':
    # NOTE to be executed from the main folder (above 'src' and 'dataset')

    print( "Executing:", "gray_frame_S" )
    gray_same( "gray_frame_S", 100 )

    print( "Executing:", "gray_frame_L" )
    gray_same( "gray_frame_L" )

    print( "Executing:", "rgb_frame_S" )
    rgb_same( "rgb_frame_S", 100 )

    print( "Executing:", "rgb_frame_L" )
    rgb_same( "rgb_frame_L" )


    print( "Executing:", "rgb_lane_S" )
    rgb_segm_valid( "rgb_lane_S", 'lane', size=100 )

    print( "Executing:", "rgb_lane_L" )
    rgb_segm_valid( "rgb_lane_L", 'lane' )

    print( "Executing:", "rgb_cars_S" )
    rgb_segm_valid( "rgb_cars_S", 'cars', size=100 )

    print( "Executing:", "rgb_cars_L" )
    rgb_segm_valid( "rgb_cars_L", 'cars' )


# not used anymore
"""
rgb_segm( "rgb_segm_S", 100 )
rgb_segm( "rgb_segm_L" )
"""
