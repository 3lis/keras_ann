"""
#############################################################################################################

Standard AutoEncoder architecture

    Alice   2018

#############################################################################################################
"""

import  os
import  sys
import  time

import  msg         as ms
import  cnfg        as cf
import  nn_arch     as na
import  nn_train    as nt
import  nn_test     as ns


FRMT                = "%y-%m-%d_%H-%M-%S"
STDERR              = True

dir_dset            = "dataset"
dir_cnfg            = "config"
dir_res             = "res"
dir_current         = None

dir_test            = os.path.join( dir_dset, "frames_L/test/frames" )  # overwritten in case of TRAIN=True


# ===========================================================================================================


# create directory for the current execution
dir_current         = os.path.join( dir_res, time.strftime( FRMT ) )
os.makedirs( dir_current )

# redirect stderr in log file
if STDERR:
    log         = os.path.join( dir_current, 'train.err' )
    sys.stderr  = open( log, 'w' )

# -----------------------------------------------------------------------------------------------------------

# read configs from command line arguments
args        = cf.get_args()

# load architecture config
cf.load_config( os.path.join( dir_cnfg, args[ 'ARCH' ] ), na.cnfg )
na.RGB              = na.cnfg[ 'img_size' ][ -1 ] > 1
na.dir_current      = dir_current

# create network model
nn      = na.create_model()
na.model_graph( nn )

# -----------------------------------------------------------------------------------------------------------

if args[ 'TRAIN' ] is not None:
    # redirect stdout in log file
    log         = os.path.join( dir_current, 'train.log' )
    sys.stdout  = open( log, 'w' )
    
    # load training config
    cf.load_config( os.path.join( dir_cnfg, args[ 'TRAIN' ] ), nt.cnfg )
    nt.cnfg[ 'dir_dset' ]   = os.path.join( dir_dset, nt.cnfg[ 'dir_dset' ] )
    nt.RGB                  = na.RGB
    nt.img_size             = na.cnfg[ 'img_size' ]
    nt.dir_current          = dir_current
    dir_test                = os.path.join( nt.cnfg[ 'dir_dset' ], 'test/frames' )

    # train model
    nt.train_model( nn )
    nt.save_model( nn )

    # restore stdout
    sys.stdout  = sys.__stdout__

# -----------------------------------------------------------------------------------------------------------

# TODO put it on top and load entire folder
if args[ 'LOAD' ] is not None:
    nn.load_weights( args[ 'LOAD' ] )

# -----------------------------------------------------------------------------------------------------------

if args[ 'TEST' ]:
    # set up test config
    ns.RGB              = na.RGB
    ns.img_size         = na.cnfg[ 'img_size' ]
    ns.dir_current      = dir_current

    # test routines
    nn_test = ns.create_test_model( nn )
    ns.model_dead( nn_test, 1, 100, dir_test )

    ns.model_outputs( nn_test, 1, dir_test=dir_test )
    #ns.model_outputs( nn_test, "i.png" )

    ns.model_weights( nn )

# -----------------------------------------------------------------------------------------------------------

if args[ 'ARCHIVE' ] > 0:
    # save config files
    if args[ 'ARCHIVE' ] >= 1:
        os.makedirs( os.path.join( dir_current, 'config' ) )
        cfile   = os.path.join( dir_cnfg, args[ 'ARCH' ] )
        if args[ 'TRAIN' ] is not None:
            cfile   += ' ' + os.path.join( dir_cnfg, args[ 'TRAIN' ] )
        os.system( "cp {} {}".format( cfile, os.path.join( dir_current, 'config' ) ) )

    # save python sources
    if args[ 'ARCHIVE' ] >= 2:
        os.makedirs( os.path.join( dir_current, 'src' ) )
        pfile   = "cnfg.py msg.py nn_arch.py nn_main.py nn_test.py nn_train.py"
        os.system( "cp {} {}".format( pfile, os.path.join( dir_current, 'src' ) ) )


