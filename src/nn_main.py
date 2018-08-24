"""
#############################################################################################################

Standard AutoEncoder architecture

    Alice   2018

#############################################################################################################
"""

# TODO try load all layers, not only the one untrainable ! ! !

# TODO check better handling of error prints and exceptions
# TODO comment global arguments
# TODO Gabor?


import  os
import  sys
import  time

import  msg         as ms
import  cnfg        as cf
import  nn_arch     as na
import  nn_train    as nt
import  nn_test     as ns
import  nn_dset     as nd


FRMT                = "%y-%m-%d_%H-%M-%S"

dset_root           = "dataset"
dset_gray           = os.path.join( dset_root, "dset_gray" )
dset_rgb            = os.path.join( dset_root, "dset_rgb" )

dir_cnfg            = "config"
dir_res             = "res"
dir_current         = None

train_err           = "train.err"
train_log           = "train.log"
train_time          = "train.time"



# ===========================================================================================================


# create directory for the current execution
dir_current         = os.path.join( dir_res, time.strftime( FRMT ) )
os.makedirs( dir_current )

# read configs from command line arguments
args        = cf.get_args()

# redirect stderr in log file
if args[ 'STDERR' ]:
    log         = os.path.join( dir_current, train_err )
    sys.stderr  = open( log, 'w' )

# -----------------------------------------------------------------------------------------------------------

# get configs
cnfg                    = cf.get_config( args[ 'CONFIG' ] )
cnfg[ 'dir_dset' ]      = os.path.join( dset_root, cnfg[ 'dir_dset' ] )
cnfg[ 'dir_current' ]   = dir_current

cf.load_config( cnfg, na.cnfg )
cf.load_config( cnfg, nt.cnfg )
cf.load_config( cnfg, ns.cnfg )
cf.load_config( cnfg, nd.cnfg )

# -----------------------------------------------------------------------------------------------------------

if args[ 'LOAD' ] is not None:
    nn  = na.load_model( args[ 'LOAD' ] )
else:
    nn  = na.create_model()

na.model_graph( nn )
na.model_summary( nn )

# -----------------------------------------------------------------------------------------------------------

if args[ 'TRAIN' ]:
    # redirect stdout in log file
    log         = os.path.join( dir_current, train_log )
    sys.stdout  = open( log, 'w' )
    
    # train model
    history     = nt.train_model( nn, train_time )
    nt.plot_history( history )
    na.save_model( nn )

    # restore stdout
    sys.stdout  = sys.__stdout__

# -----------------------------------------------------------------------------------------------------------

if args[ 'TEST' ]:
    dir_test    = os.path.join( cnfg[ 'dir_dset' ], 'test' )
    nn_test     = ns.create_test_model( nn )
    
    if cnfg[ 'arch' ] == 'AE_SEGM':
        test_frame  = os.path.join( dir_test, 'frame', 'img' )
        test_class  = os.path.join( dir_test, cnfg[ 'data_class' ].lower(), 'img' )

        ns.test_samples( nn, cnfg[ 'data_class' ].lower() )
        ns.evaluate_tset( nn, test_frame, test_class )

    if cnfg[ 'arch' ] == 'AE_SIMPLE':
        test_frame  = os.path.join( dir_test, 'img' )

        ns.test_samples( nn )
        ns.evaluate_tset( nn, test_frame, test_frame )
        ns.model_dead( nn_test, 1, 1, test_frame )

    ns.model_outputs( nn_test, 1, dtest=test_frame )
    ns.model_weights( nn )

# -----------------------------------------------------------------------------------------------------------

if args[ 'ARCHIVE' ] > 0:
    # save config files
    if args[ 'ARCHIVE' ] >= 1:
        os.makedirs( os.path.join( dir_current, 'config' ) )
        cfile   =  args[ 'CONFIG' ]
        os.system( "cp {} {}".format( cfile, os.path.join( dir_current, 'config' ) ) )

    # save python sources
    if args[ 'ARCHIVE' ] >= 2:
        os.makedirs( os.path.join( dir_current, 'src' ) )
        pfile   = "src/*.py"
        os.system( "cp {} {}".format( pfile, os.path.join( dir_current, 'src' ) ) )

"""
if __name__ == '__main__':
    main( sys.argv )
"""
