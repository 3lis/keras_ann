"""
#############################################################################################################

Utilities for loading configurations from command line arguments

The passed argumends should be text files, in which each line can contain:
    - a comment, starting with char '#'
    - an empty line
    - the name of the variable and its value

The content of those files should be fed to the configuration dicitionaries in the main python scripts.

    Alice   2018

#############################################################################################################
"""

import  os
import  sys
import  argparse

import  msg     as ms


DEBUG0          = False         # enable debugging print



def get_args():
    """ -------------------------------------------------------------------------------------------------
    Parse the command-line arguments defined by flags
    
    return:         [dict] args and their values
    ------------------------------------------------------------------------------------------------- """
    parser      = argparse.ArgumentParser()

    parser.add_argument(
            '-a',
            '--arch',
            action          = 'store',
            dest            = 'ARCH',
            type            = str,
            required        = True,
            help            = "config file describing the model architecture"
    )
    parser.add_argument(
            '-t',
            '--train',
            action          = 'store',
            dest            = 'TRAIN',
            type            = str,
            default         = None,
            help            = "config file describing the training parameters"
    )
    parser.add_argument(
            '-l',
            '--load',
            action          = 'store',
            dest            = 'LOAD',
            type            = str,
            default         = None,
            help            = "HDF5 file of model weights"
    )
    parser.add_argument(
            '-s',
            '--test',
            action          = 'store_true',
            dest            = 'TEST',
            help            = "execute testing routines"
    )
    parser.add_argument(
            '-x',
            '--save',
            action          = 'count',
            dest            = 'ARCHIVE',
            default         = 0,
            help            = "archive config files [-x] or even python scripts [-xx]"
    )

    return vars( parser.parse_args() )



def load_config( fname, cnfg ):
    """ -------------------------------------------------------------------------------------------------
    Load the content of a config file, and put the information in the config dictionary

    fname:          [str] config file (full path)
    cnfg:           [dict] to be filled with the content of the file
    ------------------------------------------------------------------------------------------------- """
    if not os.path.isfile( fname ):
        ms.print_err( "configuration file \"{}\" not found.".format( fname ) )

    if DEBUG0:
        ms.print_msg( "Reading configuration file \"{}\".\n".format( fname ) )
        os.system( "cat %s" % fname )

    with open( fname ) as doc:
        for line in doc:
            if line[ 0 ] == '#':        # comment line
                continue

            c   = line.rsplit()
            if len( c ) == 0:           # empty line
                continue

            if c[ 0 ] in cnfg:
                cnfg[ c[ 0 ] ] = eval( str().join( c[ 1: ] ) )
            else:
                ms.print_err( "configuration setting \"{}\" does not exist.".format( c[ 0 ] ) )
