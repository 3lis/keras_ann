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


DEBUG0  = False         # enable debugging print



def get_args():
    """ -----------------------------------------------------------------------------------------------------
    Parse the command-line arguments defined by flags
    
    return:         [dict] args and their values
    ----------------------------------------------------------------------------------------------------- """
    parser      = argparse.ArgumentParser()

    parser.add_argument(
            '-c',
            '--config',
            action          = 'store',
            dest            = 'CONFIG',
            type            = str,
            required        = True,
            help            = "config file describing the model architecture and training parameters"
    )
    parser.add_argument(
            '-l',
            '--load',
            action          = 'store',
            dest            = 'LOAD',
            type            = str,
            help            = "Folder or HDF5 file to load as weights or entire model"
    )
    parser.add_argument(
            '-T',
            '--train',
            action          = 'store_true',
            dest            = 'TRAIN',
            help            = "execute training of the model"
    )
    parser.add_argument(
            '-t',
            '--test',
            action          = 'store_true',
            dest            = 'TEST',
            help            = "execute testing routines"
    )
    parser.add_argument(
            '-e',
            '--err',
            action          = 'store_true',
            dest            = 'STDERR',
            help            = "redirect stderr to log file"
    )
    parser.add_argument(
            '-s',
            '--save',
            action          = 'count',
            dest            = 'ARCHIVE',
            default         = 0,
            help            = "archive config files [-s] or even python scripts [-ss]"
    )

    return vars( parser.parse_args() )



def get_config( fname ):
    """ -----------------------------------------------------------------------------------------------------
    Return the content of a config file in the form of a dictionary

    fname:          [str] config file (full path)

    return:         [dict] content of the file
    ----------------------------------------------------------------------------------------------------- """
    cnfg    = dict()

    if not os.path.isfile( fname ):
        ms.print_err( "configuration file \"{}\" not found.".format( fname ) )

    if DEBUG0:
        ms.print_msg( "Reading configuration file \"{}\".\n".format( fname ) )
        os.system( "cat %s" % fname )

    with open( fname ) as doc:
        for line in doc:
            if line[ 0 ] == '#': continue   # comment line

            c   = line.rsplit()
            if len( c ) == 0: continue      # empty line

            cnfg[ c[ 0 ] ] = eval( str().join( c[ 1: ] ) )

    return cnfg



def load_config( cnfg, dest ):
    """ -----------------------------------------------------------------------------------------------------
    Use the first dict to fill the value of the second dict, in case of common keys

    cnfg:           [dict] one with all configs
    dest:           [dict] one to be filled
    ----------------------------------------------------------------------------------------------------- """
    for k in dest.keys():
        if k in cnfg.keys():
            dest[ k ]   = cnfg[ k ]
