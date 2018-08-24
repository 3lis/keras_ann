"""
#############################################################################################################

Utilities for printing messages

    Alice   2018

#############################################################################################################
"""

import  os
import  sys
import  inspect



def print_err( msg ):
    """ -----------------------------------------------------------------------------------------------------
    Print an error messagge in stderr, including the file and line number where the print is executed

    msg:        [str] message to print
    ----------------------------------------------------------------------------------------------------- """
    LINE    = inspect.currentframe().f_back.f_lineno
    FILE    = os.path.basename( inspect.getfile( inspect.currentframe().f_back ) )

    sys.stderr.write( "ERROR [{}:{}] --> {}\n".format( FILE, LINE, msg ) )
    sys.exit( 1 )



def print_wrn( msg ):
    """ -----------------------------------------------------------------------------------------------------
    Print a warning messagge in stderr, including the file and line number where the print is executed

    msg:        [str] message to print
    ----------------------------------------------------------------------------------------------------- """
    LINE    = inspect.currentframe().f_back.f_lineno
    FILE    = os.path.basename( inspect.getfile( inspect.currentframe().f_back ) )

    sys.stderr.write( "WARNING [{}:{}] --> {}\n".format( FILE, LINE, msg ) )



def print_msg( msg ):
    """ -----------------------------------------------------------------------------------------------------
    Print a messagge in log file

    msg:        [str] message to print
    ----------------------------------------------------------------------------------------------------- """
    sys.stderr.write( msg + '\n' )



def print_line( l=70 ):
    """ -----------------------------------------------------------------------------------------------------
    Print a separation line
    ----------------------------------------------------------------------------------------------------- """
    sys.stdout.write( "\n# " + ( l * '=' ) + " #\n" )
