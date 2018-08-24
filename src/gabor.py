"""
#############################################################################################################

Mathematica-like Gabor wavelets

#############################################################################################################
"""

import  numpy       as np
from    numpy       import sin, cos, exp, sqrt, pi
from    PIL         import Image




def wght_to_img( wght, fname='k.jpg' ):
    """ -----------------------------------------------------------------------------------------------------
    Convert a (single) matix of weights into an image, for visualization

    wght:           [numpy.array] 4D matrix of shape (?, ?, 1, 1)
    fname:          [str]
    ----------------------------------------------------------------------------------------------------- """
    pixels  = wght[ :, :, 0, 0 ]
    pixels  = 255. * ( pixels - pixels.min() ) / pixels.ptp()   # normalization
    img     = Image.fromarray( pixels )
    img     = img.convert( 'RGB' )
    img.save( fname )



def f_gabor( x, s, k, f ):
    """ -----------------------------------------------------------------------------------------------------
    basic Gabor function of a single point, in the same format as from
    http://reference.wolfram.com/language/ref/GaborMatrix.html (Details and Options)

    x:  input, as integer indexes of position from center [tuple]
    s:  sigma [scalar]
    k:  wave vector [tuple]
    f:  phase shift [scalar]
    ----------------------------------------------------------------------------------------------------- """
    x1, x2  = x
    k1, k2  = k
    e       = exp( - ( x1 ** 2 + x2 ** 2 ) / ( 2 * s ** 2 ) )
    c       = cos( k1 * x1 + k2 * x2 - f )
    return e * c



def gabor_matrix_unscaled( r, s, k, f ):
    """ -----------------------------------------------------------------------------------------------------
    gives a matrix with values proportional to Mathematica GaborMatrix[ { r, s }, k, f ]
    ----------------------------------------------------------------------------------------------------- """

    rng     = np.arange( -r, r + 1 )
    xr, yr  = np.meshgrid( rng, rng )
    return f_gabor( ( xr, yr ), s, k, f )



def gabor_matrix( r, s, k, f ):
    """ -----------------------------------------------------------------------------------------------------
    gives a matrix that should correspond to Mathematica GaborMatrix[ { r, s }, k, f ]
    applying the normalization
    "so that the elements of Abs[GaborMatrix[r,k,0]+I GaborMatrix[r,k,Pi/2]] sum to 1" (Details and Options)
    ----------------------------------------------------------------------------------------------------- """

    g0  = gabor_matrix_unscaled( r, s, k, 0 )
    g1  = gabor_matrix_unscaled( r, s, k,  0.5 * pi )
    g   = sqrt( g0 ** 2 + g1 ** 2 )
    t   = g.sum()
    return gabor_matrix_unscaled( r, s, k, f ) / t



def rotate( ang, freq ):
    """ -----------------------------------------------------------------------------------------------------
    rotate the frequency freq of the given angle ang [rad]
    ----------------------------------------------------------------------------------------------------- """
    s   = sin( ang )
    c   = cos( ang )
    r   = np.array( [ [ c, -s ], [ s, c ] ] )
    f   = np.array( [ freq, freq ] )
    return np.dot( r, f )



def gabor_w( size, freq, nrot, infeat=1 ):
    """ -----------------------------------------------------------------------------------------------------
    function equivalent to GaborW[] in network.m
	input:												
		size		size of the mask							
		freq		spatial frequency							
		nrot		number of mask rotations, equals to number of output features		
		infeat		number of input features						
    the difference is in the shape of the output matrix: the "channels_last" convention is adopted,
    because it is Kera's default, while in Mathematica the "channels_first" convention is kept
    ----------------------------------------------------------------------------------------------------- """
    rk  = [ rotate( 2 * pi * a / nrot, freq ) for a in range( nrot ) ]
    sm1 = size - 1.
    mat = np.array( [ gabor_matrix( sm1 / 2, sm1 / 5, k, pi / 2 ) for k in rk ] )
    mat = np.moveaxis( mat, 0, -1 )              # move channel axis from first to last position
    mat = mat.reshape( ( size, size, infeat, nrot ) ) # add axis for input channels

    return mat
