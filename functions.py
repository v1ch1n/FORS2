from re import X
import numpy as np
from astropy.wcs import WCS
from PythonPhot import getpsf
from PythonPhot import aper
from astroquery.vizier import Vizier
import astropy.coordinates as coord
import astropy.units as u
from astropy.table import Table

# This file contains a set of useful functions in order to use it with the PSF program
######################################################################################

# Convert RA and DEC to (x,y) coordinates
def wcstoxy(imagename, ra, dec):
    w = WCS(imagename)
    x, y = w.all_world2pix(ra, dec, 1)
    # Add 1px to be consistent with DS9
    x1 = x
    y1 = y
    return x1, y1

# Convert (x,y) coordinates to RA and DEC
def xytowcs(imagename, x, y):
    w = WCS(imagename)
    ra, dec = w.all_pix2world(x, y, 0)
    return ra, dec

# Make a list of stars to extract the PSF from
# Image = Data of the image (hdulist[0].data)
# Imagename = Name of the image (with extension)
def makelist(image, imagename, RA, DEC, pathstars='./starstore',magrange=[16.5,19.0],radiussearch='3',fwhmnominal=4.5):
    # Find the dimensions of the image
    x_dim = int(len(image[0]))
    y_dim = int(len(image))

    #Query the GSC catalog to find stars usable as reference:
    Vizier.ROW_LIMIT = 999999999 
    gsc = Vizier.query_region(coord.SkyCoord(ra=RA, dec=DEC,unit=(u.deg, u.deg),radius=radiussearch+'m',catalog="GSC2.3"))

    # Has any source been found during the query? True if yes, False if not
    presentsource = (gsc!=[])
    
    # Verify that at least one of the queried sources falls within the CCD and within the magnitude range and is not coincident with the candidate
    z = 0
    intheCCD = False
    inthemagrange = False
    notcoincidentwiththecand = False

    if presentsource:
        while (z < len(gsc[0]["RAJ2000"])  and  intheCCD == False  and  inthemagrange == False):
            Xstar, Ystar = wcstoxy(imagename, gsc[0]["RAJ2000"][z], gsc[0]["DEJ2000"][z])
            
            # Fill posible masked value with 0.0
            gsc['I/305/out']["Fmag"].fill_value = 0.0

            # Check location and magnitude to fall within acceptable values
            checklocation = (Xstar > 2*fwhmnominal and Xstar < (x_dim-2*fwhmnominal) and  Ystar > 2*fwhmnominal and Ystar < (y_dim-2*fwhmnominal))
            checkmagnitude = (gsc[0]["Fmag"][z] >= magrange[0]) and (gsc[0]["Fmag"][z] < magrange[1] and gsc[0]["Class"][z] == 0)

            # Distance between the source and the candidate must be more than 2 arcsec
            dist = np.sqrt((np.mean(gsc[0]["RAJ2000"][z]) - RA)**2 + (np.mean(gsc[0]["DEJ2000"][z]) - DEC)**2)
            checkcoincidence = (dist >= 2.0/3600.0)

            # Change the values of the variables dependent of the results of the previous checks
            if checklocation:
                intheCCD = True
            if checkmagnitude:
                inthemagrange = True
            if checkcoincidence:
                notcoincidentwiththecand = True

            z += 1

    # Has any source good for the PSF extraction been found?
    if (presentsource and intheCCD and inthemagrange and notcoincidentwiththecand):
        OKgo = True
        print(">>> Your query returned at least one star suitable for the PSF extraction.")
    else:
        OKgo = False
        print("!Watch out! Your query returned no star suitable for the PSF extraction. Try again with a larger radius or a different range of magnitudes. ")

    # If there is at least one source good for PSF extraction found during the query, append the coordinates and the magnitudes to the lists
    if OKgo:
        # Fill posible masked value with 0.0
        gsc['I/305/out']["Fmag"].fill_value = 0.0

        # Open the file, that will be in the same directory of the images and will end with _stars.txt
        file = open(pathstars + '/' + imagename[:-5] + "_stars.txt", "w")
        for i in range(0, len(gsc[0]["Fmag"])):
            # Check that the source lies within the specified range of magnitude and that is catalogued as "Star"
            if ((gsc[0]["Fmag"][i]) >= magrange[0] and gsc[0]["Fmag"][i] <= magrange[1] and  gsc[0]["Class"][i]==0):
                
                # Check that the source actually falls INSIDE the CCD
                # First, convert RA and DEC to (x,y) coordinates
                Xstar, Ystar = wcstoxy(imagename, gsc[0]["RAJ2000"][i], gsc[0]["DEJ2000"][i])
                # Check that the star is within the limits of the CCD, assuming a frame of 20 px from the borders
                checklocation = (Xstar > 20 and Xstar < (x_dim-20) and  Ystar > 20 and Ystar < (y_dim-20))
                if checklocation:
                    file.write(str(gsc[0]["RAJ2000"][i]) + " " + str(gsc[0]["DEJ2000"][i]) + "\n")
        file.close()

# Get the PSF of the image
# Stars are already in (x,y) coordinates, so there is no need to transform them from RA and DEC
# Also, starslist is a table with the brightest stars in the image
def getPSFimage(image, imagename, starslist, Xcand, Ycand, ronois, phpadu, individual=False, pathpsf="./psfstore", fwhmnominal=5):
    # Dimension of the image
    x_dim = int(len(image[0]))
    y_dim = int(len(image))

    # Flag to check if we are calculating the FWHM of a single star (therefore, a single row in starslist Table)
    if individual:
        Xstars = [starslist['X']]
        Ystars = [starslist['Y']]

    # To obtain the PSF of the image using the brightest extracted stars
    else:
        # Create a list of stars that fall within acceptable values
        starslistOK = Table([[],[],[]], names=('ID','X','Y'))

        for index, Xstar, Ystar in zip(starslist['ID'], starslist['X'], starslist['Y']):
            # Check location to fall within acceptable values (not too close to the borders)
            checklocation = (Xstar > 2*fwhmnominal and Xstar < (x_dim-2*fwhmnominal) and  Ystar > 2*fwhmnominal and Ystar < (y_dim-2*fwhmnominal))
            if checklocation:
                # Add the star to the ok stars list
                starslistOK.add_row([index, Xstar, Ystar])

        # Obtain all X and Y coordinates of the stars
        Xstars = starslistOK['X'].data
        Ystars = starslistOK['Y'].data

    # Obtain the PSF of the image
    mag, magerr, flux, fluxerr, sky, skyerr, badflag, outstr = aper.aper(image, Xstars, Ystars, phpadu=1,apr=10,zeropoint=25,skyrad=[40,50],badpix=[-12000,60000],exact=True, verbose=False)
    
    # If the mag value is nan, return error values
    if individual:
        if np.isnan(mag).any():
            return [-1,-1,-1,-1,-1], 1, 1

    # Use the stars at those coordinates to generate a PSF model (Gauss)
    imagenameshort = imagename.split("/")[-1][:-5]

    print(np.where( np.array(list(mag.reshape(1,len(mag))[0]))))
    print('mag: ', np.array(list(mag.reshape(1,len(mag))[0])))

    notnan = np.where( np.isnan(np.array(list(mag.reshape(1,len(mag))[0]))) == False)[0]

    if individual:
        sky = [sky]
        gauss,psf,psfmag = getpsf.getpsf(image, Xstars, Ystars, mag, sky, ronois, phpadu, np.arange(len(Xstars)), 15, 3, pathpsf + '/' + imagenameshort+'_psf_residuals.fits', verbose=False)
    else:
        # Parameters: Image array, X coordinates, Y coordinates, vector of magnitudes, vector of sky values, ronois, phpadu, idpsf, psfrad, fitrad, psfname, zeropoint, verbose
        gauss,psf,psfmag = getpsf.getpsf(image, Xstars[notnan], Ystars[notnan], mag[notnan], sky[notnan], ronois, phpadu, np.arange(len(Xstars[notnan])), 15, 3, pathpsf + '/' + imagenameshort+'_psf_residuals.fits', verbose=False)

    return gauss, psf, psfmag