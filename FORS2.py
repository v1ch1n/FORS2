from PythonPhot import aper
from PythonPhot import pkfit
import astropy.io.fits as pyfits
import astropy.coordinates as coord
import astropy.units as u
from astropy.wcs import WCS
from astropy.table import Table
from astroquery.vizier import Vizier
from astropy.stats import sigma_clip
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import time
from scipy.spatial import KDTree
import functions as pf
from requests.exceptions import ConnectionError
import shutil

# Warnings
import warnings
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)

# Functions to be used in the main program


def create_valid_stars(image):
    # Create the directory which will contain all the Sextractor related files with the name of the image
    current_directory = os.getcwd()
    image_directory = os.path.join(current_directory + '/results', r'{}'.format(image[:-5]))

    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    # Number of valids stars and total stars
    valid_stars = 0
    total_stars = 0

    # File with valid stars (stars with flag value = 0)
    valids = open(image_directory + "/" + image[:-5] + '.sex2d.valid.txt', 'w')
    # Reg file with valid stars (for DS9)
    reg_valids = open(image_directory + '/' + image[:-5] + '.sex2d.valid.reg', 'w')

    # Open the file generated with Sextractor to obtain the non saturated stars
    with open(image[:-5]+'.sex2d.txt', 'r') as fi:
        for line in fi:
            line = line.split()
            parameters = len(line)
            if '#' not in line:
                total_stars += 1
                # If the flag value of the star is 0, then the star is valid, and we write it in the file and in the reg file
                if int(line[-1]) == 0:
                    # Write in the reg file 
                    reg_valids.write('image; circle '+line[0]+' '+line[1]+' 3 \n' )
                    # Write valid stars to file, with all the parameters except the flag value
                    valids.write(str(valid_stars)+' ')
                    # X and Y coordinates are always the first two parameters, all the rest are distinct
                    for i in range(parameters - 1):
                        valids.write(line[i]+ ' ')
                    valids.write('\n')
                    valid_stars += 1

    valids.close()
    reg_valids.close()

    return valid_stars, total_stars

def match_stars(image_path, extracted_stars, catalog, radius):
    # Obtain the header of the image, to query the center coordinates of it
    header = pyfits.getheader(image_path)
    center = pf.xytowcs(image_path, header['NAXIS1']/2, header['NAXIS2']/2)

    # Query the given catalogue according to the filter of the image
    try:
        filter = header['HIERARCH ESO INS FILT1 NAME']
        # USNO-B1 catalog
        if catalog == 'USNO-B1':
            catalog_stars = Vizier.query_region(coord.SkyCoord(ra=center[0], dec=center[1], unit='deg'), radius=0.1*u.deg, catalog='USNO-B1')
            # Extract the RA and DEC coordinates of the catalog
            ra_values = catalog_stars[0]['RAJ2000'].data
            dec_values = catalog_stars[0]['DEJ2000'].data
        elif catalog == 'GSC':
            catalog_stars = Vizier.query_region(coord.SkyCoord(ra=center[0], dec=center[1], unit='deg'), radius=0.1*u.deg, catalog='GSC2')
            # Obtain only the stars from the catalog (i.e. the ones where Class column is equal to 0)
            catalog_stars = catalog_stars[2][catalog_stars[2]['Class'] == 0]
            #catalog_stars = catalog_stars[catalog_stars['Vmag'] >= 10]
            #catalog_stars = catalog_stars[catalog_stars['Vmag'] <= 20]
            ra_values = catalog_stars['RA_ICRS'].data
            dec_values = catalog_stars['DE_ICRS'].data
    except KeyError:
        # If the image does not have a filter, use the USNO-B1 catalog
        catalog_stars = Vizier.query_region(coord.SkyCoord(ra=center[0], dec=center[1], unit='deg'), radius=0.1*u.deg, catalog='USNO-B1')

    catalog_x = list()
    catalog_y = list()
    
    # Transform the coordinates in the catalog stars to pixel values, to query the KD-tree
    w = WCS(header)
    for ra, dec in zip(ra_values, dec_values):
        sky = coord.SkyCoord(ra, dec, frame='icrs', unit='deg')
        x, y = w.world_to_pixel(sky)
        catalog_x.append(x)
        catalog_y.append(y)

    pixel_catalog_stars = np.c_[catalog_x, catalog_y]

    # Query the KD-tree to match the extracted stars with the catalog stars in the given radius
    kd_tree = KDTree(pixel_catalog_stars)
    dd, ii = kd_tree.query(extracted_stars, k=1, distance_upper_bound=radius)

    # Create a dictionary with the matched stars, where the key is the extracted star and it's value is the index of the catalog star in the catalog
    matched_stars = {k: v for k, v in zip(range(len(extracted_stars)), ii) if v != len(pixel_catalog_stars)}

    return matched_stars

def write_results(image):
    # Directory of the result of the image
    current_directory = os.getcwd()
    image_directory = os.path.join(current_directory + '/results', r'{}'.format(image[:-5]))

    # Path of the actual image
    image_path = os.path.join(fors2_folder, image)

    # Obtain the header of the image
    header = pyfits.getheader(image_path)

    # Create the file which will contain the results
    with open(image_directory + '/results.txt', 'w') as f:
        # First line of the results file contain the following:
        # Filter, Avg FWHM, Median FWHM, Avg ellipticity, Median ellipticity (Sextractor, PSF, in that order), Zeropoint
        # Rest of the lines contain the following, for every star detected with Sextractor:
        # X, Y, RA, DEC, Flux, Fluxerr, Catalog Mag, RA Cat, DEC Cat, PSF Mag, PSF Mag err
        f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(header['HIERARCH ESO INS FILT1 NAME'],\
            header['EXTNAME'], average_fwhm_sextractor, median_fwhm_sextractor, average_ellipticity_sextractor,\
            median_ellipticity_sextractor, average_fwhm_psf, median_fwhm_psf, average_ellipticity_psf,\
            median_ellipticity_psf, real_zeropoint))

    # Create the file which will contain the different values of star features
    table_to_file.write(image_directory + '/results_stars.txt', format='ascii', delimiter=',')

# Parameters to be set manually
##############################################

# Path where to store the PSF models and the lists of stars
path_psf = './psfstore'

# Recentering during the PSF fitting? ("YES"/"NO")
recenter="YES"

#Zero point: ()
zeropoint_value=27.531354844160536

# Number of brightest stars to be used from the image to fit the PSF model
bright_stars = 20

#Rough value for the FWHM (in pixel)
fwhm_nominal=8.0  #pixel, mainly used to establish the distance of the source from the border.

#Photons per ADU:
phpadu=2.4

#Readout noise per pixel (scalar)
ronois=4.9 #Rough re

# Folder where the FORS2 files are contained
fors2_folder = './FORS2Archives'

# Folder where the different results are stored
results_folder = './results'

for image in os.listdir(fors2_folder):
    # Relative path of the image
    image_path = os.path.join(fors2_folder, image)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Extract all the stars from the image using Sextractor
    os.system('sex ' + image_path + ' -CATALOG_NAME ' + image[:-5] + '.sex2d.txt')  

    # Create the directory which will contain all the Sextractor related files with the name of the image
    current_directory = os.getcwd()
    image_directory = os.path.join(current_directory + '/results', r'{}'.format(image[:-5]))

    valid_stars, total_stars = create_valid_stars(image)

    """ print("-------------------------------------------------------------------")
    print(f'Number of valid stars, i.e. stars with flag value 0: {valid_stars}')
    print(f'Number of total stars: {total_stars}')
    print(f'Percentage of valid stars: {valid_stars/total_stars*100}%') """

    valid_stars_table = Table.read(image_directory + '/' + image[:-5] + '.sex2d.valid.txt', format='ascii', delimiter=' ')

    # Obtain the FWHM and ellipticity of the valid extracted stars
    # 4th and 8th parameters in the Sextractor valid file are the ellipticity and the FWHM, respectively
    valid_stars_table.rename_column('col4', 'ellipticity')
    valid_stars_table.rename_column('col8', 'fwhm')

    fwhm_sextractor = valid_stars_table['fwhm']
    ellipticity_sextractor = valid_stars_table['ellipticity']

    # Filter the 0 values of FWHM?
    fwhm_sextractor = fwhm_sextractor[fwhm_sextractor != 0]

    # Compute the average and median FWHM and ellipticity
    average_fwhm_sextractor = np.mean(fwhm_sextractor)
    median_fwhm_sextractor = np.median(fwhm_sextractor)
    average_ellipticity_sextractor = np.mean(ellipticity_sextractor)
    median_ellipticity_sextractor = np.median(ellipticity_sextractor)

    """ print(f"Average FWHM: {average_fwhm_sextractor}")
    print(f"Median FWHM: {median_fwhm_sextractor}")
    print(f"Average ellipticity: {average_ellipticity_sextractor}")
    print(f"Median ellipticity: {median_ellipticity_sextractor}") """

    # Read the list of coordinates obtained with sextractor for the given image, which flag values are 0 (i.e. not saturated stars)
    stars_file = image_directory + "/" + image[:-5] + '.sex2d.valid.txt' 
    stars_list = Table.read(stars_file, format='ascii', delimiter=' ')
    # Rename all the columns to what they mean in Sextractor
    stars_list.rename_column('col1','ID')
    stars_list.rename_column('col2','X')
    stars_list.rename_column('col3','Y')
    stars_list.rename_column('col4','ellipticity')
    stars_list.rename_column('col5', 'flux')
    stars_list.rename_column('col6', 'fluxerr')
    stars_list.rename_column('col7', 'mag')
    stars_list.rename_column('col8', 'fwhm')

    # Arrays which will contain the individual values of FWHM and Ellipticity for every valid star detected
    fwhm_psf = []
    ellipticity_psf = []

    # Get dimension of the image
    data_image = pyfits.getdata(image_path)

    x_dim = int(len(data_image[0]))
    y_dim = int(len(data_image))

    # Computing the FWHM and ellipticity of every star using PSF fitting
    for i in range(valid_stars):
        star = stars_list[i]
        # Check that the star is located within acceptable values (not too close to the borders)
        checklocation = (star['X'] > 2.5*fwhm_nominal and star['X'] < (x_dim-2.5*fwhm_nominal) and  star['Y'] > 2.5*fwhm_nominal and star['Y'] < (y_dim-2.5*fwhm_nominal))
        if not checklocation:
            #print(f"Star #{i} too close to the borders, skipping...")   
            continue
        # PSF of the image using a single star (star #i)
        gauss, psf_residuals, psfmag = pf.getPSFimage(data_image, image_path, star, star['X'], star['Y'], ronois, phpadu, individual=True, pathpsf=path_psf)
        
        # Error, no convergence for the star
        if np.array_equal(gauss, np.array([-1,-1,-1,-1,-1])):
            continue
        
        # Compute the FWHM
        sigma_star = np.mean([gauss[3], gauss[4]])
        fwhm_star = 2.355*sigma_star
        fwhm_psf.append(fwhm_star)

        # Obtain the header of the PSF model generated
        hpsf = pyfits.getheader(path_psf + '/' + image[:-5] + '_psf_residuals.fits')
        # Determine the semi major and semi minor axis to obtain the ellipticity
        direction_x = hpsf['GAUSS4']
        direction_y = hpsf['GAUSS5']

        ellipticity_star = 0.0

        # Compute the ellipticity using the sextractor formula (1 - (semiminor axis)/(semimajor axis))
        if direction_x >= direction_y:
            ellipticity_star = 1 - (direction_y)/(direction_x)
        else:
            ellipticity_star = 1 - (direction_x)/(direction_y)

        ellipticity_psf.append(ellipticity_star)

    # Compute the average and median FWHM and ellipticity obtained with PSF fitting
    average_fwhm_psf = np.mean(fwhm_psf)
    median_fwhm_psf = np.median(fwhm_psf)
    average_ellipticity_psf = np.mean(ellipticity_psf)
    median_ellipticity_psf = np.median(ellipticity_psf)

    # Compute the PSF fitting of the image using the brightest stars (determined by the variable bright_stars)

    print("Processing image: ", image)

    # Sort the stars by their flux and select the n brightest stars (where n = bright_stars)
    brightest_stars = Table([[],[],[],[]], names=('ID','X','Y','flux'))
    brightest_stars = stars_list[np.argsort(stars_list['flux'])[::-1]][0:bright_stars]

    # Testing with a random candidate (brighest star in the image)
    # It appears that the candidate doesn't matter for the PSF model (it can be any star in the list)
    candidate = (brightest_stars['X'][0], brightest_stars['Y'][0])

    # Read the image as an array
    hdulist = pyfits.open(image_path)
    # Read the header and the data of the image
    header_image = hdulist[0].header
    data_image = hdulist[0].data

    # Acquire information about the PSF of the image
    gauss, psf_residuals_model, psfmag = pf.getPSFimage(data_image, image_path, brightest_stars, candidate[0], candidate[1], ronois, phpadu, individual=False, pathpsf=path_psf)

    # Compute the real FWHM of the image
    sigma_image = np.mean([gauss[3], gauss[4]])
    fwhm_image = 2.355*sigma_image

    # Define the sky radius
    skyRmin = 3*fwhm_image
    skyRmax = 5*fwhm_image

    # Aperture photometry to get magnitudes and sky values for specified coordinates
    mag,magerr,flux,fluxerr,sky,skyerr,badflag,outstr = \
            aper.aper(data_image,candidate[0],candidate[1],phpadu=phpadu,apr=fwhm_image,zeropoint=zeropoint_value, \
            skyrad=[skyRmin,skyRmax],badpix=[-12000,60000],exact=True, verbose=False)

    #print('Sigma image: '+str(sigma_image)+' FWHM image: '+str(fwhm_image))

    # Check that the PSF was succesfully generated in the specified folder
    PSFresult = os.path.isfile(path_psf + '/' + image[:-5] + '_psf_residuals.fits')
    if PSFresult:
        print("PSF model of the image succesfully generated!")

    # Obtain the header of the PSF model generated
    hpsf = pyfits.getheader(path_psf + '/' + image[:-5] + '_psf_residuals.fits')

    # Perform the PSF fitting
    if recenter == 'YES':
        # Load the pkfit class (we can add mask and noise image)
        pk = pkfit.pkfit_class(data_image, gauss, psf_residuals_model, ronois, phpadu)
        errmag, chi, sharp, niter, scale = pk.pkfit(1, candidate[0], candidate[1], sky, 4)
    else:
        from PythonPhot import pkfit_norecenter
        # Load the pkfit class (we can add mask and noise image)
        pk = pkfit_norecenter.pkfit_class(data_image, gauss, psf_residuals_model, ronois, phpadu)
        errmag, chi, sharp, niter, scale = pk.pkfit_norecenter(1, candidate[0], candidate[1], sky, 4)

    # Compute flux and magnitude
    fluxx = (scale / header_image['EXPTIME'])*10**(0.4*(25.0-hpsf['PSFMAG']))
    dflux = errmag*10**(0.4*(25.0-hpsf['PSFMAG']))
    magvalue = zeropoint_value - 2.5*np.log10(fluxx)

    # Estimate a conservative measure for the error
    magerror = np.absolute(zeropoint_value-2.5*np.log10(fluxx-dflux) - magvalue)

    # If the error of the magnitude is not a number, assign it to 0
    magerror = 0 if np.isnan(magerror) else magerror

    """ print("Flux for specified coordinates: "+str(flux))
    print("Computed flux: "+str(fluxx))
    print("Magnitude: "+str(mag))
    print("Computed magnitude: "+str(magvalue)) """

    # Obtain the coordinates of the extracted stars
    X_stars = stars_list['X'].data
    Y_stars = stars_list['Y'].data

    # Query regions (this has to be done in the for loop of the images)
    # Query the coordinates of all the stars in the image (Has to be done in RA and DEC? or I can also do it in X and Y?)
    # How to know which catalogue to query? If the query returns more than one table, which one to select??
    Vizier.ROW_LIMIT = -1
    # Obtain the header and the dimension of the image, to query the region with the center coordinates of the image
    header = pyfits.getheader(image_path)
    wcs_candidate = pf.xytowcs(image_path, header['NAXIS1']/2, header['NAXIS2']/2)
    # For the first iteration, query the USNO-B1 catalogue independent of the filter
    try:
        filter = header['HIERARCH ESO INS FILT1 NAME']
        try:
            catalog = Vizier.query_region(coord.SkyCoord(ra=wcs_candidate[0], dec=wcs_candidate[1], unit='deg'), radius=0.1*u.deg, catalog="USNO-B1")
        except ConnectionError:
            print("ConnectionError! Trying again...")
            # Wait 10 seconds and query the catalog again
            time.sleep(10)
            catalog = Vizier.query_region(coord.SkyCoord(ra=wcs_candidate[0], dec=wcs_candidate[1], unit='deg'), radius=0.1*u.deg, catalog="USNO-B1")
    # Other filters?
    # For now, raise an exception it the file does not have a filter, and query the USNO-B1 catalogue
    except KeyError:
        catalog = catalog = Vizier.query_region(coord.SkyCoord(ra=wcs_candidate[0], dec=wcs_candidate[1], unit='deg'), radius=0.1*u.deg, catalog="USNO-B1")

    # Extract the RA and DEC of all the stars in the table
    # USNO-B1 catalog
    raj = catalog[0]['RAJ2000'].data
    decj = catalog[0]['DEJ2000'].data

    # Obtain the x and y coordinates of the catalog stars in the image
    x_test = []
    y_test = []

    # First matching iteration
    test_image = pyfits.open(image_path)
    w = WCS(test_image[0].header)

    for ra, dec in zip(raj, decj):
        sky = coord.SkyCoord(ra, dec, frame='icrs', unit='deg')
        x,y = w.world_to_pixel(sky)
        x_test.append(x)
        y_test.append(y)

    # Create two arrays with the coordinates of the stars, one for the catalog and one for the extracted
    catalog_stars = np.c_[x_test, y_test]
    extracted_stars = np.c_[X_stars, Y_stars]

    # Create a dictionary with the matched stars (key: extracted star; value: index of catalog star in the catalog) to extract the magnitudes
    extracted_catalog = match_stars(image_path, extracted_stars, 'USNO-B1', 10)

    # If there are no stars matched, continue with the next image and delete the folder with the results
    if len(extracted_catalog) == 0:
        print("No stars matched in image "+image+"!")
        shutil.rmtree(results_folder+'/'+image[:-5])
        continue

    # Create a table with RA, DEC, X, Y, catalog Mag, flux, calculated zero point for every star
    matched_stars = Table([[],[],[],[],[],[]], names=(['RA', 'DEC', 'X', 'Y', 'Catalog Mag', 'Flux']))

    # Array of offsets to apply to the magnitudes
    offset_array = []
    zeropoint_array = []

    # Array of magnitude catalogs and calculate catalogs to plot them
    catalog_magnitudes = []
    calculated_magnitudes = []
    flux_values = []

    # Array with the magnitude errors of the stars
    magnitude_errors = []

    header_has_filter = True
    for i in extracted_catalog:
        # Obtain the RA and DEC of the matched stars in the catalog
        ra = catalog[0][extracted_catalog[i]]['RAJ2000']
        dec = catalog[0][extracted_catalog[i]]['DEJ2000']

        # Obtain the (x,y) coordinates from the star in the image
        x = extracted_stars[i][0]
        y = extracted_stars[i][1]

        # Define the extinction coefficient, and assign it depending on the filter 
        # (https://www.eso.org/observing/dfo/quality/FORS2/pipeline/coeffs.txt)
        extinction_coefficient = 0.0

        # Extract the magnitude using the appropiate filter
        try:
            filter = header['HIERARCH ESO INS FILT1 NAME']
            # Chech if the values are masked. If both are unmasked, take the mean value
            if filter == 'b_HIGH' or filter == 'v_HIGH':
                # USNO-B1 catalog
                if catalog[0][extracted_catalog[i]]['B1mag'] is not ma.masked and catalog[0][extracted_catalog[i]]['B2mag'] is not ma.masked:
                    b_mag = np.mean([catalog[0][extracted_catalog[i]]['B1mag'], catalog[0][extracted_catalog[i]]['B2mag']])
                elif catalog[0][extracted_catalog[i]]['B1mag'] is ma.masked and catalog[0][extracted_catalog[i]]['B2mag'] is not ma.masked:
                    b_mag = catalog[0][extracted_catalog[i]]['B2mag']
                elif catalog[0][extracted_catalog[i]]['B1mag'] is ma.masked and catalog[0][extracted_catalog[i]]['B2mag'] is ma.masked:
                    continue
                else:
                    b_mag = catalog[0][extracted_catalog[i]]['B1mag']
                mag_catalog = b_mag
                extinction_coefficient = 0.216
            elif filter == 'R_SPECIAL':
                if catalog[0][extracted_catalog[i]]['R1mag'] is not ma.masked and catalog[0][extracted_catalog[i]]['R2mag'] is not ma.masked:
                    r_mag = np.mean([catalog[0][extracted_catalog[i]]['R1mag'], catalog[0][extracted_catalog[i]]['R2mag']])
                elif catalog[0][extracted_catalog[i]]['R1mag'] is ma.masked and catalog[0][extracted_catalog[i]]['R2mag'] is not ma.masked:
                    r_mag = catalog[0][extracted_catalog[i]]['R2mag']
                else:
                    r_mag = catalog[0][extracted_catalog[i]]['R1mag']
                mag_catalog = r_mag
                extinction_coefficient = 0.081
            elif filter == 'I_BESS':
                mag_catalog = catalog[0][extracted_catalog[i]]['Imag']
                extinction_coefficient = 0.059
        except KeyError:
            print("No filter found!\n")
            # Don't calculate the zeropoint (in case there is no filter)
            header_has_filter = False
            break

        # Obtain the flux performing apperture photometry for the specified coordinates 
        magstar, magerrstar, fluxstar, fluxerr, skystar, skyerrstar, badflagstar, outstrstar = \
            aper.aper(data_image, x, y, phpadu=phpadu, apr=fwhm_image, zeropoint=zeropoint_value, \
            skyrad=[skyRmin, skyRmax], badpix=[-12000, 30000], exact=True, verbose=False)

        # PSF photometry
        pk = pkfit.pkfit_class(data_image, gauss, psf_residuals_model, ronois, phpadu)
        errmag, chi, sharp, niter, scale = pk.pkfit(1, x, y, skystar, 5)

        # Compute the flux and magnitude
        flux_value = (scale*10**(0.4*(25.0-hpsf['PSFMAG'])))/(header['EXPTIME'])
        dflux_value = errmag*10**(0.4*(25.0-hpsf['PSFMAG']))
        
        mag_value = zeropoint_value-2.5*np.log10(flux_value)
        
        # Compute the zeropoint of the star using the magnitude catalog and the extinction coefficient
        airmass = np.mean([header['HIERARCH ESO TEL AIRM END'], header['HIERARCH ESO TEL AIRM START']])
        zeropoint_star = mag_catalog + 2.5*np.log10(flux_value) + (extinction_coefficient * airmass)

        # Compute the magnitude error using the error propagation equation
        # sqrt(d(mag_value)/d(flux_value)**2 * dflux_value**2)
        mag_error = np.sqrt((-1.08573/flux_value)**2 * dflux_value**2)
        magnitude_errors.append(mag_error)

        # Append the catalog and calculated magnitudes
        catalog_magnitudes.append(mag_catalog)
        calculated_magnitudes.append(mag_value)

        # Compute the offset and append it to the offset array
        offset_array.append(mag_catalog - mag_value)
        zeropoint_array.append(zeropoint_star)

        flux_values.append(flux_value)
        
    # Compute the offset to apply for the zeropoint correction of the image
    if header_has_filter:
        offset = np.median(offset_array)
        real_zeropoint = np.mean(zeropoint_array)

        magvalue_calibrated = magvalue + offset

    extracted_stars_ra = []
    extracted_stars_dec = []
    catalog_stars_ra = []
    catalog_stars_dec = []

    for i in extracted_catalog:
        # Extracted and matched stars
        w = WCS(header)
        sky = w.pixel_to_world(extracted_stars[i][0], extracted_stars[i][1])
        extracted_stars_ra.append(sky.ra.deg)
        extracted_stars_dec.append(sky.dec.deg)
        # Catalog stars
        catalog_ra = catalog[0][extracted_catalog[i]]['RAJ2000']
        catalog_dec = catalog[0][extracted_catalog[i]]['DEJ2000']
        catalog_stars_ra.append(catalog_ra)
        catalog_stars_dec.append(catalog_dec)

    ra_difference = np.array(catalog_stars_ra) - np.array(extracted_stars_ra)
    dec_difference = np.array(catalog_stars_dec) - np.array(extracted_stars_dec)

    ra_dec_conc = np.c_[ra_difference, dec_difference]

    sigma_clipped = sigma_clip(ra_dec_conc, sigma=3, maxiters=None, masked=False, axis=(0,1))

    # Remove the tuples with contains NaN values
    sigma_clipped = np.asarray([t for t in sigma_clipped if not any(isinstance(n, float) and np.isnan(n) for n in t)])

    sigma_clipped = sigma_clipped.flatten()

    filtered_ra = sigma_clipped[::2]
    filtered_dec = sigma_clipped[1::2]

    filtered_data = np.c_[filtered_ra, filtered_dec]

    # Create the astrometry corrected file
    original_data, original_header = pyfits.getdata(image_path, header=True)
    astrometry_name = image[:-5] + '_astrometry.fits'

    pyfits.writeto(astrometry_name, original_data, original_header, output_verify='ignore', overwrite='True')

    with pyfits.open(astrometry_name, 'update') as f:
        for hdu in f:
            hdu.header['CRVAL1'] += np.mean(filtered_ra)
            hdu.header['CRVAL2'] += np.mean(filtered_dec)

    # Obtain the header and the dimension of the image, to query the region with the center coordinates of the image
    wcs_candidate = pf.xytowcs(astrometry_name, header['NAXIS1']/2, header['NAXIS2']/2)
    # Query the catalogue of bright stars according to the filter of the image
    try:
        filter = header['HIERARCH ESO INS FILT1 NAME']
        try:
            catalog = Vizier.query_region(coord.SkyCoord(ra=wcs_candidate[0], dec=wcs_candidate[1], unit='deg'), radius=0.1*u.deg, catalog="GSC2")
            catalog = catalog[2][catalog[2]['Class'] == 0]
            #catalog = catalog[catalog['Vmag'] >= 10]
            #catalog = catalog[catalog['Vmag'] <= 20]
        except ConnectionError:
            print("ConnectionError! Trying again...")
            # Wait 10 seconds and query the catalog again
            time.sleep(10)
            catalog = Vizier.query_region(coord.SkyCoord(ra=wcs_candidate[0], dec=wcs_candidate[1], unit='deg'), radius=0.1*u.deg, catalog="GSC2")
            catalog = catalog[2][catalog[2]['Class'] == 0]
            #catalog = catalog[catalog['Vmag'] >= 10]
            #catalog = catalog[catalog['Vmag'] <= 20]
    # Other filters?
    # For now, raise an exception it the file does not have a filter, and query the USNO-B1 catalogue
    except KeyError:
        catalog = catalog = Vizier.query_region(coord.SkyCoord(ra=wcs_candidate[0], dec=wcs_candidate[1], unit='deg'), radius=0.1*u.deg, catalog="USNO-B1")

    # GSC catalog
    raj = catalog['RA_ICRS'].data
    decj = catalog['DE_ICRS'].data

    # Obtain the x and y coordinates of the catalog stars in the image
    x_test = []
    y_test = []

    # Second matching iteration
    test_image = pyfits.open(astrometry_name)
    w = WCS(test_image[0].header)

    for ra, dec in zip(raj, decj):
        sky = coord.SkyCoord(ra, dec, frame='icrs', unit='deg')
        x,y = w.world_to_pixel(sky)
        x_test.append(x)
        y_test.append(y)

    # Create two arrays with the coordinates of the stars, one for the catalog and one for the extracted
    catalog_stars = np.c_[x_test, y_test]
    extracted_stars = np.c_[X_stars, Y_stars]

    # Execute the matching with the astrometry corrected file
    astrometry_stars = match_stars(astrometry_name, extracted_stars, 'GSC', 3)

    # Get the data of the astrometry corrected file
    data_image = pyfits.getdata(astrometry_name)

    # Array of offsets to apply to the magnitudes
    offset_array = []
    zeropoint_array = []

    # Array of magnitude catalogs and calculate catalogs to plot them
    catalog_magnitudes = dict()
    calculated_magnitudes = dict()
    flux_values = []

    # Array with the magnitude errors of the stars
    magnitude_errors = dict()

    header_has_filter = True
    for i in astrometry_stars:
        # Obtain the RA and DEC of the matched stars in the catalog

        # GSC catalog
        ra = catalog[astrometry_stars[i]]['RA_ICRS']
        dec = catalog[astrometry_stars[i]]['DE_ICRS']

        # Obtain the (x,y) coordinates from the star in the image
        x = extracted_stars[i][0]
        y = extracted_stars[i][1]

        # Define the extinction coefficient, and assign it depending on the filter 
        # (https://www.eso.org/observing/dfo/quality/FORS2/pipeline/coeffs.txt)
        extinction_coefficient = 0.0

        header = pyfits.getheader(astrometry_name)

        # Extract the magnitude using the appropiate filter
        try:
            filter = header['HIERARCH ESO INS FILT1 NAME']
            # Chech if the values are masked. If both are unmasked, take the mean value
            if filter == 'b_HIGH':
                # GSC catalog
                if catalog[astrometry_stars[i]]['Bmag'] is ma.masked:
                    #print(f"Invalid magnitude for star {i}")
                    continue
                mag_catalog = catalog[astrometry_stars[i]]['Bmag']
                extinction_coefficient = 0.216
            elif filter == 'R_SPECIAL':
                if catalog[astrometry_stars[i]]['rmag'] is ma.masked:
                    #print(f"Invalid magnitude for star {i}")
                    continue
                mag_catalog = catalog[astrometry_stars[i]]['rmag']
                extinction_coefficient = 0.081
            elif filter == 'I_BESS':
                if catalog[astrometry_stars[i]]['imag'] is ma.masked:
                    #print(f"Invalid magnitude for star {i}")
                    continue
                mag_catalog = catalog[astrometry_stars[i]]['imag']
                extinction_coefficient = 0.059
            elif filter == 'v_HIGH':
                zeropoint_value = 28.2
                mag_catalog = 1.0
                extinction_coefficient = 0.118
        except KeyError:
            print("No filter found!\n")
            # Don't calculate the zeropoint (in case there is no filter)
            header_has_filter = False
            break

        # Obtain the flux performing apperture photometry for the specified coordinates 
        magstar, magerrstar, fluxstar, fluxerr, skystar, skyerrstar, badflagstar, outstrstar = \
            aper.aper(data_image, x, y, phpadu=phpadu, apr=fwhm_image, zeropoint=zeropoint_value, \
            skyrad=[skyRmin, skyRmax], badpix=[-12000, 30000], exact=True, verbose=False)

        # PSF photometry
        pk = pkfit.pkfit_class(data_image, gauss, psf_residuals_model, ronois, phpadu)
        errmag, chi, sharp, niter, scale = pk.pkfit(1, x, y, skystar, 5)

        # Compute the flux and magnitude
        flux_value = (scale*10**(0.4*(25.0-hpsf['PSFMAG'])))/(header['EXPTIME'])
        dflux_value = errmag*10**(0.4*(25.0-hpsf['PSFMAG']))
        
        mag_value = zeropoint_value-2.5*np.log10(flux_value)
        
        # Compute the zeropoint of the star using the magnitude catalog and the extinction coefficient
        airmass = np.mean([header['HIERARCH ESO TEL AIRM END'], header['HIERARCH ESO TEL AIRM START']])
        zeropoint_star = mag_catalog + 2.5*np.log10(flux_value) + (extinction_coefficient * airmass)

        # Compute the magnitude error using the error propagation equation
        # sqrt(d(mag_value)/d(flux_value)**2 * dflux_value**2)
        mag_error = np.sqrt((-1.08573/flux_value)**2 * dflux_value**2)
        magnitude_errors[i] = mag_error

        # Append the catalog and calculated magnitudes
        catalog_magnitudes[i] = mag_catalog
        calculated_magnitudes[i] = mag_value

        # Compute the offset and append it to the offset array
        offset_array.append(mag_catalog - mag_value)
        zeropoint_array.append(zeropoint_star)

        flux_values.append(flux_value)
        
    # Compute the offset to apply for the zeropoint correction of the image
    if header_has_filter:
        offset = np.median(offset_array)
        if filter == 'v_HIGH':
            real_zeropoint = 28.2
        else:
            real_zeropoint = np.mean(zeropoint_array)

        magvalue_calibrated = magvalue + offset

    # Create a table with the extracted stars after astrometry
    astrometry_table = Table([[],[]], names=('X', 'Y'))

    extracted_keys = [k for k in astrometry_stars.keys()]
    for i in range(len(astrometry_stars)):
        astrometry_table.add_row([extracted_stars[extracted_keys[i]][0], extracted_stars[extracted_keys[i]][1]])

    fwhm_psf_2 = []
    ellipticity_psf_2 = []

    # Get dimension of the image
    x_dim = int(len(data_image[0]))
    y_dim = int(len(data_image))

    # Computing the FWHM and ellipticity of every star using PSF fitting with the astrometry corrected data
    for i in range(len(astrometry_stars)):
        star = astrometry_table[i]
        # Check that the star is located within acceptable values (not too close to the borders)
        checklocation = (star['X'] > 2.5*fwhm_nominal and star['X'] < (x_dim-2.5*fwhm_nominal) and  star['Y'] > 2.5*fwhm_nominal and star['Y'] < (y_dim-2.5*fwhm_nominal))
        if not checklocation:
            #print(f"Star #{i} too close to the borders, skipping...")   
            continue
        # PSF of the image using a single star (star #i)
        gauss, psf_residuals, psfmag = pf.getPSFimage(data_image, image_path, star, star['X'], star['Y'], ronois, phpadu, individual=True, pathpsf=path_psf)
        
        # Error, no convergence for the star
        if np.array_equal(gauss, np.array([-1,-1,-1,-1,-1])):
            continue
        
        # Compute the FWHM
        sigma_star = np.mean([gauss[3], gauss[4]])
        fwhm_star = 2.355*sigma_star
        fwhm_psf_2.append(fwhm_star)

        # Obtain the header of the PSF model generated
        hpsf = pyfits.getheader(path_psf + '/' + image[:-5] + '_psf_residuals.fits')
        # Determine the semi major and semi minor axis to obtain the ellipticity
        direction_x = hpsf['GAUSS4']
        direction_y = hpsf['GAUSS5']

        ellipticity_star = 0.0

        # Compute the ellipticity using the sextractor formula (1 - (semiminor axis)/(semimajor axis))
        if direction_x >= direction_y:
            ellipticity_star = 1 - (direction_y)/(direction_x)
        else:
            ellipticity_star = 1 - (direction_x)/(direction_y)

        ellipticity_psf_2.append(ellipticity_star)

    # File formatting to save the results
    stars_list.remove_columns(['ellipticity', 'mag', 'fwhm'])
    ra_table, dec_table = pf.xytowcs(image_path, stars_list['X'], stars_list['Y'])

    # Add the stars matched in the 2nd iteration, with it's catalog magnitude, RA and DEC
    catalog_magnitudes_tables = list()
    catalog_ra_table = list()
    catalog_dec_table = list()
    psf_magnitudes = list()
    psf_magnitudes_error = list()

    for i in range(len(extracted_stars)):
        if i in astrometry_stars and i in calculated_magnitudes:
            # Add the RA and DEC values of the matched star in the catalog
            catalog_ra_table.append(catalog[astrometry_stars[i]]['RA_ICRS'])
            catalog_dec_table.append(catalog[astrometry_stars[i]]['DE_ICRS'])
            # Add the catalog magnitude value of the matched star
            catalog_magnitudes_tables.append(catalog_magnitudes[i])
            # Add the PSF calculated magnitude and magnitude error
            psf_magnitudes.append(calculated_magnitudes[i])
            psf_magnitudes_error.append(magnitude_errors[i])
            
        else:
            catalog_ra_table.append(None)
            catalog_dec_table.append(None)
            catalog_magnitudes_tables.append(None)
            psf_magnitudes.append(None)
            psf_magnitudes_error.append(None)

    # Columns names
    stars_list['RA'] = ra_table
    stars_list['DEC'] = dec_table
    stars_list['Mag_CAT'] = catalog_magnitudes_tables
    stars_list['RA_CAT'] = catalog_ra_table
    stars_list['DEC_CAT'] = catalog_dec_table
    stars_list['Mag_PSF'] = psf_magnitudes
    stars_list['Mag_PSF_err'] = psf_magnitudes_error

    # Columns order
    column_order = ['X', 'Y', 'RA', 'DEC', 'flux', 'fluxerr', 'Mag_CAT', 'RA_CAT', 'DEC_CAT', 'Mag_PSF', 'Mag_PSF_err']
    table_to_file = stars_list[column_order]

    write_results(image)

    # Second Sextraction with a lower detection threshold, to detect fainter sources
    # <sigmas> or <threshold>,<ZP> in mag.arcsec-2, to detect fainter sources
    detect_thresh = 1.5

    # Extract all the stars from the image using Sextractor
    os.system('sex ' + image_path + ' -DETECT_THRESH ' + str(detect_thresh) + ' -CATALOG_NAME ' + image[:-5] + '.sex2d.txt')

    # Create the directory which will contain all the Sextractor related files with the name of the image
    current_directory = os.getcwd()
    image_directory = os.path.join(current_directory + '/results', r'{}'.format(image[:-5]))

    valid_stars, total_stars = create_valid_stars(image)

    """ print(f'Number of valid stars, i.e. stars with flag value 0: {valid_stars}')
    print(f'Number of total stars: {total_stars}')
    print(f'Percentage of valid stars: {valid_stars/total_stars*100}%') """

    # Read the list of coordinates obtained with sextractor for the given image, which flag values are 0 (i.e. not saturated stars)
    stars_file = image_directory + "/" + image[:-5] + '.sex2d.valid.txt' 
    stars_list = Table.read(stars_file, format='ascii', delimiter=' ')
    # Rename the first 3 columns, as index, X coordinate and Y coordinate. The starlist has another parameters, but for now they are not important
    # Also rename the 5th column, because the flux is important to filter bright and faint stars
    stars_list.rename_column('col1','ID')
    stars_list.rename_column('col2','X')
    stars_list.rename_column('col3','Y')
    stars_list.rename_column('col5', 'flux')
    stars_list.rename_column('col6', 'fluxerr')

    # Define the header and data of the image
    image_data = pyfits.getdata(image_path, ext=0)
    image_header = pyfits.getheader(image_path, ext=0)

    # Array which contains the measured magnitudes of the stars
    measured_magnitudes = list()
    magnitude_errors = list()

    for x, y in zip(stars_list['X'], stars_list['Y']):
        # Obtain the flux performing apperture photometry for the specified coordinates 
        magstar, magerrstar, fluxstar, fluxerr, skystar, skyerrstar, badflagstar, outstrstar = \
            aper.aper(image_data, x, y, phpadu=phpadu, apr=fwhm_image, zeropoint=real_zeropoint, \
            skyrad=[skyRmin, skyRmax], badpix=[-12000, 30000], exact=True, verbose=False)

        # PSF photometry
        pk = pkfit.pkfit_class(image_data, gauss, psf_residuals_model, ronois, phpadu)
        errmag, chi, sharp, niter, scale = pk.pkfit(1, x, y, skystar, 5)

        # Compute the flux and magnitude
        flux_value = (scale*10**(0.4*(25.0-hpsf['PSFMAG'])))/(header['EXPTIME'])
        dflux_value = errmag*10**(0.4*(25.0-hpsf['PSFMAG']))
        
        mag_value = real_zeropoint-2.5*np.log10(flux_value)

        # Compute the magnitude error
        mag_error = np.sqrt((-1.08573/flux_value)**2 * dflux_value**2)

        # Append the measured magnitude to the list
        measured_magnitudes.append(mag_value)
        magnitude_errors.append(mag_error)

    # File formatting (same as before)
    stars_list.remove_columns(['col4', 'col7', 'col8'])
    ra_table, dec_table = pf.xytowcs(image_path, stars_list['X'], stars_list['Y'])
    # Add the RA and DEC values of the stars in the table
    stars_list['RA'] = ra_table
    stars_list['DEC'] = dec_table
    # Add the measured magnitudes and magnitude errors to the table
    stars_list['Mag_PSF'] = measured_magnitudes
    stars_list['Mag_PSF_err'] = magnitude_errors

    # Columns order
    column_order = ['X', 'Y', 'RA', 'DEC', 'flux', 'fluxerr', 'Mag_PSF', 'Mag_PSF_err']
    table_to_file = stars_list[column_order]

    table_to_file.write(image_directory + '/results_faint.txt', format='ascii', delimiter=',')

    # Delete all files that end with '.sex2d.txt' and with '_astrometry.fits'
    for file in os.listdir(os.getcwd()):
        if file.endswith('.sex2d.txt') or file.endswith('_astrometry.fits'):
            os.remove(file)