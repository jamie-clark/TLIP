"""
Created on DATE

@author: jamieclark
"""
import numpy as np
from LIA import models, microlensing_classifier, noise_models, training_set
from pyDANDIA import phot_db
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.table import Table, vstack
from astropy.io import ascii
from time import time
import matplotlib.pyplot as plt
import random
from pyLIMA import event, telescopes, microlmodels
from contextlib import contextmanager
import sys, os
import importlib
from tqdm import tqdm

# Script to temporarily suppress output from functions
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def ROME_classification_script(ra, dec, radius, db_file, LIA_directory, filt_choice='3', tel_choices=[1, 2], mag_cutoff=[14,17], mag_err_cutoff=0.1):
	"""Runs the LIA classification algorithm on all stars within a certain region of a ROME FIELD,
	writing the results to text files on the local system drive. Requires a DB of ROME data.
	Parameters
	__________
	ra : string
		Central RA(J2000) in 'hh:mm:ss' string format.
	dec : string
		Central DEC(J2000) in 'hh:mm:ss' string format.
	radius: float
		Radius for box search in the database in arcminutes.
	db_file : string
		System path of the database file on the machine.
	LIA_directory : string
		System path of the directory that the all_features.py and pca_features.py files are in.
	filt_choice : string, optional
		Number of corresponding telescope filter requested. List of all options can be found below.
		Defaults to '3' for I filterband.
	tel_choices : array, optional
		List of corresponding telescopes requested. List of all options can be found below.
		Defaults to [1, 2] for lsc-doma in Chile. Recommend keeping sites seperate.
	mag_cutoff: array, optional
		Sets the upper and lower cutoff points for what we count as too bright/dim of a star of a star.
		Stars with magnitudes that frequently dip below this point will be excluded. Defaults to [14,17].
	mag_err_cutoff: int, optional
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we discard them from our calculations. Defaults to 0.1.

	Outputs
	_______
	results : txt file
		A txt file containing all results of the classification script in table format, 
		included in columns of star id, ra, dec, filter and telescope used, predicted class,
		and prediction probability. 
	results_truncated : txt file
		A txt file containing truncated results of the classification script, printing
		a list of all microlensing candidates in detail.
	"""
	# generates the LIA models
	rf, pca = models.create_models(str(LIA_directory)+'all_features.txt', str(LIA_directory)+'pca_features.txt')

	
	# search db for stars in proper region
	conn = phot_db.get_connection(dsn=db_file)
	center = SkyCoord(ra,dec, frame='icrs', unit=(units.hourangle, units.deg))
	results = phot_db.box_search_on_position(conn, center.ra.deg, center.dec.deg, radius/60.0, radius/60.0)

	if len(results) == 0: 
		print("Error: No stars found in this region!")
		exit()

	star_ids = []
	predictions = []
	probabilities = []
	microlensing_probabilities = []

	info_list = []
	print(results['star_id'])
	print("Beginning database query and calculating predictions on " + str(len(results)) + " stars...")
	for star_id in tqdm(results['star_id']) :
		prediction, probability, ML_probability, mag, magerr = [[], [], [], [], []]
		try:
			for tel_choice in tel_choices: 
				mag, magerr = np.append([mag, magerr], extract_lightcurve(star_id, db_file, filt_choice, tel_choice, mag_err_cutoff)[1:3], axis=1)
			if len(mag) <= 10: continue
			prediction, probability, ML_probability = extract_results(mag, magerr, rf, pca, mag_cutoff, mag_err_cutoff)
			ra, dec = extract_ra_dec(star_id, db_file)
			info_list.append([star_id, ra, dec, str(filt_choice)+"_"+"-".join(str(x) for x in tel_choices), prediction, probability, ML_probability])
		except:
			pass

	timestamp = time()
	# Generate results.txt file
	try:
		results_table = Table(rows=info_list, names=('star_id', 'ra', 'dec', 'filter_telescope', 'prediction', 'probability', 'ml_probability'), meta={'name': 'full results table'})
		results_table.sort('prediction', 'probability')
		ascii.write(results_table, "results_"+str(timestamp)+".txt", overwrite=True)
	except:
		print("Writing table failed.")

	# Generate results_truncated.txt file
	try:
		results_table.add_index('prediction')
		ml_table = results_table.loc['ML']
		ascii.write(ml_table, "results_truncated.txt", overwrite=True)
	except:
		print("No microlensing events detected.")
	print("Text files generated. Program complete.")


def extract_results(mag, magerr, rf, pca, mag_cutoff, mag_err_cutoff=0.1):
	"""Runs LIA on a lightcurve and outputs the parameters of the classification.

	Parameters
	__________
	mag : array
		Mag is the magnitude of the object.
	magerr : array
		Magerr is the uncertainty in the magnitude measurement.
	rf : LIA model
		Refers to the random forest LIA model generated by LIA's models.create_models() function.
		See ROME_classification_script() for when these are generated.
	pca : LIA model
		Refers to the PCA LIA model generated by LIA's models.create_models() function.
		See ROME_classification_script() for when these are generated.
	mag_err_cutoff: int, optional
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we may choose to discard them from our calculations. Defaults to 0.1.

	Outputs
	_______
	prediction : string
		The predicted class of the object. Options are 'VAR' for Variable, 'CONS' for Constant,
		'CV' for Cataclysmic Variable, and 'ML' for Microlensing.
	probability : float
		The probability of certainty in this classification as reported by LIA.
	ml_prediction : float
		The probability that this event could be a microlensing event, as reported by LIA.
	"""

	# Cut out entries where the magnitude is dimmer/brighter than the cutoff values, 
	# entries where the image is completely saturated, entries with less than three entries, 
	# and entries where there are errors
	if (not all(i <= mag_cutoff[1] for i in mag)) or (not all(i >= mag_cutoff[0] for i in mag)) or (len(list(set(mag))) < 3) or (True in np.isnan(mag)):
		return()
	else:
		prediction, ml_pred, cons_pred, cv_pred, var_pred = microlensing_classifier.predict(mag, magerr, rf, pca)[0:5]
		probability = max(ml_pred[0], cons_pred[0], cv_pred[0], var_pred[0])
		result = [prediction, probability, ml_pred[0]]
		return(result)

def extract_lightcurve(star_id, db_file, filt_choice='3', tel_choice=2, mag_err_cutoff=0.1):
	"""Given the ID of a star and parameters, extracts the lightcurve (hjd, mag, magerr) of
	the object from the DB.

	Parameters
	__________
	star_id : int
		ID of the specified star in the specified database.
	db_file : string
		System path of the database file on the machine.
	filt_choice : string, optional
		Number of corresponding telescope filter requested. List of all options can be found below.
		Defaults to '3' for I filterband.
	tel_choice : int, optional
		Number of corresponding site location requested. List of all options can be found below.
		Defaults to 2 for lsc-doma-1m0a-fl15 in Chile.
	mag_err_cutoff: int, optional
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we discard them from our calculations. Defaults to 0.1.

	Outputs
	_______
	hjd : float
		HJD refers to the Heliocentric Julian Date, the timestamp of the event.
	mag : float
		Mag is the magnitude of the object.
	magerr : float
		Magerr is the uncertainty in the magnitude measurement.
	"""
	# connect to db and query all lightcurve data for specified object
	conn = phot_db.get_connection(dsn=db_file)
	query = 'SELECT hjd, image, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(star_id)+'" AND filter="'+str(filt_choice)+'" AND facility="'+str(tel_choice)+'"'
	phot_table = phot_db.query_to_astropy_table(conn, query, args=())
	
	# mask data based on magnitude (ignore -9999 values) and discard points with high uncertainty
	exposures = []
	for img_id in np.asarray(phot_table['image']):
		query = 'SELECT exposure_time FROM images WHERE img_id="'+str(img_id)+'"'
		exposure_time = phot_db.query_to_astropy_table(conn, query, args=())
		exposures.append(exposure_time[0][0])
	exposures = np.asarray(exposures)
	mag = np.asarray(phot_table['calibrated_mag'])
	magerr = np.asarray(phot_table['calibrated_mag_err'])
	hjd = np.asarray(phot_table['hjd'])
	image = np.asarray(phot_table['image'])
	mask = np.all([mag >0, magerr <mag_err_cutoff, exposures==300], axis=0)
	mag = mag[mask]
	magerr = magerr[mask]
	hjd = hjd[mask]
	image = image[mask]
	# include systematic errors associated with images
	errors = np.loadtxt('systematic_errors.txt')
	errors = Table(rows=errors, names=('image', 'sys_error'))
	#errors = errors[0]

	data = Table([mag, magerr, hjd, image], names=('mag', 'magerr', 'hjd', 'image'))

	mags2 = []
	magerr2 = []
	hjd2 = []

	for mag,magerr,hjd,image in data:
		try:
			result = np.where(np.asarray(errors['image']) == image)
			location = result[0][0]
			error = errors[location][1]
			new_error = np.sqrt(float(magerr)**2+float(error)**2)
			
		except:
			#print("We found an error.")
			new_error = magerr
		mags2.append(mag)
		hjd2.append(hjd)
		magerr2.append(new_error)
	mask = np.asarray(magerr2) <mag_err_cutoff
	mag = np.asarray(mags2)[mask]
	magerr = np.asarray(magerr2)[mask]
	hjd = np.asarray(hjd2)[mask]
	return(hjd, mag, magerr)

def plot_lightcurve(hjd, mag, magerr):
	"""Given a lightcurve, plots the results for the user to see using matplotlib.

	Parameters
	__________
	hjd : float
		HJD refers to the Heliocentric Julian Date, the timestamp of the event.
	mag : float
		Mag is the magnitude of the object.
	magerr : float
		Magerr is the uncertainty in the magnitude measurement.

	Outputs
	_______
	plot : plot
		Results are plotted on the user's screen.
	"""
	plt.scatter(np.asarray(hjd-2450000), mag)
	plt.gca().invert_yaxis()
	plt.errorbar(np.asarray(hjd)-2450000, mag, yerr=magerr, linestyle="None")
	plt.show()

def extract_ra_dec(star_id, db_file):
	"""Given a star ID and DB, extracts the RA and Dec of the star.

	Parameters
	__________
	star_id : int
		ID of the specified star in the specified database.
	db_file : string
		System path of the database file on the machine.

	Outputs
	_______
	ra : string
		The right ascension(J2000) in hh:mm:ss format.
	dec : string
		The declination(J2000) in dd:mm:ss format.
	"""
	conn = phot_db.get_connection(dsn=db_file)
	query = 'SELECT ra, dec FROM stars WHERE star_id="'+str(star_id)+'"'
	phot_table = phot_db.query_to_astropy_table(conn, query, args=())
	ra = phot_table['ra'][0]
	dec = phot_table['dec'][0]
	center = SkyCoord(ra,dec, frame='icrs', unit=(units.deg, units.deg)).to_string("hmsdms")
	x = center.split(" ")
	x[0] = x[0].replace("h", ":"); x[0] = x[0].replace("m", ":"); x[0] = x[0].replace("s", "");
	x[1] = x[1].replace("d", ":"); x[1] = x[1].replace("m", ":"); x[1] = x[1].replace("s", "");
	ra, dec = x
	return (ra, dec)


def plot_all_lightcurves(star_id, db_file, mag_err_cutoff=0.1):
	"""Given the ID of a star, plots lightcurves from all filter and site combinations
	for the user to view on one graph.

	Parameters
	__________
	star_id : int
		ID of the specified star in the specified database.
	db_file : string
		System path of the database file on the machine.
	mag_err_cutoff: int, optional
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we discard them from our calculations. Defaults to 0.1.

	Outputs
	_______
	plot : plot
		Results are plotted on the user's screen.
	"""
	color_list = ['b', 'r', 'k']
	marker_list = ['.', 'o', '^', 's', 'x', '+', 'D']
	filter_list = [1,2,3]
	#site_list = [[[1,2], "LSC-DOMA"], [[8, 12], "CPT-DOMA"], [[3], "LSC-DOMB"], [[4, 5], "LSC-DOMC"], [[6, 10], "COJ-DOMA"], [[7, 11], "COJ-DOMB"], [[9], "CPT-DOMC"]]
	site_list =[[[1,4], "LSC-DOMA"], [[2,5], "COJ-DOMA"], [[3, 6], "CPT-DOMA"], [[7, 8], "LSC-DOMC"], [[9], "LSC-DOMB"], [[10, 12], "COJ-DOMB"], [[11], "CPT-DOMC"]]
	plt.gca().invert_yaxis()
	
	for x,filt in enumerate(filter_list):
		color = color_list[x]
		for y,site in enumerate(site_list):
			hjd, mag, magerr = [[], [], []]
			tel_choices = site[0]
			site_name = site[1]
			marker = marker_list[y]
			for tel_choice in tel_choices: 
				hjd, mag, magerr = np.append([hjd, mag, magerr], extract_lightcurve(star_id=star_id, db_file=db_file, filt_choice=filt, tel_choice=tel_choice, mag_err_cutoff=mag_err_cutoff), axis=1)
			plt.scatter(np.asarray(hjd-2450000), mag, c=color, marker=marker, label=site_name)
			plt.errorbar(np.asarray(hjd)-2450000, mag, c=color, marker=marker, yerr=magerr, linestyle="None")	
	
	plt.legend(loc='best')
	plt.xlabel('HJD', fontsize=15)
	plt.ylabel('Magnitude',fontsize=15)
	plt.show()


def create_training_set(db_file, filt_choice, tel_choices, mag_err_cutoff=0.1):
	"""Generates an LIA training set with specified parameters and a random sampling
	of survey cadence from the DB.

	Parameters
	__________
	db_file : string
		System path of the database file on the machine.
	filt_choice : string, optional
		Number of corresponding telescope filter requested. List of all options can be found below.
		Defaults to '3' for I filterband.
	tel_choices : array, optional
		List of corresponding telescopes requested. List of all options can be found below.
		Recommend keeping sites seperate.
	mag_err_cutoff: int, optional
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we discard them from our calculations. Defaults to 0.1.

	Outputs (from LIA)
	_______
	lightcurves : FITS
        All simulated lightcurves in a FITS file, sorted by class and ID
    all_features : txt file
        A txt file containing all the features plus class label.
    pca_stats : txt file
        A txt file containing all PCA features plus class label. 
	"""
	random_integers = []
	for i in tqdm(range(1, 501)):
		random_integers.append(random.randint(1, 162355))

	timestamps = []
	for star_id in tqdm(random_integers):
		hjd = []
		try:
			for tel_choice in tel_choices: 
				hjd = np.append(hjd, extract_lightcurve(star_id, db_file, filt_choice, tel_choice, mag_err_cutoff)[0])
			if len(hjd) > 10:
				timestamps.append(hjd)
		except:
			pass

	# PTF noise model
	median = [14.3, 14.75, 15.3, 15.8, 16.3, 16.8, 17.4, 17.9, 18.4, 18.8, 19.3, 19.8, 20.4, 20.9, 21.5]
	rms = [0.01, 0.0092, 0.0094, 0.01, 0.012, 0.014, 0.018, 0.022, 0.032, 0.048, 0.065, 0.09, 0.11, 0.17, 0.2]
	ptf_model = noise_models.create_noise(median, rms)


	# ROME noise model
	median = [13.5,14.0,14.5,15.0,15.5,16.0,16.5,17.0,17.5,18.0,18.5,19.0]
	rms = [0.01,0.02,0.03, 0.05,0.07,0.08,0.13,0.17,0.4,0.5,0.6,1.0]
	rome_model = noise_models.create_noise(median, rms)


	# Calculated noise model
	median = [13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5]
	rms = [0.042, 0.024 , 0.013, 0.015 , 0.034, 0.075, 0.144, 0.244, 0.380, 0.558, 0.782, 1.056, 1.386, 1.777, 2.233]
	first_extracted_model = noise_models.create_noise(median, rms)

	median = [13. , 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14. ,14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15. , 15.1,15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16. , 16.1, 16.2,16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17. , 17.1, 17.2, 17.3,17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 18. , 18.1, 18.2, 18.3, 18.4,18.5, 18.6, 18.7, 18.8, 18.9, 19.]
	rms = [0.0067, 0.0068, 0.007 , 0.0072, 0.0074, 0.0076, 0.0079, 0.0083,0.0086, 0.0091, 0.0096, 0.0102, 0.0108, 0.0115, 0.0124, 0.0133,0.0143, 0.0155, 0.0168, 0.0182, 0.0197, 0.0215, 0.0234, 0.0255,0.0278, 0.0304, 0.0332, 0.0363, 0.0397, 0.0435, 0.0476, 0.0521,0.0571, 0.0626, 0.0685, 0.0751, 0.0823, 0.0902, 0.0989, 0.1084,0.1188, 0.1302, 0.1428, 0.1565, 0.1716, 0.1882, 0.2063, 0.2262,0.248 , 0.2719, 0.2981, 0.3269, 0.3584, 0.3929, 0.4309, 0.4724,0.518 , 0.5679, 0.6227, 0.6828, 0.7487]
	second_extracted_model = noise_models.create_noise(median, rms)

	median = np.array([12. , 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13. ,
       13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14. , 14.1,
       14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15. , 15.1, 15.2,
       15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16. , 16.1, 16.2, 16.3,
       16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17. , 17.1, 17.2, 17.3, 17.4,
       17.5, 17.6, 17.7, 17.8, 17.9, 18. , 18.1, 18.2, 18.3, 18.4, 18.5,
       18.6, 18.7, 18.8, 18.9, 19.])
	median = median[::2]
	rms = np.array([0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
       0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
       0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
       0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
       0.01      , 0.01      , 0.01      , 0.01      , 0.01      ,
       0.01      , 0.01      , 0.01008718, 0.01106038, 0.01212746,
       0.0132975 , 0.01458042, 0.01598711, 0.01752952, 0.01922073,
       0.02107511, 0.0231084 , 0.02533786, 0.02778241, 0.03046281,
       0.03340181, 0.03662435, 0.0401578 , 0.04403216, 0.0482803 ,
       0.05293829, 0.05804569, 0.06364583, 0.06978626, 0.07651912,
       0.08390154, 0.09199621, 0.10087184, 0.11060377, 0.12127463,
       0.13297498, 0.14580417, 0.15987109, 0.17529517, 0.19220733,
       0.21075115, 0.23108404, 0.25337861, 0.27782412, 0.30462809,
       0.33401806, 0.36624352, 0.40157803, 0.44032156, 0.48280299,
       0.52938295])
	rms = rms[::2]

	third_extracted_model = noise_models.create_noise(median, rms)

	training_set.create(timestamps, min_mag=13, max_mag=18, noise=third_extracted_model, n_class=500)
	print("Training set generated. Program complete.")


def drip_feed(star_id, db_file, LIA_directory, filt_choice, tel_choices, mag_cutoff, mag_err_cutoff):
	"""Generates an "drip feed" plot where the user can see the evolution of the LIA prediction
	over time.

	Parameters
	__________
	star_id : int
		ID of the specified star in the specified database.
	db_file : string
		System path of the database file on the machine.
	LIA_directory : string
		System path of the directory that the all_features.py and pca_features.py files are in.
	filt_choice : string, optional
		Number of corresponding telescope filter requested. List of all options can be found below.
		Defaults to '3' for I filterband.
	tel_choices : array, optional
		List of corresponding telescopes requested. List of all options can be found below.
		Recommend keeping sites seperate.
	mag_cutoff: array
		Sets the upper and lower cutoff points for what we count as too bright/dim of a star of a star.
		Stars with magnitudes that frequently dip below this point will be excluded. Defaults to [14,17].
	mag_err_cutoff: int
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we discard them from our calculations. Defaults to 0.1.

	Outputs 
	_______
	plot : plot
		Results are plotted on the user's screen.
	"""
	# daniel's code
	rf, pca = models.create_models(str(LIA_directory)+'all_features.txt', str(LIA_directory)+'pca_features.txt')
	mjd, mag, magerr = [[], [], []]
	#try:
	for tel_choice in tel_choices:
		mjd, mag, magerr = np.append([mjd, mag, magerr], extract_lightcurve(star_id, db_file, filt_choice, tel_choice, mag_err_cutoff),axis=1)
	#if len(mjd) == 0:
	#	print("No data was found for this object with these settings.")
	#	continue
	mjd = np.asarray(mjd) - 2450000
	sosort = np.array([mjd,mag,magerr]).T
	sosort = sosort[sosort[:,0].argsort(),]

	mjd = sosort[:,0]
	mag = sosort[:,1]
	magerr = sosort[:,2]

	prob_pred_list=[]
	prob_pred_list2=[]
	prob_pred_list3=[]
	prob_pred_list4=[]

	for i in tqdm(range(3, len(mjd)-1, 1)):
	               
	                current_mag = mag[0:i+1]
	                current_magerr = magerr[0:i+1]
	               
	                try:
	                    pred, prob_pred, prob_pred2, prob_pred3, prob_pred4 = microlensing_classifier.predict(current_mag, current_magerr, rf, pca)
	                except ValueError or ZeroDivisionError:
	                    pred = 'BAD'
	                    prob_pred = 0.0
	                    prob_pred2 = 0.0
	                    prob_pred3 = 0.0
	                    prob_pred4 = 0.0

	                prob_pred_list.append(prob_pred)
	                prob_pred_list2.append(prob_pred2)
	                prob_pred_list3.append(prob_pred3)
	                prob_pred_list4.append(prob_pred4) #ML


	#PLOT WITH SHARED AXIS
	import matplotlib.pyplot as plt

	fig=plt.figure()
	ax1=plt.subplot(2,1,1)
	plt.errorbar(mjd, mag, yerr=magerr, fmt='ro', label = 'i')
	plt.legend(loc=2,prop={'size': 20})
	plt.ylabel('Magnitude', fontsize=30)
	plt.title('ID: '+str(star_id), fontsize=30)	
	plt.xlim(mjd[0], mjd[-2])
	#plt.title('OGLE-1999-BUL-40', fontsize=40)
	ax1.tick_params(axis='y', labelsize=20)
	ax1.tick_params(axis='x', labelsize=20)
	plt.gca().invert_yaxis()

	ax2=plt.subplot(2,1,2)
	plt.plot(mjd[3:len(mjd)-1], prob_pred_list2, 'yo-', label = 'CONS')
	plt.plot(mjd[3:len(mjd)-1], prob_pred_list3, 'rv-', label = 'CV')
	plt.plot(mjd[3:len(mjd)-1], prob_pred_list4, 'bs-', label = 'VAR')
	plt.plot(mjd[3:len(mjd)-1], prob_pred_list, 'g>-', label = 'ML')
	plt.legend(loc = 2, prop={'size': 17})
	plt.xlim(mjd[0], mjd[-2])
	plt.ylim(0, 1.05)
	plt.xlabel('HJD', fontsize=30)
	plt.ylabel('Probability Prediction',fontsize=30)

	ax2.tick_params(axis='x', labelsize=20)
	ax2.tick_params(axis='y', labelsize=20)
	ax1.get_shared_x_axes().join(ax1, ax2)
	plt.show()

def extract_lightcurves_on_position(ra, dec, radius, db_file, filt_choice='3', tel_choices=[1, 2], mag_err_cutoff=0.1):
	"""Similar to the ROME_classification_script, this script uses ra, dec, radius
	to find stars in a region, and then extracts the lightcurves of these stars, writing
	them to a file.

	Parameters
	__________
	ra : string
		Central RA(J2000) in 'hh:mm:ss' string format.
	dec : string
		Central DEC(J2000) in 'hh:mm:ss' string format.
	radius: float
		Radius for box search in the database in arcminutes.
	db_file : string
		System path of the database file on the machine.
	filt_choice : string, optional
		Number of corresponding telescope filter requested. List of all options can be found below.
		Defaults to '3' for I filterband.
	tel_choices : array, optional
		List of corresponding telescopes requested. List of all options can be found below.
		Defaults to [1,2] for Dome A in Chile. Recommend keeping sites seperate.
	mag_err_cutoff: int, optional
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we discard them from our calculations. Defaults to 0.1.

	Outputs
	_______
	lightcurves : txt file
		Text file containing the lightcurves of all stars in the region.
	"""
	# SEARCH DB FOR STARS IN PROPER REGION
	conn = phot_db.get_connection(dsn=db_file)
	center = SkyCoord(ra,dec, frame='icrs', unit=(units.hourangle, units.deg))
	results = phot_db.box_search_on_position(conn, center.ra.deg, center.dec.deg, radius/60.0, radius/60.0)

	if len(results) == 0: 
		print("Error: No stars found in this region!")
		exit()

	star_ids = []
	times = []
	mags = []
	magerrs = []

	info_list = []


	print("Beginning database query and calculating predictions on " + str(len(results)) + " stars...")

	for star_idx,star_id in enumerate(results['star_id']) :
		time, mag, magerr = [[], [], []]
		try:
			for tel_choice in tel_choices: 
				print(mag)	
				time, mag, magerr = np.append([time, mag, magerr], extract_lightcurve(star_id, db_file, filt_choice, tel_choice, mag_err_cutoff), axis=1)
				print(mag)
			star_ids = np.append(star_ids, np.repeat(star_id, len(mag)))
			times, mags, magerrs = np.append([times, mags, magerrs], [time, mag, magerr], axis=1)
		except:
			pass


	# Generate results.txt file
	print(mags)
	print(star_ids)
	results_table = Table({'star_id': star_ids, 'hjd': times, 'mag': mags, 'magerr': magerrs}, names=('star_id', 'hjd', 'mag', 'magerr'))
	ascii.write(results_table, "lightcurves.txt", overwrite=True)
	print("Text file generated. Program complete.")

def pyLIMA_plot_from_db(ID, db,filt_choice='3', tel_choices = [6,10], mag_err_cutoff=0.5,debug=False):
	"""Similar to the ROME_classification_script, this script uses ra, dec, radius
	to find stars in a region, and then extracts the lightcurves of these stars, writing
	them to a file.

	Parameters
	__________
	ID : int
		ID of the specified star in the specified database.
	db : string
		System path of the database file on the machine.
	filt_choice : string, optional
		Number of corresponding telescope filter requested. List of all options can be found below.
		Defaults to '3' for I filterband.
	tel_choices : array, optional
		List of corresponding telescopes requested. List of all options can be found below.
		Defaults to [1,2] for Dome A in Chile. Recommend keeping sites seperate.
	mag_err_cutoff: int, optional
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we discard them from our calculations. Defaults to 0.1.
	debug : boolean, optional
		if enabled (set to True), imports python debugger and activates it. Defaults to False.
	Outputs
	_______
	plot : plot
		Results are plotted on the user's screen.
	"""
	if debug: import pdb; pdb.set_trace();

	ra, dec = extract_ra_dec(ID, db)

	center = SkyCoord(ra,dec, frame='icrs', unit=(units.hourangle, units.deg))


	your_event = event.Event()
	your_event.name = str(ID)
	your_event.ra = float(center.to_string("decimal").split()[0])
	your_event.dec = float(center.to_string("decimal").split()[1])

	for tel_choice in tel_choices:
		photometry = extract_lightcurve(ID, db, filt_choice=filt_choice, tel_choice=tel_choice, mag_err_cutoff=mag_err_cutoff)
		photo_table = Table(photometry, names=['hjd', 'mag', 'magerr'])	
		if tel_choice == tel_choices[0]:
			ascii.write(photo_table, "Survey_A.dat", overwrite=True, format='no_header')
			data_1 = np.loadtxt('./Survey_A.dat')
		else:
			ascii.write(photo_table, "Survey_A.dat", overwrite=True, format='no_header')
			data_2 = np.loadtxt('./Survey_A.dat')
			try: data_1 = np.append(data_1, data_2, axis=0)
			except: pass
			

	#print(data_1)

	with suppress_stdout():
	
		telescope_1 = telescopes.Telescope(name='LCO', camera_filter='I', light_curve_magnitude=data_1)

		your_event.telescopes.append(telescope_1)
		your_event.find_survey('LCO')
		your_event.check_event()

		### Construct the model you want to fit. Let's go basic with a PSPL, without second_order effects :
		model_1 = microlmodels.create_model('PSPL', your_event)

		### Let's try with the simplest Levenvberg_Marquardt algorithm :
		your_event.fit(model_1,'LM')

		### Let's see some plots.
		your_event.fits[0].produce_outputs()

	print('Chi2_LM :',your_event.fits[0].outputs.fit_parameters.chichi)
	#import pdb; pdb.set_trace()
#	print(your_event.fits[0].outputs.fit_parameters.te)
	plt.show()
	plt.clf()
	plt.cla()
	plt.close()


def pyLIMA_classification(IDs, db ,filt_choice='3', tel_choices = [1,2],mag_err_cutoff=0.1, debug=False):
	"""Similar to the ROME_classification_script, this script uses ra, dec, radius
	to find stars in a region, and then extracts the lightcurves of these stars, writing
	them to a file.

	Parameters
	__________
	IDs : array
		Array containing the IDs of specified stars in the specified database.
	db : string
		System path of the database file on the machine.
	filt_choice : string, optional
		Number of corresponding telescope filter requested. List of all options can be found below.
		Defaults to '3' for I filterband.
	tel_choices : array, optional
		List of corresponding telescopes requested. List of all options can be found below.
		Defaults to [1,2] for Dome A in Chile. Recommend keeping sites seperate.
	mag_err_cutoff: int, optional
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we discard them from our calculations. Defaults to 0.1.
	debug : boolean, optional
		if enabled (set to True), imports python debugger and activates it. Defaults to False.
	Outputs
	_______
	IDs : array
		returns IDs that passed the criteria (chi squared < 10, 0 < uo <= 1, 10 <= tE <= 50)
	"""
	if debug: import pdb; pdb.set_trace();
	errors = list()
	print("Running pyLIMA classification on "+str(len(IDs))+" objects...")
	print("Estimated runtime of "+str(len(IDs)/222.3)+" minutes.")
	results = []
	for ID in tqdm(IDs):
		ra, dec = extract_ra_dec(ID, db)

		center = SkyCoord(ra,dec, frame='icrs', unit=(units.hourangle, units.deg))

		your_event = event.Event()
		your_event.name = 'OGLE-2018-BLG-0022'
		your_event.ra = float(center.to_string("decimal").split()[0])
		your_event.dec = float(center.to_string("decimal").split()[1])

		

		for tel_choice in tel_choices:
			photometry = extract_lightcurve(ID, db, tel_choice=tel_choice, mag_err_cutoff=mag_err_cutoff)
			photo_table = Table(photometry, names=['hjd', 'mag', 'magerr'])	
			if tel_choice == tel_choices[0]:
				ascii.write(photo_table, "Survey_A.dat", overwrite=True, format='no_header')
				data_1 = np.loadtxt('./Survey_A.dat')
			else:
				ascii.write(photo_table, "Survey_A.dat", overwrite=True, format='no_header')
				data_2 = np.loadtxt('./Survey_A.dat')
				try: data_1 = np.append(data_1, data_2, axis=0)
				except: pass
		try:		
			with suppress_stdout():
				#print(data_1)
				if len(data_1) < 6: continue
				telescope_1 = telescopes.Telescope(name='LCO', camera_filter='I', light_curve_magnitude=data_1)

				your_event.telescopes.append(telescope_1)
				your_event.find_survey('LCO')
				your_event.check_event()

				### Construct the model you want to fit. Let's go basic with a PSPL, without second_order effects :
				model_1 = microlmodels.create_model('PSPL', your_event)

				### Let's try with the simplest Levenvberg_Marquardt algorithm :
				your_event.fit(model_1,'LM')

				### Let's see some plots.
				your_event.fits[0].produce_outputs()



			outputs = your_event.fits[0].outputs.fit_parameters
			if outputs.chichi < 10 and 0 < outputs.uo <= 1 and 10 <= outputs.tE <= 50:
				results.append(ID)
		except:
			errors.append(ID)
	importlib.reload(plt)
	if len(errors) > 0: 
		print("We had "+str(len(errors))+" errors on targets: "+str(errors))
	else:
		print("We had no pyLIMA errors.")
	print("Out of "+str(len(IDs))+" ML candidates, we found "+str(len(results))+" targets that passed our criteria.")
	return results

def full_pipeline():
	"""Similar to the ROME_classification_script, this script uses ra, dec, radius
	to find stars in a region, and then extracts the lightcurves of these stars, writing
	them to a file.

	Parameters
	__________
	###
	###
	Outputs
	_______
	###
	###
	"""
	ID = 107503 
	db = '/home/jclark/examples/ROME-FIELD-16_phot.db'
	tel_choices = [1,4]
	mag_err_cutoff=0.5
	prob_cutoff = 0.20
	prob_override = 0.50
	radius=60

	#import pdb; pdb.set_trace()
	#print(extract_ra_dec(33065, '/home/jclark/examples/ROME-FIELD-16_phot.db'))

	#drip_feed(ID, '/home/jclark/examples/ROME-FIELD-16_phot.db', 
	#'/home/jclark/examples/','3',[6,10],20,0.25)
	ROME_classification_script('17:59:27.3125', '-28:36:34.1901', radius,
	'/home/jclark/examples/ROME-FIELD-16_phot.db', '/home/jclark/examples/','3',tel_choices,[14,21],mag_err_cutoff)

	#create_training_set('/home/jclark/examples/ROME-FIELD-16_phot.db','3',[1,2],17.5,0.25)

	#extract_lightcurves_on_position('17:59:27.04', '-28:36:37.00', 0.1,
	#'/home/jclark/examples/ROME-FIELD-16_phot.db', '3',[1,2],[13,20],0.25)

	#plot_all_lightcurves(ID, '/home/jclark/examples/ROME-FIELD-16_phot.db', 0.25)
	#for ID in [36055, 107503, 42866, 10203, 100, 75000, 3003, 135000, 22031, 83000]:
	#	extract_lightcurve(ID,db,mag_err_cutoff=0.5)
	#star_id ra dec filter_telescope prediction probability ml_probability

	ml_candidates = np.loadtxt('results_truncated.txt', skiprows=1, usecols=0)
	ml_candidates_mask = np.loadtxt('results_truncated.txt', skiprows=1, usecols=5) > prob_cutoff
	ml_candidates = ml_candidates[ml_candidates_mask]


	results = pyLIMA_classification(ml_candidates, db, tel_choices=tel_choices,mag_err_cutoff=mag_err_cutoff)
	overrides = ml_candidates[np.loadtxt('results_truncated.txt', skiprows=1, usecols=5) > prob_override]
	#import pdb; pdb.set_trace()
	y = np.asarray([results.count(i) for i in overrides])
	overrides = overrides[y == 0]
	print("We had "+str(len(overrides))+" overrides.")
	for x in results:
		print("Result ID: "+str(x))
		#pyLIMA_plot_from_db(x,db,tel_choices=tel_choices,mag_err_cutoff=mag_err_cutoff)
	for x in overrides:
		#pyLIMA_plot_from_db(x,db,tel_choices=tel_choices,mag_err_cutoff=mag_err_cutoff)
		print("Override ID: "+str(x))
#	for target in np.append(results, overrides):
		# process target
#		if seperation < Xarcseconds:
			# target identified
			# add it to the page and cross listed catalog


def new_extract_med_std(conn,ra,dec,radius,filt_choice,telos,overriding_results=[]):


	

	if len(overriding_results) != 0:
		results = Table({'star_id':overriding_results})
	else:
		center= SkyCoord(ra,dec, frame='icrs', unit=(units.hourangle, units.deg))
		results = phot_db.box_search_on_position(conn, center.ra.deg, center.dec.deg, radius, radius)

	medians = []
	stds = []
	stars = []
	if len(results) > 0:
		
		for star_id in tqdm(results['star_id']):
			for telo in telos:
				query = 'SELECT filter, facility, hjd, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(star_id)+'" AND filter="'+str(filt_choice)+'" AND facility="'+str(telo)+'"'	    	
				if telo == telos[0]:
					phot_table = phot_db.query_to_astropy_table(conn, query, args=())
				else:
					phot_table = vstack([phot_table, phot_db.query_to_astropy_table(conn, query, args=())])
			#import pdb; pdb.set_trace()
			mask =  phot_table['calibrated_mag'] >0
			uncerts = np.asarray(phot_table['calibrated_mag_err'][mask])
			mags = np.asarray(phot_table['calibrated_mag'][mask])
			mean = weighted_mean(mags, uncerts)
			std = weighted_standard_deviation(mags, uncerts, mean)
			if (mean>10) and (len(mags)>5) and (std>0):
			
				medians.append(mean)
				stds.append(std)
				stars.append(star_id)
			

	return medians,stds,stars


def plot_cam12_lightcurves(star_id, db_file, mag_err_cutoff):
	color_list = ['k']
	marker_list = ['.', 'x']
	filter_list = [3]
	site_list = [[[1], "CAM 1"], [[2], "CAM 2"]]
	plt.gca().invert_yaxis()
	for x,filt in enumerate(filter_list):
		color = color_list[x]
		for y,site in enumerate(site_list):
			hjd, mag, magerr = [[], [], []]
			tel_choices = site[0]
			site_name = site[1]
			marker = marker_list[y]
			for tel_choice in tel_choices: 
				hjd, mag, magerr = np.append([hjd, mag, magerr], extract_lightcurve(star_id=star_id, db_file=db_file, filt_choice=filt, tel_choice=tel_choice, mag_err_cutoff=mag_err_cutoff), axis=1)
			#print(hjd,mag,magerr)
			plt.scatter(np.asarray(hjd-2450000), mag, c=color, marker=marker, label=site_name)
			plt.errorbar(np.asarray(hjd)-2450000, mag, c=color, marker=marker, yerr=magerr, linestyle="None")	
	plt.legend(loc='best')
	plt.show()


def weighted_mean(data, uncert):
	w_i = np.power(uncert, -2)
	sum_top = np.sum(data*w_i)
	sum_bottom = np.sum(w_i)
	if sum_bottom == 0:
		return 0
	else:
		return sum_top/sum_bottom

def weighted_standard_deviation(data, uncert, weighted_mean):
	w_i = np.power(uncert, -2)
	M = len(data[uncert!=0])
	if M == 0: return 0;
	sum_top = np.sum(w_i*np.square(np.asarray(data)-weighted_mean))
	sum_bottom = (M-1)*np.sum(w_i)/M
	return np.sqrt(sum_top/sum_bottom)

def extract_med_std(conn,ra,dec,radius,filt,telo):


	center= SkyCoord(ra,dec, frame='icrs', unit=(units.hourangle, units.deg))
	results = phot_db.box_search_on_position(conn, center.ra.deg, center.dec.deg, radius, radius)



	medians = []
	stds = []
	if len(results) > 0:
		
		for star_id in tqdm(results['star_id']) :


			query = 'SELECT filter, facility, hjd, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(star_id)+'" AND filter="'+str(filt_choice)+'" AND facility="'+str(telo)+'"'	    
			phot_table = phot_db.query_to_astropy_table(conn, query, args=())
			#import pdb; pdb.set_trace()

			datasets = identify_unique_datasets(phot_table,facilities,filters)
			mask =  phot_table['calibrated_mag'] >0
			uncerts = np.asarray(phot_table['calibrated_mag_err'][mask])
			mags = np.asarray(phot_table['calibrated_mag'][mask])
			mean = weighted_mean(mags, uncerts)
			std = weighted_standard_deviation(mags, uncerts, mean)
			if (mean>10) and (len(mags)>5) and (std>0):
			
				medians.append(mean)
				stds.append(std)
			

	return medians,stds

def extract_airmass(star,db,filt,tel_choices):
	conn = phot_db.get_connection(dsn=db)
	airmasses = []
	image_ids = []
	hjds = []
	mags = []
	magerrs = []
	for telo in tel_choices:
		query = 'SELECT airmass, img_id, date_obs_jd FROM images WHERE facility="'+str(telo)+'" AND filter="'+str(filt)+ '"'
		phot_table = phot_db.query_to_astropy_table(conn, query, args=())
		
		for airmass, image, hjd in phot_table:
			try:
				query = 'SELECT hjd, image, calibrated_mag, calibrated_mag_err FROM phot WHERE image="'+str(image)+'" AND facility="'+str(telo)+'" AND filter="'+str(filt)+'" AND star_id="'+str(star)+'"'
				temp_table = phot_db.query_to_astropy_table(conn, query, args=())
				hjds = np.append(hjds, temp_table[0][0])
				mags = np.append(mags, temp_table[0][2])
				magerrs = np.append(magerrs, temp_table[0][3])
				airmasses = np.append(airmasses, airmass)
				image_ids = np.append(image_ids, image)
			except:
				pass
	airmasses = airmasses[mags >0]
	magerrs = magerrs[mags >0]
	image_ids = image_ids[mags >0]
	hjds = hjds[mags >0]
	mags = mags[mags >0]

	errors = np.loadtxt('systematic_errors.txt')
	errors = Table(rows=errors, names=('image', 'sys_error'))
	#errors = errors[0]
	#import pdb; pdb.set_trace()
	data = Table([mags, magerrs, hjds, image_ids], names=('mag', 'magerr', 'hjd', 'image'))

	mags2 = []
	magerr2 = []
	hjd2 = []

	for mag,magerr,hjd,image in data:
		try:
			result = np.where(np.asarray(errors['image']) == image)
			location = result[0][0]
			error = errors[location][1]
			new_error = np.sqrt(float(magerr)**2+float(error)**2)
			
		except:
			#print("We found an error.")
			new_error = magerr
		mags2.append(mag)
		hjd2.append(hjd)
		magerr2.append(new_error)


	mean = weighted_mean(mags2, magerr2)
	residuals = mags2 - mean
	plt.scatter(mags2, airmasses)
	plt.xlabel('Magnitude', fontsize=15)
	plt.ylabel('Airmass',fontsize=15)
	plt.title('ID: '+str(star)+", mean mag = "+str(np.round(mean, 2)))
	plt.show()




def extract_airmass_residuals(star,db,filt,tel_choices):
	conn = phot_db.get_connection(dsn=db)
	airmasses = []
	image_ids = []
	hjds = []
	mags = []
	magerrs = []
	for telo in tel_choices:
		query = 'SELECT airmass, img_id, date_obs_jd FROM images WHERE facility="'+str(telo)+'" AND filter="'+str(filt)+ '"'
		phot_table = phot_db.query_to_astropy_table(conn, query, args=())
		
		for airmass, image, hjd in phot_table:
			try:
				query = 'SELECT hjd, image, calibrated_mag, calibrated_mag_err FROM phot WHERE image="'+str(image)+'" AND facility="'+str(telo)+'" AND filter="'+str(filt)+'" AND star_id="'+str(star)+'"'
				temp_table = phot_db.query_to_astropy_table(conn, query, args=())
				hjds = np.append(hjds, temp_table[0][0])
				mags = np.append(mags, temp_table[0][2])
				magerrs = np.append(magerrs, temp_table[0][3])
				airmasses = np.append(airmasses, airmass)
				image_ids = np.append(image_ids, image)
			except:
				pass
	airmasses = airmasses[mags >0]
	magerrs = magerrs[mags >0]
	image_ids = image_ids[mags >0]
	hjds = hjds[mags >0]
	mags = mags[mags >0]

	errors = np.loadtxt('systematic_errors.txt')
	errors = Table(rows=errors, names=('image', 'sys_error'))
	#errors = errors[0]
	#import pdb; pdb.set_trace()
	data = Table([mags, magerrs, hjds, image_ids], names=('mag', 'magerr', 'hjd', 'image'))

	mags2 = []
	magerr2 = []
	hjd2 = []

	for mag,magerr,hjd,image in data:
		try:
			result = np.where(np.asarray(errors['image']) == image)
			location = result[0][0]
			error = errors[location][1]
			new_error = np.sqrt(float(magerr)**2+float(error)**2)
			
		except:
			#print("We found an error.")
			new_error = magerr
		mags2.append(mag)
		hjd2.append(hjd)
		magerr2.append(new_error)


	mean = weighted_mean(mags2, magerr2)
	residuals = mags2 - mean
	plt.scatter(airmasses, residuals)
	plt.ylabel('Residual', fontsize=15)
	plt.xlabel('Airmass',fontsize=15)
	plt.title('ID: '+str(star)+", mean mag = "+str(np.round(mean, 2)))
	plt.gca().invert_yaxis()
	plt.show()

def extract_background_residuals(star,db,filt,tel_choices):
	conn = phot_db.get_connection(dsn=db)
	airmasses = []
	image_ids = []
	hjds = []
	mags = []
	magerrs = []
	backgrounds = []
	for telo in tel_choices:
		query = 'SELECT airmass, img_id, date_obs_jd FROM images WHERE facility="'+str(telo)+'" AND filter="'+str(filt)+ '"'
		phot_table = phot_db.query_to_astropy_table(conn, query, args=())
		
		for airmass, image, hjd in phot_table:
			try:
				query = 'SELECT hjd, image, calibrated_mag, calibrated_mag_err, local_background FROM phot WHERE image="'+str(image)+'" AND facility="'+str(telo)+'" AND filter="'+str(filt)+'" AND star_id="'+str(star)+'"'
				temp_table = phot_db.query_to_astropy_table(conn, query, args=())
				hjds = np.append(hjds, temp_table[0][0])
				mags = np.append(mags, temp_table[0][2])
				magerrs = np.append(magerrs, temp_table[0][3])
				airmasses = np.append(airmasses, airmass)
				image_ids = np.append(image_ids, image)
				backgrounds = np.append(backgrounds, temp_table[0][4])
			except:
				pass
	airmasses = airmasses[mags >0]
	magerrs = magerrs[mags >0]
	image_ids = image_ids[mags >0]
	hjds = hjds[mags >0]
	backgrounds = backgrounds[mags >0]
	mags = mags[mags >0]

	errors = np.loadtxt('systematic_errors.txt')
	errors = Table(rows=errors, names=('image', 'sys_error'))
	#errors = errors[0]
	data = Table([mags, magerrs, hjds, image_ids], names=('mag', 'magerr', 'hjd', 'image'))

	mags2 = []
	magerr2 = []
	hjd2 = []
	#import pdb; pdb.set_trace()


	for mag,magerr,hjd,image in data:
		try:
			result = np.where(np.asarray(errors['image']) == image)
			location = result[0][0]
			error = errors[location][1]
			new_error = np.sqrt(float(magerr)**2+float(error)**2)
			
		except:
			#print("We found an error.")
			new_error = magerr
		mags2.append(mag)
		hjd2.append(hjd)
		magerr2.append(new_error)


	mean = weighted_mean(mags2, magerr2)
	residuals = mags2 - mean
	plt.scatter(backgrounds, residuals)
	plt.ylabel('Residual', fontsize=15)
	plt.xlabel('Sky Background',fontsize=15)
	plt.title('ID: '+str(star)+", mean mag = "+str(np.round(mean, 2)))
	plt.gca().invert_yaxis()
	plt.show()


def extract_airmass_vs_hjd(star,db,filt,tel_choices):
	conn = phot_db.get_connection(dsn=db)
	airmasses = []
	image_ids = []
	hjds = []
	mags = []
	magerrs = []
	for telo in tel_choices:
		query = 'SELECT airmass, img_id, date_obs_jd FROM images WHERE facility="'+str(telo)+'" AND filter="'+str(filt)+ '"'
		phot_table = phot_db.query_to_astropy_table(conn, query, args=())
		
		for airmass, image, hjd in phot_table:
			try:
				query = 'SELECT hjd, image, calibrated_mag, calibrated_mag_err FROM phot WHERE image="'+str(image)+'" AND facility="'+str(telo)+'" AND filter="'+str(filt)+'" AND star_id="'+str(star)+'"'
				temp_table = phot_db.query_to_astropy_table(conn, query, args=())
				hjds = np.append(hjds, temp_table[0][0])
				mags = np.append(mags, temp_table[0][2])
				magerrs = np.append(magerrs, temp_table[0][3])
				airmasses = np.append(airmasses, airmass)
				image_ids = np.append(image_ids, image)
			except:
				pass
	airmasses = airmasses[mags >0]
	magerrs = magerrs[mags >0]
	image_ids = image_ids[mags >0]
	hjds = hjds[mags >0]
	mags = mags[mags >0]

	plt.scatter(hjds, airmasses)
	plt.xlabel('HJD', fontsize=15)
	plt.ylabel('Airmass',fontsize=15)
	plt.title('ID: '+str(star))
	plt.gca().invert_yaxis()
	plt.show()

def extract_blank_residuals(star,db,filt,tel_choices,quantity):
	#import pdb; pdb.set_trace()
	conn = phot_db.get_connection(dsn=db)
	airmasses = []
	image_ids = []
	hjds = []
	mags = []
	magerrs = []
	exps = []
	for telo in tel_choices:
		query = 'SELECT '+str(quantity)+', img_id, date_obs_jd, exposure_time FROM images WHERE facility="'+str(telo)+'" AND filter="'+str(filt)+ '"'
		phot_table = phot_db.query_to_astropy_table(conn, query, args=())
		
		for airmass, image, hjd,exp in phot_table:
			try:
				query = 'SELECT hjd, image, calibrated_mag, calibrated_mag_err FROM phot WHERE image="'+str(image)+'" AND facility="'+str(telo)+'" AND filter="'+str(filt)+'" AND star_id="'+str(star)+'"'
				temp_table = phot_db.query_to_astropy_table(conn, query, args=())
				hjds = np.append(hjds, temp_table[0][0])
				mags = np.append(mags, temp_table[0][2])
				magerrs = np.append(magerrs, temp_table[0][3])
				airmasses = np.append(airmasses, airmass)
				image_ids = np.append(image_ids, image)
				exps = np.append(exps, np.round(exp, 0))
			except:
				pass
	airmasses = airmasses[mags >0]
	magerrs = magerrs[mags >0]
	image_ids = image_ids[mags >0]
	hjds = hjds[mags >0]
	exps = exps[mags >0]
	mags = mags[mags >0]

	errors = np.loadtxt('systematic_errors.txt')
	errors = Table(rows=errors, names=('image', 'sys_error'))
	#errors = errors[0]
	#import pdb; pdb.set_trace()
	data = Table([mags, magerrs, hjds, image_ids], names=('mag', 'magerr', 'hjd', 'image'))

	mags2 = []
	magerr2 = []
	hjd2 = []

	for mag,magerr,hjd,image in data:
		try:
			result = np.where(np.asarray(errors['image']) == image)
			location = result[0][0]
			error = errors[location][1]
			new_error = np.sqrt(float(magerr)**2+float(error)**2)
			
		except:
			#print("We found an error.")
			new_error = magerr
		mags2.append(mag)
		hjd2.append(hjd)
		magerr2.append(new_error)


	mean = weighted_mean(mags2, magerr2)
	residuals = mags2 - mean
	airmasses_1 = airmasses[exps ==300]
	airmasses_2 = airmasses[exps !=300]
	residuals_1 = residuals[exps ==300]
	residuals_2 = residuals[exps !=300]
	plt.scatter(airmasses_1, residuals_1, color='blue', label='300s exposure')
	plt.scatter(airmasses_2, residuals_2, color='red', label='not 300s exposure')
	plt.ylabel('Residual', fontsize=15)
	plt.xlabel(str(quantity),fontsize=15)
	plt.legend()
	plt.title('ID: '+str(star)+", mean mag = "+str(np.round(mean, 2)))
	plt.gca().invert_yaxis()
	plt.show()

def extract_hjds_residuals(star,db,filt,tel_choices):
	conn = phot_db.get_connection(dsn=db)
	airmasses = []
	image_ids = []
	hjds = []
	mags = []
	magerrs = []
	for telo in tel_choices:
		query = 'SELECT airmass, img_id, date_obs_jd FROM images WHERE facility="'+str(telo)+'" AND filter="'+str(filt)+ '"'
		phot_table = phot_db.query_to_astropy_table(conn, query, args=())
		
		for airmass, image, hjd in phot_table:
			try:
				query = 'SELECT hjd, image, calibrated_mag, calibrated_mag_err FROM phot WHERE image="'+str(image)+'" AND facility="'+str(telo)+'" AND filter="'+str(filt)+'" AND star_id="'+str(star)+'"'
				temp_table = phot_db.query_to_astropy_table(conn, query, args=())
				hjds = np.append(hjds, temp_table[0][0])
				mags = np.append(mags, temp_table[0][2])
				magerrs = np.append(magerrs, temp_table[0][3])
				airmasses = np.append(airmasses, airmass)
				image_ids = np.append(image_ids, image)
			except:
				pass
	airmasses = airmasses[mags >0]
	magerrs = magerrs[mags >0]
	image_ids = image_ids[mags >0]
	hjds = hjds[mags >0]
	mags = mags[mags >0]

	errors = np.loadtxt('systematic_errors.txt')
	errors = Table(rows=errors, names=('image', 'sys_error'))
	#errors = errors[0]
	#import pdb; pdb.set_trace()
	data = Table([mags, magerrs, hjds, image_ids], names=('mag', 'magerr', 'hjd', 'image'))

	mags2 = []
	magerr2 = []
	hjd2 = []

	for mag,magerr,hjd,image in data:
		try:
			result = np.where(np.asarray(errors['image']) == image)
			location = result[0][0]
			error = errors[location][1]
			new_error = np.sqrt(float(magerr)**2+float(error)**2)
			
		except:
			#print("We found an error.")
			new_error = magerr
		mags2.append(mag)
		hjd2.append(hjd)
		magerr2.append(new_error)


	mean = weighted_mean(mags2, magerr2)
	residuals = mags2 - mean
	plt.scatter(hjds, residuals)
	plt.ylabel('Residual', fontsize=15)
	plt.xlabel('HJD',fontsize=15)
	plt.title('ID: '+str(star)+", mean mag = "+str(np.round(mean, 2)))
	plt.gca().invert_yaxis()
	plt.show()