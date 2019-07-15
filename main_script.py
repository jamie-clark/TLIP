"""
Created on DATE

@author: jamieclark
"""
import numpy as np
from LIA import models, microlensing_classifier, noise_models, training_set
from pyDANDIA import phot_db
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.table import Table
from astropy.io import ascii
import time
import matplotlib.pyplot as plt
import random

def ROME_classification_script(ra, dec, radius, db_file, LIA_directory, filt_choice='3', tel_choices=[1, 2], mag_cutoff=[14,17], mag_err_cutoff=0.1):
	"""Creates...

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
	tel_choice : int, optional
		Number of corresponding site location requested. List of all options can be found below.
		Defaults to 2 for lsc-doma-1m0a-fl15 in Chile.
	mag_cutoff: int, optional
		Sets the cutoff point for what counts as too dim of a star.
		Stars with magnitudes that frequently dip below this point will be excluded. Defaults to 17.
	mag_err_cutoff: int, optional
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we discard them from our calculations. Defaults to 0.1.

	Outputs
	_______
	results_* : txt file
		A txt file containing all results of the classification script in table format, 
		included in columns of star id, ra, dec, filter and telescope used, predicted class,
		and prediction probability. 
	results_ml_* : txt file
		A txt file containing truncated results of the classification script, printing
		a list of all microlensing candidates in detail.
	ml_lightcurves_* : txt file
		A txt file containing the lightcurves of all microlensing candidates.
	"""
	# generates the LIA models
	rf, pca = models.create_models(str(LIA_directory)+'all_features.txt', str(LIA_directory)+'pca_features.txt')

	
	# SEARCH DB FOR STARS IN PROPER REGION
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

	print("Beginning database query and calculating predictions on " + str(len(results)) + " stars...")

	for star_idx,star_id in enumerate(results['star_id']) :
		prediction, probability, ML_probability, mag, magerr = [[], [], [], [], []]
		progress = 100*star_idx/len(results)
		print("Progress at " + str(round(progress, 2)) + "%.")
		try:
			for tel_choice in tel_choices: 
				mag, magerr = np.append([mag, magerr], extract_lightcurve(star_id, db_file, filt_choice, tel_choice, mag_cutoff, mag_err_cutoff)[1:3], axis=1)
			prediction, probability, ML_probability = extract_results(mag, magerr, rf, pca, mag_cutoff, mag_err_cutoff)
			ra, dec = extract_ra_dec(star_id, db_file)
			info_list.append([star_id, ra, dec, str(filt_choice)+"_"+"-".join(str(x) for x in tel_choices), prediction, probability, ML_probability])
			print(len(mag))
		except:
			pass


	print("Progress at 100.00%")
	#print(info_list)
	# Generate timestamp
	timestamp = time.time()

	# Generate results_*.txt file
	#print(info_list)
	results_table = Table(rows=info_list, names=('star_id', 'ra', 'dec', 'filter_telescope', 'prediction', 'probability', 'ml_probability'), meta={'name': 'full results table'})
	results_table.sort('prediction', 'probability')
	ascii.write(results_table, "results.txt", overwrite=True)
	
	# Generate results_truncated_*.txt file
	try:
		results_table.add_index('prediction')
		ml_table = results_table.loc['ML']
		ascii.write(ml_table, "results_truncated_"+str(timestamp)+".txt")
	except:
		print("No microlensing events detected.")
	# Generate ml_lightcurves_*.txt file
#	ml_lightcurves = []
#	for ml_target in results_table.loc['ML']:
#		star_id = ml_target['star_id']
#		hjd, mag, magerr = extract_lightcurve(star_id, db_file, mag_cutoff=float("inf"))
#		ml_lightcurves.append([star_id, hjd, mag, magerr])
#	ml_lightcurves_table = Table()
	print("Text files generated. Program complete.")



def extract_results(mag, magerr, rf, pca, mag_cutoff, mag_err_cutoff):


	# Cut out entries where the magnitude is dimmer than the cutoff, 
	# entries where the image is completely saturated, entries with less than three entries, 
	# and entries where there are errors
	if (not all(i <= mag_cutoff[1] for i in mag)) or (not all(i >= mag_cutoff[0] for i in mag)) or (len(list(set(mag))) < 3) or (True in np.isnan(mag)):
		return()
	else:
		#import pdb; pdb.set_trace()
		prediction, ml_pred, cons_pred, cv_pred, var_pred = microlensing_classifier.predict(mag, magerr, rf, pca)[0:5]
		probability = max(ml_pred[0], cons_pred[0], cv_pred[0], var_pred[0])
		result = [prediction, probability, ml_pred[0]]
		return(result)

def extract_lightcurve(star_id, db_file, filt_choice='3', tel_choice=2, mag_cutoff=[14,float("inf")], mag_err_cutoff=0.1):
	"""Creates...

	Parameters
	__________
	star_id : string
		ID of the specified star in the specified database.
	db_file : string
		System path of the database file on the machine.
	filt_choice : string, optional
		Number of corresponding telescope filter requested. List of all options can be found below.
		Defaults to '3' for I filterband.
	tel_choice : int, optional
		Number of corresponding site location requested. List of all options can be found below.
		Defaults to 2 for lsc-doma-1m0a-fl15 in Chile.
	mag_cutoff: int, optional
		Sets the cutoff point for what counts as too dim of a star.
		Stars with magnitudes that frequently dip below this point will be excluded.
		Defaults to Infinity (exclude nothing).
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
	conn = phot_db.get_connection(dsn=db_file)
	query = 'SELECT hjd, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(star_id)+'" AND filter="'+str(filt_choice)+'" AND facility="'+str(tel_choice)+'"'
	phot_table = phot_db.query_to_astropy_table(conn, query, args=())
	mag = np.asarray(phot_table['calibrated_mag'])
	magerr = np.asarray(phot_table['calibrated_mag_err'])
	hjd = np.asarray(phot_table['hjd'])
	mask = np.all([mag >0, magerr <mag_err_cutoff], axis=0)
	mag = mag[mask]
	magerr = magerr[mask]
	hjd = hjd[mask]
	return(hjd, mag, magerr)

def plot_lightcurve(hjd, mag, magerr):
	"""Creates...

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


def plot_all_lightcurves(star_id, db_file, mag_err_cutoff):
	color_list = ['g', 'r', 'k']
	marker_list = ['.', 'o', '^', 's', 'x', '+', 'D']
	filter_list = [1,2,3]
	site_list = [[[1,2], "LSC-DOMA"], [[8, 12], "CPT-DOMA"], [[3], "LSC-DOMB"], [[4, 5], "LSC-DOMC"], [[6, 10], "COJ-DOMA"], [[7, 11], "COJ-DOMB"], [[9], "CPT-DOMC"]]
	plt.gca().invert_yaxis()
	for x,filt in enumerate(filter_list):
		color = color_list[x]
		for y,site in enumerate(site_list):
			hjd, mag, magerr = [[], [], []]
			tel_choices = site[0]
			site_name = site[1]
			marker = marker_list[y]
			for tel_choice in tel_choices: 
				hjd, mag, magerr = np.append([hjd, mag, magerr], extract_lightcurve(star_id=star_id, db_file=db_file, filt_choice=filt, tel_choice=tel_choice, mag_cutoff=float("inf"), mag_err_cutoff=mag_err_cutoff), axis=1)
			#print(hjd,mag,magerr)
			plt.scatter(np.asarray(hjd-2450000), mag, c=color, marker=marker, label=site_name)
			plt.errorbar(np.asarray(hjd)-2450000, mag, c=color, marker=marker, yerr=magerr, linestyle="None")	
	plt.legend(loc='best')
	plt.show()


def create_training_set(db_file, filt_choice, tel_choices, mag_cutoff, mag_err_cutoff):

	random_integers = []
	for i in range(1, 501):
		random_integers.append(random.randint(1, 162355))

	timestamps = []
	for star_id in random_integers:
		hjd = []
		try:
			for tel_choice in tel_choices: 
				hjd = np.append(hjd, extract_lightcurve(star_id, db_file, filt_choice, tel_choice, mag_cutoff, mag_err_cutoff)[0])
			if len(hjd) > 10:
				timestamps.append(hjd)
		except:
			pass

	# PTF noise model
	median = [14.3, 14.75, 15.3, 15.8, 16.3, 16.8, 17.4, 17.9, 18.4, 18.8, 19.3, 19.8, 20.4, 20.9, 21.5]
	rms = [0.01, 0.0092, 0.0094, 0.01, 0.012, 0.014, 0.018, 0.022, 0.032, 0.048, 0.065, 0.09, 0.11, 0.17, 0.2]
	ptf_model = noise_models.create_noise(median, rms)

	print(ptf_model)

	# ROME noise model
	median = [13.5,14.0,14.5,15.0,15.5,16.0,16.5,17.0,17.5,18.0,18.5,19.0]
	rms = [0.01,0.02,0.03, 0.05,0.07,0.08,0.13,0.17,0.4,0.5,0.6,1.0]
	rome_model = noise_models.create_noise(median, rms)

	print(rome_model)

	g = [len(hjd) for hjd in timestamps]
	print(g)
	training_set.create(timestamps, min_mag=14, max_mag=17, noise=rome_model, n_class=500)
	print("Training set generated. Program complete.")


def drip_feed(star_id, db_file, LIA_directory, filt_choice, tel_choices, mag_cutoff, mag_err_cutoff):
	# daniel's code
	rf, pca = models.create_models(str(LIA_directory)+'all_features.txt', str(LIA_directory)+'pca_features.txt')
	mjd, mag, magerr = [[], [], []]
	for tel_choice in tel_choices:
		mjd, mag, magerr = np.append([mjd, mag, magerr], extract_lightcurve(star_id, db_file, filt_choice, tel_choice, mag_cutoff, mag_err_cutoff),axis=1)
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

	for i in range(3, len(mjd)-1, 1):
	               
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

def extract_lightcurves_on_position(ra, dec, radius, db_file, filt_choice='3', tel_choices=[1, 2], mag_cutoff=[14,17], mag_err_cutoff=0.1):
	"""Creates...

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
	tel_choice : int, optional
		Number of corresponding site location requested. List of all options can be found below.
		Defaults to 2 for lsc-doma-1m0a-fl15 in Chile.
	mag_cutoff: int, optional
		Sets the cutoff point for what counts as too dim of a star.
		Stars with magnitudes that frequently dip below this point will be excluded. Defaults to 17.
	mag_err_cutoff: int, optional
		Sets the cutoff point for what counts as too inaccurate of a datapoint.
		Data points with very high uncertainties can throw off our detection algorithm,
		so we discard them from our calculations. Defaults to 0.1.

	Outputs
	_______
	hjd
	mag
	magerr

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
				time, mag, magerr = np.append([time, mag, magerr], extract_lightcurve(star_id, db_file, filt_choice, tel_choice, mag_cutoff, mag_err_cutoff), axis=1)
				print(mag)
			star_ids = np.append(star_ids, np.repeat(star_id, len(mag)))
			times, mags, magerrs = np.append([times, mags, magerrs], [time, mag, magerr], axis=1)
		except:
			pass


	# Generate results_*.txt file
	print(mags)
	print(star_ids)
	results_table = Table({'star_id': star_ids, 'hjd': times, 'mag': mags, 'magerr': magerrs}, names=('star_id', 'hjd', 'mag', 'magerr'))
	ascii.write(results_table, "lightcurves.txt", overwrite=True)
	print("Text file generated. Program complete.")