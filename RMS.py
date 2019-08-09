from sys import argv
import sqlite3
from os import getcwd, path, remove, environ
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.table import Table
import matplotlib.pyplot as plt
from pyDANDIA import phot_db
from pyDANDIA.lightcurves import identify_unique_datasets
import os.path
from matplotlib.colors import LogNorm
from tqdm import tqdm
from main_script import extract_lightcurve
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 30}

plt.rc('font', **font)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)






def plot_RMS(med,std,filt,log=None):

	if log:

		plt.yscale('log')
		plt.ylim([0.001,3])
	else:
		plt.ylim([-0.1,3])


	plt.scatter(med,std,s=2)
	plt.xlabel(r'$'+filt+' $ [mag]')
	plt.ylabel(r'$\sigma $ [mag]')

	mag = np.arange(10,24,0.1)
	flux = 10**((ZP-mag)/2.5)
	
	emag = 2.5/np.log(10)*1/flux**0.5
	back = 2.5/np.log(10)*BACKGROUND**0.5/flux

	plt.plot(mag,emag,lw=2,c='r',label = 'ZP = '+str(ZP))
	plt.plot(mag,back,lw=2,c='orange',linestyle='--',label = 'BACK = '+str(BACKGROUND))
	plt.legend(loc=2,fontsize=15)
	plt.xlim([13,21])
	plt.show()


def plot_RMS_density(med,std,filt,log=None):

	if log:

		plt.yscale('log')

	plt.hist2d(med,std,50,norm=LogNorm())
	plt.xlabel(r'$'+filt+' $ [mag]')
	plt.ylabel(r'$\sigma $ [mag]')
	plt.colorbar()
	plt.axis([13,21,-0.1,1.1])
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


			#query = 'SELECT filter, facility, hjd, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(star_id)+'" AND filter="'+str(filt_choice)+'" AND facility="'+str(telo)+'"'	    
			#phot_table = phot_db.query_to_astropy_table(conn, query, args=())
			#import pdb; pdb.set_trace()
			#datasets = identify_unique_datasets(phot_table,facilities,filters)
			#mask =  phot_table['calibrated_mag'] >0
			#uncerts = np.asarray(phot_table['calibrated_mag_err'][mask])
			#mags = np.asarray(phot_table['calibrated_mag'][mask])
			#mean = weighted_mean(mags, uncerts)
			#std = weighted_standard_deviation(mags, uncerts, mean)
			hjd, mags, uncerts = extract_lightcurve(star_id=star_id, db_file=db_dir, filt_choice=filt_choice, tel_choice=tel_choice, mag_err_cutoff=mag_err_cutoff)
			mags = np.asarray(mags)
			uncerts = np.asarray(uncerts)
			mean = weighted_mean(mags, uncerts)
			std = weighted_standard_deviation(mags, uncerts, mean)


			if (mean>10) and (len(mags)>5) and (std>0):
			
				medians.append(mean)
				stds.append(std)
			

	return medians,stds

def extract_med_poisson(conn,ra,dec,radius,filt,telo):


	center= SkyCoord(ra,dec, frame='icrs', unit=(units.hourangle, units.deg))
	results = phot_db.box_search_on_position(conn, center.ra.deg, center.dec.deg, radius, radius)



	medians = []
	stds = []
	if len(results) > 0:
		
		for star_id in tqdm(results['star_id']) :


			query = 'SELECT filter, image, facility, hjd, calibrated_mag, calibrated_mag_err, calibrated_flux, calibrated_flux_err FROM phot WHERE star_id="'+str(star_id)+'" AND filter="'+str(filt_choice)+'" AND facility="'+str(telo)+'"'	    
			phot_table = phot_db.query_to_astropy_table(conn, query, args=())
			#import pdb; pdb.set_trace()

			datasets = identify_unique_datasets(phot_table,facilities,filters)
			mask =  phot_table['calibrated_mag'] >0
			uncerts = np.asarray(phot_table['calibrated_mag_err'][mask])
			mags = np.asarray(phot_table['calibrated_mag'][mask])
			flux = np.asarray(phot_table['calibrated_flux'])
			fluxuncerts = np.asarray(phot_table['calibrated_flux_err'])
			exposure_times = []
			for image in np.asarray(phot_table['image']):
				query = 'SELECT exposure_time FROM images WHERE img_id = "'+str(image)+'"'
				phot_table = phot_db.query_to_astropy_table(conn, query, args=())
				exposure_times.append(phot_table[0][0])
			flux = flux*exposure_times
			fluxuncerts = fluxuncerts*exposure_times
			mean = weighted_mean(mags, uncerts)
			std = np.sqrt(weighted_mean(flux,fluxuncerts))
			if (mean>10) and (len(mags)>5) and (std>0):
			
				medians.append(mean)
				stds.append(std)
			

	return medians,stds


def fit_spline(mag, std):
	from scipy import interpolate 
	x = std
	y = mag
	temp_table = Table([x,y])
	temp_table.sort('col1')
	x = np.asarray(temp_table['col0'])
	y = np.asarray(temp_table['col1'])
	pdb.set_trace()
	#tck = interpolate.splrep(y, x, s=float(len(x))-np.sqrt(2*float(len(x))))
	xnew = np.arange(13.5,20.6,0.25)
	tck = interpolate.CubicSpline(y, x)
	z = np.polyfit(y,x,3)
	z2 = np.poly1d(z)
	ynew = z2(xnew)
	return(xnew, ynew)
#FIELD-12
#ra = '17:52:43.2412'	
#dec = '-29:16:42.6459'
#db_dir = '/home/ebachelet/pyDANDIA/Analysis/ROME-FIELD-04_coj-doma-1m0-11-fa12.db'



#query = 'SELECT filter, facility, hjd, calibrated_flux FROM phot WHERE star_id="'+str(star_id)+'" AND image ="' + str(image_id) + '" AND filter="'+str(filt_choice)+'" AND facility="'+str(tel_choice)+'"'	    
#select flux and calibrated_flux from phot

#FIELD-16
ra = '18:00:17.9956'	
dec = '-28:32:15.2109'
db_dir = '/home/jclark/examples/ROME-FIELD-16_phot.db'

ZP = 25
BACKGROUND = 15000

filt_choice = '3'
tel_choice = 2
mag_err_cutoff = 500


conn = phot_db.get_connection(dsn=db_dir)
facilities = phot_db.fetch_facilities(conn)
filters = phot_db.fetch_filters(conn)
code_id = phot_db.get_stage_software_id(conn,'stage6')


	

center= SkyCoord(ra,dec, frame='icrs', unit=(units.hourangle, units.deg))
    
radius = 5/ 60.0 #arcmins


if __name__ == '__main__':
	print(facilities)
	print(filters)
	meds,stds = extract_med_std(conn,ra,dec,radius,filt_choice,tel_choice)
	#mags,poissons = extract_med_poisson(conn,ra,dec,radius,filt_choice,tel_choice)
	import pdb; pdb.set_trace()
	median, rms = fit_spline(meds,stds)[0:2]
	print(median)
	print(rms)
	plt.plot(median,rms)
	plot_RMS(meds,stds,filters[int(filt_choice)-1][1],log=True)
#
#plt.plot(median, rms)
#plt.show()
	plot_RMS_density(meds,stds,filters[int(filt_choice)-1][1])


