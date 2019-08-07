from main_script import *
from tqdm import tqdm
import random
from astropy.table import Table, hstack, vstack

#db = '/home/jclark/examples/ROME-FIELD-16_phot.db'


#FIELD-16
ra = '18:00:17.9956'	
dec = '-28:32:15.2109'
db_dir = '/home/jclark/examples/ROME-FIELD-16_phot_3sites_ip.db'

ZP = 25
BACKGROUND = 15000

filt_choice = '3'
tel_choice = 2

site_list =[[[1,4], "LSC-DOMA"], [[2,5], "COJ-DOMA"], [[3, 6], "CPT-DOMA"]]

for site in site_list:
	tel_choices = site[0]
	site_name = site[1]

	conn = phot_db.get_connection(dsn=db_dir)
	facilities = phot_db.fetch_facilities(conn)
	filters = phot_db.fetch_filters(conn)
	code_id = phot_db.get_stage_software_id(conn,'stage6')


	center= SkyCoord(ra,dec, frame='icrs', unit=(units.hourangle, units.deg))
	    
	radius = 60.0/ 60.0 #arcmins

	random_integers = []
	for i in range(1, 1001):
		random_integers.append(random.randint(1, 162355))

	import pdb; pdb.set_trace()	
	# get weighted mean for every star chosen
	meds,stds,stars = new_extract_med_std(conn,ra,dec,radius,filt_choice,tel_choices,overriding_results=np.asarray(random_integers))
	star_meds_table = Table({'star_id': stars, 'mag': meds}, names=('star_id', 'mag'))

	residuals = []
	hjds = []
	star_ids = []
	star_mags = []
	errors = []
	images = []
	times = []
	for telo in tel_choices:
		query = 'SELECT img_id, date_obs_jd FROM images WHERE facility="'+str(telo)+'"'	    
		results = phot_db.query_to_astropy_table(conn, query, args=())
		images = np.append(images, results['img_id'])
		times = np.append(times, results['date_obs_jd'])
	results = Table({'img_id':images, 'date_obs_jd':times}, names=('img_id', 'date_obs_jd'))
	if len(results['img_id']) != len(list(set(results['img_id']))):
		print("error! error!")
		exit()

	for image_id,hjd in tqdm(results):
		R_top=0
		R_bottom=0
		for star_id,star_mag in star_meds_table:
			query = 'SELECT filter, facility, hjd, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(star_id)+'" AND image ="' + str(image_id) + '" AND filter="'+str(filt_choice)+'"'	    
			phot_table = phot_db.query_to_astropy_table(conn, query, args=())
			if len(phot_table) > 2: continue;
			if len(phot_table) == 0: continue;
			mag = float(phot_table['calibrated_mag'][0])
			uncert = float(phot_table['calibrated_mag_err'][0])
			mask = mag >0
			if mask == False: continue;
			w_i=np.power(uncert, -2)		
			star_mag = float(star_mag)
			R_j_top = w_i*(mag - star_mag)
			R_j_bottom = w_i
			R_top += R_j_top
			R_bottom += R_j_bottom
		if R_top == 0 or R_bottom == 0:
			continue
		else:
			R_total = R_top/R_bottom
			residuals.append(R_total)
			hjds.append(hjd)
		#R_2_top=0
	#	R_2_bottom=0
	#	for star_id,star_mag in star_meds_table:
	# ADD RESIDUALS TO RESULTS BASED ON 
	import pdb; pdb.set_trace()
	t = results
	t.add_index('date_obs_jd')
	t = t.loc[hjds]
	t = Table([np.asarray(t['img_id']), np.asarray(t['date_obs_jd']), np.asarray(residuals)])

	for image_id,hjd,residual in tqdm(t):
		R_top=0
		R_bottom=0
		n=0
		N=0
		for star_id,star_mag in star_meds_table:
			query = 'SELECT filter, facility, hjd, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(star_id)+'" AND image ="' + str(image_id) + '" AND filter="'+str(filt_choice)+'"'	    
			phot_table = phot_db.query_to_astropy_table(conn, query, args=())
			if len(phot_table) > 2: continue;
			if len(phot_table) == 0: continue;
			mag = float(phot_table['calibrated_mag'][0])
			uncert = float(phot_table['calibrated_mag_err'][0])
			w_i=np.power(uncert, -2)		
			star_mag = float(star_mag)
			R_j_top = w_i*((mag - star_mag)-residual)**2
			R_j_bottom = w_i
			R_top += R_j_top
			R_bottom += R_j_bottom
			N += 1
		if R_top == 0 or R_bottom == 0:
			continue
		else:
			R_total = (R_top/R_bottom)*(N/(N-1))
			errors.append(R_total)
	#import pdb; pdb.set_trace()
	temp_table = Table({'img_id': np.asarray(t['col0']), 'error': errors}, names=('img_id', 'error'))
	ascii.write(temp_table, "systematic_errors_"+str(site_name)+".txt", overwrite=True, format='no_header')

import pdb; pdb.set_trace()
LSC_DOMA_Values = np.loadtxt("systematic_errors_"+site_list[0][1]+".txt")
COJ_DOMA_Values = np.loadtxt("systematic_errors_"+site_list[1][1]+".txt")
CPT_DOMA_Values = np.loadtxt("systematic_errors_"+site_list[2][1]+".txt")

values = np.concatenate([LSC_DOMA_Values, COJ_DOMA_Values, CPT_DOMA_Values])

results_table = Table(rows=values)
results_table = results_table[results_table['col1'] > 0]

print(len(results_table))

ascii.write(results_table, "systematic_errors.txt", overwrite=True, format='no_header')

fig, ax = plt.subplots()

x = np.asarray(hjds)-2450000
y = np.log10(np.abs(residuals))

x_A = x[y >= 0]
y_A = y[y >= 0]

x_B = x[np.all([y < 0, y >= -1],axis=0)]
y_B = y[np.all([y < 0, y >= -1],axis=0)]

x_C = x[np.all([y < -1, y >= -2],axis=0)]
y_C = y[np.all([y < -1, y >= -2],axis=0)]

x_D = x[y < -2]
y_D = y[y < -2]

print("y >= 0: "+str(x_A))
print("-1 <= y < 0"+str(x_B))
print("-2 <= y < -1: "+str(x_C))
print("y < -2: "+str(x_D))

for x,y,color in [[x_A, y_A, 'red'], [x_B, y_B, 'orange'], [x_C, y_C, 'yellow'], [x_D, y_D, 'black']]:
	ax.scatter(x, y, c=color)
plt.xlabel('HJD')
plt.ylabel('log10(Residual)')	
plt.show()