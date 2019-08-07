from main_script import *

db = '/home/jclark/examples/ROME-FIELD-16_phot.db'
ID = 107961
#ID = 81365
#ID = 60174
#ID = 107961



IDs = [20041,
  20005,
  19708,
  19664,
  19390,
  19113]  

IDs = [60751,
  60706,
  60590,
  60508,
  60420,
  60256,
  60174,
  59887]

#plot_all_lightcurves(ID, db, 500)

IDs = [81365, 76197, 74609, 74525, 78373, 84428, 77477, 85785, 92367, 66221,
107961, 60420,
99309, 98402, 105836, 68914, 69516]

#import pdb; pdb.set_trace()
#print(extract_ra_dec(ID, db))
#drip_feed(ID, db, '/home/jclark/examples/', '3', [1,4],20,2)
#ROME_classification_script('18:00:12.01','-28:22:10.8', 60, db, '/home/jclark/examples/','3',[1,4],[13,18],0.5)
for ID in IDs:
	try:
		pass
		# LSC DOMA
		#drip_feed(ID, db, '/home/jclark/examples/','3',[1,4],20,2)
		# COJ DOMA
		#drip_feed(ID, db, '/home/jclark/examples/', '3', [2,5],20,2)
		# CPT DOMA
		#drip_feed(ID, db, '/home/jclark/examples/', '3', [3,6],20,2)
	except:
		print("Error on star ID "+str(ID)+". Continuing to the next ID...")
#create_training_set(db,'3',[1,4],0.25)
#extract_lightcurves_on_position('17:59:27.04', '-28:36:37.00', 0.1,
#db, '3',[1,2],[13,20],0.25)

#extract_lightcurve(107961, db, tel_choice=3, mag_err_cutoff=500)
#for ID in IDs: 
#
#for ID in [36055, 107503, 42866, 10203, 100, 75000, 3003, 135000, 22031, 83000]:
#print(extract_lightcurve(ID,db,mag_err_cutoff=0.5))
#star_id ra dec filter_telescope prediction probability ml_probability

#pyLIMA_plot_from_db(ID, db, tel_choices=[2])
#full_pipeline()
#plot_cam12_lightcurves(10855, db, 500)
for ID in IDs[12:]:
	try:
		print(ID)
		extract_blank_residuals(ID, db,'3',[1,2,3,4,5,6],"airmass")
	except:
		pass


for ID in IDs[12:]:
	try:
		print(ID)
		#extract_background_residuals(ID, db,'3',[1,2,3,4,5,6])
	except:
		pass