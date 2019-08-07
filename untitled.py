from main_script import *

ID = 107503 
db = '/home/jclark/examples/ROME-FIELD-16_phot.db'
tel_choices = [1,4]
mag_err_cutoff=0.5
prob_cutoff = 0.45
prob_override = 0.60
radius=60

#import pdb; pdb.set_trace()
#print(extract_ra_dec(33065, '/home/jclark/examples/ROME-FIELD-16_phot.db'))


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
import pdb; pdb.set_trace()
y = np.asarray([results.count(i) for i in overrides])
overrides = overrides[y == 0]