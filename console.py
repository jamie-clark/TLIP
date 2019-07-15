from main_script import *

#print(extract_ra_dec(33065, '/home/jclark/examples/ROME-FIELD-16_phot.db'))

#drip_feed(107503, '/home/jclark/examples/ROME-FIELD-16_phot.db', 
#'/home/jclark/examples/','3',[1,2],20,1)
#ROME_classification_script('17:59:27.04', '-28:36:37.00', 0.1,
#'/home/jclark/examples/ROME-FIELD-16_phot.db', '/home/jclark/examples/','3',[1,2],[13,20],0.25)

#create_training_set('/home/jclark/examples/ROME-FIELD-16_phot.db','3',[1,2],17.5,0.25)

extract_lightcurves_on_position('17:59:27.04', '-28:36:37.00', 0.1,
'/home/jclark/examples/ROME-FIELD-16_phot.db', '3',[1,2],[13,20],0.25)

#plot_all_lightcurves(33065, '/home/jclark/examples/ROME-FIELD-16_phot.db', 1)