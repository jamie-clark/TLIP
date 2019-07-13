from main_script import *

#print(extract_ra_dec(42866, '/home/jclark/examples/ROME-FIELD-16_phot.db'))

#drip_feed(107628, '/home/jclark/examples/ROME-FIELD-16_phot.db', 
#'/home/jclark/examples/','3',[1,2],17,1.5)
#ROME_classification_script('17:59:27.05', '-28:36:37.0', 0.2,
#'/home/jclark/examples/ROME-FIELD-16_phot.db', '/home/jclark/examples/','3',[1,2],[10,20],1.5)

create_training_set('/home/jclark/examples/ROME-FIELD-16_phot.db','3',[1,2],17.5,0.25)


#plot_all_lightcurves(33065, '/home/jclark/examples/ROME-FIELD-16_phot.db', 0.25)