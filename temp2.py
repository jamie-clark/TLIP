from main_script import *
from time import time

db = '/home/jclark/examples/ROME-FIELD-16_phot.db'
radius=60
tel_choices = [1,2,3,4,5,6,7,8,9,10,11,12]

filt = '3'
mag_err_cutoff=500
mag_cutoff = [12.75,15.25]

create_training_set(db,filt,tel_choices,mag_err_cutoff)
ROME_classification_script('17:59:27.3125', '-28:36:34.1901', radius,
	'/home/jclark/examples/ROME-FIELD-16_phot.db', '/home/jclark/examples/',filt,tel_choices,mag_cutoff,mag_err_cutoff)
print(time())
filt = '2'

create_training_set(db,filt,tel_choices,mag_err_cutoff)
ROME_classification_script('17:59:27.3125', '-28:36:34.1901', radius,
	'/home/jclark/examples/ROME-FIELD-16_phot.db', '/home/jclark/examples/',filt,tel_choices,mag_cutoff,mag_err_cutoff)
print(time())
filt = '1'

create_training_set(db,filt,tel_choices,mag_err_cutoff)
ROME_classification_script('17:59:27.3125', '-28:36:34.1901', radius,
	'/home/jclark/examples/ROME-FIELD-16_phot.db', '/home/jclark/examples/',filt,tel_choices,mag_cutoff,mag_err_cutoff)
print(time())
filt = '3'
mag_err_cutoff=0.5
mag_cutoff = [12.75,15.25]

create_training_set(db,filt,tel_choices,mag_err_cutoff)
ROME_classification_script('17:59:27.3125', '-28:36:34.1901', radius,
	'/home/jclark/examples/ROME-FIELD-16_phot.db', '/home/jclark/examples/',filt,tel_choices,mag_cutoff,mag_err_cutoff)
print(time())
filt = '2'

create_training_set(db,filt,tel_choices,mag_err_cutoff)
ROME_classification_script('17:59:27.3125', '-28:36:34.1901', radius,
	'/home/jclark/examples/ROME-FIELD-16_phot.db', '/home/jclark/examples/',filt,tel_choices,mag_cutoff,mag_err_cutoff)
print(time())
filt = '1'

create_training_set(db,filt,tel_choices,mag_err_cutoff)
ROME_classification_script('17:59:27.3125', '-28:36:34.1901', radius,
	'/home/jclark/examples/ROME-FIELD-16_phot.db', '/home/jclark/examples/',filt,tel_choices,mag_cutoff,mag_err_cutoff)
print(time())