from main_script import *

db = '/home/jclark/examples/ROME-FIELD-16_phot_3sites_ip.db'
star_id = ID = 107961
tel_choice = 7
filt_choice = '3'
conn = phot_db.get_connection(dsn=db)
query = 'SELECT hjd, image, calibrated_mag, calibrated_mag_err FROM phot WHERE star_id="'+str(star_id)+'" AND filter="'+str(filt_choice)+'" AND facility="'+str(tel_choice)+'"'
phot_table = phot_db.query_to_astropy_table(conn, query, args=())
print(phot_table)