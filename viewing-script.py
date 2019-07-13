import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
hdul = fits.open('lightcurves.fits')

import pdb; pdb.set_trace()


data = hdul[1].data.tolist()

for j in [1.0, 2.0, 3.0, 4.0, 5.0, 1100.0, 1101.0, 1102.0, 1103.0, 1104.0, 1536.0, 1537.0, 1538.0, 1539.0, 1540.0, 2203.0, 2204.0, 2205.0, 2206.0, 2207.0]:
	#import pdb; pdb.set_trace()

	magnitudes = []
	times = []

	results = [t for t in data if t[1] == j]
	
	for i in results: 
		magnitudes.append(i[3]); 
		times.append(i[2]);
	print(results[0][0])
	plt.scatter(times, magnitudes)
	plt.gca().invert_yaxis()
	plt.show()
	print(len(times))
import pdb; pdb.set_trace()

