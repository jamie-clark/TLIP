# daniels code
from __future__ import division
from main_script import *
#from tsne_python import tsne as TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsne
from LIA import extract_features, models

LIA_directory = '/home/jclark/examples/'
pca_model = models.create_models(str(LIA_directory)+'all_features.txt', str(LIA_directory)+'pca_features.txt')[1]
db = '/home/jclark/examples/ROME-FIELD-16_phot.db'
import pdb; pdb.set_trace()

data = np.loadtxt('pca_features.txt',usecols=np.arange(1,45))
#data = np.loadtxt('all_features.txt',usecols=np.arange(2,49))
x_data = np.asarray(data).astype('float64')
y_data = [1]*500+[2]*500+[3]*500+[4]*500+[5]*10+[6]*5+[7]*2 #V,C,ML,CV
for VAR in [107961]:
#for xxx,VAR in enumerate([74609, 74609]):
	mag, magerr = [[], []]
	try:
		for tel_choice in [1,2]: 
			mag, magerr = np.append([mag, magerr], extract_lightcurve(VAR, db, tel_choice=tel_choice,mag_err_cutoff=0.5)[1:3], axis=1)
	#	if xxx == 1:
	#		magerr = np.asarray(magerr)/1
		array=[]
		stat_array = array.append(extract_features.extract_all(mag, magerr, convert=True))
		array=np.array([i for i in array])
		stat_array = pca_model.transform(array)
	except:
		continue
	x_data = np.append(x_data, stat_array,axis=0)
#for CONS in [99309, 98402, 105836, 68914, 69516]:
#	pass
#for ML in [33065, 107503]:
#	pass
data = [np.asarray(i) for i in x_data]

vis_data = tsne(n_components=2).fit_transform(data)
vis_x = vis_data[:,0]
vis_y = vis_data[:,1]

#plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet",4))
#plt.title('Feature Space Distribution', fontsize=40)
#plt.colorbar(ticks=range(4))
#plt.
#plt.show()




colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'black']
fig, ax = plt.subplots()
for i,object_class in enumerate(['Variable', 'Constant', 'CV', 'ML']):
    x=vis_x[i*500:(i+1)*500]
    y=vis_y[i*500:(i+1)*500]
    ax.scatter(x, y, c=colors[i], label=object_class)



x=vis_x[2000:2008]
y=vis_y[2000:2008]
object_class='Real Variable'
ax.scatter(x, y, s=1000, c=colors[4], label=object_class)
#ax.text()

#x=vis_x[2010:2015]
#y=vis_y[2010:2015]
#object_class='Real Constant'
#ax.scatter(x, y, s=100, c=colors[5], label=object_class)


#x=vis_x[2015:2017]
#y=vis_y[2015:2017]
#object_class='Real ML'
#ax.scatter(x, y, s=100, c=colors[6], label=object_class)


ax.legend()
ax.grid(True)
ax.set_title('Feature Space Distribution', fontsize=40)
plt.show()


















