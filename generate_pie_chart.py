from main_script import *
#import pdb; pdb.set_trace()
filenames = ['results_1565199596.2127893.txt', 'results_1565199732.7445877.txt', 'results_1565200011.1949725.txt', 'results_1565270066.2696369.txt', 'results_1565305626.7650695.txt', 'results_1565333078.415825.txt']
for file in filenames:
	data = np.loadtxt(file,dtype='O',skiprows=1)
	data = Table(rows=data)
	predictions = data['col4'].tolist()
	quantities = [predictions.count('CONSTANT'), predictions.count('VARIABLE'), predictions.count('CV'), predictions.count('ML')]
	percentages = np.asarray(quantities)/len(predictions)
	print("For file: "+str(file)+ ", we have "+str(quantities[0])+" constants, "+str(quantities[1])+" variable stars, "+str(quantities[2])+" CVs, and "+str(quantities[3])+" microlensing events.")





 


  
  

