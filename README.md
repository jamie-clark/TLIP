# Tools for LIA Integration with other LCO Packages

TLIP is an open-source collection of tools to use in combination with the LCO open-source microlensing detection program LIA and other LCO programs such as pyDANDIA and pyLIMA. The collection of tools allows the user to pull data from a database of stars using pyDANDIA, run them through the machine learning algorithm LIA, and process the results using microlensing fitting software pyLIMA. This code is currently written for use with ROME/REA data but could be adapted fairly easily for use with other surveys. The machine learning algorithm must be trained with a training set, the parameters to do this are set by the data.

# Installation

Requires Python3.6 -- to install all dependencies run

```
pip install -r requirements.txt

or

pip install --user -r requirements.txt

if necessary
```

from the TLIP directory.

# Performing a test


The **main_script.py** file contains all the primary functions provided for use in the toolkit.

```
from main_script import *

# put the directory to your database here as a string
db = '/home/jclark/examples/ROME-FIELD-16_phot.db'

# customize these parameters as you see fit, and replace "star_id" with the ID of a star in your DB (try positive integer values)
hjd, mag, magerr = extract_lightcurve(star_id, db, filt_choice='3', tel_choice=2, mag_err_cutoff=0.1)
plot_lightcurve(hjd, mag, magerr)
```

If this runs successfully, you should see a plot of a lightcurve. This is a good test to see that the algorithm is working. Next we will create the training set.

```
# modify this with parameters
create_training_set(db,'3',[1,4],0.25)
```
Ideally, this will work properly. If this errors, fiddle around with the magnitude range on the final command of the "create_training_set" function.

Now, it is time to run the actual classification! 

```
# fill these in with your own parameters. See the function for specific descriptions of all parameters
ROME_classification_script('18:00:12.01','-28:22:10.8', 0.2, db, '/home/jclark/examples/','3',[1,4],[13,18],0.5)
```
This will write a "results.txt" file in the working directory. This is the result of the classification script, it will print a list of star IDs with the predicted class and probability.

# pyLIMA

"We find that in practice the algorithm flags << 1% of lightcurves as microlensing, with false-alerts being prominent when data quality is bad. This is difficult to circumnavigate as we can only train with what we expect the survey to detect, and as such simple anomalies in the photometry can yield unpredictable results. In practice, we strongly recommend fitting each microlensing candidate LIA detects with [pyLIMA](https://github.com/ebachelet/pyLIMA), an open-source program for modeling microlensing events. By restricting microlensing parameters to reasonable observables, this fitting algorithm acts as a great second filter in the search for these rare transient events. We’ve had great success by restricting our PSPL parameters to the following:

* 0 <= tE <= 1000
* uo < 2.0
* Reduced Chi2 <= 10

These are mere suggestions, and as pyLIMA provides more information than this, we suggest you explore a parameter space that best fits your needs." -- from LIA github. Applies here
 
# How to Contribute?

Want to contribute? Bug detections? Comments? Suggestions? Please email us : 
jamesclark@ucsb.edu, etibachelet@gmail.com, rstreet@lcogt.net
