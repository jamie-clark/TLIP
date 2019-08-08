# Tools for LIA Integration with other LCO Packages


# Tools for LIA Integration with other LCO Packages

TLIP is an open-source collection of tools to use in combination with the LCO open-source microlensing detection program LIA and other LCO programs such as pyDANDIA and pyLIMA. The collection of tools allows the user to pull data from a database of stars using pyDANDIA, run them through the machine learning algorithm LIA, and process the results using microlensing fitting software pyLIMA. This code is currently written for use with ROME/REA data but could be adapted fairly easily for use with other surveys. The machine learning algorithm must be trained with a training set, the parameters to do this are set by the data.

# Installation

Requires Python3.6 -- to install all dependencies run

```
pip install -r requirements.txt
```

from the TLIP directory.

# Performing a test


The **simulate** module contains the framework necessary for simulating all individual classes. For simulating a complete training set, we’ve simplified the process by including all necessary steps within the **create_training** module. The ‘hard’ part is aggregating the necessary timestamps you expect the survey to measure in. These can be simulated, or be derived from real lightcurves if the survey is already underway. In this example we will assume a year-long space survey with daily cadence, hence only one timestamp for which to simulate our classes (please note that the set of timestamps must be appended to a list). We will also assume the survey has limiting magnitudes of 15 and 20, and as we don’t know the noise model of this imaginary survey, we will default to applying a Gaussian model — although the **noise_models** module contains a function for creating your own. Now, let’s simulate 500 of each class:

```
from main_script import *

# put the directory to your database here as a string
db = '/home/jclark/examples/ROME-FIELD-16_phot.db'


```

This function will output a FITS file titled ‘lightcurves’ that will contain the photometry for your simulated classes, sorted by ID number and class. It will also save two text files with labeled classes. The file titled ‘all_features’ contains the class label and the ID number corresponding to each light curve in the FITS file, followed by the 47 statistical values computed, while the other titled ‘pca_features’ contains only the class label followed by the principal components. We need these two text files to construct the required models.

```
from LIA import models

rf, pca = models.create_models(‘all_features.txt’, ‘pca_features.txt’)
```
With the RF model trained and the PCA transformation saved, we are ready to classify any light curve.

```
# fill these in with your own parameters. See the function for specific descriptions of all parameters
ROME_classification_script('18:00:12.01','-28:22:10.8', 0.2, db, '/home/jclark/examples/','3',[1,4],[13,18],0.5)
```
We’re interested only in the first two outputs which are the predicted class and the probability it’s microlensing, but by default the **predict** function will output the probability predictions for all classes. For more information please refer to the documentation available in the specific modules.

# pyLIMA

"We find that in practice the algorithm flags << 1% of lightcurves as microlensing, with false-alerts being prominent when data quality is bad. This is difficult to circumnavigate as we can only train with what we expect the survey to detect, and as such simple anomalies in the photometry can yield unpredictable results. In practice, we strongly recommend fitting each microlensing candidate LIA detects with [pyLIMA](https://github.com/ebachelet/pyLIMA), an open-source program for modeling microlensing events. By restricting microlensing parameters to reasonable observables, this fitting algorithm acts as a great second filter in the search for these rare transient events. We’ve had great success by restricting our PSPL parameters to the following:

* 0 <= tE <= 1000
* uo < 2.0
* Reduced Chi2 <= 10

These are mere suggestions, and as pyLIMA provides more information than this, we suggest you explore a parameter space that best fits your needs." 

# Test Script

To make sure that the algorithm is working, please run the following test scripts located in the **test** folder:

* test_features
* test_classifier

If both test scripts work you are good to go!
 
# How to Contribute?

Want to contribute? Bug detections? Comments? Suggestions? Please email us : 
jamesclark@ucsb.edu, etibachelet@gmail.com, rstreet@lcogt.net
