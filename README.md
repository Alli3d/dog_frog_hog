# DogFrogHog
ML classifier using computer vision to classify photos of dogs, frogs, and hogs.

Uses GridSearchCV to test which algorithm (SVM, random forest, log reg) works best, including a parameter sweep. Creates a classification report and a visualised confusion matrix from the best (manually selected) combination. 

Also conducts a small sanity test for the OpenCV function imread (was, for whatever reason, unstable on local machine but presents no issues when run elsewhere).

## Usage
`main.py` will not runon your local machine without access to the photos used for training (picture data omitted, approx 160 images at 80Mb).

This file cleans the images, encodes them for testing, then conducts the GridSearchCV parameter sweeps. The best combination is then trained and pickled.

`dogfroghog.pickle` is the complete algorithm. You do not need to run `main.py` first. Running `main.py` first will overwrite the current pickled algorithm.

## Output
Result of the project is a pickle file containing the algorithm. It usually reaches 70% accuracy (as opposed to a 33.3% accuracy of random guess).

## Project status
Project will likely be improved in the future by:
- Changing the image cleaning for ROI from Haar cascade to an algorithm designed to work with these animals. This may require deep learning methods.
- Adding more images/balancing out the ratio of the images (slightly less frog images than hog and dog images).