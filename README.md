# K-Nearest-Neighbors
My implementation of KNN for my Intro to Artificial Intelligence class.

We are given a set of training data, with 1469 items, and testing data, with 3429 items.

Each point has 11 attributes (or dimensions) and a class label.

To give an idea of the computation that occurs: The distance between 2 points, in 11-dimensional space, is calculated 5,037,201 times, for each value of 'k'.

-----------------------------------------
The general pseudocode is:
- For each item X in testing data:
  - Calculate distance to each point Y in training data
  - Argsort these distances in increasing order
  - From the sorted list, select top 'k' items
  - Find the most frequent class from these 'k' items, this is the predicted class for X
    - Ties should be broken in favor of the class that comes first in the data file (point with lowest index)
- Compare predicted label to true label, count correct/incorrect
