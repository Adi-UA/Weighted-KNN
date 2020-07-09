# Weighted-KNN

This was my attempt at writing the K-Nearest Neighbors algorithm, and I decided to make a few modifications along the way to see how much of a difference it would make. Specifically, after I implemented the basic K-NN algorithm, I augmented it to accept weights from the user for each feature in a vector and for how the neartest neighbors affected the final output.

## The Model

### Basics 

For those of you unfamiliar with how the KNN algorithm works, it can at a high level be described as:
* Compare each input feature for one test data point against the same inputs of everything in the training dataset.
* Store the results of that comparison as the distance between the various points in the train set.
* Sort the results to find K entries in the train set that have the least distance from the test data.
* Use the known output values of those K results to predict which category the input data belongs to.

Look [here](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) for a more in-depth but easy-to-understand of KNN.

### Specifics (assuming you know the basics)

#### Distances
This model compares the i-th feature of the input vector with the i-th feature of the other training vector with optional feature weights. The user can specify the importance given to a certain feature while calculating total distance by specifying a weight for the feature typically between 0 and 1 inclusive. For example, if input vectors had n features then the user would input a tuple/list of size n where a feature's weight corresponded to its index in the input vector; weight for the feature on index 3 would occupy the 3rd index in the weights iterable and so on.

These weights can be negative, but it's completely useless to do so because the algorithm caluclates the absolute distance between correponding features before multiplying the weight, and values < 0 are automatically not considered. Conversely, specifying a weight to be greater than 1 could have strange effects if you don't scale the other weights properly.

The absolute distance is not the same as the euclidean distance used in standard K-NN. The formula doesn't square differences because weights can be added manually.

In sumary, the standard formula for two vectors x and y is: **sqrt( sum( (y[i] - x[i])^2))**, and
 
the modified formula is: **sqrt(sum( abs(y[i] - x[i]) * weight[i] ))**

When he distances have been calculated for all values in the train set, the K nearest neighbors are chosen and are stored in a list of tuples as (category, distance)
.
#### Classification

After, the distances have been calculated as specified above and the K nearest neighbors have been found, the following algorithm is used to classify the input: 

* Go through each of the K neighbors (in the list of tuples from the previous step)
* Count the number of times each category occurs and store it in a category count dictionary.
* For each category occuerence, extract the distance, negate it and add ( _distance_weight_ * negated distance) to the dictionary of the category's distance total score. 
* Once the neighbors have been iterated through, add the (respective count scores * _count weight_) to the distance scores of the corresponding categories
* Find the maximum resulting score and report the associated category as the output.

What the algorithm is doing through the above steps is keeping a score for each category. The score lowers with greater distance values and improves with larger count values. This approach works quite well in general. Consider the following example:

We have 3 nearest neighbors classified into category A or B as: [(A:0),(B:1),(B:2)]
Since the algorithm disapproves of larger distances, the net score for cateory B is lower than that of category A. In this case category A is a perfect match with the input since the distance is 0, and despite only one of the 3 nearest neighbors being from category A, the algorithm is able to figure out that A is the right choice since it is a perfect match.

I have added another facet to this algorithm in the form of count_weight and distance_weight. These default to 1, but can be used to set how much each aspect of the score affects the final classification. That is, a lower count_weight means more occurences of a category in the nearest neighbors isn't as important in the overall classification as the distance score. Conversely, the same is true for the distance weight. If we set the distance weight to 0 for example, we would essentially be saying that classify from the nearest neighbors based only on which category occurs most frequently. Playing around with these parameters has given me some interesing results (Sample results given below).


## Using

### Before Running (Assuming you have pip, git and an appropriate version of python)

* If you're not worried about breaking anything in your environment:
    
    * Run `pip install -r requirements.txt`

* Otherwise, if you're trying to install the dependencies individually:
    * Run `pip install numpy`
    * Run `pip install pandas`
    
Then, clone the repo with: `git clone https://github.com/Adi-UA/Weighted-KNN.git`.

**Note:** I used `python 3.7.x`,`numpy 1.19.0`, and `pandas 1.0.5`.

### Running

How do you use the model? Refer to the file `main.py` where I've set up a simple example on the [white wine dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality).

It should be reasonably easy to figure out what has to be done from looking at the file, but basically, once the input has been formatted into a list of lists representing a list of vectors either using your own functions or using the ones I
povided in `data_handlers.py` you should simply be able to pass in the train and test data to the `run_model` function. You will of course also need to specify a K value but everthing else is optional. If you don't give any weights it will simply run like a normal K-NN (all weights default to 1).

## Sample Results (Rounded):
K = 3 for all the following results with a 75-25 train-test split.

* [White Wines](https://archive.ics.uci.edu/ml/datasets/wine+quality): Standard K-NN --> 55% accuracy (seed = 3)
  * With only feature weights = (0.2, 0.7, 0.2, 0.5, 0.2, 0.3, 0.25, 0.3, 0.3, 0.6, 1, 1) --> 55% accuracy (seed = 3)
  * With only distance weight = 0.5 --> 56% accuracy (seed = 3)
  * With both feature and distance weights --> 62% accuracy (seed = 3)

* [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris): Standard K-NN --> 97% (seed = 3)
  * With distance weight 0.5 --> 97% (seed = 3)
                
* [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database): Standard K-NN --> 64% (seed = 3)
  * With distance weight = 0 --> 69% (seed = 3)
  * With distance weight = 0 --> 71% (seed = 5)
* [Banknote Authentication Dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication): Standard KNN --> 100% (seed = 3)

Within the repo these datasets can be found under `data/`
