from data_handlers import *
import numpy as np
import wknn


def run_model(train_set,
              test_set,
              k,
              weights=None,
              count_weight=1,
              distance_weight=1):
    """
    This function runs the KNN model based on the input parameters

    Args:
        train_set (list of lists): Training dataset where each row of the list of lists represents an entry's vector with the last index of the list being the output value.
        test_set (list of lists): Testing dataset where each row of the list of lists represents an entry's vecto with the last index of the list being the expected output.
        k (int): The number of neigbors to check against
        weights (list, optional): The weights to assign each feature of the input vector in sequential order. Defaults to None which runs the model with everything weighted to 1.
        count_weight (int, optional): How much should the classification favour the number of a certain category present among the neighbors of a testcase. Defaults to 1.
        distance_weight (int, optional): How much should the classification favour the distance of each entry present among the neighbors of a testcase. Defaults to 1.

    Returns:
        tuple: Returns a list of known output, predicted output and model accuracy on the test input.
    """

    length = len(test_set)
    y = np.zeros((length))
    y_pred = np.zeros((length))

    if weights is None:
        weights = np.ones((length))  # Default all weights to 1
    weights_str = ' '.join(map(str, weights))

    train_set_str = ' '.join(map(str, train_set[0]))
    for i in range(1, len(train_set)):
        train_set_str += ("|" + ' '.join(map(str, train_set[i]))
                          )  # Create formatted string representing test set

    # Run on Test Data
    for i, test in enumerate(test_set):
        test_str = ' '.join(map(str, test))
        final_class = wknn.classify(train_set_str, test_str, weights_str, k,
                                    count_weight, distance_weight)

        y[i] = test[-1]
        y_pred[i] = final_class

    # Find Accuracy on Test Data
    correct = np.sum(y == y_pred)

    accuracy = correct / length * 100
    return accuracy


# Read the data
data_file = "data/winequality-white.csv"  # Looks inside the current directory
df = pd.read_csv(data_file, sep=';')

# Run Model
wt = (0.2, 0.7, 0.2, 0.5, 0.2, 0.3, 0.25, 0.3, 0.3, 0.5, 1, 1)
train, test = random_train_test_split(df, 0.75, 3)

accuracy = run_model(train, test, 3, weights=wt, distance_weight=0.5)

# Print result
print("Model Accuracy: " + str(accuracy) + " %")
