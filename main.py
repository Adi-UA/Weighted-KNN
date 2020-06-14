from model import *
from data_handlers import *


def run_model(
        train_set,
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

    y = []
    y_pred = []

    # Run on Test Data
    for test in test_set:
        if weights is not None:
            candidates = find_k_nearest(train, test, k, weights)
        else:
            candidates = find_k_nearest(train, test, k)

        final_class = classify(candidates, count_weight, distance_weight)
        y.append(test[-1])
        y_pred.append(final_class)

    # Find Accuracy on Test Data
    correct = 0
    total_preds = len(y_pred)
    for i in range(total_preds):
        if y_pred[i] == y[i]:
            correct += 1

    accuracy = correct / total_preds * 100
    return y, y_pred, accuracy


# Read the data
data_file = "data/winequality-white.csv"  # Looks inside the current directory
df = pd.read_csv(data_file, sep=';')

# Run Model
wt = (0.2, 0.7, 0.2, 0.5, 0.2, 0.3, 0.25, 0.3, 0.3, 0.6, 1, 1)
train, test = random_train_test_split(df, 0.75, 3)
y, y_pred, accuracy = run_model(
    train, test, 3, weights=wt, distance_weight=0.5)

# Print result
print("Model Accuracy: " + str(accuracy) + " %")
