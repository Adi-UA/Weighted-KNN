import numpy as np
from math import sqrt


def weighted_euclidean_distance(vec1, vec2, weights):
    """
    Uses the features and their respective weights to calculate the euclidean distance between two vectors.
    The formula is: sqrt(sum(weight[i]*(vec2[i]-vec1[i]))) where i is the index of the feature.

    Args:
        vec1 (list): A vector with n features. The n-th feature should be the output which is not used in the calculation.
        vec2 (list): Another vector with n features. The n-th feature should be the output which is not used in the calculation.
        weights (list): The weights of n features in sequence. Each weight should typically be between 0.0 and 1. n-the feature weight doesn't matter.

    Returns:
        float: The calculated euclidean distance. It will always be greater than 0.
    """
    distance = 0.0
    
    for i in range(len(vec1) - 1):
        weight = weights[i]
        contribution = weight * abs(vec2[i] - vec1[i])

        distance += contribution
    
    return sqrt(distance)


def find_k_nearest(train, test, k, weights = None):
    """
    Given a dataset of train vectors to check against, this function finds the k nearest samples and returns them.

    Args:
        train (list of lists): The list of vectors to check against.
        test (list of lists): The vector to find the neighbors of.
        k (int): The number of neighbors to shortlist.
        weights (list, optional): The weights of n features in sequence. Each weight should typically be between 0.0 and 1. n-the feature weight doesn't matter. If nothing is given it sets all feature weights to 1.

    Returns:
        list: A list of the (cateegory, distance) where distance is the calculated distance from the test input.
    """
    candidates = []

    if weights is None:
        weights = [1]*len(test)

    for vec in train:
        if k<=0:
            break
        distance = weighted_euclidean_distance(test,vec,weights)
        result = (vec[-1], distance)
        candidates.append(result)
    
    candidates.sort(key=lambda item: item[1])
    return candidates[:k]


def find_category_scores(candidates, count_weight, distance_weight):
    """
    Given the candidates returned by the find_k_nearest function, this function calculates scores for each category to determine their probability.

    Args:
        candidates (list): The (category, distance) tuples
        count_weight (float): This determines how much importance is given to the number of a certain category present in the candidates list. Higher weights means a category scores more when more of it is present in candidates.
        distance_weight (float): This determines how much importance is given to the distance of a certain category present in the candidates list. Higher weights means a category scores more when it is LESS distant from the test input.

    Returns:
        dict: The keys are the category and the values are their score
    """
    category_count = {}
    category_scores = {}

    for candidate in candidates:
        category = candidate[0]
        distance = -candidate[1]  # negative because larger distaces lower scores

        if category not in category_count:
            category_count[category] = 1
            category_scores[category] = distance_weight * distance
        else:
            category_count[category] = category_count[category] + 1
            category_scores[category] = category_scores[category] + (distance_weight * distance)
    
    for category in category_scores:
        category_scores[category] = category_scores[category] + (count_weight * category_count[category])
    
    return category_scores


def classify(candidates, count_weight = 1, distance_weight = 1):
    """
    This function uses the calculated scores to classify the test input by choosing the maximum score to be the "winner".

    Args:
        candidates (list): The (category, distance) tuples
        count_weight (float, optional): This determines how much importance is given to the number of a certain category present in the candidates list. Higher weights means a category scores more when more of it is present in candidates. It is 1 by default.
        distance_weight (float, optional): This determines how much importance is given to the distance of a certain category present in the candidates list. Higher weights means a category scores more when it is LESS distant from the test input. It is 1 by default

    Returns:
        The final category as determined by the algorithm.
    """
    category_scores = find_category_scores(candidates, count_weight, distance_weight)
    max_score = max(category_scores.values())

    final_category = None
    for category in category_scores:
        if max_score == category_scores[category]:
            final_category = category
            break
    
    return final_category
