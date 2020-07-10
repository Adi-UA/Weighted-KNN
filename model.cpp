#include <Python.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <unordered_map>
#include <string>
#include <tuple>
#include <vector>

using namespace std;


/**
 * @brief Converts a string of the form "1 2 3.7 4 5.1 6 7" into a vector of
 * double with a single whitespace used as the delimiter.
 *
 * @param charPtr The string to convert
 * @return vector<double> The converted vector
 */
vector<double> stringToDoubleVector(const char* charPtr) {
    vector<double> vec;
    string s(charPtr);
    int length = s.length();

    string temp = "";
    for (int i = 0; i < length; i++) {
        char ch = s.at(i);
        if (ch == ' ') {
            vec.push_back(stod(temp));
            temp = "";
        }
        else {
            temp += ch;
        }
    }
    vec.push_back(stod(temp));

    return vec;
}

/**
 * @brief Converts a string of the form "1 2 3.7|4 5.1 6 7" into a vector of
 * double vectors. Each vector in the list fo vectors is delimited by '|'
 *
 * The example gets converted into a vector of two vectors such that the two
 * vectors are: [1, 2, 3.7] and [4, 5.1, 6, 7]
 *
 * @param charPtr The string to convert
 * @return vector<vector<double>> The vector containing the list fo vectors.
 */
vector<vector<double>> stringToNestedDoubleVector(const char* charPtr) {
    vector<vector<double>> retval;
    string s(charPtr);
    int length = s.length();

    string temp = "";
    for (int i = 0; i < length; i++) {
        char ch = s.at(i);
        if (ch == '|') {
            retval.push_back(stringToDoubleVector(temp.c_str()));
            temp = "";
        }
        else {
            temp += ch;
        }
    }
    retval.push_back(stringToDoubleVector(temp.c_str()));

    return retval;
}

/**
 * @brief  Uses the features and their respective weights to calculate the
 euclidean distance between two vectors. The formula is:
 sqrt(sum(weight[i]*abs(vec2[i]-vec1[i]))) where i is the index of the feature.
 *
 * @param vec1 A vector with n features. The n-th feature should be the output
 which is not used in the calculation.
 * @param vec2 Another vector with n features. The n-th feature should be the
 output which is not used in the calculation.
 * @param weights The weights of n features in sequence. Each weight should
 typically be between 0.0 and 1. The n-th feature weight doesn't matter.
 * @return double The calculated euclidean distance. It will always be greater
 than 0.
 */
double weightedEDistance(vector<double>& vec1,
                         vector<double>& vec2,
                         vector<double>& weights) {
    double distance = 0.0;

    int length = vec1.size();

    for (int i = 0; i < length - 1; i++) {
        double weight = weights.at(i);
        double contribution = weight * abs(vec2.at(i) - vec1.at(i));
        distance += contribution;
    }

    return sqrt(distance);
}

/**
 * @brief Given a dataset of train vectors to check against, this function finds
 the k nearest samples and returns them.
 *
 * @param trainSet The vector of vectors to check against.
 * @param test  The vector to find the neighbors of.
 * @param k The number of neighbors to shortlist.
 * @param weights The weights of n features in sequence. Each weight should
 typically be between 0.0 and 1. The n-th feature weight doesn't matter.

 * @return vector<tuple<double, double>> A vector of the tuple containing
 (category, distance) pairs where distance is the calculated distance from the
 test input.
 */
vector<tuple<int, double>> findKNearest(vector<vector<double>>& trainSet,
                                           vector<double>& test,
                                           int k,
                                           vector<double>& weights) {

    vector<tuple<int, double>> candidates;

    for (vector<double> & vec : trainSet) {
        if (k <= 0) {
            break;
        }
        else {
            double distance = weightedEDistance(test, vec, weights);
            tuple<int, double> candidate;
            candidate = make_tuple(vec.back(), distance);
            candidates.push_back(candidate);
        }
    }

    sort(
        candidates.begin(),
        candidates.end(),
        [](const tuple<int, double>& lhs, const tuple<int, double>& rhs) {
            return get<1>(lhs) < get<1>(rhs);
        });

    vector<tuple<int, double>> kCandidates;
    for(int i=0; i< k; i++){
        kCandidates.push_back(candidates.at(i));
    }

    return kCandidates;
}

/**
 * @brief This function finds the K nearest neighbors and places them into a
 * vector of candidates which it uses to calculate scores for each category to
 * determine their probability.
 *
 * @param trainSet vector of vectors to check against.
 * @param test  The vector to find the neighbors of.
 * @param k The number of neighbors to shortlist.
 * @param weights The weights of n features in sequence. Each weight should
 * typically be between 0.0 and 1. The n-th feature weight doesn't matter.
 * @param count_weight This determines how much importance is given to the
 * number of a certain category present in the candidates list. Higher weights
 * means a category scores more when more of it is present in candidates.
 * @param distance_weight This determines how much importance is given to the
 * distance of a certain category present in the candidates list. Higher weights
 * means a category scores more when it is LESS distant from the test input.
 * @return map<double, double> The keys are the category and the values are
 * their score
 */
unordered_map<int, double> findCategoryScores(vector<vector<double>>& trainSet,
                                       vector<double>& test,
                                       int k,
                                       vector<double>& weights,
                                       double count_weight,
                                       double distance_weight) {

    unordered_map<int, int> categoryCount;
    unordered_map<int, double> categoryScores;

    vector<tuple<int, double>> candidates =
        findKNearest(trainSet, test, k, weights);

    for (auto& candidate : candidates) {
        int category = get<0>(candidate);
        double distance = (get<1>(candidate)) * (-1);

        if (categoryCount.count(category) == 0) {
            categoryCount.insert(pair<int, int>(category, 1));
            categoryScores.insert(
                pair<int, double>(category, distance_weight * distance));
        }
        else {
            int oldCount = categoryCount.at(category);
            categoryCount.insert(pair<int, int>(category, oldCount + 1));
            double oldScore = categoryScores.at(category);
            categoryScores.insert(pair<int, double>(
                category, oldScore + (distance_weight * distance)));
        }
    }

    unordered_map<int, double>::iterator itr;
    for (itr = categoryScores.begin(); itr != categoryScores.end(); ++itr) {
        int category = itr->first;
        double oldScore = itr->second;
        categoryScores.insert(pair<int, double>(
            category, oldScore + (count_weight * categoryCount.at(category))));
    }

    return categoryScores;
}

/**
 * @brief This function finds the K nearest neighbors and places them into a
 * vector of candidates which it uses the find category scores. These category
 * scores in turn are then weighted based on the count and distance weights and
 * a final category result is returned.
 *
 * @param trainSet The vector of vectors to check against.
 * @param test  The vector to find the neighbors of.
 * @param k The number of neighbors to shortlist into a candidates list.
 * @param weights The weights of n features in sequence. Each weight should
 * typically be between 0.0 and 1. The n-th feature weight doesn't matter.
 * @param count_weight This determines how much importance is given to the
 * number of a certain category present in the candidates list. Higher weights
 * means a category scores more when more of it is present in the calculated
 * candidates.
 * @param distance_weight This determines how much importance is given to the
 * distance of a certain category present in the candidates list. Higher weights
 * means a category score is more when it is LESS distant from the test input.
 *
 * @return double The final category the input vector was classified as.
 */
int classifyVec(vector<vector<double>>& trainSet,
                vector<double>& test,
                int k,
                vector<double>& weights,
                double count_weight = 1,
                double distance_weight = 1) {

    unordered_map<int, double> categoryScores = findCategoryScores(
        trainSet, test, k, weights, count_weight, distance_weight);

    bool isFirst = true;
    double maxScore = 0;
    unordered_map<int, double>::iterator itr;
    for (itr = categoryScores.begin(); itr != categoryScores.end(); ++itr) {
        double score = itr->second;

        if (isFirst) {
            maxScore = score;
            isFirst = false;
        }
        else {
            if (score > maxScore) {
                maxScore = score;
            }
        }
    }

    int finalCategory = 0;
    for (itr = categoryScores.begin(); itr != categoryScores.end(); ++itr) {
        int category = itr->first;
        if (maxScore == categoryScores.at(category)) {
            finalCategory = category;
            break;
        }
    }

    return finalCategory;
}

/**
 * @brief This function is called by the Python code to actually run the
 *internal model code. The parameters  and return value documented below are as
 *they would be seen and should be used within python.
 *
 *
 * @param train_vector_set The list of vectors in the train set used to classify
 *the test vector. Must be passed as a string such that each vector of the train
 *set is separated by '|' and in each vector the values are separated by a
 *single whitespace. For example, the vectors [[1,2,3],[4,5,6]] would be input
 *as "1 2 3|4 5 6".
 *@param test_vector The vector representing the input to be classified. Must be
 *passed in as a string. Since this is a single vector it is represented as
 *values separated by a single white space. So a vector [1,2,3] becomes "1 2 3".
 * @param weight_vector The vector representing the weights assigned to each
 *feature during calculation. The n-th weight is ignored. Input this in the same
 *format as test_vector.
 * @param k An integer representing the number of k values to check against.
 * @param count_weight This determines how much importance is given to the
 *number of a certain category present in the candidates list. Higher weights
 *means a category scores more when more of it is present in the calculated
 *candidates.
 * @param distance_weight This determines how much importance is given to the
 *distance of a certain category present in the candidates list. Higher weights
 *means a category score is more when it is LESS distant from the test input.
 * @return double The classification
 */
static PyObject* classify(PyObject* self, PyObject* args) {
    char* trainSetChars;
    char* testVectorChars;
    char* weightChars;
    int k;
    double count_weight;
    double distance_weight;

    if (!PyArg_ParseTuple(args,
                          "sssidd",
                          &trainSetChars,
                          &testVectorChars,
                          &weightChars,
                          &k,
                          &count_weight,
                          &distance_weight)) {
        return NULL;
    }

    vector<vector<double>> trainSet = stringToNestedDoubleVector(trainSetChars);
    vector<double> test = stringToDoubleVector(testVectorChars);
    vector<double> weights = stringToDoubleVector(weightChars);
    int classification =
        classifyVec(trainSet, test, k, weights, count_weight, distance_weight);

    return Py_BuildValue("i", classification);
}

/**
 * @brief Simply defines the name given to the relevant methods when they are
 * called from python and what function they should call internally.
 */
static PyMethodDef myMethods[] = {
    {"classify",
     classify,
     METH_VARARGS,
     "Classify a test vector using the weighted KNN model"},
    {NULL, NULL, 0, NULL}};

/**
 * @brief Define the module
 */
static struct PyModuleDef wknn = {
    PyModuleDef_HEAD_INIT, "wknn", "Weighted KNN", -1, myMethods};

/**
 * @brief Function that allows the module to be built.
 */
PyMODINIT_FUNC PyInit_wknn(void) { return PyModule_Create(&wknn); }
