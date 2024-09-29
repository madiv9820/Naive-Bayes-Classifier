# Naive Bayes Classification

__Naive Bayes__ classification is a probabilistic machine learning technique based on Bayes' theorem, particularly useful for classification tasks. It assumes that the features used to predict the class are conditionally independent given the class label, which is why it's called __"naive"__.

## Key Formula

The formula for Bayes' theorem is:

$$
P(C | X) = \frac{P(X | C) \cdot P(C)}{P(X)}
$$

Where:
- \(P(C | X)\) is the posterior probability of class \(C\) given features \(X\).
- \(P(X | C)\) is the likelihood of features \(X\) given class \(C\).
- \(P(C)\) is the prior probability of class \(C\).
- \(P(X)\) is the total probability of features \(X\).

## Assumption

Naive Bayes assumes that the features are conditionally independent given the class label:

$$
P(X | C) = \prod_{i=1}^{n} P(X_i | C)
$$

## Example

Here’s a simple example of how Naive Bayes might work in a text classification task:
- __Data:__
    - Classes: Spam, Not Spam
    - Features: Words (e.g., "free", "win", "money")

- __Training:__
    - Calculate the prior probabilities P(Spam) and P(Not Spam).
    - For each word, calculate the likelihood P(word∣Spam) and P(word∣Not Spam).

- __Prediction:__
    - For a new email, calculate the probability of it being Spam and Not Spam based on the words it contains, and choose the class with the highest probability.

## Real-Life Applications

1. __Spam Detection:__ Naive Bayes is widely used in email filtering systems to classify emails as spam or not spam based on the frequency of certain words.

2. __Sentiment Analysis:__ Businesses use Naive Bayes to analyze customer reviews and feedback to determine whether the sentiment is positive, negative, or neutral.

3. __Document Categorization:__ News organizations and blogs can use Naive Bayes to categorize articles into predefined categories like sports, politics, and entertainment based on the content.

## Usage

You can implement Naive Bayes using Python's `sklearn` library.

## Python Implementation

```
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Example data
X = np.array([[1, 2], [1, 4], [1, 0],
              [2, 2], [2, 4], [2, 0]])
y = np.array([0, 0, 0, 1, 1, 1])  # Class labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```
<hr>

# Created Algorithm

## `GaussianNB`

This class implements the Naive Bayes classification algorithm for both categorical and continuous features.

__Attributes__
- `__count_Dict`: A private dictionary to hold counts and statistics for each class.
- `__Inputs`: Stores the input features.
- `__Outputs`: Stores the output labels.

__Methods__
- `fit(x: np.ndarray, y: np.ndarray)`: Fits the model to the training data.
- `predict(x: np.ndarray) -> np.ndarray`: Predicts class labels for input samples.
- `score(x: np.ndarray, y: np.ndarray) -> np.float64`: Calculates the accuracy of the model on test data.

### Method Descriptions

### `fit`

```python
def fit(self, x: np.ndarray, y: np.ndarray) -> None
```
__Parameters:__
- x: Input features as a NumPy array.
- y: Output labels as a NumPy array.

__Description:__ This method calculates the necessary statistics (mean, variance, counts) for each class based on the training data.

### `predict`

```python
def predict(self, x: np.ndarray) -> np.ndarray
```

__Parameters:__
- `x`: Input samples for which predictions are to be made.
Returns: An array of predicted class labels.

__Description:__ This method calculates the log probabilities for each class and predicts the class with the highest probability.

### `score`
```python
def score(self, x: np.ndarray, y: np.ndarray) -> np.float64
```

__Parameters:__
- `x`: Input features for testing.
- `y`: True output labels for testing.

__Returns:__ The accuracy of the model as a float.

__Description:__ This method evaluates the model by comparing predicted labels to true labels.
<hr>

# Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request with your changes.

# References
- [Naive Bayes Classifier - Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Understanding Naive Bayes - Towards Data Science](https://towardsdatascience.com/all-about-naive-bayes-8e13cef044cf)