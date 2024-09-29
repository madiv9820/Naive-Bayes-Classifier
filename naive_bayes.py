import numpy as np

class GaussianNB:
    def __init__(self) -> None:
        self.__count_Dict = {}  # Dictionary to hold counts and statistics for each class
        self.__Inputs = None  # Input features
        self.__Outputs = None  # Output labels

    # String representation of the class
    def __str__(self) -> str:
        return 'naive_bayes.GaussianNB()'

    def __create_Dictionary(self) -> None:
        # Determine if a feature is categorical or continuous based on unique value counts
        is_Feature_Distinct = [(np.unique(self.__Inputs[:, feature_No])).shape[0] < 5 
                                for feature_No in range(self.__Inputs.shape[1])]
        
        # Iterate over each unique output class
        for output in np.unique(self.__Outputs):
            self.__count_Dict[output] = {}
            self.__count_Dict[output]['Count'] = np.sum(self.__Outputs == output)  # Count occurrences of the class

            # Iterate over each feature
            for feature_No in range(self.__Inputs.shape[1]):
                if is_Feature_Distinct[feature_No]:  # Categorical feature
                    self.__count_Dict[output][feature_No] = {}
                    
                    unique_Features = np.unique(self.__Inputs[:, feature_No])  # Unique values of the feature
                    
                    for feature in unique_Features:
                        count = np.sum((self.__Inputs[:, feature_No] == feature) & (self.__Outputs == output))
                        self.__count_Dict[output][feature_No][feature] = count
                
                else:  # Continuous feature
                    column = self.__Inputs[self.__Outputs == output][:, feature_No]
                    
                    variance = np.var(column, ddof=1)  # Sample variance
                    mean = np.mean(column)  # Mean
                    
                    self.__count_Dict[output][feature_No] = {
                        'variance': variance,
                        'mean': mean
                    }

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.__Inputs = x
        self.__Outputs = y
        self.__create_Dictionary()  # Build the count dictionary

    def __calculate_log_likelihood(self, sample, output):
        """Calculate the log-likelihood of a sample given a class."""
        log_likelihood = np.log(self.__count_Dict[output]['Count'])  # Initialize log likelihood
        
        for feature_No in range(len(sample)):
            if feature_No in self.__count_Dict[output]:
                if 'mean' in self.__count_Dict[output][feature_No]:
                    # Continuous feature: calculate Gaussian log-likelihood
                    mean = self.__count_Dict[output][feature_No]['mean']
                    variance = self.__count_Dict[output][feature_No]['variance']
                    
                    log_likelihood += (-0.5 * np.log(2 * np.pi * variance) - 
                                        (((sample[feature_No] - mean) ** 2) / (2 * variance)))
                
                else:
                    # Categorical feature with Laplace correction
                    feature_value = sample[feature_No]

                    # Add 1 for Laplace
                    count = self.__count_Dict[output][feature_No].get(feature_value, 0) + 1  
                    # Total unique categories
                    total_count = self.__count_Dict[output]['Count'] + len(self.__count_Dict[output][feature_No])  
                    
                    log_likelihood += np.log(count) - np.log(total_count)  # Log probability
        
        return log_likelihood

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in x."""
        predictions = []
        total_samples = len(self.__Outputs)  # Total number of samples
        
        for sample in x:
            class_log_probabilities = {}  # Store log probabilities for each class
            
            # Calculate log probabilities for each class
            for output in self.__count_Dict:
                prior = np.log(self.__count_Dict[output]['Count']) - np.log(total_samples)  # Log of prior probability
                likelihood = self.__calculate_log_likelihood(sample, output)  # Log likelihood
                class_log_probabilities[output] = prior + likelihood  # Total log probability for the class
            
            # Choose the class with the highest log probability
            predicted_class = max(class_log_probabilities, key = class_log_probabilities.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)

    def score(self, x: np.ndarray, y: np.ndarray) -> np.float64:
        """Calculate the accuracy of the model on test data."""
        outputs = self.predict(x)  # Get predictions
        return np.sum(outputs == y) / y.shape[0]  # Calculate accuracy