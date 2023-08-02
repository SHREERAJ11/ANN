import numpy as np

# Training data
training_data = {
    '0': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    '1': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    '2': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    '3': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    '4': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '5': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    '6': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    '7': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    '8': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    '9': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

# Even numbers will be labeled as 1, odd numbers as -1
labels = {
    '0': 1,
    '1': -1,
    '2': 1,
    '3': -1,
    '4': 1,
    '5': -1,
    '6': 1,
    '7': -1,
    '8': 1,
    '9': -1
}

class DeltaRulePerceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        # Step function as the activation function
        return 1 if x >= 0 else -1

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)

    def train(self, training_data, labels):
        for _ in range(self.epochs):
            for data, label in zip(training_data.values(), labels.values()):
                prediction = self.predict(data)
                error = label - prediction
                self.weights += self.learning_rate * error * np.array(data)
                self.bias += self.learning_rate * error

if __name__ == "__main__":
    perceptron = DeltaRulePerceptron(input_size=10, learning_rate=0.1, epochs=100)
    perceptron.train(training_data, labels)

    # Test the trained perceptron
    test_numbers = ['3', '8', '5']
    for number in test_numbers:
        binary_representation = training_data[number]
        prediction = perceptron.predict(binary_representation)
        result = "Even" if prediction == 1 else "Odd"
        print(f"Number {number} is {result}.")
