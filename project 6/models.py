import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)
    

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    converged = False

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Define layer sizes for easier adjustments and reuse
        input_to_hidden = 20
        hidden_to_output = 1

        # Initialize weights and biases
        self.W1 = nn.Parameter(1, input_to_hidden)
        self.b1 = nn.Parameter(1, input_to_hidden)
        self.W2 = nn.Parameter(input_to_hidden, hidden_to_output)
        self.b2 = nn.Parameter(1, hidden_to_output)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        first_layer_output = nn.Linear(x, self.W1)
        biased_first_layer = nn.AddBias(first_layer_output, self.b1)
        activated_first_layer = nn.ReLU(biased_first_layer)

        # Second layer processing
        second_layer_output = nn.Linear(activated_first_layer, self.W2)
        output = nn.AddBias(second_layer_output, self.b2)

        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(x)
        # Calculate and return the square loss between predictions and true labels
        loss = nn.SquareLoss(predictions, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss_threshold = 0.01  # Target loss to achieve before stopping training
        learning_rate = -0.01  # Negative learning rate for parameter update
        loss_value = float('inf')  # Initialize with maximum possible loss

        # Training continues until loss is reduced below the threshold
        while loss_value > loss_threshold:
            # Get loss node
            loss_node = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            
            # Compute gradients for all parameters
            params = [self.W1, self.b1, self.W2, self.b2]
            gradients = nn.gradients(loss_node, params)
            
            # Update parameters using the calculated gradients
            for param, grad in zip(params, gradients):
                param.update(grad, learning_rate)

            # Recalculate loss to determine if further training is needed
            loss_value = nn.as_scalar(loss_node)


                

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        input_size = 784  # Input dimension (flattened 28x28 image)
        hidden_size1 = 250  # Neurons in the first hidden layer
        hidden_size2 = 150  # Neurons in the second hidden layer
        output_size = 10    # Number of classes (digits 0-9)
        
        # Initialize weights and biases for the three layers
        self.W1 = nn.Parameter(input_size, hidden_size1)
        self.b1 = nn.Parameter(1, hidden_size1)
        self.W2 = nn.Parameter(hidden_size1, hidden_size2)
        self.b2 = nn.Parameter(1, hidden_size2)
        self.W3 = nn.Parameter(hidden_size2, output_size)
        self.b3 = nn.Parameter(1, output_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        first_linear_output = nn.Linear(x, self.W1)
        first_biased_output = nn.AddBias(first_linear_output, self.b1)
        first_activation_output = nn.ReLU(first_biased_output)

        # Second hidden layer processing
        second_linear_output = nn.Linear(first_activation_output, self.W2)
        second_biased_output = nn.AddBias(second_linear_output, self.b2)
        second_activation_output = nn.ReLU(second_biased_output)

        # Output layer processing
        output_linear = nn.Linear(second_activation_output, self.W3)
        output_biased = nn.AddBias(output_linear, self.b3)

        return output_biased
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_scores = self.run(x)
        # Calculate and return the softmax loss
        softmax_loss = nn.SoftmaxLoss(predicted_scores, y)
        return softmax_loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        minimum_accuracy = 0.976  # Accuracy threshold to stop training
        learning_rate = -0.34     # Learning rate for updates

        current_accuracy = float('-inf')  # Initialize with the lowest possible value
        while current_accuracy < minimum_accuracy:
            for inputs, labels in dataset.iterate_once(60):
                # Compute gradients for each parameter based on the current batch loss
                gradients = nn.gradients(
                    self.get_loss(inputs, labels),
                    [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
                )

                # Update each parameter using its corresponding gradient
                for param, grad in zip([self.W1, self.b1, self.W2, self.b2, self.W3, self.b3], gradients):
                    param.update(grad, learning_rate)

            # Evaluate the current model accuracy on the validation set
            current_accuracy = dataset.get_validation_accuracy()
            print(f"Current Validation Accuracy: {current_accuracy:.4f}")
