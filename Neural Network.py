# Imported packages
import math
import random
import threading
import queue
import csv
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Constants
NEURON_SCALE = 5  # Scale for neuron visualization
SCREEN_PADDING = 25  # Padding around the canvas


# Sigmoid activation function maps any real value to a range of (0, 1)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Tanh activation function maps values to (-1, 1) for better gradient propagation
def tanh(x):
    return math.tanh(x)

# ReLU activation function outputs 0 for negative inputs, x for positive inputs
def relu(x):
    return max(0, x)


# Derivative of sigmoid, needed for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Derivative of tanh, used for backpropagation
def tanh_derivative(x):
    return 1 - x ** 2

# Derivative of ReLU, which is either 0 or 1
def relu_derivative(x):
    return 1.0 if x > 0 else 0.0


# Normalizes data to range [0, 1] for better numerical stability in training
def normalize_data(data):
    normalized_data = []
    for col in zip(*data):
        col_min = min(col)
        col_max = max(col)
        if col_max == col_min:  # Avoid division by zero
            normalized_col = [0.5] * len(col)  # Neutral value if column is constant
        else:
            normalized_col = [(x - col_min) / (col_max - col_min) for x in col]
        normalized_data.append(normalized_col)
    return list(zip(*normalized_data))  # Transpose back to original structure


# Represents the dataset, including input features and labels
class Dataset:
    def __init__(self):
        self.x = []  # Input features
        self.y = []  # Corresponding labels
        self.input_size = 0  # Number of features in each sample
        self.output_size = 0  # Number of unique classes
        self.label_mapping = {}  # Mapping of class labels to integers
        self.dataset_loaded = False  # Indicates whether data is loaded
        self.random_data = False  # Indicates if data is random

    # Loads dataset from a CSV file and processes it
    def load_data(self, filepath):
        try:
            with open(filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                data = list(reader)[1:]  # Skip header
                random.shuffle(data)  # Shuffle for random training order

                # Separate features and labels, normalize features
                features = normalize_data([list(map(float, row[:-1])) for row in data])
                labels = [row[-1] for row in data]

                # Map unique labels to integers for one-hot encoding
                unique_labels = sorted(set(labels))
                self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
                numerical_labels = [self.label_mapping[label] for label in labels]

                self.x = features
                self.y = numerical_labels
                self.input_size = len(self.x[0])  # Set input size to feature count
                self.output_size = len(unique_labels)  # Output size equals class count
                self.dataset_loaded = True  # Mark dataset as loaded
        except Exception as e:
            print(f"Error loading data: {e}")
            self.dataset_loaded = False

    # Returns input features and labels
    def get_data(self):
        return self.x, self.y

# Represents a single neuron in the network
class Neuron:
    def __init__(self, layer_index, neuron_index, x, y):
        self.layer_index = layer_index  # Index of the layer this neuron belongs to
        self.neuron_index = neuron_index  # Index of the neuron in its layer
        self.x = x  # x-coordinate for visualization
        self.y = y  # y-coordinate for visualization
        self.output = 0.0  # Output of the neuron after activation
        self.delta = 0.0  # Delta used for backpropagation
        self.bias = random.uniform(-1, 1)  # Random initial bias

    # Draws the neuron as a circle on the canvas
    def draw(self, canvas, font_size=8):
        canvas.create_oval(self.x - NEURON_SCALE, self.y - NEURON_SCALE,
                           self.x + NEURON_SCALE, self.y + NEURON_SCALE,
                           fill='white', outline='black')
        canvas.create_text(self.x, self.y - NEURON_SCALE - 10,
                           text=f"{self.output:.2f}",
                           font=("Arial", font_size), fill='blue')

# Represents the connection between two neurons
class Axon:
    def __init__(self, from_neuron, to_neuron):
        self.from_neuron = from_neuron  # Neuron sending the signal
        self.to_neuron = to_neuron  # Neuron receiving the signal
        self.value = random.uniform(-1, 1)  # Random initial weight

    # Draws the axon as a line connecting neurons
    def draw(self, canvas, color='gray', font_size=8):
        canvas.create_line(self.from_neuron.x, self.from_neuron.y,
                           self.to_neuron.x, self.to_neuron.y, fill=color)
        mid_x = (self.from_neuron.x + self.to_neuron.x) / 2
        mid_y = (self.from_neuron.y + self.to_neuron.y) / 2
        canvas.create_text(mid_x, mid_y,
                           text=f"{self.value:.2f}",
                           font=("Arial", font_size), fill='red')

# Represents the neural network
class Network:
    def __init__(self, layers, ui):
        self.layers = layers  # List of layer sizes
        self.neurons = []  # List of neuron objects
        self.weights = []  # List of axon objects representing weights
        self.ui = ui  # Reference to the UI for visualization
        self.activation_function = sigmoid  # Default activation function
        self.activation_derivative = sigmoid_derivative  # Derivative of activation function
        self.create_network()

    # Sets the activation function and its derivative
    def set_activation_function(self, name):
        if name == 'Sigmoid':
            self.activation_function = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif name == 'Tanh':
            self.activation_function = tanh
            self.activation_derivative = tanh_derivative
        elif name == 'ReLU':
            self.activation_function = relu
            self.activation_derivative = relu_derivative

    # Initializes the neurons and axons of the network
    def create_network(self):
        canvas_width = self.ui.canvas.winfo_width()
        canvas_height = self.ui.canvas.winfo_height()
        num_layers = len(self.layers)
        layer_width = (canvas_width - 2 * SCREEN_PADDING) / max(1, num_layers - 1)

        # Create neurons for each layer
        for l_index, num_neurons in enumerate(self.layers):
            layer_neurons = []
            layer_height = (canvas_height - 2 * SCREEN_PADDING) / num_neurons
            for n_index in range(num_neurons):
                x = SCREEN_PADDING + l_index * layer_width
                y = SCREEN_PADDING + n_index * layer_height + layer_height / 2
                neuron = Neuron(l_index, n_index, x, y)
                layer_neurons.append(neuron)
            self.neurons.append(layer_neurons)

        # Create axons (connections) between layers
        for l in range(len(self.neurons) - 1):
            for from_neuron in self.neurons[l]:
                for to_neuron in self.neurons[l + 1]:
                    self.weights.append(Axon(from_neuron, to_neuron))

    # Performs forward propagation of inputs through the network
    def forward_prop(self, inputs):
        for i, value in enumerate(inputs):
            self.neurons[0][i].output = value  # Input layer outputs are the inputs

        for l in range(1, len(self.layers)):
            for neuron in self.neurons[l]:
                total_input = neuron.bias
                for prev_neuron in self.neurons[l - 1]:
                    for weight in self.weights:
                        if weight.from_neuron == prev_neuron and weight.to_neuron == neuron:
                            total_input += prev_neuron.output * weight.value
                neuron.output = self.activation_function(total_input)  # Apply activation function
        return [neuron.output for neuron in self.neurons[-1]]  # Return outputs of the last layer

    # Performs backpropagation to adjust weights and biases based on errors
    def back_prop(self, targets, learning_rate):
        # Calculate deltas for the output layer
        for i, neuron in enumerate(self.neurons[-1]):
            error = targets[i] - neuron.output  # Error is target - actual
            neuron.delta = error * self.activation_derivative(neuron.output)  # Compute delta

        # Calculate deltas for hidden layers
        for l in reversed(range(1, len(self.layers) - 1)):  # Skip input and output layers
            for neuron in self.neurons[l]:
                error = sum(weight.to_neuron.delta * weight.value
                            for weight in self.weights if weight.from_neuron == neuron)
                neuron.delta = error * self.activation_derivative(neuron.output)

        # Update weights based on deltas and learning rate
        for weight in self.weights:
            change = learning_rate * weight.from_neuron.output * weight.to_neuron.delta
            weight.value += change  # Adjust weight by the calculated change

        # Update biases for all neurons except the input layer
        for layer in self.neurons[1:]:  # Skip input layer
            for neuron in layer:
                neuron.bias += learning_rate * neuron.delta  # Adjust bias by delta

    # Train the network using a dataset for a specified number of epochs
    def train(self, training_data, epochs, learning_rate, update_queue):
        update_queue.put({'type': 'print', 'message': "\nTraining Started:"})
        for epoch in range(epochs):
            total_error = 0
            correct_predictions = 0

            # Iterate over each sample in the training data
            for inputs, targets in training_data:
                outputs = self.forward_prop(inputs)  # Forward pass
                error = sum((t - o) ** 2 for t, o in zip(targets, outputs)) / 2  # Mean squared error
                total_error += error
                self.back_prop(targets, learning_rate)  # Backpropagation

                # Calculate prediction accuracy for classification tasks
                predicted_label = outputs.index(max(outputs))  # Predicted class (highest output)
                actual_label = targets.index(max(targets))  # Actual class (highest target value)
                if predicted_label == actual_label:
                    correct_predictions += 1

            # Calculate average error and accuracy after each epoch
            avg_error = total_error / len(training_data)
            accuracy = (correct_predictions / len(training_data)) * 100
            message = f"Epoch {epoch + 1}/{epochs} - Error: {avg_error:.4f} - Accuracy: {accuracy:.2f}%"
            update_queue.put({'type': 'print', 'message': message})

            # Update the UI with metrics for this epoch
            update_queue.put({
                'type': 'update_metrics',
                'error': avg_error,
                'accuracy': accuracy,
                'epoch': epoch + 1,
                'total_epochs': epochs
            })

        update_queue.put({'type': 'print', 'message': "Training completed."})

        # Provide final weights and neuron outputs for visualization or logging
        update_queue.put({
            'type': 'training_complete',
            'weights': self.get_weights(),
            'neuron_outputs': self.get_neuron_outputs()
        })

    # Makes predictions for a given input using the trained network
    def predict(self, inputs):
        outputs = self.forward_prop(inputs)  # Perform forward propagation
        return outputs.index(max(outputs))  # Return the index of the highest output value

    # Retrieves the weights of the network organized by layers
    def get_weights(self):
        weights_per_layer = []
        for l in range(len(self.neurons) - 1):
            layer_weights = []
            for from_neuron in self.neurons[l]:
                weights_to_next_layer = []
                for to_neuron in self.neurons[l + 1]:
                    for weight in self.weights:
                        if weight.from_neuron == from_neuron and weight.to_neuron == to_neuron:
                            weights_to_next_layer.append(weight.value)
                            break  # Found the correct weight
                layer_weights.append(weights_to_next_layer)
            weights_per_layer.append(layer_weights)
        return weights_per_layer

    # Retrieves the outputs of all neurons organized by layers
    def get_neuron_outputs(self):
        outputs_per_layer = []
        for layer in self.neurons:
            outputs = [neuron.output for neuron in layer]
            outputs_per_layer.append(outputs)
        return outputs_per_layer


# User Interface class for the neural network
class UI(tk.Tk):
    def __init__(self):
        # Initialize the Tkinter window
        tk.Tk.__init__(self)

        # Window properties
        self.title("Real-Valued Neural Network")  # Title of the window
        self.state("zoomed")  # Start the window in fullscreen mode
        self.option_add("*tearOff", False)  # Disable menu tear-off feature

        # Create control panel (left side)
        self.create_control_panel()

        # Create canvas for the network visualization
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Additional properties
        self.dataset = Dataset()  # Dataset object to manage data loading
        self.network = None  # Neural network object
        self.training_thread = None  # Thread for training
        self.update_queue = queue.Queue()  # Queue for inter-thread communication

        # Training data
        self.epoch_data = []  # Data for training progress
        self.accuracy_sigmoid = []  # Accuracy for sigmoid function
        self.accuracy_tanh = []  # Accuracy for tanh function
        self.accuracy_relu = []  # Accuracy for ReLU function

    # Creates the control panel with various buttons and inputs
    def create_control_panel(self):
        control_frame = tk.Frame(self, bg="lightgray", width=200)  # Frame for control elements
        control_frame.pack(side="left", fill="y", padx=10, pady=10, anchor="nw")

        # Network configuration section
        tk.Label(control_frame, text="Network Configuration",
                 font=("Arial", 14, "bold"),
                 bg="lightgray").pack(pady=5)

        tk.Label(control_frame, text="Hidden Layers (e.g., 3,2,4):", bg="lightgray").pack(pady=5)
        self.layer_entry = tk.Entry(control_frame)  # Entry for layer configuration
        self.layer_entry.pack(pady=5)
        self.layer_entry.insert(0, "3,2,4")  # Default value

        tk.Button(control_frame, text="New Network", command=self.generate_network).pack(pady=5)
        tk.Button(control_frame, text="Reset Network", command=self.reset_network).pack(pady=10)

        # Performance section
        tk.Label(control_frame, text="Performance",
                 font=("Arial", 14, "bold"),
                 bg="lightgray").pack(pady=5)

        tk.Label(control_frame, text="Epochs:", bg="lightgray").pack(pady=1)
        self.epochs_entry = tk.Entry(control_frame)  # Entry for number of epochs
        self.epochs_entry.pack(pady=5)
        self.epochs_entry.insert(0, "10")

        tk.Label(control_frame, text="Learning Rate:", bg="lightgray").pack(pady=1)
        self.lr_entry = tk.Entry(control_frame)  # Entry for learning rate
        self.lr_entry.pack(pady=5)
        self.lr_entry.insert(0, "0.1")

        tk.Label(control_frame, text="Activation Function:", bg="lightgray").pack(pady=5)
        self.activation_var = tk.StringVar(value="Sigmoid")  # Dropdown for activation function
        tk.OptionMenu(control_frame, self.activation_var, "Sigmoid", "Tanh", "ReLU").pack(pady=5)

        # Buttons for training and testing
        tk.Button(control_frame, text="Train w/ Dataset", command=self.train_dataset).pack(pady=5)
        tk.Button(control_frame, text="Train w/ Random Data", command=self.train_random_data).pack(pady=5)
        tk.Button(control_frame, text="Test Network", command=self.test_network).pack(pady=5)

        # Output display
        tk.Label(control_frame, text="Test Accuracy", bg="lightgray").pack(pady=1)
        self.output_box = tk.Text(control_frame, height=2, width=8, wrap="word")  # Output box for accuracy
        self.output_box.pack(pady=1)
        self.output_box.config(state="disabled")  # Make it read-only

        # Frame for the accuracy graph
        self.graph_frame = tk.Frame(control_frame, bg="white", width=200, height=200)
        self.graph_frame.pack(side="bottom", fill="x", pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 4))  # Matplotlib figure for accuracy
        self.ax.set_title("Training Accuracy")
        self.ax.set_xlabel("Epochs")
        self.ax.grid(True)

        # Ensure x-axis values are integers
        self.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Initially hide the entire graph
        self.ax.set_visible(False)

        # Add separate lines for each activation function
        self.sigmoid_line, = self.ax.plot([], [], label="Sigmoid", color="blue")
        self.tanh_line, = self.ax.plot([], [], label="Tanh", color="green")
        self.relu_line, = self.ax.plot([], [], label="ReLU", color="red")

        self.ax.legend()
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)

    # Generates a network based on input size, output size, and hidden layer parameters
    def generate_network(self):
        self.load_data()  # Ensure data is loaded

        try:
            hidden_layers = [int(s) for s in self.layer_entry.get().split(',') if s.strip()]
            layers = [self.dataset.input_size] + hidden_layers + [self.dataset.output_size]

            if not hasattr(self.dataset, 'original_x'):
                self.dataset.original_x = self.dataset.x[:]  # Save original features
                self.dataset.original_y = self.dataset.y[:]  # Save original labels

            self.network = Network(layers, self)  # Create the network
            self.network.set_activation_function(self.activation_var.get())
            self.draw_network(self.network)  # Visualize the network

        except Exception as e:
            print(f"Error generating network: {e}")

    # Loads a new dataset from a CSV file
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.dataset.load_data(file_path)

    # Resets the current network and visualization
    def reset_network(self):
        self.canvas.delete("all")  # Clear the canvas
        self.network = None  # Reset network object
        self.accuracy_sigmoid.clear()  # Clear accuracy data
        self.accuracy_tanh.clear()
        self.accuracy_relu.clear()
        self.sigmoid_line.set_data([], [])
        self.tanh_line.set_data([], [])
        self.relu_line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas_plot.draw()

    # Trains the network with the loaded dataset
    def train_dataset(self):
        self.load_data()

        if not self.dataset.dataset_loaded:
            print("No dataset loaded. Please load a valid dataset first.")
            return

        if not self.network:
            try:
                hidden_layers = [int(s) for s in self.layer_entry.get().split(',') if s.strip()]
                layers = [self.dataset.input_size] + hidden_layers + [self.dataset.output_size]
                self.network = Network(layers, self)
            except Exception as e:
                print(f"Error generating network: {e}")
                return

        try:
            dataset_size = len(self.dataset.x)
            training_size = int(0.2 * dataset_size)  # Use 20% of data for training

            x, y = self.dataset.get_data()
            num_classes = self.dataset.output_size
            y_one_hot = [[1 if i == label else 0 for i in range(num_classes)] for label in y]

            self.training_data = list(zip(x[:training_size], y_one_hot[:training_size]))
            self.testing_data = list(zip(x[training_size:], y[training_size:]))

            learning_rate = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())

            self.network.set_activation_function(self.activation_var.get())
            self.training_thread = threading.Thread(
                target=self.run_training_thread,
                args=(epochs, learning_rate)
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            self.after(100, self.run_nn)  # noqa: specific-warning-code

        except Exception as e:
            print(f"Error during training with dataset: {e}")

    # Runs a training thread for random data
    def train_random_data(self):
        if not self.network:
            print("Please generate a network first.")
            return

        if not self.dataset.dataset_loaded:
            print("No dataset loaded. Please load a dataset first.")
            return

        try:
            dataset_size = len(self.dataset.x)
            training_size = int(0.2 * dataset_size)  # 20% for training
            num_features = self.dataset.input_size
            num_classes = self.dataset.output_size

            features = [[random.uniform(0, 1) for _ in range(num_features)] for _ in range(training_size)]
            labels = [random.randint(0, num_classes - 1) for _ in range(training_size)]
            one_hot_labels = [[1 if i == label else 0 for i in range(num_classes)] for label in labels]
            features = normalize_data(features)

            self.training_data = list(zip(features, one_hot_labels))
            self.testing_data = []

            learning_rate = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())

            self.network.set_activation_function(self.activation_var.get())
            self.training_thread = threading.Thread(
                target=self.run_training_thread,
                args=(epochs, learning_rate)
            )
            self.training_thread.daemon = True
            self.training_thread.start()
            self.after(100, self.run_nn)  # noqa: specific-warning-code

        except Exception as e:
            print(f"Error during training with random data: {e}")

    # Tests the network using a portion of the dataset
    def test_network(self):
        if not self.network:
            print("Please generate and train the network first.")
            return

        if not hasattr(self.dataset, 'original_x') or not hasattr(self.dataset, 'original_y'):
            print("No original dataset available for testing. Please generate the network first.")
            return

        x, y = self.dataset.original_x, self.dataset.original_y  # noqa: specific-warning-code

        if isinstance(y[0], list):
            y = [label.index(1) for label in y]

        split_index = int(0.8 * len(x))
        testing_data = list(zip(x[split_index:], y[split_index:]))

        if not testing_data:
            print("No testing data available.")
            return

        correct_predictions = 0
        total_predictions = len(testing_data)

        for inputs, label in testing_data:
            prediction = self.network.predict(inputs)
            if prediction == label:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        print(f"\nTesting Results:")
        print(f"Testing completed. Accuracy: {accuracy * 100:.2f}% ({correct_predictions}/{total_predictions})")
        self.update_output_box(f"{accuracy * 100:.2f}%")

    # Draws the neural network on the canvas
    def draw_network(self, network):
        self.canvas.delete("all")
        for weight in network.weights:
            weight.draw(self.canvas, font_size=8)
        for layer in network.neurons:
            for neuron in layer:
                neuron.draw(self.canvas, font_size=10)

    # Updates the training accuracy graph
    def update_graph(self, accuracy):
        activation_function = self.activation_var.get()

        if activation_function == "Sigmoid":
            self.accuracy_sigmoid.append(accuracy)
            self.sigmoid_line.set_data(range(len(self.accuracy_sigmoid)), self.accuracy_sigmoid)
        elif activation_function == "Tanh":
            self.accuracy_tanh.append(accuracy)
            self.tanh_line.set_data(range(len(self.accuracy_tanh)), self.accuracy_tanh)
        elif activation_function == "ReLU":
            self.accuracy_relu.append(accuracy)
            self.relu_line.set_data(range(len(self.accuracy_relu)), self.accuracy_relu)

        if not self.ax.get_visible():
            self.ax.set_visible(True)
            self.ax.set_xlabel("Epochs")
            self.ax.set_ylabel("Accuracy (%)")

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas_plot.draw()

    # Updates the output box with a given message
    def update_output_box(self, message):
        self.output_box.config(state="normal")
        self.output_box.insert("end", message + "\n")
        self.output_box.see("end")
        self.output_box.config(state="disabled")

    # Runs the training thread and updates the UI periodically
    def run_training_thread(self, epochs, learning_rate):
        self.network.train(self.training_data, epochs, learning_rate, self.update_queue)

    def run_nn(self):
        try:
            while True:
                task = self.update_queue.get_nowait()
                if task['type'] == 'update_network':
                    self.draw_network(task['network'])
                elif task['type'] == 'print':
                    print(task['message'])
                elif task['type'] == 'update_metrics':
                    accuracy = task['accuracy']
                    self.update_graph(accuracy)
                elif task['type'] == 'training_complete':
                    weights = task['weights']
                    neuron_outputs = task['neuron_outputs']
                    print("\nFinal Weights per Layer:")
                    for l, layer_weights in enumerate(weights):
                        print(f"\nLayer {l} to Layer {l+1} Weights:")
                        for from_neuron_idx, weights_to_next_layer in enumerate(layer_weights):
                            print(f"  Neuron {from_neuron_idx} weights: {weights_to_next_layer}")
                    print("\nFinal Neuron Outputs per Layer:")
                    for l, outputs in enumerate(neuron_outputs):
                        print(f"Layer {l} Outputs: {outputs}")
        except queue.Empty:
            pass
        if self.training_thread and self.training_thread.is_alive():
            self.after(100, self.run_nn)  # noqa: specific-warning-code
        else:
            self.after(1000, self.run_nn)  # noqa: specific-warning-code

# Entry point for the application
if __name__ == '__main__':
    ui = UI()
    ui.mainloop()
