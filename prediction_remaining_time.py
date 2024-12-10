# pip install qiskit-terra==0.24.1
# pip install qiskit-aer==0.12.0
# pip install qiskit-optimization==0.5.0
# pip install qiskit==0.41.0
# pip install qiskit-optimization==0.5.0
# pip install qiskit-machine-learning==0.6.1
# pip install torch==2.0.1
# pip install numpy==1.24.3
# pip install scikit-learn==1.2.2

# Below is a fully worked-out example demonstrating how one might approach the prediction of remaining
# time in a process mining scenario using quantum neural networks (QNN) with Qiskit. This is a conceptual
# demonstration rather than a guaranteed best-practice solution. The goal is to show how Qiskit’s QNN framework
# can be integrated with PyTorch for a regression task.
#
# What This Example Does:
#
# Synthetic Data Generation:
# We simulate process traces composed of events (a_i, d_i), where a_i is an activity (e.g., A, B, C, D), and d_i
# is the duration of that event. Each trace ends at a certain total time. For each partial prefix of a trace,
# the "remaining time" is the total time of the future events not yet executed.
#
# Feature Encoding:
# We encode each prefix into a numeric feature vector. A simple encoding is to take the frequency of each
# activity in the prefix and normalize it. We also add a time-related feature, such as the elapsed time so far
# or average event duration. This is a simplistic encoding for illustration.
#
# Quantum Neural Network Setup:
# We use TwoLayerQNN from Qiskit Machine Learning, which creates a simple feature map + ansatz structure.
# We integrate this QNN with PyTorch using the TorchConnector. This allows us to define a loss (MSE) and train
# the parameters of the quantum circuit to predict a continuous value (remaining time).
#
# Training:
# We run a training loop in PyTorch to fit the QNN to the training data, adjusting the circuit parameters
# to minimize mean squared error.

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes

##################################################
# Step 1: Generate Synthetic Process Mining Data
##################################################
# Let's assume we have 4 possible activities: A, B, C, D
activities = ['A', 'B', 'C', 'D']
num_activities = len(activities)
np.random.seed(42)

num_traces = 10
max_trace_length = 10
min_trace_length = 5

# Generate random traces, each event has a random duration between 1 and 10 time units
traces = []
for _ in range(num_traces):
    length = np.random.randint(min_trace_length, max_trace_length + 1)
    # Randomly select activities
    trace_activities = np.random.choice(activities, size=length)
    # Random durations
    durations = np.random.randint(1, 11, size=length)
    traces.append(list(zip(trace_activities, durations)))

# For each prefix, we want to predict the sum of the durations of the remaining events.
prefixes = []
remaining_times = []
for trace in traces:
    total_time = sum(d for _, d in trace)
    for i in range(len(trace) - 1):
        prefix = trace[:i + 1]
        completed_time = sum(d for _, d in prefix)
        future_time = total_time - completed_time
        prefixes.append(prefix)
        remaining_times.append(float(future_time))

prefixes = np.array(prefixes, dtype=object)
remaining_times = np.array(remaining_times, dtype=float)

##################################################
# Step 2: Feature Encoding
##################################################
# We'll use a simple encoding: sum frequencies of each activity in the prefix.
# Also include a scaled "elapsed time" feature.

encoder = OneHotEncoder(categories=[activities], sparse_output=False)
encoder.fit(np.array(activities).reshape(-1, 1))


def encode_prefix(prefix):
    # prefix is a list of (activity, duration)
    acts = [p[0] for p in prefix]
    durations = [p[1] for p in prefix]

    prefix_array = np.array(acts).reshape(-1, 1)
    one_hots = encoder.transform(prefix_array)
    freq_vector = one_hots.sum(axis=0)  # frequency of each activity

    # Normalize frequency vector
    norm = np.linalg.norm(freq_vector)
    if norm != 0:
        freq_vector = freq_vector / norm

    # Add a time-related feature: normalized elapsed time
    elapsed_time = sum(durations)
    # For normalization, consider max possible elapsed time ~ max_trace_length * 10 (worst case)
    max_possible_time = max_trace_length * 10
    time_feature = elapsed_time / max_possible_time

    # Final feature vector: activity frequencies + time feature
    feature_vector = np.concatenate([freq_vector, [time_feature]])
    return feature_vector


X = np.array([encode_prefix(p) for p in prefixes])
y = remaining_times

print("aaaaaaa")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

##################################################
# Step 3: Set Up the QNN (TwoLayerQNN)
##################################################
# Our feature dimension:
feature_dim = X_train.shape[1]  # number_of_activities + 1 time feature

# Create a QuantumInstance for simulation
backend = BasicAer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend=backend, shots=1, seed_simulator=42, seed_transpiler=42)

# Define a feature map and ansatz
feature_map = ZFeatureMap(feature_dimension=feature_dim, reps=1)
ansatz = RealAmplitudes(num_qubits=feature_dim, reps=1)

# TwoLayerQNN combines a feature map, ansatz, and measurement.
# For regression, the QNN outputs an expectation value (1D).
qnn = TwoLayerQNN(
    feature_map=feature_map,
    ansatz=ansatz,
    input_gradients=True,  # allows backpropagation
    quantum_instance=quantum_instance
)

# TorchConnector integrates QNN with PyTorch
model = TorchConnector(qnn)

##################################################
# Step 4: Training the QNN with PyTorch
##################################################
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.02)

# Simple training loop
epochs = 50
for epoch in range(epochs):
    print(epoch)
    optimizer.zero_grad()
    predictions = model(X_train_torch)  # Forward pass
    loss = criterion(predictions, y_train_torch)  # MSE Loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Training Loss = {loss.item()}")
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Training Loss = {loss.item()}")

##################################################
# Step 5: Evaluation
##################################################
model.eval()
with torch.no_grad():
    test_preds = model(X_test_torch)
    test_loss = criterion(test_preds, y_test_torch).item()

print("Test MSE:", test_loss)

# You can also look at some predictions vs actual values:
for i in range(5):
    print(f"Prefix {i}, Actual Remaining Time: {y_test[i]}, Predicted: {test_preds[i].item()}")
