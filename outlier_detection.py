# pip install qiskit-terra==0.24.1
# pip install qiskit-aer==0.12.0
# pip install qiskit-optimization==0.5.0
# pip install qiskit==0.41.0
# pip install qiskit-optimization==0.5.0
# pip install qiskit-machine-learning==0.6.1
# pip install torch==2.0.1
# pip install numpy==1.24.3
# pip install scikit-learn==1.2.2

# Below is a step-by-step example of how you might use quantum computing techniques (via Qiskit)
# to perform outlier detection on a small synthetic dataset. The idea is to leverage a quantum kernel
# as a similarity measure and then use a classical outlier detection algorithm (like a One-Class SVM)
# that operates on this quantum-derived kernel matrix. This approach combines the power of quantum
# kernel methods from Qiskit with a classical anomaly detection framework.
#
# Key Concepts
#
# Event Log and Feature Representation:
# In process mining, event logs are collections of events (with timestamps, activity names, etc.)
# recorded during process execution. To apply machine learning or quantum methods, you first need
# to represent each trace (sequence of events) as a numerical vector. For demonstration, we’ll use
# a simple synthetic dataset. In a real scenario, you might:
#
#  - Encode each trace as a frequency vector of activities.
#  - Encode trace patterns using embeddings (e.g., vectorizing sequences or using more complex
#    encoding schemes).
#
# Quantum Kernel:
# A quantum kernel maps classical data into a Hilbert space via a quantum feature map and measures
# similarities between data points using a quantum circuit. This can provide a non-trivial similarity
# measure that may capture complex relationships better than classical kernels for certain datasets.
#
# Outlier Detection with One-Class SVM:
# The One-Class SVM is a classical technique that tries to find a boundary around the bulk of the
# data points (considered “normal”). Any point falling outside that boundary is considered an outlier.
# Here, we will supply the kernel matrix computed by our quantum kernel to the One-Class SVM.


import numpy as np
from sklearn.svm import OneClassSVM
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import algorithm_globals

# Set a seed for reproducibility
algorithm_globals.random_seed = 42

#########################################
# Step 1: Create a Synthetic Dataset
#########################################
# Imagine we have numerical features for traces. For simplicity,
# let's say we have 2D vectors representing simplified process features.

# "Normal" data cluster
normal_data = np.random.randn(20, 2) * 0.3 + np.array([0.5, 0.5])
# A few outliers far away from the normal cluster
outliers = np.array([[2.0, 2.0], [2.5, 2.5], [-1.5, 2.0]])

# Combine them to form our dataset
X = np.vstack([normal_data, outliers])

# In outlier detection, we don't have labels for training in a supervised manner.
# The One-Class SVM is trained only on the "normal" data,
# and then we test how well it scores the entire dataset.
X_train = normal_data  # Train only on the normal cluster
X_test = X             # Test on all data (normal + outliers)

#########################################
# Step 2: Define a Quantum Feature Map
#########################################
# A quantum kernel relies on a feature map. Here we use a simple ZZFeatureMap.
# This feature map encodes 2D data into quantum states. Adjust 'reps' as needed.

feature_map = ZZFeatureMap(feature_dimension=2, reps=2, entanglement='full')

#########################################
# Step 3: Set up a Quantum Instance
#########################################
backend = Aer.get_backend('statevector_simulator')
qi = QuantumInstance(backend=backend)

#########################################
# Step 4: Create a Quantum Kernel
#########################################
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=qi)

# Compute the kernel matrix for training data
kernel_matrix_train = quantum_kernel.evaluate(X_train, X_train)

#########################################
# Step 5: Train a One-Class SVM on the Quantum Kernel
#########################################
# We will pass precomputed kernels to the One-Class SVM.
# Note: One-Class SVM expects a kernel="precomputed" if we provide a matrix directly.
oc_svm = OneClassSVM(kernel="precomputed", gamma='auto', nu=0.1)
oc_svm.fit(kernel_matrix_train)

#########################################
# Step 6: Evaluate on Test Data
#########################################
# Compute kernel matrix between test and train data
kernel_matrix_test = quantum_kernel.evaluate(X_test, X_train)

# Predict: +1 = inlier, -1 = outlier
predictions = oc_svm.predict(kernel_matrix_test)

#########################################
# Step 7: Analyze Results
#########################################
normal_predictions = predictions[:len(normal_data)]
outlier_predictions = predictions[len(normal_data):]

print("Normal data predictions:", normal_predictions)
print("Outlier data predictions:", outlier_predictions)

num_outliers_correct = np.sum(outlier_predictions == -1)
print(f"Detected {num_outliers_correct} out of {len(outliers)} true outliers correctly.")
