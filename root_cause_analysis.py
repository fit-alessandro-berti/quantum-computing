# pip install qiskit-terra==0.24.1
# pip install qiskit-aer==0.12.0
# pip install qiskit-optimization==0.5.0
# pip install qiskit==0.41.0
# pip install qiskit-optimization==0.5.0
# pip install qiskit-machine-learning==0.6.1
# pip install torch==2.0.1
# pip install numpy==1.24.3
# pip install scikit-learn==1.2.2

# Below is a conceptual, end-to-end example of applying quantum computing to root-cause analysis
# in process mining, using synthetic data and simplified assumptions. In practice, one would use
# domain-specific logic, more advanced feature engineering, and sophisticated interpretability methods.
#
# Scenario:
# We have an event log with traces represented by numerical features (e.g., activity frequencies,
# waiting times, resource utilization). Some traces are normal, others are deviant (e.g., violating
# SLAs or involving prohibited actions).
#
# Goal:
#  - Use a quantum kernel (via Qiskit) to measure similarity between traces.
#  - Train a quantum-kernel-based SVM classifier (QSVC) to distinguish normal vs. deviant traces.
#  - Explore which features most strongly influence the classifier, hinting at root causes.
#
# While QSVC doesn't provide direct feature importance, we can:
#  - Observe changes in classification performance when removing or altering certain features.
#  - Compare kernel distance patterns for normal and deviant classes.
#  - Test subsets of features to see which best separate classes.
#
# This helps form root-cause hypotheses: if including a certain feature (e.g., frequency of a
# specific activity) significantly improves class separation, that feature may be key to the
# deviation's cause.
#
# Steps:
# 1. Data Generation: Simulate traces with four features (two activity frequencies, waiting time,
#    resource utilization), making deviant traces differ notably in some attributes.
# 2. Quantum Kernel Setup: Use Qiskit feature maps and quantum kernels to compute trace similarities.
# 3. Classification (QSVC): Train a QSVC on the training set to classify normal vs. deviant traces.
# 4. Root-Cause Clue Examination: Check how performance changes without certain features, and
#    inspect kernel distances. Improved separation with specific features suggests potential root causes.


import numpy as np
from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set a random seed for reproducibility
algorithm_globals.random_seed = 42

#########################################
# Step 1: Synthetic Data Generation
#########################################
# Let's say we have:
# Feature 0: Frequency of activity A
# Feature 1: Frequency of activity B
# Feature 2: Average waiting time between certain events
# Feature 3: Resource utilization score

# Generate normal traces (class = 0)
normal_traces = np.zeros((50, 4))
normal_traces[:, 0] = np.random.normal(loc=5, scale=1.0, size=50)    # Activity A ~ N(5,1)
normal_traces[:, 1] = np.random.normal(loc=5, scale=1.0, size=50)    # Activity B ~ N(5,1)
normal_traces[:, 2] = np.random.normal(loc=30, scale=5, size=50)     # Waiting time ~ N(30,5)
normal_traces[:, 3] = np.random.normal(loc=0.5, scale=0.1, size=50)  # Utilization ~ N(0.5,0.1)

# Generate deviant traces (class = 1)
# Deviant traces might have abnormally high frequency of Activity B and higher waiting times
deviant_traces = np.zeros((25, 4))
deviant_traces[:, 0] = np.random.normal(loc=5, scale=1.0, size=25)   # Activity A ~ similar
deviant_traces[:, 1] = np.random.normal(loc=9, scale=1.0, size=25)   # Activity B higher ~ N(9,1)
deviant_traces[:, 2] = np.random.normal(loc=45, scale=5, size=25)    # Waiting time higher ~ N(45,5)
deviant_traces[:, 3] = np.random.normal(loc=0.5, scale=0.1, size=25) # Utilization ~ similar

X = np.vstack([normal_traces, deviant_traces])
y = np.array([0]*50 + [1]*25)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#########################################
# Step 2: Define a Quantum Feature Map and Kernel
#########################################
# Use a ZZFeatureMap which can capture pairwise feature interactions
feature_map = ZZFeatureMap(feature_dimension=X.shape[1], reps=2, entanglement='full')

backend = Aer.get_backend('statevector_simulator')
qi = QuantumInstance(backend=backend)

quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=qi)

#########################################
# Step 3: Train QSVC with the Quantum Kernel
#########################################
# QSVC from Qiskit Machine Learning can use a QuantumKernel.
qsvc = QSVC(quantum_kernel=quantum_kernel)

# Fit the QSVC on the training data
qsvc.fit(X_train, y_train)

# Predict on the test set
y_pred = qsvc.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

#########################################
# Step 4: Investigating Root-Cause Clues
#########################################

# One approach: Check accuracy when removing certain features:
# This is a heuristic approach to see which features are crucial.

def do_test_feature_importance(features_to_keep):
    # Retrain and test the model using only a subset of features
    X_train_sub = X_train[:, features_to_keep]
    X_test_sub = X_test[:, features_to_keep]
    # Recreate feature map with updated dimension
    sub_feature_map = ZZFeatureMap(feature_dimension=len(features_to_keep), reps=2, entanglement='full')
    sub_quantum_kernel = QuantumKernel(feature_map=sub_feature_map, quantum_instance=qi)
    sub_qsvc = QSVC(quantum_kernel=sub_quantum_kernel)
    sub_qsvc.fit(X_train_sub, y_train)
    y_pred_sub = sub_qsvc.predict(X_test_sub)
    return y_pred_sub

all_features = [0,1,2,3]
accuracy_full = np.mean(y_pred == y_test)

print(f"Accuracy with all features: {accuracy_full:.2f}")

for f in all_features:
    keep = [ff for ff in all_features if ff != f]
    y_pred_sub = do_test_feature_importance(keep)
    accuracy_sub = np.mean(y_pred_sub == y_test)
    print(f"Accuracy without feature {f}: {accuracy_sub:.2f}")

# Interpretation:
# If removing a certain feature drastically reduces accuracy, that feature likely plays a key role
# in distinguishing normal and deviant traces. This can help form hypotheses about root causes.

#########################################
# Additional Analysis (Kernel Distance Patterns)
#########################################
# Another way is to inspect how the quantum kernel separates classes.
# Let's compute the kernel matrix for test data vs. train data and look at distances.

kernel_train = quantum_kernel.evaluate(X_train)
kernel_test = quantum_kernel.evaluate(X_test, X_train)

# For a given test sample, see if it's closer (in kernel similarity) to normal or deviant training samples.
test_idx_example = 0
test_sample_kernel_row = kernel_test[test_idx_example]
normal_indices_train = np.where(y_train == 0)[0]
deviant_indices_train = np.where(y_train == 1)[0]

avg_sim_normal = np.mean(test_sample_kernel_row[normal_indices_train])
avg_sim_deviant = np.mean(test_sample_kernel_row[deviant_indices_train])

print(f"Test sample {test_idx_example} average similarity to normal class: {avg_sim_normal:.3f}")
print(f"Test sample {test_idx_example} average similarity to deviant class: {avg_sim_deviant:.3f}")

# If deviant samples have higher similarity, it suggests the key features that define the quantum feature map
# (and thus the kernel) are capturing the patterns in deviant data.

# This kind of analysis can guide further root-cause investigation:
# For example, if we find that removing feature 1 (Activity B frequency) drastically decreases
# classification accuracy, we might hypothesize that higher occurrences of Activity B is a key root cause
# for the deviations. Similarly, if waiting time is crucial, it might indicate that delays cause the deviations.

#########################################
# Conclusion:
#########################################
# In this demonstration:
# - We used a quantum kernel to classify normal vs. deviant traces.
# - By experimenting with feature subsets, we can gauge which features are most important.
# - This knowledge can be combined with domain expertise to form hypotheses about root causes of deviant behavior.
