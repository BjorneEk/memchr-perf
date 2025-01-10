import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import sys
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def read_file(filename):
	with open(filename, 'r') as file:
		lines = file.readlines()

	# First line contains N and M
	N, M = map(int, lines[0].strip().split())

	names = lines[1].strip().split()

	# Third line contains the data
	data = lines[2].split(',')
	matrix = [[] for _ in range(M)]

	for row in data:
		row = row.strip()
		if row:
			values = list(map(float, row.split()))
			for i in range(M):
				matrix[i].append(values[i])
	return N, M, names, matrix
def human_readable_bytes(value):
	"""
	Convert a byte value into a human-readable string with appropriate prefixes.
	"""
	prefixes = [' ', 'K', 'M', 'G', 'T', 'P', 'E']
	index = 0
	while value >= 1024 and index < len(prefixes) - 1:
		value /= 1024.0
		index += 1
	return f"{value:.1f}{prefixes[index]}"

def human_readable_time(value):
	"""
	Convert a time value into a human-readable string with appropriate prefixes.
	"""
	prefixes = ['', 'm', 'μ', 'n', 'p', 'f', 'a']  # Time prefixes
	index = 0
	while value < 1 and index < len(prefixes) - 1:
		value *= 1000.0
		index += 1
	return f"{value:.1f}{prefixes[index]}s"

def plot_data(N, M, names, matrix, filename):
	x = range(1, N + 1)  # X-axis: 1 to N

	plt.figure(figsize=(10, 6))

	# Plot each line
	for i in range(M):
		plt.plot(x, matrix[i], label=names[i], linewidth=1)

	plt.xlabel('Bytes')
	plt.ylabel('Time')
	plt.title('Comparison of memcmp implementations')
	plt.grid(True)

	plt.xlim(left=0)  # Minimum x-axis value is 0
	plt.ylim(bottom=0)  # Minimum y-axis value is 0

	# Customize x-axis labels to human-readable byte prefixes
	xticks = plt.xticks()[0]  # Get current x-ticks
	xtick_labels = [human_readable_bytes(int(tick)) for tick in xticks]
	plt.xticks(xticks, xtick_labels)
	# Customize y-axis labels to human-readable time prefixes
	yticks = plt.yticks()[0]  # Get current y-ticks
	ytick_labels = [human_readable_time(tick) for tick in yticks]
	plt.yticks(yticks, ytick_labels)

	plt.legend()

	plt.tight_layout()
	plt.savefig(filename, format="png", bbox_inches="tight")
	print(f"Plot saved as: {filename}")
	plt.show()

def plot_data_log(N, M, names, matrix, filename):
	x = range(1, N + 1)  # X-axis: 1 to N

	plt.figure(figsize=(10, 6))

	for i in range(M):
		plt.plot(x, matrix[i], label=names[i], linewidth=1)

	plt.xlabel('Bytes')
	plt.ylabel('Time')
	plt.title('Comparison of memchr implementations')
	plt.grid(True)

	#plt.xscale('log')
	plt.yscale('log')

	#plt.xlim(left=1)  # Minimum x-axis value is 1 (log scale cannot include 0)
	#plt.ylim(bottom=1e-9)  # Set a reasonable lower bound for time
	plt.xlim(left=0)  # Minimum x-axis value is 0
	mins = [min(ds) for ds in matrix]
	plt.ylim(bottom=min(mins))  # Minimum y-axis value is 0
	xticks = plt.xticks()[0]  # Get current x-ticks
	xtick_labels = [human_readable_bytes(int(tick)) for tick in xticks]
	plt.xticks(xticks, xtick_labels)

	yticks = plt.yticks()[0]  # Get current y-ticks
	ytick_labels = [human_readable_time(tick) for tick in yticks]
	plt.yticks(yticks, ytick_labels)

	plt.legend()

	plt.tight_layout()
	plt.savefig(filename, format="png", bbox_inches="tight")
	print(f"Plot saved as: {filename}")
	plt.show()

#def plot_data(N, M, names, matrix):
#	x = range(1, N + 1)  # X-axis: 1 to N
#
#	plt.figure(figsize=(10, 6))
#
#	# Plot each line
#	for i in range(M):
#		plt.plot(x, matrix[i], label=names[i], linewidth=1)
#
#	plt.xlabel('N')
#	plt.ylabel('Time')
#	plt.title('Comparison of functions')
#	plt.grid(True)
#
#	plt.legend()
#
#	plt.tight_layout()
#	plt.show()

def remove_outliers(matrix, iqr_multiplier=1.0):
	cleaned_matrix = []
	for dataset in matrix:  # Process each sequence independently
		dataset = np.array(dataset)
		q1 = np.percentile(dataset, 25)  # 1st quartile
		q3 = np.percentile(dataset, 75)  # 3rd quartile
		iqr = q3 - q1                     # Interquartile range
		lower_bound = q1 - iqr_multiplier * iqr
		upper_bound = q3 + iqr_multiplier * iqr

		cleaned_dataset = dataset.copy()
		for i in range(len(dataset)):
			if dataset[i] < lower_bound or dataset[i] > upper_bound:
				# Replace outlier with interpolated value
				if 0 < i < len(dataset) - 1:
					cleaned_dataset[i] = (dataset[i - 1] + dataset[i + 1]) / 2
				elif i == 0:
					cleaned_dataset[i] = dataset[i + 1]  # First element uses next value
				elif i == len(dataset) - 1:
					cleaned_dataset[i] = dataset[i - 1]  # Last element uses previous value

		cleaned_matrix.append(cleaned_dataset.tolist())
	return cleaned_matrix

def lowpass(matrix, window_size=3):
	filtered_matrix = []
	for dataset in matrix:  # Process each sequence independently
		dataset = np.array(dataset)
		filtered_data = dataset.copy()
		for i in range(len(dataset)):
			# Define the window range
			start = max(0, i - window_size // 2)
			end = min(len(dataset), i + window_size // 2 + 1)
			filtered_data[i] = sum(dataset[start:end]) / (end - start)
		filtered_matrix.append(filtered_data.tolist())
	return filtered_matrix

def accelerated_filter(matrix, window_size=3):
	# Convert input matrix to a TensorFlow tensor
	matrix_tensor = tf.convert_to_tensor(matrix, dtype=tf.float32)

	# Create the moving average kernel
	kernel = tf.ones([window_size], dtype=tf.float32) / window_size

	# Reshape kernel for 1D convolution
	kernel = kernel[:, tf.newaxis, tf.newaxis]

	# Add batch and channel dimensions to matrix_tensor for 1D convolution
	matrix_tensor = matrix_tensor[:, :, tf.newaxis]

	# Apply 1D convolution
	filtered_tensor = tf.nn.conv1d(matrix_tensor, kernel, stride=1, padding="SAME")

	# Remove channel dimension and convert back to NumPy
	filtered_matrix = tf.squeeze(filtered_tensor).numpy()

	return filtered_matrix
def gaussian_kernel(size, sigma=1.0):
	x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
	g = tf.exp(-0.5 * (x / sigma) ** 2)
	return g / tf.reduce_sum(g)

def gaussian_filter(matrix, window_size=3, sigma=1.0):
	# Convert input matrix to a TensorFlow tensor
	matrix_tensor = tf.convert_to_tensor(matrix, dtype=tf.float32)

	# Create the Gaussian kernel
	kernel = gaussian_kernel(window_size, sigma)

	# Reshape kernel for 1D convolution
	kernel = kernel[:, tf.newaxis, tf.newaxis]

	# Add batch and channel dimensions to matrix_tensor for 1D convolution
	matrix_tensor = matrix_tensor[:, :, tf.newaxis]

	# Apply 1D convolution
	filtered_tensor = tf.nn.conv1d(matrix_tensor, kernel, stride=1, padding="SAME")

	# Convert back to NumPy
	filtered_matrix = tf.squeeze(filtered_tensor).numpy()

	return filtered_matrix

def score_list(matrix):
	#times = []
	#for dataset in matrix:
	#	dataset = np.array(dataset)
	#	times.append(np.sum(dataset).tolist())
	#return times
	times = [np.sum(dataset) for dataset in matrix]
	print("times: ", times)
	small = min(times)
	#print("small: ", small)
	scores = [x - small for x in times]
	scores = [ (x * x) * 100 for x in scores]
	print("scores: ", scores)
	return scores

def main():
	if len(sys.argv) != 3:
		print("Usage: python plot.py <input> <output>")
		sys.exit(1)
	filename = sys.argv[1]  # Replace with your filename
	output_file = sys.argv[2]
	N, M, names, matrix = read_file(filename)
	scores = score_list(matrix)
	names = [f"{s}(+{v:.1f})" for s, v in zip(names, scores)]
	#matrix = remove_outliers(matrix, 1.5)
	#print("TensorFlow version:", tf.__version__)
	# Check if GPU (MPS) is available
	#print("Is GPU available (MPS):", tf.config.list_physical_devices('GPU'))
	matrix = accelerated_filter(matrix, 1000)
	#matrix = gaussian_filter(matrix, 500, 50)
	matrix = remove_outliers(matrix, 1.5)

	plot_data_log(N, M, names, matrix, output_file)

if __name__ == "__main__":
	main()