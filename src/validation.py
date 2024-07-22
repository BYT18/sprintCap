import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.stats import pearsonr
import math


# Detect spike ranges
def detect_spike_ranges(data, threshold):
    mean = np.mean(data)
    std = np.std(data)
    spike_indices = np.where(np.abs(data - mean) > threshold * std)[0]
    return spike_indices

#
def downsample_with_pandas(data, target_length):
    df = pd.DataFrame(data)
    df_downsampled = df.iloc[::(len(df) // target_length), :]
    return df_downsampled.values.flatten()

# Used for downsampling
def interpolate_to_target_length(data, target_length):
    original_length = len(data)
    x = np.linspace(0, 1, original_length)
    f = interp1d(x, data, kind='linear')
    x_new = np.linspace(0, 1, target_length)
    return f(x_new)

# Function to calculate angles between three specified markers
def calculate_angles3(df, marker_triples):
    angles = {}
    for marker1, marker2, marker3 in marker_triples:
        vec1_start = df[[f"{marker1}_X", f"{marker1}_Y", f"{marker1}_Z"]].values
        vec1_end = df[[f"{marker2}_X", f"{marker2}_Y", f"{marker2}_Z"]].values
        vec2_end = df[[f"{marker3}_X", f"{marker3}_Y", f"{marker3}_Z"]].values

        # Compute vectors from marker1 to marker2 and marker2 to marker3
        vector1 = vec1_end - vec1_start
        vector2 = vec2_end - vec1_end

        # Compute the angle between these vectors
        dot_product = np.einsum('ij,ij->i', vector1, vector2)
        norms = np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2, axis=1)
        angles[f"{marker1}-{marker2}-{marker3}"] = np.arccos(dot_product / norms) * (180.0 / np.pi)

    return angles

# Plotting function
def plot_angles(angles):
    num_angles = len(angles)
    fig, axes = plt.subplots(num_angles, 1, figsize=(12, 3 * num_angles))

    for ax, (key, angle) in zip(axes, angles.items()):
        ax.plot(angle, label=key)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f'Angle: {key}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def compute_cross_correlation(x, y):
    """Compute the cross-correlation of two signals."""
    x = np.asarray(x)
    y = np.asarray(y)
    correlation = correlate(x, y, mode='full')
    lag = np.arange(-len(x) + 1, len(y))
    #correlation, lag = pearsonr(x,y)
    return correlation, lag
    #nsig1 = x - np.mean(x)  # Demean
    #nsig2 = y - np.mean(y)  # Demean
    #nsig1 = x
    #nsig2 = y

    #corr = np.correlate(nsig1,nsig2,'full')
    #corr = sm.tsa.stattools.ccf(y, x, adjusted=False)
    #return corr, range(len(corr))

# Define a function to calculate mean
def mean(arr):
    return sum(arr) / len(arr)


# function to calculate cross-correlation
def cross_correlation(x, y):
    # Calculate means
    #print(x)
    #print(y)
    x_mean = mean(x)
    y_mean = mean(y)
    #print(x_mean)
    #print(y_mean)
    # Calculate numerator
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    #print(numerator)
    # Calculate denominators
    x_sq_diff = sum((a - x_mean) ** 2 for a in x)
    y_sq_diff = sum((b - y_mean) ** 2 for b in y)
    denominator = math.sqrt(x_sq_diff * y_sq_diff)
    #print(denominator)
    correlation = numerator / denominator

    return correlation


import numpy as np


def normalized_cross_correlation(signal1, signal2):
    """
    Compute the normalized cross-correlation of two signals.

    Parameters:
    signal1 (numpy.ndarray): First input signal.
    signal2 (numpy.ndarray): Second input signal.

    Returns:
    numpy.ndarray: Normalized cross-correlation.
    numpy.ndarray: Corresponding lags.
    """

    # Ensure the signals are numpy arrays
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)

    # Check if the lengths of the signals are the same
    #if len(signal1) != len(signal2):
    #    raise ValueError("Signals must have the same length.")

    # Calculate the means
    mean1 = np.mean(signal1)
    mean2 = np.mean(signal2)

    # Subtract the means from the signals
    signal1_centered = signal1 - mean1
    signal2_centered = signal2 - mean2

    # Calculate the standard deviations
    std1 = np.std(signal1)
    std2 = np.std(signal2)

    # Ensure standard deviations are not zero to avoid division by zero
    if std1 == 0 or std2 == 0:
        raise ValueError("Standard deviation of a signal is zero, cannot normalize cross-correlation.")

    # Calculate the cross-correlation
    cross_corr = np.correlate(signal1_centered, signal2_centered, mode='full')

    # Normalize the cross-correlation
    normalized_cross_corr = cross_corr / (len(signal1) * std1 * std2)

    # Create an array of lags
    lags = np.arange(-len(signal1) + 1, len(signal1))

    return normalized_cross_corr, lags


"""# Read the entire TSV file into a list of lines
with open('Random_Moves.tsv', 'r') as f:
    lines = f.readlines()

# Extract metadata
metadata = {}
index = 0
while index < len(lines) and not lines[index].startswith('-'):
    parts = lines[index].strip().split('\t')
    key = parts[0]
    value = parts[1:]
    metadata[key] = value if len(value) > 1 else value[0]
    index += 1

# Extract the marker names and trajectory types
marker_names = metadata['MARKER_NAMES']
trajectory_types = metadata['TRAJECTORY_TYPES']

# Read the trajectory data starting from the line that starts with the first number
trajectory_data = []
while index < len(lines):
    if lines[index].strip() != "":
        trajectory_data.append(list(map(float, lines[index].strip().split('\t'))))
    index += 1

# Convert the trajectory data to a numpy array
trajectory_data = np.array(trajectory_data)

# Print the metadata
print("Metadata:")
for key, value in metadata.items():
    print(f"{key}: {value}")

# Print the marker names and trajectory types
print("\nMarker Names:", marker_names)
print("Trajectory Types:", trajectory_types)

# Print the trajectory data
print("\nTrajectory Data:")
print(trajectory_data)

# Create a pandas DataFrame from the trajectory data
num_markers = len(marker_names)
columns = []
for marker in marker_names:
    columns.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])

df = pd.DataFrame(trajectory_data, columns=columns)
print("\nDataFrame:")
print(df)

# Example marker triples for angle calculation (user-defined)
marker_triples = [
    ('Sh', 'El', 'Wr'),
    ('Sh', 'Hi', 'Kn'),
    ('Hi', 'Kn', 'An')
]

# Calculate angles
qualysis_angles = calculate_angles3(df, marker_triples)

# Find spike indices
spike_indices = detect_spike_ranges(qualysis_angles['Sh-Hi-Kn'],2.5)
print(spike_indices)

sp = pd.DataFrame({'time': range(len(qualysis_angles['Sh-Hi-Kn'])), 'signal': qualysis_angles['Sh-Hi-Kn']})

# In a real scenario, you might use a threshold to identify spikes
spike_indices = range(spike_indices[0]-1, spike_indices[len(spike_indices)-1]+1)

# Mask the spikes
sp.loc[spike_indices, 'signal'] = np.nan

# Interpolate to fill the spikes
sp['signal'] = sp['signal'].interpolate()
print(sp)
qualysis_angles['Sh-Hi-Kn'] = sp['signal']

df = pd.read_csv('positions.csv')

phone_angles = { "e": df['elbow'],"h": df['hip'],"k": df['knee']}
# Process the angles: take the absolute value and subtract 180
phone_angles["e"] = np.abs(phone_angles["e"] - 180)
phone_angles["k"] = np.abs(phone_angles["k"] - 180)
phone_angles["h"] = np.abs(phone_angles["h"] - 180)

phone_angles["e"] = phone_angles["e"][37:690]
phone_angles["k"] = phone_angles["k"][37:690]
phone_angles["h"] = phone_angles["h"][37:690]

target_length = len(phone_angles['k'])  # Length of the shorter dataset
#signal2_interpolated = interpolate_to_target_length(qualysis_angles['Hi-Kn-An'], target_length)
#qualysis_angles['Hi-Kn-An'] = signal2_interpolated

#signal2_interpolated = interpolate_to_target_length(qualysis_angles['Sh-El-Wr'], target_length)
#qualysis_angles['Sh-El-Wr'] = signal2_interpolated

#signal2_interpolated = interpolate_to_target_length(qualysis_angles['Sh-Hi-Kn'], target_length)
#qualysis_angles['Sh-Hi-Kn'] = signal2_interpolated

#signal2_downsampled = downsample_with_pandas(qualysis_angles['Hi-Kn-An'], target_length)
#qualysis_angles['Hi-Kn-An'] = signal2_downsampled

#signal2_downsampled = downsample_with_pandas(qualysis_angles['Sh-El-Wr'], target_length)
#qualysis_angles['Sh-El-Wr'] = signal2_downsampled

#signal2_downsampled = downsample_with_pandas(qualysis_angles['Sh-Hi-Kn'], target_length)
#qualysis_angles['Sh-Hi-Kn'] = signal2_downsampled

# Upsample by repeating every frame 3 times
data = {
    'frame': range(len(qualysis_angles['Sh-El-Wr'])),
    'e': qualysis_angles['Sh-El-Wr'],
    'h': qualysis_angles['Sh-Hi-Kn'],
    'k': qualysis_angles['Hi-Kn-An']
}
df = pd.DataFrame(data)
df_upsampled = df.loc[df.index.repeat(3)].reset_index(drop=True)

# Downsample by taking every 20th frame
#df_downsampled = df_upsampled.iloc[::18].reset_index(drop=True)
df_downsampled = df_upsampled.iloc[::20].reset_index(drop=True)
qualysis_angles['Sh-El-Wr'] = df_downsampled['e']
qualysis_angles['Sh-Hi-Kn'] = df_downsampled['h']
qualysis_angles['Hi-Kn-An'] = df_downsampled['k']


# Plot angles
plot_angles(qualysis_angles)
plot_angles(phone_angles)

# Compute cross-correlation between knee angles and hip positions
correlation, lag = compute_cross_correlation(phone_angles['k'], qualysis_angles['Hi-Kn-An'])
#correlation /= np.max(correlation)
#correlation /= (len(phone_angles['k']) * np.std(phone_angles['k']) * np.std(qualysis_angles['Hi-Kn-An'])) # Normalization

correlation2, lag2 = compute_cross_correlation(phone_angles['e'], qualysis_angles['Sh-El-Wr'])
#correlation2 /= np.max(correlation2)

correlation3, lag3 = compute_cross_correlation(phone_angles['h'], qualysis_angles['Sh-Hi-Kn'])
#correlation3 /= np.max(correlation3)
print(correlation)
print(correlation2)
print(correlation3)


# Plot cross-correlation between knee angles and hip positions
plt.figure(figsize=(12, 6))

# Plot cross-correlation between knee angles and elbow positions
plt.subplot(3, 1, 1)
plt.plot(lag2, correlation2)
plt.title('Cross-Correlation between Elbow Angles')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')

# Plot cross-correlation between knee angles and elbow positions
plt.subplot(3, 1, 2)
plt.plot(lag3, correlation3)
plt.title('Cross-Correlation between Hip Angles')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')

plt.subplot(3, 1, 3)
plt.plot(lag, correlation)
plt.title('Cross-Correlation between Knee Angles')
plt.xlabel('Lag')
plt.ylabel('Cross-Correlation')

plt.tight_layout()
plt.show()

# Find the lag with the maximum cross-correlation coefficient
mean1 = (correlation2[np.argmax(correlation2)]+correlation[np.argmax(correlation2)]+correlation3[np.argmax(correlation2)])/3
mean2 = (correlation[np.argmax(correlation)]+correlation2[np.argmax(correlation)]+correlation3[np.argmax(correlation)])/3
mean3 = (correlation3[np.argmax(correlation3)]+correlation[np.argmax(correlation3)]+correlation2[np.argmax(correlation3)])/3
print(mean1)
print(mean2)
print(mean3)
#max_av_cor = max(np.argmax(correlation2)+correlation[np.argmax(correlation2)]+correlation3[np.argmax(correlation2)],)
shift_lag = lag2[np.argmax(correlation2)] - 40

#shift_lag = lag[np.argmax(correlation)]

print(f'Lag with maximum cross-correlation coefficient: {shift_lag}')
print(max(correlation2))
print(correlation[np.argmax(correlation2)])
print(correlation3[np.argmax(correlation2)])

# Shift signal2 by the found lag
#aligned_signal2 =  np.roll(qualysis_angles['Sh-El-Wr'],shift_lag)
aligned_signal2 = pd.Series(phone_angles['e']).shift(shift_lag)
aligned_signal3 = pd.Series(phone_angles['h']).shift(shift_lag)
aligned_signal = pd.Series(phone_angles['k']).shift(shift_lag)

# Plot original and aligned signals
plt.figure(figsize=(12, 6))
plt.plot(qualysis_angles['Sh-El-Wr'], label='Signal 1')
plt.plot(aligned_signal2, label=f'Signal 2 (shifted by {lag} frames)')
plt.title('Overlay of Signal 1 and Aligned Signal 2')
plt.xlabel('Frame')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot original and aligned signals
plt.figure(figsize=(12, 6))
plt.plot(qualysis_angles['Sh-Hi-Kn'], label='Signal 1')
plt.plot(aligned_signal3, label=f'Signal 2 (shifted by {lag} frames)')
plt.title('Overlay of Signal 1 and Aligned Signal 2')
plt.xlabel('Frame')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plot original and aligned signals
plt.figure(figsize=(12, 6))
plt.plot(qualysis_angles['Hi-Kn-An'], label='Signal 1')
plt.plot(aligned_signal, label=f'Signal 2 (shifted by {lag} frames)')
plt.title('Overlay of Signal 1 and Aligned Signal 2')
plt.xlabel('Frame')
plt.ylabel('Value')
plt.legend()
plt.show()

print(len(aligned_signal))
print(len(qualysis_angles['Hi-Kn-An']))
print('lag:', shift_lag)
correlation = cross_correlation(aligned_signal, qualysis_angles['Hi-Kn-An'])
print('Correlation:', correlation)
correlation = cross_correlation(aligned_signal3, qualysis_angles['Sh-Hi-Kn'])
print('Correlation:', correlation)
correlation = cross_correlation(aligned_signal2, qualysis_angles['Sh-El-Wr'])
print('Correlation:', correlation)"""