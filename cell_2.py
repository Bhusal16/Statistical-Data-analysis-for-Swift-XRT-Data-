import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm, lognorm

# Set background to white and font size to 14
plt.style.use('seaborn-white')
plt.rcParams.update({'font.size': 14})

# Load data
df = pd.read_csv('csub_t3.txt', header=None, names=['time', 'rate', 'error', 'nump'])

# Calculate the average rate and standard deviation
average_rate = df['rate'].mean()
std_dev = df['rate'].std()

# Calculate max limit of x-axis
max_time = df['time'].max()

# Plotting time vs rate
plt.figure(figsize=(10, 8))
plt.plot(df['time'], df['rate'], color='black')
plt.xlabel('Time')
plt.ylabel('Rate')
plt.title(f'Time vs Rate (Max Time: {max_time:.2f}, Average Rate: {average_rate:.2f})')
plt.grid(True)
plt.show()

# Define bin edges from 0 to 12
bin_edges = np.arange(0, 13, 1)

# Plot histogram of rates
plt.figure(figsize=(10, 8))
count_bins, _, _ = plt.hist(df['rate'], bins=bin_edges, alpha=0.75, label='Observed rates', color='gray', edgecolor='black')

# Generate a range of count values for Poisson, normal, and lognormal distributions
x_poisson = np.arange(0, 13)
x_normal = np.linspace(0, 12, 100)
x_lognormal = np.linspace(0, 12, 1000)

# Fit Poisson distribution to the data (use the average rate as the parameter)
poisson_fit = poisson.pmf(x_poisson, mu=average_rate)

# Fit Normal distribution (using mean and standard deviation)
normal_fit_params = norm.fit(df['rate'])
normal_pdf = norm.pdf(x_normal, *normal_fit_params)

# Fit Lognormal distribution (log-transform the data and fit the lognormal parameters)
lognormal_fit_params = lognorm.fit(df['rate'][df['rate'] > 0])  # Ensure positive rates
lognormal_pdf = lognorm.pdf(x_lognormal, *lognormal_fit_params)

# Scale normal PDF and lognormal PDF to fit histogram scale
scale_factor = len(df['rate']) * (bin_edges[1] - bin_edges[0])
normal_scaled = normal_pdf * scale_factor
lognormal_scaled = lognormal_pdf * scale_factor

# Plot Poisson distribution
plt.plot(x_poisson, poisson_fit * scale_factor, color='blue', linestyle='-', linewidth=2, label='Poisson fit')

# Plot Normal distribution
plt.plot(x_normal, normal_scaled, color='orange', linestyle='--', linewidth=2, label='Normal fit')

# Plot Lognormal distribution
plt.plot(x_lognormal, lognormal_scaled, color='green', linestyle=':', linewidth=2, label='Lognormal fit')

plt.xlabel('Rate')
plt.ylabel('Frequency')
plt.title('Histogram of Rates with Poisson, Normal, and Lognormal Fits')
plt.legend()
plt.grid(True)
plt.show()

# Create table for bin counts and calculated frequencies
bin_counts = count_bins

# Compute Poisson, normal, and lognormal frequencies for bins
poisson_freq = poisson.pmf(bin_edges[:-1], mu=average_rate) * scale_factor
normal_freq = norm.pdf(bin_edges[:-1], *normal_fit_params) * scale_factor
lognormal_freq = lognorm.pdf(bin_edges[:-1], *lognormal_fit_params) * scale_factor

# Function to compute chi-squared value
def chi_squared(observed, expected):
    return np.sum((observed - expected) ** 2 / expected)

# Compute chi-squared for Poisson, Normal, and Lognormal fits
chi_squared_poisson = chi_squared(bin_counts, poisson_freq)
chi_squared_normal = chi_squared(bin_counts, normal_freq)
chi_squared_lognormal = chi_squared(bin_counts, lognormal_freq)

# Create DataFrame for table
table_data = {
    'Bin': bin_edges[:-1],
    'Bin Count': bin_counts.astype(int),
    'Poisson Frequency': poisson_freq,
    'Normal Frequency': normal_freq,
    'Lognormal Frequency': lognormal_freq
}

table_df = pd.DataFrame(table_data)

# Print the table and chi-squared values
print("Table of Bin Counts and Fitted Frequencies:")
print(table_df)

print("\nChi-Squared Values:")
print(f"Poisson: {chi_squared_poisson:.2f}")
print(f"Normal: {chi_squared_normal:.2f}")
print(f"Lognormal: {chi_squared_lognormal:.2f}")

# Number of observations
n = len(df['rate'])

# For Poisson: 1 parameter (mu)
# Log-likelihood for Poisson: sum of log PMF values
log_likelihood_poisson = np.sum(poisson.logpmf(df['rate'].astype(int), mu=average_rate))
bic_poisson = 1 * np.log(n) - 2 * log_likelihood_poisson

# For Normal: 2 parameters (mean and std)
log_likelihood_normal = np.sum(norm.logpdf(df['rate'], loc=average_rate, scale=std_dev))
bic_normal = 2 * np.log(n) - 2 * log_likelihood_normal

# For Log-Normal: 2 parameters (log-mean and log-std)
log_likelihood_lognormal = np.sum(lognorm.logpdf(df['rate'][df['rate'] > 0], log_std, scale=np.exp(log_mean)))
bic_lognormal = 2 * np.log(len(df['rate'][df['rate'] > 0])) - 2 * log_likelihood_lognormal

# Print the BIC values
print("\nBayesian Information Criterion (BIC):")
print(f"Poisson BIC: {bic_poisson:.2f}")
print(f"Normal BIC: {bic_normal:.2f}")
print(f"Log-Normal BIC: {bic_lognormal:.2f}")

