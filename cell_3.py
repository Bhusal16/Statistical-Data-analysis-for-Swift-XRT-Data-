import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm, lognorm

# Set the background to white
plt.style.use('seaborn-white')
plt.rcParams.update({'font.size': 14})

# Read data from 'csub_t0.txt'
df_t0 = pd.read_csv('csub_t0.txt', header=None, names=['time', 'rate', 'error', 'nump'])

# Read data from 'csub_t3.txt'
df_t3 = pd.read_csv('csub_t3.txt', header=None, names=['time', 'rate', 'error', 'nump'])

# Adjust time values of df_t3 to start where df_t0 ends
max_time_t0 = df_t0['time'].max()
df_t3['time'] += max_time_t0

# Concatenate the dataframes
df_combined = pd.concat([df_t0, df_t3])

# Calculate the maximum time and average rate
max_time = df_combined['time'].max()
average_rate = df_combined['rate'].mean()
std_dev = df_combined['rate'].std()

# Plotting time vs rate from combined data
plt.figure(figsize=(10, 8))
plt.plot(df_combined['time'], df_combined['rate'], color='black')  # Line color: Black
plt.xlabel('Time')
plt.ylabel('Rate')
plt.title(f'Time vs Rate\nMax Time: {max_time:.2f}, Average Rate: {average_rate:.2f}')
plt.grid(True)
plt.show()

# Plot histogram of rates
plt.figure(figsize=(10, 8))
count_bins = plt.hist(df_combined['rate'], bins=30, alpha=0.75, label='Observed rates', color='gray', edgecolor='black')

# Generate a range of count values for Poisson, normal, and lognormal distributions
max_rate = df_combined['rate'].max()
x_poisson = range(int(max_rate) + 1)
x_normal = np.linspace(0, max_rate, 100)
x_lognormal = np.linspace(0, max_rate, 1000)

# Compute Poisson distribution with the mean rate
poisson_prob = poisson.pmf(x_poisson, mu=average_rate)

# Compute normal distribution PDF
normal_pdf = norm.pdf(x_normal, loc=average_rate, scale=std_dev)

# Log-transform the rates for lognormal fitting
log_rates = np.log(df_combined['rate'][df_combined['rate'] > 0])  # ensuring positive rates
log_mean = np.mean(log_rates)
log_std = np.std(log_rates)

# Compute lognormal distribution PDF
lognormal_pdf = lognorm.pdf(x_lognormal, log_std, scale=np.exp(log_mean))

# Scale normal PDF and lognormal PDF to fit histogram scale
scale_factor = len(df_combined['rate']) * (count_bins[1][1] - count_bins[1][0])
normal_scaled = normal_pdf * scale_factor
lognormal_scaled = lognormal_pdf * scale_factor

# Plot Poisson distribution
plt.plot(x_poisson, poisson_prob * scale_factor, color='blue', linestyle='-', linewidth=2, label='Poisson fit')

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

# Define bin edges from 0 to 50 for chi-squared calculation
bin_edges = np.arange(1, 50, 1)
count_bins, _ = np.histogram(df_combined['rate'], bins=bin_edges)

# Compute Poisson, normal, and lognormal frequencies for bins
poisson_freq = poisson.pmf(bin_edges[:-1], mu=average_rate) * scale_factor
normal_freq = norm.pdf(bin_edges[:-1], loc=average_rate, scale=std_dev) * scale_factor
lognormal_freq = lognorm.pdf(bin_edges[:-1], log_std, scale=np.exp(log_mean)) * scale_factor

# Compute chi-squared values
chi_squared_poisson = np.sum((count_bins - poisson_freq) ** 2 / poisson_freq)
chi_squared_normal = np.sum((count_bins - normal_freq) ** 2 / normal_freq)
chi_squared_lognormal = np.sum((count_bins - lognormal_freq) ** 2 / lognormal_freq)

# Create table for bin counts and calculated frequencies
table_data = {
    'Bin': bin_edges[:-1],
    'Bin Count': count_bins,
    'Poisson Frequency': poisson_freq,
    'Normal Frequency': normal_freq,
    'Lognormal Frequency': lognormal_freq
}
table_df = pd.DataFrame(table_data)

print("Table of Bin Counts and Frequencies:")
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
