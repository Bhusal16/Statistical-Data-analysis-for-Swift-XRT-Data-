import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm, lognorm

# Set the background to white
plt.style.use('seaborn-white')
plt.rcParams.update({'font.size': 14})

# Read data
df = pd.read_csv('combined_light_curve.txt', header=None, names=['time', 'rate', 'error', 'nump'])

# Apply binning (adjust 'bin_size' as needed)
bin_size = 50  # Number of points per bin
df['bin_index'] = df.index // bin_size

# Aggregate data by bins (mean of rate, first value of time in each bin)
df_binned = df.groupby('bin_index').agg({'time': 'first', 'rate': 'mean'}).reset_index(drop=True)

# Extract useful statistics
max_time = df_binned['time'].max()
average_rate = df_binned['rate'].mean()
std_dev = df_binned['rate'].std()

# Plot Time vs Rate (Binned Data)
plt.figure(figsize=(10, 8))
plt.plot(df_binned['time'], df_binned['rate'], color='black')  # Line color: Black
plt.ylim(0, 4000)  # Limit y-axis to 4000
plt.xlabel('Time')
plt.ylabel('Rate')
plt.title(f'Binned Time vs Rate\nMax Time: {max_time:.2f}, Average Rate: {average_rate:.2f}')
plt.grid(True)
plt.show()

# Plot histogram of rates
plt.figure(figsize=(10, 8))
count_bins = plt.hist(df_binned['rate'], bins=30, alpha=0.75, label='Observed rates', color='gray', edgecolor='black')

# Generate range of count values for distributions
max_rate = df_binned['rate'].max()
x_poisson = range(int(max_rate) + 1)
x_normal = np.linspace(0, max_rate, 100)
x_lognormal = np.linspace(0, max_rate, 1000)

# Compute Poisson distribution
poisson_prob = poisson.pmf(x_poisson, mu=average_rate)

# Compute normal distribution PDF
normal_pdf = norm.pdf(x_normal, loc=average_rate, scale=std_dev)

# Log-transform rates for lognormal fitting
log_rates = np.log(df_binned['rate'][df_binned['rate'] > 0])  # Ensuring positive rates
log_mean = np.mean(log_rates)
log_std = np.std(log_rates)

# Compute lognormal distribution PDF
lognormal_pdf = lognorm.pdf(x_lognormal, log_std, scale=np.exp(log_mean))

# Scale distributions to match histogram
scale_factor = len(df_binned['rate']) * (count_bins[1][1] - count_bins[1][0])
normal_scaled = normal_pdf * scale_factor
lognormal_scaled = lognormal_pdf * scale_factor

# Plot Poisson fit
plt.plot(x_poisson, poisson_prob * scale_factor, color='blue', linestyle='-', linewidth=2, label='Poisson fit')

# Plot Normal fit
plt.plot(x_normal, normal_scaled, color='orange', linestyle='--', linewidth=2, label='Normal fit')

# Plot Lognormal fit
plt.plot(x_lognormal, lognormal_scaled, color='green', linestyle=':', linewidth=2, label='Lognormal fit')

plt.ylim(0, 000)  # Limit y-axis to 4000
plt.xlabel('Rate')
plt.ylabel('Frequency')
plt.title('Histogram of Rates with Poisson, Normal, and Lognormal Fits')
plt.legend()
plt.grid(True)
plt.show()

# Define bin edges for chi-square calculation
bin_edges = np.arange(1, min(4000, max_rate), 1)
count_bins, _ = np.histogram(df_binned['rate'], bins=bin_edges)

# Compute expected frequencies
poisson_freq = poisson.pmf(bin_edges[:-1], mu=average_rate) * scale_factor
normal_freq = norm.pdf(bin_edges[:-1], loc=average_rate, scale=std_dev) * scale_factor
lognormal_freq = lognorm.pdf(bin_edges[:-1], log_std, scale=np.exp(log_mean)) * scale_factor

# Compute chi-squared values
chi_squared_poisson = np.sum((count_bins - poisson_freq) ** 2 / poisson_freq)
chi_squared_normal = np.sum((count_bins - normal_freq) ** 2 / normal_freq)
chi_squared_lognormal = np.sum((count_bins - lognormal_freq) ** 2 / lognormal_freq)

# Create a table for bin counts and calculated frequencies
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

