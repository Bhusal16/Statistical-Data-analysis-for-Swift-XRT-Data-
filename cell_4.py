import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, norm, lognorm

# Set Matplotlib background to white
plt.style.use('seaborn-white')

# Read data from files
df_t0 = pd.read_csv('csub_t0.txt', header=None, names=['time', 'rate', 'error', 'nump'])
df_t3 = pd.read_csv('csub_t3.txt', header=None, names=['time', 'rate', 'error', 'nump'])

# Adjust time values
df_t3['time'] += df_t0['time'].max()
df_combined = pd.concat([df_t0, df_t3])

df_t5 = pd.read_csv('csub_t5.txt', header=None, names=['time', 'rate', 'error', 'nump'])
df_t5['time'] += df_combined['time'].max()
df_combined = pd.concat([df_combined, df_t5])

# Calculate max time and average rate
max_time = df_combined['time'].max()
average_rate = df_combined['rate'].mean()

# Plot time vs rate
fig, ax = plt.subplots(facecolor='white')  # White background
ax.set_facecolor('white')
ax.plot(df_combined['time'], df_combined['rate'], color='black')

# Set labels, title, and tick sizes
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Rate', fontsize=14)
ax.set_title(f'Time vs Rate\nMax Time: {max_time:.2f}, Average Rate: {average_rate:.2f}', fontsize=14)
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, color='lightgray')

plt.show()

# Histogram of rates
fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
ax.set_facecolor('white')

count_bins = ax.hist(df_combined['rate'], bins=30, alpha=0.75, label='Observed rates', color='gray', edgecolor='black')

# Generate x values for distributions
max_rate = df_combined['rate'].max()
x_poisson = range(int(max_rate) + 1)
x_normal = np.linspace(0, max_rate, 100)
x_lognormal = np.linspace(0, max_rate, 1000)

# Compute Poisson, Normal, and Lognormal distributions
poisson_prob = poisson.pmf(x_poisson, mu=average_rate)
std_dev = df_combined['rate'].std()
normal_pdf = norm.pdf(x_normal, loc=average_rate, scale=std_dev)
log_rates = np.log(df_combined['rate'][df_combined['rate'] > 0])
log_mean, log_std = np.mean(log_rates), np.std(log_rates)
lognormal_pdf = lognorm.pdf(x_lognormal, log_std, scale=np.exp(log_mean))

# Scale distributions
scale_factor = len(df_combined['rate']) * (count_bins[1][1] - count_bins[1][0])
normal_scaled = normal_pdf * scale_factor
lognormal_scaled = lognormal_pdf * scale_factor

# Plot distributions
ax.plot(x_poisson, poisson_prob * scale_factor, color='black', linestyle='-', linewidth=2, label='Poisson fit')
ax.plot(x_normal, normal_scaled, color='black', linestyle='--', linewidth=2, label='Normal fit')
ax.plot(x_lognormal, lognormal_scaled, color='black', linestyle=':', linewidth=2, label='Lognormal fit')

# Set labels, title, and legend
ax.set_xlabel('Rate', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.set_title('Histogram of Rates with Poisson, Normal, and Lognormal Fits', fontsize=14)
ax.tick_params(axis='both', labelsize=14)
ax.legend(fontsize=14)
ax.grid(True, color='lightgray')

plt.show()

# Chi-Square Calculation
bin_edges = np.arange(1, int(max_rate) + 1, 1)
observed_counts, _ = np.histogram(df_combined['rate'], bins=bin_edges)

# Expected frequencies
poisson_freq = poisson.pmf(bin_edges[:-1], mu=average_rate) * scale_factor
normal_freq = norm.pdf(bin_edges[:-1], loc=average_rate, scale=std_dev) * scale_factor
lognormal_freq = lognorm.pdf(bin_edges[:-1], log_std, scale=np.exp(log_mean)) * scale_factor

# Chi-square values
chi_squared_poisson = np.sum((observed_counts - poisson_freq) ** 2 / poisson_freq)
chi_squared_normal = np.sum((observed_counts - normal_freq) ** 2 / normal_freq)
chi_squared_lognormal = np.sum((observed_counts - lognormal_freq) ** 2 / lognormal_freq)

# Table of bin counts and frequencies
table_df = pd.DataFrame({
    'Bin': bin_edges[:-1],
    'Bin Count': observed_counts,
    'Poisson Frequency': poisson_freq,
    'Normal Frequency': normal_freq,
    'Lognormal Frequency': lognormal_freq
})

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

