import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, lognorm
import numpy as np

# Set the background to white
plt.style.use('seaborn-white')

# Load data
df = pd.read_csv('csub_t0.txt', header=None, names=['time', 'rate', 'error', 'nump'])

# Calculate max time and average rate
max_time = df['time'].max()
average_rate = df['rate'].mean()

# Plotting time vs rate
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(df['time'], df['rate'], color='black')

# Setting labels and title with font size 14
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Rate', fontsize=14)
ax.set_title(f'Time vs Rate (Max Time: {max_time:.2f}, Average Rate: {average_rate:.2f})', fontsize=14)

# Setting tick parameters with font size 14
ax.tick_params(axis='both', labelsize=14)

ax.grid(True)
plt.show()

# Log-transform the rates for log-normal fitting
log_rates = np.log(df['rate'][df['rate'] > 0])  # ensuring positive rates
log_mean = np.mean(log_rates)
log_std = np.std(log_rates)

# Specify the desired bin edges
bin_edges = np.arange(10, 52, 1)  # Bin edges from 10 to 50

# Plot histogram of rates
fig, ax = plt.subplots(figsize=(10, 8))
count_bins, _, _ = ax.hist(df['rate'], bins=bin_edges, alpha=0.75, label='Observed rates', color='gray', edgecolor='black')

# Calculate the average rate and standard deviation
std_dev = df['rate'].std()

# Compute Poisson distribution with the mean rate
x_poisson = np.arange(10, 51)
poisson_prob = poisson.pmf(x_poisson, mu=average_rate)

# Compute normal distribution PDF
x_normal = np.linspace(10, 50, 100)
normal_pdf = norm.pdf(x_normal, loc=average_rate, scale=std_dev)

# Compute log-normal distribution PDF
x_lognormal = np.linspace(10, 50, 100)
lognormal_pdf = lognorm.pdf(x_lognormal, log_std, scale=np.exp(log_mean))

# Scale normal PDF and log-normal PDF to fit histogram scale
scale_factor = len(df['rate']) * (bin_edges[1] - bin_edges[0])
normal_scaled = normal_pdf * scale_factor
lognormal_scaled = lognormal_pdf * scale_factor

# Plot Poisson distribution
ax.plot(x_poisson, poisson_prob * scale_factor, linestyle='-', linewidth=2, label='Poisson fit')

# Plot Normal distribution
ax.plot(x_normal, normal_scaled, linestyle='--', linewidth=2, label='Normal fit')

# Plot Log-normal distribution
ax.plot(x_lognormal, lognormal_scaled, linestyle=':', linewidth=2, label='Lognormal fit')

# Plot the data points
ax.scatter(df['rate'], np.zeros_like(df['rate']), marker='x', color='black', label='Data')

# Setting labels, title, and legend with font size 14
ax.set_xlabel('Rate', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.set_title('Histogram of Rates with Poisson, Normal, and Log-Normal Fits', fontsize=14)
ax.legend(fontsize=14)

# Setting tick parameters with font size 14
ax.tick_params(axis='both', labelsize=14)

ax.grid(True, linestyle='--', linewidth=0.5)
plt.show()

# Calculate frequencies of Poisson fit for counts ranging from 10 to 50
counts = np.arange(10, 51)
poisson_frequencies = poisson.pmf(counts, mu=average_rate) * scale_factor

# Calculate frequencies of Normal distribution fit for counts ranging from 10 to 50
normal_frequencies = norm.pdf(counts, loc=average_rate, scale=std_dev) * scale_factor

# Calculate frequencies of Log-Normal distribution fit for counts ranging from 10 to 50
lognormal_frequencies = lognorm.pdf(counts, log_std, scale=np.exp(log_mean)) * scale_factor

# Create a DataFrame for the table
distribution_table = pd.DataFrame({
    'Bin Edges': bin_edges[:-1],
    'Histogram Bin Counts': count_bins.astype(int),
    'Poisson Frequency': poisson_frequencies,
    'Normal Frequency': normal_frequencies,
    'Log-Normal Frequency': lognormal_frequencies
})

print("Table for Poisson, Normal, and Log-Normal Distributions:")
print(distribution_table)

# Calculate the observed and expected values from the DataFrame
observed_counts = distribution_table['Histogram Bin Counts']
poisson_expected = distribution_table['Poisson Frequency']
normal_expected = distribution_table['Normal Frequency']
lognormal_expected = distribution_table['Log-Normal Frequency']

# Calculate the squared differences between observed and expected values, divided by expected values
poisson_squared_diff = ((observed_counts - poisson_expected) ** 2) / poisson_expected
normal_squared_diff = ((observed_counts - normal_expected) ** 2) / normal_expected
lognormal_squared_diff = ((observed_counts - lognormal_expected) ** 2) / lognormal_expected

# Sum up the squared differences to get the chi-squared value
chi_squared_poisson = poisson_squared_diff.sum()
chi_squared_normal = normal_squared_diff.sum()
chi_squared_lognormal = lognormal_squared_diff.sum()

print("\nChi-squared value for Poisson fit:", chi_squared_poisson)
print("Chi-squared value for Normal fit:", chi_squared_normal)
print("Chi-squared value for Log-Normal fit:", chi_squared_lognormal)
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
