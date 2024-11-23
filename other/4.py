import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

# Step 1: Define prior and posterior parameters
alpha_prior = 3
beta_prior = 5

# First survey: 70 likes out of 100
likes_first_survey = 70
total_first_survey = 100

# Second survey: 40 dislikes out of the 70 who liked the smartphone
dislikes_second_survey = 40
likes_second_survey = likes_first_survey - dislikes_second_survey

# Posterior after first survey: Beta(alpha + likes, beta + (total - likes))
alpha_post_first = alpha_prior + likes_first_survey
beta_post_first = beta_prior + (total_first_survey - likes_first_survey)

# Posterior after second survey: Beta(alpha + likes + dislikes, beta + (total - likes - dislikes))
alpha_post_second = alpha_post_first + likes_second_survey
beta_post_second = beta_post_first + (total_first_survey - likes_first_survey - dislikes_second_survey)

# Step 2: Define a range of p values for plotting
p_values = np.linspace(0, 1, 1000)

# Step 3: Plot the prior and posterior distributions
plt.figure(figsize=(12, 8))

# Plot Prior Distribution
plt.subplot(3, 1, 1)
prior_pdf = beta.pdf(p_values, alpha_prior, beta_prior)
plt.plot(p_values, prior_pdf, label='Prior Distribution (Beta(3, 5))', color='blue')
plt.fill_between(p_values, prior_pdf, alpha=0.3, color='blue')
plt.title('Prior Distribution of p')
plt.xlabel('p')
plt.ylabel('Density')
plt.legend()

# Plot Posterior Distribution after First Survey
plt.subplot(3, 1, 2)
posterior_pdf_first = beta.pdf(p_values, alpha_post_first, beta_post_first)
plt.plot(p_values, posterior_pdf_first, label=f'Posterior Distribution after First Survey (Beta({alpha_post_first}, {beta_post_first}))', color='green')
plt.fill_between(p_values, posterior_pdf_first, alpha=0.3, color='green')
plt.title('Posterior Distribution of p after First Survey')
plt.xlabel('p')
plt.ylabel('Density')
plt.legend()

# Plot Posterior Distribution after Second Survey
plt.subplot(3, 1, 3)
posterior_pdf_second = beta.pdf(p_values, alpha_post_second, beta_post_second)
plt.plot(p_values, posterior_pdf_second, label=f'Posterior Distribution after Second Survey (Beta({alpha_post_second}, {beta_post_second}))', color='red')
plt.fill_between(p_values, posterior_pdf_second, alpha=0.3, color='red')
plt.title('Posterior Distribution of p after Second Survey')
plt.xlabel('p')
plt.ylabel('Density')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()