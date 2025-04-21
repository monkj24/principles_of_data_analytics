# z ChatGPI jak wyglada correlation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)
data = {
    "Age": np.random.randint(18, 65, 100),
    "Salary": np.random.randint(30000, 120000, 100),
    "Experience": np.random.randint(1, 40, 100),
    "Performance_Score": np.random.uniform(0, 10, 100)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

plt.title("Feature Correlation Heatmap")
plt.show()


