from utils.visualization.graph_utils import *
import pandas as pd
import numpy as np


# Generate sample age data based on normal distribution
np.random.seed(42)
rng = np.random.default_rng(seed=42)

age_data = rng.integers(1, 100, size=100)

# Fit the data into a DataFrame with column "Age"
df = pd.DataFrame(age_data, columns=["Age"])
plot_column_hist_with_normal_dist(df, "Age", title="Age Distribution")
