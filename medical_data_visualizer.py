import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv") # Read the csv and put the dataframe in df

# 2
df['overweight'] = df['weight'] / (df['height'] / 100) ** 2 # Create the column 'overweight' and give it the BMI value
df['overweight'] = df['overweight'].map(lambda x: 1 if x > 25 else 0) # Map each BMI value in 'overweight' to 0 or 1 depending if it's above 25 or not

# 3
df['cholesterol'] = df['cholesterol'].map(lambda x: 1 if x > 1 else 0) # Map each value to 0 or 1 depending if it's above 1 or not
df['gluc'] = df['gluc'].map(lambda x: 1 if x > 1 else 0) # Map each value to 0 or 1 depending if it's above 1 or not


# 4
def draw_cat_plot():
    # 5
    # pd.melt reshapes the dataframe into a long format, that is, the columns are merged into one, where each rows represents an instance of the variable. In each row, it puts the name of the variable and it's corresponding value for each instance.
    df_cat = pd.melt(
        df, # Specify that df is the dataframe to be melted
        # 6
        id_vars=['cardio'], # Keep the 'cardio' column intact for grouping the data later
        # 7
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'] # Specify all the columns to be melted
    )

    df_cat.columns = ['cardio', 'variable', 'value'] # Rename the columns

    # 8
    fig = sns.catplot(
        x='variable',  # Put categorical variables on the x-axis
        hue='value',   # Change the color based on the value
        col='cardio',  # Separate plots for each value of 'cardio'
        data=df_cat,   # Use df_cat's data to create the plot
        kind='count',  # Use the number of occurrences
    )

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df # Copies the data
    df_heat = df_heat[df['ap_lo'] <= df['ap_hi']] # Only keep rows where this is true
    df_heat = df_heat[df['height'] >= df['height'].quantile(0.025)] # Only keep rows where this is true
    df_heat = df_heat[df['height'] <= df['height'].quantile(0.975)] # Only keep rows where this is true
    df_heat = df_heat[df['weight'] >= df['weight'].quantile(0.025)] # Only keep rows where this is true
    df_heat = df_heat[df['weight'] <= df['weight'].quantile(0.975)] # Only keep rows where this is true

    # 12
    corr = df_heat.corr() # Creates a correlation matrix (a pd.Dataframe)

    # 13
    mask = np.triu(np.ones(corr.shape), k=0) # Creates a matrix filled with ones and sets the upper triangle to 1 and the lower to 0

    # 14
    fig, ax = plt.subplots(figsize=(12, 8)) # Creates the plot and sets it's size

    # 15
    sns.heatmap(
        data=corr,
        mask=mask,
        annot=True,       # Show the correlation values
        fmt=".1f",        # Correlation values to 1 decimal place
        cmap='coolwarm',  # Use the coolwarm color map
        square=True,      # Make cells square
        vmin=-0.7,        # Set the minimum value of the colormap (helps white being in the middle)
        vmax=0.7          # Set the maximum value of the colormap
    )

    # 16
    fig.savefig('heatmap.png') # Saving the heatmap
    return fig
