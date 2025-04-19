import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def to_dataframe(data, feature):
    """
    Convert a customer object or a list of objects to a DataFrame.

    Parameters:
    data (dict, object, or list): Customer data in dictionary form, a single object, or a list of objects.
    features (list, optional): List of attribute names to include in the DataFrame.

    Returns:
    pd.DataFrame: A DataFrame representation of the input data.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    elif isinstance(data, dict):
        return pd.DataFrame([{feature: data.get(feature, None)}])
    elif isinstance(data, list):
        return  pd.DataFrame([{feature: getattr(obj, feature)  for obj in data if hasattr(obj, feature)}])
    else:
        raise ValueError("Input must be a DataFrame, dictionary, class instance, or list of class instances.")

def plot_numeric_distribution(
    sample_data,
    similar_data,
    feature,
    figsize=(10, 5),
    title_fontsize=16,
    label_fontsize=14,
    density_color='blue',
    similar_color='orange',
    grid=True
):
    """
    Plot the distribution of a numeric feature.

    Parameters:
    sample_data (pd.DataFrame, dict, object, or iterable): Data for sample instances.
    similar_data (pd.DataFrame, dict, object, or iterable): Data for similar instances.
    feature (str): The numeric feature name to plot.
    figsize (tuple): Size of the figure.
    title_fontsize (int): Font size for the title.
    label_fontsize (int): Font size for the labels.
    density_color (str): Color for the sample instances density plot.
    similar_color (str): Color for the similar instances density plot.
    """
    sample_df = to_dataframe(sample_data, feature=feature)
    similar_df = to_dataframe(similar_data, feature=feature)

    plt.figure(figsize=figsize)
    sns.kdeplot(sample_df[feature], color=density_color, label='Sample Instances', fill=True, alpha=0.5)
    sns.kdeplot(similar_df[feature], color=similar_color, label='Similar Instances', fill=True, alpha=0.5)

    plt.title(f'Distribution of {feature}', fontsize=title_fontsize)
    plt.xlabel(feature, fontsize=label_fontsize)
    plt.ylabel('Density', fontsize=label_fontsize)
    plt.legend()
    if grid:
        plt.grid()
    plt.show()

def plot_categorical_distribution(
    sample_data,
    similar_data,
    feature,
    figsize=(10, 5),
    title_fontsize=16,
    label_fontsize=14,
    sample_color='blue',
    similar_color='orange',
    grid=True
):
    """
    Plot the distribution of a categorical feature.

    Parameters:
    sample_data (pd.DataFrame, dict, object, or iterable): Data for sample instances.
    similar_data (pd.DataFrame, dict, object, or iterable): Data for similar instances.
    feature (str): The categorical feature name to plot.
    figsize (tuple): Size of the figure.
    title_fontsize (int): Font size for the title.
    label_fontsize (int): Font size for the labels.
    sample_color (str): Color for the sample instances bars.
    similar_color (str): Color for the similar instances bars.
    """
    sample_df = to_dataframe(sample_data, feature=feature)
    similar_df = to_dataframe(similar_data, feature=feature)

    plt.figure(figsize=figsize)

    # Count occurrences and compute proportions
    count_df1 = sample_df[feature].value_counts(normalize=True).reset_index()
    count_df1.columns = [feature, 'Count Instance Sample']
    
    count_df2 = similar_df[feature].value_counts(normalize=True).reset_index()
    count_df2.columns = [feature, 'Count Instance Similar']

    # Merge counts
    merged_counts = pd.merge(count_df1, count_df2, on=feature, how='outer').fillna(0)

    sns.barplot(x=feature, y='Count Instance Sample', data=merged_counts, color=sample_color, alpha=0.6, label='Sample Instances')
    sns.barplot(x=feature, y='Count Instance Similar', data=merged_counts, color=similar_color, alpha=0.6, label='Similar Instances')

    plt.title(f'Distribution of {feature}', fontsize=title_fontsize)
    plt.xlabel(feature, fontsize=label_fontsize)
    plt.ylabel('Proportion', fontsize=label_fontsize)
    plt.legend()
    if grid:
        plt.grid()
    plt.show()
