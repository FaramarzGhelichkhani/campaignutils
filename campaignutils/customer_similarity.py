import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class SimilarFinder:
    @classmethod
    def _to_dataframe(cls, data):
        """
        Convert a customer object or a list of objects to a DataFrame.

        Parameters:
        data (dict, object, or list): Customer data in dictionary form, a single object, or a list of objects.

        Returns:
        pd.DataFrame: A DataFrame representation of the input data.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame([{attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('__') and not callable(getattr(obj, attr))} for obj in data])
        elif isinstance(data, object):
            # Convert a single object to a DataFrame
            return pd.DataFrame([{attr: getattr(data, attr) for attr in dir(data) if not attr.startswith('__') and not callable(getattr(data, attr))}])
        else:
            raise ValueError("Input must be a dictionary, class instance, or list of class instances.")

    @classmethod
    def _preprocess(cls, target_df, sample_df, numeric_features, categorical_features):
        """
        Preprocess the source and sample DataFrames to prepare for KNN.

        Parameters:
        df1 (pd.DataFrame): Source DataFrame.
        sample_df (pd.DataFrame): Sample DataFrame.
        numeric_features (list): List of numeric feature names.
        categorical_features (list): List of categorical feature names.
        key_field (str): The name of the key field to drop from the DataFrame.

        Returns:
        tuple: Processed source DataFrame, processed sample DataFrame, list of features.
        """
        # Select only the relevant features
        # Use .loc to ensure no SettingWithCopyWarning
        target_df = target_df.loc[:, numeric_features + categorical_features]
        sample_df = sample_df.loc[:, numeric_features + categorical_features]

        # Scaling numeric features
        scaler = StandardScaler()
        target_df.loc[:, numeric_features] = scaler.fit_transform(target_df[numeric_features])
        sample_df.loc[:, numeric_features] = scaler.transform(sample_df[numeric_features])

        # One-hot encoding for categorical features
        out1 = pd.get_dummies(target_df, columns=categorical_features, drop_first=True)
        out2 = pd.get_dummies(sample_df, columns=categorical_features, drop_first=True)

        # Ensure the same columns after one-hot encoding
        out1, out2 = out1.align(out2, join='outer', axis=1, fill_value=0)

        return out1, out2, out1.columns.tolist()
        
    @classmethod
    def find_neighbors(cls, target_data, sample_data, numeric_features=None, categorical_features=None, n_neighbors=5, key_feature='customerCode'):
        """
        Find similar customers based on the sample data and source data.

        Parameters:
        source_data (pd.DataFrame, dict, object, or list): DataFrame containing source customer data or a list of customer object(s)/dict(s).
        sample_data (pd.DataFrame, dict, object, or list): DataFrame containing the sample customer data or a list of customer object(s)/dict(s).
        n_neighbors (int): Number of neighbors to find.
        threshold (float): Threshold for filtering similar customers.
        numeric_features (list): List of numeric feature names.
        categorical_features (list): List of categorical feature names.

        Returns:
        list: List of similar customer identifiers.
        """
        
        # Default features if not provided
        if numeric_features is None:
            numeric_features = ['age', 'totalPortfolioValue']
        if categorical_features is None:
            categorical_features = ['genderName']

        # Convert source_data and sample_data to DataFrame if they are not
        source_df = cls._to_dataframe(target_data)
        sample_df = cls._to_dataframe(sample_data)
       
        if key_feature not in sample_df.columns or key_feature not in source_df.columns:
            raise ValueError(f"The key feature '{key_feature}' must be present in both source and sample DataFrames.")
        
        source_df_processed, sample_df_processed, features = cls._preprocess(source_df, sample_df, numeric_features, categorical_features)

        # Fit KNN
        neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        neigh.fit(source_df_processed[features])

        # Find neighbors
        sample_neighbors = neigh.kneighbors(sample_df_processed[features], return_distance=False)
        
        similar_customers = []
        
        for i, neighbors in enumerate(sample_neighbors):
            customers = source_df.iloc[neighbors]
            similar_customers.extend(customers[key_feature].to_list())
        
        return similar_customers
    