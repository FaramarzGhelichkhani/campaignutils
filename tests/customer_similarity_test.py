import sys
import os
import pytest
import pandas as pd

# Ensure the parent directory is in the path for importing the CustomerSimilarityFinder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from campaignutils.customer_similarity import CustomerSimilarityFinder  # Adjusted import

class Customer:
    def __init__(self, customerKey, age, totalPortfolioValue, genderName):
        self.customerKey = customerKey
        self.age = age
        self.totalPortfolioValue = totalPortfolioValue
        self.genderName = genderName

# Sample Data for Tests
source_data = [
    {'customerKey': 1, 'age': 25, 'totalPortfolioValue': 50000, 'genderName': 'Male'},
    {'customerKey': 2, 'age': 30, 'totalPortfolioValue': 60000, 'genderName': 'Female'},
    {'customerKey': 3, 'age': 22, 'totalPortfolioValue': 45000, 'genderName': 'Male'},
    {'customerKey': 4, 'age': 35, 'totalPortfolioValue': 70000, 'genderName': 'Female'},
    # {'customerKey': 5, 'age': 90, 'totalPortfolioValue': 545141, 'genderName': 'Female'},
]

# Convert the sample data into a DataFrame
source_df = pd.DataFrame(source_data)

# Create Customer instances for testing
single_customer = Customer(customerKey=10, age=28, totalPortfolioValue=55000, genderName='Male')
customer_list = [Customer(customerKey=11, age=29, totalPortfolioValue=58000, genderName='Male')]

class TestCustomerSimilarityFinder:
    
    def test_to_dataframe_single_object(self):
        df = CustomerSimilarityFinder._to_dataframe(single_customer)
        assert df.shape == (1, 4)  # 1 row, 4 columns

    def test_to_dataframe_list_of_objects(self):
        df = CustomerSimilarityFinder._to_dataframe(customer_list)
        assert df.shape == (1, 4)  # 1 row, 4 columns

    def test_to_dataframe_dict(self):
        df = CustomerSimilarityFinder._to_dataframe(source_data[0])
        assert df.shape == (1, 4)  # 1 row, 4 columns

    def test_to_dataframe_invalid_input(self):
        with pytest.raises(ValueError) as excinfo:
            CustomerSimilarityFinder._to_dataframe(123)  # Invalid input
        assert "Input must be a dictionary, class instance, or list of class instances." in str(excinfo.value)

    def test_preprocess(self):
        processed_source, _, features = CustomerSimilarityFinder._preprocess(source_df, source_df.copy(), ['age', 'totalPortfolioValue'], ['genderName'])
        assert processed_source.shape == (4, 6)  # Should have 4 rows and 6 columns after one-hot encoding (genderName has two categories)
        assert 'customerKey' not in features

    def test_find_neighbors(self):
        similar_customers = CustomerSimilarityFinder.find_neighbors(source_df, single_customer, n_neighbors=2)
        assert isinstance(similar_customers, list)
        assert len(similar_customers) > 0  # At least one similar customer should be found
        assert all(customer in source_df['customerKey'].values for customer in similar_customers)  # All found keys should exist in the source DataFrame

    def test_find_neighbors_with_list(self):
        similar_customers = CustomerSimilarityFinder.find_neighbors(source_df, customer_list, n_neighbors=2)
        assert isinstance(similar_customers, list)
        assert len(similar_customers) > 0  # At least one similar customer should be found
        assert all(customer in source_df['customerKey'].values for customer in similar_customers)  # All found keys should exist in the source DataFrame

    def test_find_neighbors_invalid_key(self):
        with pytest.raises(ValueError) as excinfo:
            CustomerSimilarityFinder.find_neighbors(source_df, {'age': 28})  # Missing customerKey
        assert "The key feature 'customerKey' must be present in both source and sample DataFrames." in str(excinfo.value)

if __name__ == "__main__":
    pytest.main()