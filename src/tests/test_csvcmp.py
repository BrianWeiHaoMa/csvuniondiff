from unittest import TestCase

import pandas as pd

from src.csvcmp import (
    CSVCmp, 
    CommandOptions,
    ParallelInput,
    ParallelInputArgs,
)


class CommandRes:
    def __init__(
        self, 
        obj: CSVCmp,
        left_df: pd.DataFrame,  
        right_df: pd.DataFrame, 
        left_expected_df: pd.DataFrame,
        right_expected_df: pd.DataFrame,       
    ):
        pass


class OnlyInTest(TestCase):
    test_set_folder = f"./src/tests/test-data/only-in/"

    def test_match_rows_true(self):
        self._test_match_rows_true_helper(
            input_dir=f"{self.test_set_folder}testset-1/",
            left_csv="test1.csv",
            right_csv="test2.csv",
        )
        self._test_match_rows_true_helper(
            input_dir=f"{self.test_set_folder}testset-3/",
            left_csv="test1.xlsx",
            right_csv="test2.xlsx",
        )

    def test_match_rows_false(self):
        self._test_match_rows_false_helper(
            input_dir=f"{self.test_set_folder}testset-1/",
            left_xlsx="test1.csv",
            right_xlsx="test2.csv",
        )
        self._test_match_rows_false_helper(
            input_dir=f"{self.test_set_folder}testset-3/",
            left_xlsx="test1.xlsx",
            right_xlsx="test2.xlsx",
        )

    def _test_match_rows_true_transform_helper(self):
        obj = CSVCmp(f"{self.test_set_folder}testset-2/", None)

        def left_df_trans(df: pd.DataFrame) -> pd.DataFrame:
            def email_trans(row):
                arr = row["Email"].split("@")
                return arr[0] + str(row["Age"]) + "@" + arr[1]
            df["Email"] = df.apply(email_trans, axis=1)

            df.reindex(columns=["Name", "Email", "Age"])
            return df

        left_dfs, right_dfs = obj.only_in(
            ParallelInputArgs(
                ["test1.csv"], 
                ["test2.csv"], 
                left_trans_funcs=[left_df_trans],
                right_trans_funcs=[lambda x: x],
            ),
            CommandOptions(match_rows=True, enable_printing=False),
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'Name': ['Michael Brown', 'Sarah Wilson', 'David Thompson'],
            'Age': [28, 32, 45],
            'Email': ['michaelbrown28@example.com', 'sarahwilson32@example.com', 'davidthompson45@example.com']
        }, index=[4, 5, 6])

        expected_right_df = pd.DataFrame({
            'Name': ['Brian Harris'],
            'Email': ['brianharris33@example.com'],
            'Age': [33]
        }, index=[7])

        return left_df, right_df, expected_left_df, expected_right_df

    def test_match_rows_true_transform(self):
        left_df, right_df, expected_left_df, expected_right_df = self._test_match_rows_true_transform_helper()
        
        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))

    def _test_match_rows_true_helper(
            self, 
            input_dir: str, 
            left_csv: str, 
            right_csv: str
        ):
        obj = CSVCmp(input_dir, None)
        left_dfs, right_dfs = obj.only_in(
            ParallelInputArgs(
                [left_csv], 
                [right_csv],
            ), 
            CommandOptions(match_rows=True, enable_printing=False),
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'Name': ['Michael Wilson', 'Michael Wilson', 'Bob Thompson', 'Emily Davis', 'Michael Wilson'],
            'Age': [32, 32, 35, 27, 32],
            'Email': ['michaelwilson@example.com', 'michaelwilson@example.com', 'bobthompson@example.com', 'emilydavis@example.com', 'michaelwilson@example.com']
        }, index=[3, 4, 5, 6, 7])

        expected_right_df = pd.DataFrame({
            'Name': ['John Doe', 'Jane Smith', 'John Smith__1', 'Michael Johnson__1', 'Emily Davis__1'],
            'Age': [25, 30, 35, 32, 27],
            'Email': ['johndoe@example.com', 'janesmith@example.com', 'johnsmith@example.com', 'michaeljohnson@example.com', 'emilydavis@example.com']
        }, index=[1, 3, 6, 7, 8])

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))

    def _test_match_rows_false_helper(self, input_dir: str, left_xlsx: str, right_xlsx: str):
        obj = CSVCmp(input_dir, None)
        left_dfs, right_dfs = obj.only_in(
            ParallelInputArgs(
                [left_xlsx], 
                [right_xlsx], 
            ),
            CommandOptions(match_rows=False, enable_printing=False),
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'Name': ['Michael Wilson', 'Michael Wilson', 'Bob Thompson', 'Emily Davis', 'Michael Wilson'],
            'Age': [32, 32, 35, 27, 32],
            'Email': ['michaelwilson@example.com', 'michaelwilson@example.com', 'bobthompson@example.com', 'emilydavis@example.com', 'michaelwilson@example.com']
        }, index=[3, 4, 5, 6, 7])

        expected_right_df = pd.DataFrame({
            'Name': ['John Smith__1', 'Michael Johnson__1', 'Emily Davis__1'],
            'Age': [35, 32, 27],
            'Email': ['johnsmith@example.com', 'michaeljohnson@example.com', 'emilydavis@example.com']
        }, index=[6, 7, 8])

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))


class IntersectionTest(TestCase):
    test_set_folder = f"./src/tests/test-data/intersection/"

    def test_match_rows_true(self):
        obj = CSVCmp(f"{self.test_set_folder}testset-1", None)
        left_dfs, right_dfs = obj.intersection(
            ParallelInputArgs(
                ["test1.csv"], 
                ["test2.csv"], 
            ),
            CommandOptions(match_rows=True, enable_printing=False),
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'Name': ['John', 'Michael', 'Sophia', 'Daniel', 'Emma', 'Emma', 'Oliver', 'Ava', 'William', 'James', 'Mia', 'Benjamin', 'Charlotte', 'Alexander', 'Amelia', 'Henry', 'Jacob', 'Harper'],
            'Age': [25, 41, 28, 35, 29, 29, 37, 26, 33, 27, 34, 31, 38, 36, 39, 40, 42, 23],
            'City': ['New York', 'Chicago', 'San Francisco', 'Boston', 'Seattle', 'Seattle', 'Houston', 'Miami', 'Dallas', 'Denver', 'Phoenix', 'Austin', 'Portland', 'San Diego', 'Nashville', 'Philadelphia', 'Orlando', 'Washington D.C.']
        }, index=[0, 2, 3, 4, 5, 6, 9, 14, 15, 17, 18, 19, 20, 24, 25, 26, 28, 29])

        expected_right_df = pd.DataFrame({
            'Name': ['John', 'Michael', 'Sophia', 'Daniel', 'Emma', 'Emma', 'Oliver', 'Ava', 'William', 'James', 'Mia', 'Benjamin', 'Charlotte', 'Alexander', 'Amelia', 'Henry', 'Jacob', 'Harper'],
            'Age': [25, 41, 28, 35, 29, 29, 37, 26, 33, 27, 34, 31, 38, 36, 39, 40, 42, 23],
            'City': ['New York', 'Chicago', 'San Francisco', 'Boston', 'Seattle', 'Seattle', 'Houston', 'Miami', 'Dallas', 'Denver', 'Phoenix', 'Austin', 'Portland', 'San Diego', 'Nashville', 'Philadelphia', 'Orlando', 'Washington D.C.']
        }, index=[0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19])

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))

    def test_match_rows_false(self):
        obj = CSVCmp(f"{self.test_set_folder}testset-1", None)
        left_dfs, right_dfs = obj.intersection(
            ParallelInputArgs(
                ["test1.csv"], 
                ["test2.csv"], 
            ),
            CommandOptions(match_rows=False, enable_printing=False),
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'Name': ['John', 'Michael', 'Sophia', 'Daniel', 'Emma', 'Emma', 'Emma', 'Emma', 'Oliver', 'Emma', 'Emma', 'Emma', 'Emma', 'Ava', 'William', 'James', 'Mia', 'Benjamin', 'Charlotte', 'Charlotte', 'Charlotte', 'Charlotte', 'Alexander', 'Amelia', 'Henry', 'Jacob', 'Harper', 'Harper', 'Harper', 'Harper'],
            'Age': [25, 41, 28, 35, 29, 29, 29, 29, 37, 29, 29, 29, 29, 26, 33, 27, 34, 31, 38, 38, 38, 38, 36, 39, 40, 42, 23, 23, 23, 23],
            'City': ['New York', 'Chicago', 'San Francisco', 'Boston', 'Seattle', 'Seattle', 'Seattle', 'Seattle', 'Houston', 'Seattle', 'Seattle', 'Seattle', 'Seattle', 'Miami', 'Dallas', 'Denver', 'Phoenix', 'Austin', 'Portland', 'Portland', 'Portland', 'Portland', 'San Diego', 'Nashville', 'Philadelphia', 'Orlando', 'Washington D.C.', 'Washington D.C.', 'Washington D.C.', 'Washington D.C.']
        }, index=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32])

        expected_right_df = pd.DataFrame({
            'Name': ['John', 'Michael', 'Sophia', 'Daniel', 'Emma', 'Emma', 'Oliver', 'Ava', 'William', 'James', 'Mia', 'Benjamin', 'Charlotte', 'Alexander', 'Amelia', 'Henry', 'Jacob', 'Harper'],
            'Age': [25, 41, 28, 35, 29, 29, 37, 26, 33, 27, 34, 31, 38, 36, 39, 40, 42, 23],
            'City': ['New York', 'Chicago', 'San Francisco', 'Boston', 'Seattle', 'Seattle', 'Houston', 'Miami', 'Dallas', 'Denver', 'Phoenix', 'Austin', 'Portland', 'San Diego', 'Nashville', 'Philadelphia', 'Orlando', 'Washington D.C.']
        }, index=[0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19])

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))


class ParallelInputTest(TestCase):
    test_set_folder = f"./src/tests/test-data/random/"

    def test_columns_to_use(self):
        obj = ParallelInput(
            ParallelInputArgs(
                left_input=["test1.csv"], 
                right_input=["test1.csv"], 
            ),
            input_dir=f"{self.test_set_folder}",
            options=CommandOptions(columns_to_use=["column3", "column4"]),
        )

        for p_data in obj:
            self.assertTrue(p_data.columns_to_use.equals(pd.Index(["column3", "column4"])))

    def test_columns_to_ignore(self):
        obj = ParallelInput(
            ParallelInputArgs(
                left_input=["test1.csv"], 
                right_input=["test1.csv"], 
            ),
            input_dir=f"{self.test_set_folder}",
            options=CommandOptions(columns_to_ignore=["column3", "column4"]),
        )    

        for p_data in obj:
            self.assertTrue(p_data.columns_to_use.equals(pd.Index(["column1", "column2", "column5"])))

    def test_columns_to_use_and_ignore(self):
         with self.assertRaises(ValueError):
             CommandOptions(columns_to_use=["column3", "column4"], columns_to_ignore=["column3"])

    def test_align_columns(self):
        obj = ParallelInput(
            ParallelInputArgs(
                left_input=["test2.csv"], 
                right_input=["test3.csv"],    
            ),
            input_dir=f"{self.test_set_folder}",
            options=CommandOptions(align_columns=True),
        )

        for p_data in obj:
            self.assertTrue(p_data.columns_to_use.equals(pd.Index(["column1", "column2", "column3", "column4"])))
            self.assertTrue(p_data.left_df.columns.equals(pd.Index(["column1", "column2", "column3", "column4", "column5", "column6", "column7"])))
            self.assertTrue(p_data.right_df.columns.equals(pd.Index(["column1", "column2", "column3", "column4"])))
