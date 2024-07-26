from unittest import TestCase

import pandas as pd
import pytest

from csvuniondiff.src.csvuniondiff import (
    CsvUnionDiff, 
    CommandOptions,
    ParallelInput,
    ParallelInputArgs,
    change_inputs_to_dfs,
)

TEST_DATA_FOLDER_PATH = "csvuniondiff/dev/tests/test-data/"


class TestPublicFunctions:
    def test_change_inputs_to_dfs(self):
        test_set_folder = f"{TEST_DATA_FOLDER_PATH}random/"

        tmp_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [6, 7, 8, 9, 10],
            'C': [11, 12, 13, 14, 15],
            'D': [16, 17, 18, 19, 20],
            'E': [21, 22, 23, 24, 25]
        })

        test1_df = pd.read_csv(f"{test_set_folder}test1.csv")
        test2_df = pd.read_csv(f"{test_set_folder}test2.csv")
        test3_df = pd.read_csv(f"{test_set_folder}test3.csv")

        first_dfs, second_dfs, third_dfs = change_inputs_to_dfs(
            first_input=["test1.csv"],
            second_input=["test2.csv"],
            third_input=["test3.csv", tmp_df],
            input_dir=test_set_folder,
        )

        assert test1_df.equals(first_dfs[0])
        assert test2_df.equals(second_dfs[0])
        assert test3_df.equals(third_dfs[0])
        assert tmp_df.equals(third_dfs[1])
        
        test4_df = pd.read_csv(f"{test_set_folder}test4.csv")
        test4_df_nulls_filled = test4_df.fillna("NULL")
        
        fourth_dfs = change_inputs_to_dfs(
            first_input=["test4.csv"],
            input_dir=test_set_folder,
            fill_null="NULL",
        )

        assert test4_df_nulls_filled.equals(fourth_dfs[0])

        test4_df = pd.read_csv(f"{test_set_folder}test4.csv")
        test4_df_nulls_dropped = test4_df.dropna()
        
        fourth_dfs = change_inputs_to_dfs(
            first_input=["test4.csv"],
            input_dir=test_set_folder,
            drop_null=True,
        )

        assert test4_df_nulls_dropped.equals(fourth_dfs[0])


class TestCommandOptions:
    def test_check_args(self):
        with pytest.raises(ValueError):
            CommandOptions(
                use_columns=["column3", "column4"],
                ignore_columns=["column3"],
            )

        with pytest.raises(ValueError):
            CommandOptions(
                use_common_columns=True, 
                use_columns=["column3", "column4"],
            )

        with pytest.raises(ValueError):
            CommandOptions(
                use_common_columns=True, 
                ignore_columns=["column3"],
            )


class TestDiff:
    test_set_folder = f"{TEST_DATA_FOLDER_PATH}diff/"

    def test_match_rows_true(self):
        obj = CsvUnionDiff(f"{self.test_set_folder}testset-1/", None)
        parallel_input_args = ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"],
        )
        options = CommandOptions(match_rows=True, enable_printing=False)
        left_dfs, right_dfs = obj.diff(
            parallel_input_args, 
            options,
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

        assert left_df.equals(expected_left_df)
        assert right_df.equals(expected_right_df)
        
    def test_match_rows_false(self):
        obj = CsvUnionDiff(f"{self.test_set_folder}testset-1/", None)
        parallel_input_args = ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"], 
        )
        options = CommandOptions(match_rows=False, enable_printing=False)
        left_dfs, right_dfs = obj.diff(
            parallel_input_args, 
            options,
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

        assert left_df.equals(expected_left_df)
        assert right_df.equals(expected_right_df)

    def test_match_rows_false_row_count(self):
        obj = CsvUnionDiff(f"{self.test_set_folder}testset-1/", None)
        parallel_input_args = ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"], 
        )
        options = CommandOptions(
            match_rows=False, 
            enable_printing=False,
            return_row_counts=True,
        )
        left_dfs, right_dfs = obj.diff(
            parallel_input_args, 
            options,
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'Name': ['Michael Wilson', 'Bob Thompson', 'Emily Davis'],
            'Age': [32, 35, 27],
            'Email': ['michaelwilson@example.com', 'bobthompson@example.com', 'emilydavis@example.com'],
            'count': [3, 1, 1]
        }, index=[0, 1, 2])

        expected_right_df = pd.DataFrame({
            'Name': ['Emily Davis__1', 'John Smith__1', 'Michael Johnson__1'],
            'Age': [27, 35, 32],
            'Email': ['emilydavis@example.com', 'johnsmith@example.com', 'michaeljohnson@example.com'],
            'count': [1, 1, 1]
        }, index=[0, 1, 2])

        assert left_df.equals(expected_left_df)
        assert right_df.equals(expected_right_df)
    
    def test_match_rows_true_transform(self):
        obj = CsvUnionDiff(f"{self.test_set_folder}testset-2/", None)

        def left_df_trans(df: pd.DataFrame) -> pd.DataFrame:
            def email_trans(row):
                arr = row["Email"].split("@")
                return arr[0] + str(row["Age"]) + "@" + arr[1]
            df["Email"] = df.apply(email_trans, axis=1)

            df = df[["Name", "Email", "Age"]]
            return df

        parallel_input_args = ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"], 
            left_trans_funcs=[left_df_trans],
            right_trans_funcs=[lambda x: x],
        )
        options = CommandOptions(
            match_rows=True, 
            enable_printing=False,
            return_transformed_rows=False,    
        )
        left_dfs, right_dfs = obj.diff(
            parallel_input_args,
            options,
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'Name': ['Michael Brown', 'Sarah Wilson', 'David Thompson'],
            'Age': [28, 32, 45],
            'Email': ['michaelbrown@example.com', 'sarahwilson@example.com', 'davidthompson@example.com'],
        }, index=[4, 5, 6])

        expected_right_df = pd.DataFrame({
            'Name': ['Brian Harris'],
            'Email': ['brianharris33@example.com'],
            'Age': [33],
        }, index=[7])

        assert left_df.equals(expected_left_df)
        assert right_df.equals(expected_right_df)

    def test_match_rows_true_align_columns(self):
        obj = CsvUnionDiff(f"{self.test_set_folder}testset-4/", None)
        parallel_input_args = ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"], 
        )
        options = CommandOptions(
            match_rows=True, 
            align_columns=True, 
            enable_printing=False,
            use_common_columns=True,
        )
        left_dfs, right_dfs = obj.diff(
            parallel_input_args,
            options,
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'Age': [32, 32, 35, 27, 32],
            'Email': ['michaelwilson@example.com', 'michaelwilson@example.com', 'bobthompson@example.com', 'emilydavis@example.com', 'michaelwilson@example.com'],
            'Name': ['Michael Wilson', 'Michael Wilson', 'Bob Thompson', 'Emily Davis', 'Michael Wilson'],
        }, index=[3, 4, 5, 6, 7])

        expected_right_df = pd.DataFrame({
            'Age': [25, 30, 35, 32, 27],
            'Email': ['johndoe@example.com', 'janesmith@example.com', 'johnsmith@example.com', 'michaeljohnson@example.com', 'emilydavis@example.com'],
            'Name': ['John Doe', 'Jane Smith', 'John Smith__1', 'Michael Johnson__1', 'Emily Davis__1'],
            'tmp': [0] * 5,
        }, index=[1, 3, 6, 7, 8])

        assert left_df.equals(expected_left_df)
        assert right_df.equals(expected_right_df)

    def test_keep_columns(self):
        obj = CsvUnionDiff(f"{self.test_set_folder}testset-4/", None)
        parallel_input_args = ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"], 
        )
        options = CommandOptions(
            match_rows=True, 
            align_columns=True, 
            enable_printing=False,
            use_common_columns=True,
            keep_columns=["Name", "Age"],
        )
        left_dfs, right_dfs = obj.diff(
            parallel_input_args,
            options,
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'Name': ['Michael Wilson', 'Michael Wilson', 'Bob Thompson', 'Emily Davis', 'Michael Wilson'],
            'Age': [32, 32, 35, 27, 32],
        }, index=[3, 4, 5, 6, 7])

        expected_right_df = pd.DataFrame({
            'Name': ['John Doe', 'Jane Smith', 'John Smith__1', 'Michael Johnson__1', 'Emily Davis__1'],
            'Age': [25, 30, 35, 32, 27],
        }, index=[1, 3, 6, 7, 8])

        assert left_df.equals(expected_left_df)
        assert right_df.equals(expected_right_df)


class TestUnion:
    test_set_folder = f"{TEST_DATA_FOLDER_PATH}union/"

    def test_match_rows_true(self):
        obj = CsvUnionDiff(f"{self.test_set_folder}testset-1", None)
        parallel_input_args = ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"], 
        )
        options = CommandOptions(match_rows=True, enable_printing=False)
        left_dfs, right_dfs = obj.union(
            parallel_input_args,
            options,
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

        assert left_df.equals(expected_left_df)
        assert right_df.equals(expected_right_df)

    def test_match_rows_false(self):
        obj = CsvUnionDiff(f"{self.test_set_folder}testset-1", None)
        parallel_input_args = ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"], 
        )
        options = CommandOptions(match_rows=False, enable_printing=False)
        left_dfs, right_dfs = obj.union(
            parallel_input_args,
            options,
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

        assert left_df.equals(expected_left_df)
        assert right_df.equals(expected_right_df)
    
    def test_keep_columns(self):
        obj = CsvUnionDiff(f"{self.test_set_folder}testset-1", None)
        parallel_input_args = ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"], 
        )
        options = CommandOptions(
            match_rows=False, 
            enable_printing=False,
            keep_columns=["City"],
        )
        left_dfs, right_dfs = obj.union(
            parallel_input_args,
            options,
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'City': ['New York', 'Chicago', 'San Francisco', 'Boston', 'Seattle', 'Seattle', 'Seattle', 'Seattle', 'Houston', 'Seattle', 'Seattle', 'Seattle', 'Seattle', 'Miami', 'Dallas', 'Denver', 'Phoenix', 'Austin', 'Portland', 'Portland', 'Portland', 'Portland', 'San Diego', 'Nashville', 'Philadelphia', 'Orlando', 'Washington D.C.', 'Washington D.C.', 'Washington D.C.', 'Washington D.C.']
        }, index=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32])

        expected_right_df = pd.DataFrame({
            'City': ['New York', 'Chicago', 'San Francisco', 'Boston', 'Seattle', 'Seattle', 'Houston', 'Miami', 'Dallas', 'Denver', 'Phoenix', 'Austin', 'Portland', 'San Diego', 'Nashville', 'Philadelphia', 'Orlando', 'Washington D.C.']
        }, index=[0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19])

        assert left_df.equals(expected_left_df)
        assert right_df.equals(expected_right_df)

    def test_keep_columns_row_counts(self):
        obj = CsvUnionDiff(f"{self.test_set_folder}testset-1", None)
        parallel_input_args = ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"], 
        )
        options = CommandOptions(
            match_rows=False, 
            enable_printing=False,
            keep_columns=["City"],
            return_row_counts=True,
        )
        left_dfs, right_dfs = obj.union(
            parallel_input_args,
            options,
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'City': ['Seattle', 'Washington D.C.', 'Portland', 'Austin', 'Boston', 'Houston', 'Chicago', 'Dallas', 'Denver', 'New York', 'Nashville', 'Miami', 'Orlando', 'Phoenix', 'Philadelphia', 'San Francisco', 'San Diego'],
            'count': [8, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

        expected_right_df = pd.DataFrame({
            'City': ['Seattle', 'Austin', 'Boston', 'Dallas', 'Chicago', 'Houston', 'Miami', 'Nashville', 'Denver', 'New York', 'Orlando', 'Phoenix', 'Philadelphia', 'Portland', 'San Diego', 'San Francisco', 'Washington D.C.'],
            'count': [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }, index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

        assert left_df.equals(expected_left_df)
        assert right_df.equals(expected_right_df)


class TestParallelInput:
    test_set_folder = f"{TEST_DATA_FOLDER_PATH}random/"

    def test_use_columns(self):
        obj = ParallelInput(
            ParallelInputArgs(
                left_input=["test1.csv"], 
                right_input=["test1.csv"], 
            ),
            input_dir=f"{self.test_set_folder}",
            options=CommandOptions(use_columns=["column3", "column4"]),
        )

        for p_data in obj:
            assert p_data.columns_to_use.equals(pd.Index(["column3", "column4"]))

    def test_ignore_columns(self):
        obj = ParallelInput(
            ParallelInputArgs(
                left_input=["test1.csv"], 
                right_input=["test1.csv"], 
            ),
            input_dir=f"{self.test_set_folder}",
            options=CommandOptions(ignore_columns=["column3", "column4"]),
        )    

        for p_data in obj:
            assert p_data.columns_to_use.equals(pd.Index(["column1", "column2", "column5"]))

    def test_use_columns_and_ignore(self):
         with pytest.raises(ValueError):
             CommandOptions(use_columns=["column3", "column4"], ignore_columns=["column3"])

    def test_align_columns(self):
        obj = ParallelInput(
            ParallelInputArgs(
                left_input=["test2.csv"], 
                right_input=["test3.csv"],    
            ),
            input_dir=f"{self.test_set_folder}",
            options=CommandOptions(align_columns=True),
        )

        with pytest.raises(ValueError):
            for p_data in obj:
                pass

        obj = ParallelInput(
            ParallelInputArgs(
                left_input=["test2.csv"], 
                right_input=["test3.csv"],    
            ),
            input_dir=f"{self.test_set_folder}",
            options=CommandOptions(align_columns=True, use_common_columns=True),
        )

        for p_data in obj:
            assert p_data.columns_to_use.equals(pd.Index(["column1", "column2", "column3", "column4"]))
            assert p_data.left_df.columns.equals(pd.Index(["column1", "column2", "column3", "column4", "column7", "column5", "column6"]))
            assert p_data.right_df.columns.equals(pd.Index(["column1", "column2", "column3", "column4"]))
