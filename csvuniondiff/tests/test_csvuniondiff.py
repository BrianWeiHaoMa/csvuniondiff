from unittest import TestCase

import pandas as pd

from csvuniondiff.csvuniondiff import (
    CsvUnionDiff, 
    CommandOptions,
    ParallelInput,
    ParallelInputArgs,
    change_inputs_to_dfs,
)


class PublicFunctionsTest(TestCase):
    def test_change_inputs_to_dfs(self):
        test_set_folder = "./csvuniondiff/tests/test-data/random/"

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

        self.assertTrue(test1_df.equals(first_dfs[0]))
        self.assertTrue(test2_df.equals(second_dfs[0]))
        self.assertTrue(test3_df.equals(third_dfs[0]))
        self.assertTrue(tmp_df.equals(third_dfs[1]))
        
        test4_df = pd.read_csv(f"{test_set_folder}test4.csv")
        test4_df_nulls_filled = test4_df.fillna("NULL")
        
        fourth_dfs = change_inputs_to_dfs(
            first_input=["test4.csv"],
            input_dir=test_set_folder,
            fill_null="NULL",
        )

        self.assertTrue(test4_df_nulls_filled.equals(fourth_dfs[0]))

        test4_df = pd.read_csv(f"{test_set_folder}test4.csv")
        test4_df_nulls_dropped = test4_df.dropna()
        
        fourth_dfs = change_inputs_to_dfs(
            first_input=["test4.csv"],
            input_dir=test_set_folder,
            drop_null=True,
        )

        self.assertTrue(test4_df_nulls_dropped.equals(fourth_dfs[0]))


class CommandOptionsTest(TestCase):
    def test_check_args(self):
        with self.assertRaises(ValueError):
            CommandOptions(
                use_columns=["column3", "column4"],
                ignore_columns=["column3"],
            )

        with self.assertRaises(ValueError):
            CommandOptions(
                use_common_columns=True, 
                use_columns=["column3", "column4"],
            )

        with self.assertRaises(ValueError):
            CommandOptions(
                use_common_columns=True, 
                ignore_columns=["column3"],
            )


class DiffTest(TestCase):
    test_set_folder = f"./csvuniondiff/tests/test-data/diff/"

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

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))
        
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

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))
    
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
        options = CommandOptions(match_rows=True, enable_printing=False)
        left_dfs, right_dfs = obj.diff(
            parallel_input_args,
            options,
        )

        left_df = left_dfs[0]
        right_df = right_dfs[0]

        expected_left_df = pd.DataFrame({
            'Name': ['Michael Brown', 'Sarah Wilson', 'David Thompson'],
            'Email': ['michaelbrown28@example.com', 'sarahwilson32@example.com', 'davidthompson45@example.com'],
            'Age': [28, 32, 45],
        }, index=[4, 5, 6])

        expected_right_df = pd.DataFrame({
            'Name': ['Brian Harris'],
            'Email': ['brianharris33@example.com'],
            'Age': [33],
        }, index=[7])

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))

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

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))

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

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))


class UnionTest(TestCase):
    test_set_folder = f"./csvuniondiff/tests/test-data/union/"

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

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))

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

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))
    
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

        self.assertTrue(left_df.equals(expected_left_df))
        self.assertTrue(right_df.equals(expected_right_df))


class ParallelInputTest(TestCase):
    test_set_folder = f"./csvuniondiff/tests/test-data/random/"

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
            self.assertTrue(p_data.columns_to_use.equals(pd.Index(["column3", "column4"])))

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
            self.assertTrue(p_data.columns_to_use.equals(pd.Index(["column1", "column2", "column5"])))

    def test_use_columns_and_ignore(self):
         with self.assertRaises(ValueError):
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

        with self.assertRaises(ValueError):
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
            self.assertTrue(p_data.columns_to_use.equals(pd.Index(["column1", "column2", "column3", "column4"])))
            self.assertTrue(p_data.left_df.columns.equals(pd.Index(["column1", "column2", "column3", "column4", "column7", "column5", "column6"])))
            self.assertTrue(p_data.right_df.columns.equals(pd.Index(["column1", "column2", "column3", "column4"])))
