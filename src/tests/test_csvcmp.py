from unittest import TestCase

import pandas as pd

from src.csvcmp import CsvCmp


class CsvCmpTestCase(TestCase):
    def test_compare_row_existence_match_rows_true(self):
        obj = CsvCmp("./test-data/testset-1/")
        left_dfs, right_dfs = obj.compare_row_existence(["test1.csv"], ["test2.csv"], match_rows=True, enable_printing=False)

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

        self.assertEqual(left_df.equals(expected_left_df), True)
        self.assertEqual(right_df.equals(expected_right_df), True)

    def test_compare_row_existence_match_rows_false(self):
        obj = CsvCmp("./test-data/testset-1/")
        left_dfs, right_dfs = obj.compare_row_existence(["test1.csv"], ["test2.csv"], match_rows=False, enable_printing=False)

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

        self.assertEqual(left_df.equals(expected_left_df), True)
        self.assertEqual(right_df.equals(expected_right_df), True)

