import os
from unittest import TestCase

from csvcmp.command import CommandLineParser


class CommandLineParserTest(TestCase):
    def test_parse_only_in(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        only_in = command_line_parser.parse_only_in()
        self.assertEqual(only_in, ["test1.csv", "test2.csv"])

        with self.assertRaises(SystemExit, msg="exactly 2 arguments allowed"):
            command_line_parser = CommandLineParser("--only-in test1.csv".split())
            only_in = command_line_parser.parse_only_in()

        with self.assertRaises(SystemExit, msg="exactly 2 arguments allowed"):
            command_line_parser = CommandLineParser("--only-in test1.csv test2.csv test3.csv".split())
            only_in = command_line_parser.parse_only_in()

        with self.assertRaises(SystemExit):
            command_line_parser = CommandLineParser("--only-in".split())
            only_in = command_line_parser.parse_only_in()

    def test_parse_intersection(self):
        command_line_parser = CommandLineParser("--intersection test1.csv test2.csv".split())
        intersection = command_line_parser.parse_intersection()
        self.assertEqual(intersection, ["test1.csv", "test2.csv"])

        with self.assertRaises(SystemExit, msg="exactly 2 arguments allowed"):
            command_line_parser = CommandLineParser("--intersection test1.csv".split())
            intersection = command_line_parser.parse_intersection()

        with self.assertRaises(SystemExit, msg="exactly 2 arguments allowed"):
            command_line_parser = CommandLineParser("--intersection test1.csv test2.csv test3.csv".split())
            intersection = command_line_parser.parse_intersection()

        with self.assertRaises(SystemExit):
            command_line_parser = CommandLineParser("--intersection".split())
            intersection = command_line_parser.parse_intersection()

    def test_parse_align_columns(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --align-columns".split())
        align_columns = command_line_parser.parse_align_columns()
        self.assertEqual(align_columns, True)

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        align_columns = command_line_parser.parse_align_columns()
        self.assertEqual(align_columns, False)

    def test_parse_columns_to_use(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --columns-to-use a b c".split())
        columns_to_use = command_line_parser.parse_columns_to_use()
        self.assertEqual(columns_to_use, ["a", "b", "c"])

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --columns-to-use".split())
        columns_to_use = command_line_parser.parse_columns_to_use()
        self.assertEqual(columns_to_use, [])

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        columns_to_use = command_line_parser.parse_columns_to_use()
        self.assertEqual(columns_to_use, [])

    def test_parse_columns_to_ignore(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --columns-to-ignore a b c".split())
        columns_to_ignore = command_line_parser.parse_columns_to_ignore()
        self.assertEqual(columns_to_ignore, ["a", "b", "c"])

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --columns-to-ignore".split())
        columns_to_ignore = command_line_parser.parse_columns_to_ignore()
        self.assertEqual(columns_to_ignore, [])

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        columns_to_ignore = command_line_parser.parse_columns_to_ignore()
        self.assertEqual(columns_to_ignore, [])

    def test_parse_fill_null(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --fill-null 0".split())
        fill_null = command_line_parser.parse_fill_null()
        self.assertEqual(fill_null, "0")

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        fill_null = command_line_parser.parse_fill_null()
        self.assertEqual(fill_null, None)

    def test_parse_drop_null(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --drop-null".split())
        drop_null = command_line_parser.parse_drop_null()
        self.assertEqual(drop_null, True)

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        drop_null = command_line_parser.parse_drop_null()
        self.assertEqual(drop_null, False)

    def test_parse_drop_duplicates(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --drop-duplicates".split())
        drop_duplicates = command_line_parser.parse_drop_duplicates()
        self.assertEqual(drop_duplicates, True)

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        drop_duplicates = command_line_parser.parse_drop_duplicates()
        self.assertEqual(drop_duplicates, False)

    def test_parse_input_dir(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --input-dir test/".split())
        input_dir = command_line_parser.parse_input_dir()
        self.assertEqual(input_dir, "test/")

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        input_dir = command_line_parser.parse_input_dir()
        self.assertEqual(input_dir, f"{os.sep}")
    
    def test_parse_output_dir(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --output-dir test/".split())
        output_dir = command_line_parser.parse_output_dir()
        self.assertEqual(output_dir, "test/")

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        output_dir = command_line_parser.parse_output_dir()
        self.assertEqual(output_dir, None)
    
    def test_parse_match_rows(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --match-rows".split())
        match_rows = command_line_parser.parse_match_rows()
        self.assertEqual(match_rows, True)

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        match_rows = command_line_parser.parse_match_rows()
        self.assertEqual(match_rows, False)