from unittest import TestCase

from csvcmp.command import CommandLineParser


class CommandLineParserTest(TestCase):
    def test_parse_columns_to_use(self):
        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --columns-to-use a b c".split())
        columns_to_use = command_line_parser.parse_columns_to_use()
        self.assertEqual(columns_to_use, ["a", "b", "c"])

        command_line_parser = CommandLineParser("--only-in test1.csv test2.csv".split())
        columns_to_use = command_line_parser.parse_columns_to_use()
        self.assertEqual(columns_to_use, [])

        self.rai

        with self.assertRaises(Exception):
            command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --columns-to-use".split())
            columns_to_use = command_line_parser.parse_columns_to_use()

    # def test_parse_columns_to_use(self):
    #     command_line_parser = CommandLineParser("--only-in test1.csv test2.csv --columns-to-use a b c".split())
    #     columns_to_use = command_line_parser.parse_columns_to_use()
    #     self.assertEqual(columns_to_use, ["a", "b", "c"])