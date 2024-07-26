import pytest

from csvuniondiff.src.command import CommandLineParser


class TestCommandLineParser:
    def test_parse_version(self):
        command_line_parser = CommandLineParser("--version".split())
        version = command_line_parser.parse_version()
        assert version == True

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        version = command_line_parser.parse_version()
        assert version == False

    def test_parse_diff(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        only_in = command_line_parser.parse_diff()
        assert only_in == ["test1.csv", "test2.csv"]

        with pytest.raises(SystemExit):
            command_line_parser = CommandLineParser("--diff test1.csv".split())
            only_in = command_line_parser.parse_diff()

        with pytest.raises(SystemExit):
            command_line_parser = CommandLineParser("--diff test1.csv test2.csv test3.csv".split())
            only_in = command_line_parser.parse_diff()

        with pytest.raises(SystemExit):
            command_line_parser = CommandLineParser("--diff".split())
            only_in = command_line_parser.parse_diff()

    def test_parse_union(self):
        command_line_parser = CommandLineParser("--union test1.csv test2.csv".split())
        intersection = command_line_parser.parse_union()
        assert intersection == ["test1.csv", "test2.csv"]

        with pytest.raises(SystemExit):
            command_line_parser = CommandLineParser("--union test1.csv".split())
            intersection = command_line_parser.parse_union()

        with pytest.raises(SystemExit):
            command_line_parser = CommandLineParser("--union test1.csv test2.csv test3.csv".split())
            intersection = command_line_parser.parse_union()

        with pytest.raises(SystemExit):
            command_line_parser = CommandLineParser("--union".split())
            intersection = command_line_parser.parse_union()

    def test_parse_align_columns(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --align-columns".split())
        align_columns = command_line_parser.parse_align_columns()
        assert align_columns == True

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        align_columns = command_line_parser.parse_align_columns()
        assert align_columns == False

    def test_parse_columns_to_use(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --use-columns a b c".split())
        columns_to_use = command_line_parser.parse_use_columns()
        assert columns_to_use == ["a", "b", "c"]

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --use-columns".split())
        columns_to_use = command_line_parser.parse_use_columns()
        assert columns_to_use == []

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        columns_to_use = command_line_parser.parse_use_columns()
        assert columns_to_use is None

    def test_parse_columns_to_ignore(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --ignore-columns a b c".split())
        columns_to_ignore = command_line_parser.parse_ignore_columns()
        assert columns_to_ignore == ["a", "b", "c"]

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --ignore-columns".split())
        columns_to_ignore = command_line_parser.parse_ignore_columns()
        assert columns_to_ignore == []

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        columns_to_ignore = command_line_parser.parse_ignore_columns()
        assert columns_to_ignore is None

    def test_parse_fill_null(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --fill-null 0".split())
        fill_null = command_line_parser.parse_fill_null()
        assert fill_null == "0"

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        fill_null = command_line_parser.parse_fill_null()
        assert fill_null is None

    def test_parse_drop_null(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --drop-null".split())
        drop_null = command_line_parser.parse_drop_null()
        assert drop_null == True

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        drop_null = command_line_parser.parse_drop_null()
        assert drop_null == False

    def test_parse_drop_duplicates(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --drop-duplicates".split())
        drop_duplicates = command_line_parser.parse_drop_duplicates()
        assert drop_duplicates == True

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        drop_duplicates = command_line_parser.parse_drop_duplicates()
        assert drop_duplicates == False

    def test_parse_input_dir(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --input-dir test/".split())
        input_dir = command_line_parser.parse_input_dir()
        assert input_dir == "test/"

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        input_dir = command_line_parser.parse_input_dir()
        assert input_dir is None
    
    def test_parse_output_dir(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --output-dir test/".split())
        output_dir = command_line_parser.parse_output_dir()
        assert output_dir == "test/"

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        output_dir = command_line_parser.parse_output_dir()
        assert output_dir is None
    
    def test_parse_match_rows(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --match-rows".split())
        match_rows = command_line_parser.parse_match_rows()
        assert match_rows == True

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        match_rows = command_line_parser.parse_match_rows()
        assert match_rows == False

    def test_parse_keep_columns(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --keep-columns a b c".split())
        keep_columns = command_line_parser.parse_keep_columns()
        assert keep_columns == ["a", "b", "c"]

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --keep-columns".split())
        keep_columns = command_line_parser.parse_keep_columns()
        assert keep_columns == []

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        keep_columns = command_line_parser.parse_keep_columns()
        assert keep_columns is None

    def test_parse_use_common_columns(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --use-common-columns".split())
        use_common_columns = command_line_parser.parse_use_common_columns()
        assert use_common_columns == True

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        use_common_columns = command_line_parser.parse_use_common_columns()
        assert use_common_columns == False

    def test_parse_dont_add_timestamp(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --dont-add-timestamp".split())
        dont_add_timestamp = command_line_parser.parse_dont_add_timestamp()
        assert dont_add_timestamp == True

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        dont_add_timestamp = command_line_parser.parse_dont_add_timestamp()
        assert dont_add_timestamp == False
    
    def test_parse_disable_printing(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --disable-printing".split())
        disable_printing = command_line_parser.parse_disable_printing()
        assert disable_printing == True

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        disable_printing = command_line_parser.parse_disable_printing()
        assert disable_printing == False

    def test_parse_print_prepared(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --print-prepared".split())
        print_prepared = command_line_parser.parse_print_prepared()
        assert print_prepared == True

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        print_prepared = command_line_parser.parse_print_prepared()
        assert print_prepared == False
    
    def test_parse_save_file_extension(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --save-file-extension .csv".split())
        save_file_extension = command_line_parser.parse_save_file_extension()
        assert save_file_extension == ".csv"

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        save_file_extension = command_line_parser.parse_save_file_extension()
        assert save_file_extension == "csv"

    def test_parse_row_counts(self):
        command_line_parser = CommandLineParser("--diff test1.csv test2.csv --row-counts".split())
        row_counts = command_line_parser.parse_row_counts()
        assert row_counts == True

        command_line_parser = CommandLineParser("--diff test1.csv test2.csv".split())
        row_counts = command_line_parser.parse_row_counts()
        assert row_counts == False
        