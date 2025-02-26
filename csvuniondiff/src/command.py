import os
import sys
from argparse import ArgumentParser

from .csvuniondiff import CsvUnionDiff, ParallelInputArgs, CommandOptions
from csvuniondiff import __version__


def check_for_two_files(files, option_name):
    if len(files) != 2:
        raise ValueError(f"{option_name} requires 2 files to be passed in")


class CommandLineParser:
    def __init__(
            self,
            args: list[str] = sys.argv[1:],
        ):
        argument_parser = self.get_argument_parser()
        self.args = argument_parser.parse_args(args=args)
    
    def get_argument_parser(self) -> ArgumentParser:
        argument_parser = ArgumentParser(description="csvuniondiff command line parser")

        argument_parser.add_argument("--version", action="store_true", help="print the version of this package")
        argument_parser.add_argument("--diff", nargs=2, help="use the diff command, takes 2 files as arguments")
        argument_parser.add_argument("--union", nargs=2, help="use the union command, takes 2 files as arguments")

        argument_parser.add_argument("-a", "--align-columns", action="store_true", help="aligns common columns on the left sorted")
        argument_parser.add_argument("-c", "--use-columns", default=None, nargs="*", help="only use these columns for comparison")
        argument_parser.add_argument("--ignore-columns", default=None, nargs="*", help="do not use these columns for comparison")
        argument_parser.add_argument("-f", "--fill-null", nargs="?", const="NULL", type=str, help="fills null option value so that they can be compared, default is 'NULL'")
        argument_parser.add_argument("-d", "--drop-null", action="store_true", help="drop rows with nulls")
        argument_parser.add_argument("-D", "--drop-duplicates", action="store_true", help="drop duplicate rows")
        argument_parser.add_argument("-i", "--input-dir", default=None, type=str, help="use this directory path as the base for the path to the files")
        argument_parser.add_argument("-o", "--output-dir", default=None, type=str, help="save outputs from the script to this directory")
        argument_parser.add_argument("-m", "--match-rows", action="store_true", help="use the match rows algorithm for comparison")
        argument_parser.add_argument("-k", "--keep-columns", default=None, nargs="*", help="only keep these columns in the final result")
        argument_parser.add_argument("-C", "--use-common-columns", action="store_true", help="use the maximal set of common columns for comparison")
        argument_parser.add_argument("--dont-add-timestamp", action="store_true", help="don't add a timestamp directory when outputting files")
        argument_parser.add_argument("--disable-printing", action="store_true", help="disable printing to stdout")
        argument_parser.add_argument("--print-prepared", action="store_true", help="print the prepared df before comparison")
        argument_parser.add_argument("--save-file-extension", default="csv", type=str, help="the extension for output files (csv, xlsx, json, xml, or html)")
        argument_parser.add_argument("-r", "--row-counts", action="store_true", help="use the counts of each unique row in the final result instead")

        return argument_parser
    
    def check_args(self):
        self.check_for_exactly_one_command()
        self.check_for_one_or_no_use_columns()

    def check_for_exactly_one_command(self):
        cnt = 0
        if self.args.diff is not None:
            cnt += 1
        if self.args.union is not None:
            cnt += 1
        
        if cnt != 1:
            raise ValueError("Please provide exactly one command")
    
    def check_for_one_or_no_use_columns(self):
        if self.args.use_common_columns and self.args.use_columns is not None:
            raise ValueError("Please provide only one of use-common-columns or use-columns")
        
    def parse_version(self) -> bool:
        return self.args.version
    
    def parse_diff(self) -> list[str] | None:
        diff = self.args.diff
        if diff is not None:
            check_for_two_files(
                diff,
                "diff"
            )
        return diff

    def parse_union(self) -> list[str] | None:
        union = self.args.union
        if union is not None:
            check_for_two_files(
                union,
                "union"
            )
        return union
    
    def parse_align_columns(self) -> bool:
        return self.args.align_columns
    
    def parse_use_columns(self) -> list[str] | None:
        return self.args.use_columns
    
    def parse_ignore_columns(self) -> list[str] | None:
        return self.args.ignore_columns
    
    def parse_fill_null(self) -> str:
        return self.args.fill_null

    def parse_drop_null(self) -> bool:
        return self.args.drop_null
    
    def parse_drop_duplicates(self) -> bool:
        return self.args.drop_duplicates
    
    def parse_input_dir(self) -> str:
        return self.args.input_dir
    
    def parse_output_dir(self) -> str | None:
        return self.args.output_dir
    
    def parse_match_rows(self) -> bool:
        return self.args.match_rows
            
    def parse_keep_columns(self) -> list[str] | None:
        return self.args.keep_columns
    
    def parse_use_common_columns(self) -> bool:
        return self.args.use_common_columns
    
    def parse_dont_add_timestamp(self) -> bool:
        return self.args.dont_add_timestamp
    
    def parse_disable_printing(self) -> bool:
        return self.args.disable_printing
    
    def parse_print_prepared(self) -> bool:
        return self.args.print_prepared
    
    def parse_save_file_extension(self) -> str | None:
        return self.args.save_file_extension
    
    def parse_row_counts(self) -> bool:
        return self.args.row_counts


def main():
    command_line_parser = CommandLineParser(args=sys.argv[1:])

    version = command_line_parser.parse_version()
    if version:
        print(__version__)
        exit(0)
    
    command_line_parser.check_args()

    diff = command_line_parser.parse_diff()
    union = command_line_parser.parse_union()
    align_columns = command_line_parser.parse_align_columns()
    use_columns = command_line_parser.parse_use_columns()
    ignore_columns = command_line_parser.parse_ignore_columns()
    fill_null = command_line_parser.parse_fill_null()
    drop_null = command_line_parser.parse_drop_null()
    drop_duplicates = command_line_parser.parse_drop_duplicates()
    input_dir = command_line_parser.parse_input_dir()
    output_dir = command_line_parser.parse_output_dir()
    match_rows = command_line_parser.parse_match_rows()
    keep_columns = command_line_parser.parse_keep_columns()
    use_common_columns = command_line_parser.parse_use_common_columns()
    dont_add_timestamp = command_line_parser.parse_dont_add_timestamp()
    disable_printing = command_line_parser.parse_disable_printing()
    print_prepared = command_line_parser.parse_print_prepared()
    save_file_extension = command_line_parser.parse_save_file_extension()
    row_counts = command_line_parser.parse_row_counts()

    if input_dir is None:
        input_dir = f".{os.sep}"

    csvcmp = CsvUnionDiff(
        input_dir=input_dir, 
        output_dir=output_dir,
    )

    options = CommandOptions(
        align_columns=align_columns,
        use_columns=use_columns,
        ignore_columns=ignore_columns,
        fill_null=fill_null,
        drop_null=drop_null,
        match_rows=match_rows,
        enable_printing=True if not disable_printing else False,
        add_save_timestamp=True if not dont_add_timestamp else False,
        drop_duplicates=drop_duplicates,
        keep_columns=keep_columns,
        use_common_columns=use_common_columns,
        print_prepared=print_prepared,
        print_transformed=False,
        return_transformed_rows=False,
        return_row_counts=row_counts,
    )

    if diff is not None:
        csvcmp.diff(
            ParallelInputArgs(
                left_input=[diff[0]],
                right_input=[diff[1]],
                left_trans_funcs=[],
                right_trans_funcs=[],
                data_save_file_extensions=[save_file_extension],
            ),
            options=options,
        )

    if union is not None:
        csvcmp.union(
            ParallelInputArgs(
                left_input=[union[0]],
                right_input=[union[1]],
                left_trans_funcs=[],
                right_trans_funcs=[],
                data_save_file_extensions=[save_file_extension],
            ),
            options=options,
        )

if __name__ == "__main__":
    main()
    