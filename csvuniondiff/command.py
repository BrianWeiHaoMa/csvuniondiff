import os
import sys
from argparse import ArgumentParser

from csvuniondiff.csvuniondiff import CSVUnionDiff, ParallelInputArgs, CommandOptions


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
        self.check_args()
    
    def get_argument_parser(self) -> ArgumentParser:
        argument_parser = ArgumentParser(description="csvcmp command line parser")

        argument_parser.add_argument("--diff", nargs=2, help="use the diff command, takes 2 files as arguments")
        argument_parser.add_argument("--union", nargs=2, help="use the union command, takes 2 files as arguments")

        argument_parser.add_argument("--align-columns", action="count", help="if column names are not aligned in csv, but both have same column names, realigns the column names to match")
        argument_parser.add_argument("--use-columns", default=None, nargs="*", help="only use these columns for comparison")
        argument_parser.add_argument("--ignore-columns", default=None, nargs="*", help="do not use these columns for comparison")
        argument_parser.add_argument("--fill-null", nargs="?", const="NULL", type=str, help="fills null with 'NULL' so that they can be compared")
        argument_parser.add_argument("--drop-null", action="count", help="drop rows with nulls")
        argument_parser.add_argument("--drop-duplicates", action="count", help="drop duplicates")
        argument_parser.add_argument("--input-dir", default=f"{os.sep}", type=str, help="use this dir as the base for the path to the files")
        argument_parser.add_argument("--output-dir", type=str, help="use this dir as the base for the outputs of the script")
        argument_parser.add_argument("--match-rows", action="count", help="use the match rows method with the command")
        argument_parser.add_argument("--keep-columns", default=None, nargs="*", help="keep only these columns in the output")
        argument_parser.add_argument("--use-common-columns", action="count", help="use the largest set of common columns for comparison and ignores those that are not common")
        argument_parser.add_argument("--dont-add-timestamp", action="count", help="don't add a timestamp when outputting files")
        argument_parser.add_argument("--disable-printing", action="count", help="disable printing to stdout")

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
            raise ValueError("Please provide only one command")
    
    def check_for_one_or_no_use_columns(self):
        if self.args.use_common_columns is not None and self.args.use_columns is not None:
            raise ValueError("Please provide only one of use-common-columns or use-columns")
    
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
        align_columns = self.args.align_columns
        return True if align_columns is not None else False
    
    def parse_use_columns(self) -> list[str] | None:
        columns_to_use = self.args.use_columns
        return columns_to_use
    
    def parse_ignore_columns(self) -> list[str] | None:
        columns_to_ignore = self.args.ignore_columns
        return columns_to_ignore
    
    def parse_fill_null(self) -> str:
        fill_null = self.args.fill_null
        return fill_null

    def parse_drop_null(self) -> bool:
        drop_null = self.args.drop_null
        return True if drop_null is not None else False
    
    def parse_drop_duplicates(self) -> bool:
        drop_duplicates = self.args.drop_duplicates
        return True if drop_duplicates is not None else False
    
    def parse_input_dir(self) -> str:
        input_dir = self.args.input_dir
        return input_dir
    
    def parse_output_dir(self) -> str | None:
        output_dir = self.args.output_dir
        return output_dir
    
    def parse_match_rows(self) -> bool:
        match_rows = self.args.match_rows
        return True if match_rows is not None else False
    
    def parse_keep_columns(self) -> list[str] | None:
        keep_columns = self.args.keep_columns
        return keep_columns
    
    def parse_use_common_columns(self) -> bool:
        use_common_columns = self.args.use_common_columns
        return True if use_common_columns is not None else False
    
    def parse_dont_add_timestamp(self) -> bool:
        dont_add_timestamp = self.args.dont_add_timestamp
        return True if dont_add_timestamp is not None else False
    
    def parse_disable_printing(self) -> bool:
        disable_printing = self.args.disable_printing
        return True if disable_printing is not None else False
    

def main():
    command_line_parser = CommandLineParser(args=sys.argv[1:])

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

    commands_count = 0
    if diff is not None:
        commands_count += 1
    if union is not None:
        commands_count += 1
    if commands_count != 1:
        raise ValueError("Please provide exactly one command")

    if input_dir is None:
        input_dir = f".{os.sep}"

    csvcmp = CSVUnionDiff(
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
    )

    if diff is not None:
        csvcmp.diff(
            ParallelInputArgs(
                left_input=[diff[0]],
                right_input=[diff[1]],
                left_trans_funcs=[],
                right_trans_funcs=[],
                data_save_file_extensions=[],
                output_transformed_rows=False,
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
                data_save_file_extensions=[],
                output_transformed_rows=False,
            ),
            options=options,
        )


if __name__ == "__main__":
    main()
    