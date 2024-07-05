import os
import sys
from argparse import ArgumentParser

from csvcmp.csvcmp import CSVCmp, ParallelInputArgs, CommandOptions


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
    
    def get_argument_parser(self):
        argument_parser = ArgumentParser(description="csvcmp command line parser")

        argument_parser.add_argument("--only-in", nargs=2, help="use the only-in command, takes 2 files as arguments")
        argument_parser.add_argument("--intersection", nargs=2, help="use the intersection command, takes 2 files as arguments")

        argument_parser.add_argument("--align-columns", action="count", help="if column names are not aligned in csv, but both have same column names, realigns the column names to match")
        argument_parser.add_argument("--columns-to-use", default=None, nargs="+", help="only use these columns for comparison")
        argument_parser.add_argument("--columns-to-ignore", default=None, nargs="+", help="do not use these columns for comparison")
        argument_parser.add_argument("--fill-null", nargs="?", const="NULL", type=str, help="fills null with 'NULL' so that they can be compared")
        argument_parser.add_argument("--drop-null", action="count", help="drop rows with nulls")
        argument_parser.add_argument("--drop-duplicates", action="count", help="drop duplicates")
        argument_parser.add_argument("--input-dir", default=f"{os.sep}", type=str, help="use this dir as the base for the path to the files")
        argument_parser.add_argument("--output-dir", type=str, help="use this dir as the base for the outputs of the script")
        argument_parser.add_argument("--match-rows", action="count", help="use the match rows method with the command")

        return argument_parser
    
    def check_args(self):
        self.check_for_exactly_one_command()

    def check_for_exactly_one_command(self):
        cnt = 0
        if self.args.only_in is not None:
            cnt += 1
        if self.args.intersection is not None:
            cnt += 1
        
        if cnt != 1:
            raise ValueError("Please provide only one command")
    
    def parse_only_in(self):
        only_in = self.args.only_in
        if only_in is not None:
            check_for_two_files(
                only_in,
                "only-in"
            )
        return only_in

    def parse_intersection(self):
        intersection = self.args.intersection
        if intersection is not None:
            check_for_two_files(
                intersection,
                "intersection"
            )
        return intersection
    
    def parse_align_columns(self) -> bool:
        align_columns = self.args.align_columns
        return True if align_columns is not None else False
    
    def parse_columns_to_use(self) -> list[str]:
        columns_to_use = self.args.columns_to_use
        return columns_to_use
    
    def parse_columns_to_ignore(self) -> list[str]:
        columns_to_ignore = self.args.columns_to_ignore
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
    

if __name__ == "__main__":
    command_line_parser = CommandLineParser(args=sys.argv[1:])

    only_in = command_line_parser.parse_only_in()
    intersection = command_line_parser.parse_intersection()
    align_columns = command_line_parser.parse_align_columns()
    columns_to_use = command_line_parser.parse_columns_to_use()
    columns_to_ignore = command_line_parser.parse_columns_to_ignore()
    fill_null = command_line_parser.parse_fill_null()
    drop_null = command_line_parser.parse_drop_null()
    drop_duplicates = command_line_parser.parse_drop_duplicates()
    input_dir = command_line_parser.parse_input_dir()
    output_dir = command_line_parser.parse_output_dir()
    match_rows = command_line_parser.parse_match_rows()

    commands_count = 0
    if only_in is not None:
        commands_count += 1
    if intersection is not None:
        commands_count += 1
    if commands_count != 1:
        raise ValueError("Please provide exactly one command")

    if input_dir is None:
        input_dir = f".{os.sep}"

    csvcmp = CSVCmp(
        input_dir=input_dir, 
        output_dir=output_dir,
    )

    options = CommandOptions(
        align_columns=align_columns,
        columns_to_use=columns_to_use,
        columns_to_ignore=columns_to_ignore,
        fill_null=fill_null,
        drop_null=drop_null,
        match_rows=match_rows,
        enable_printing=True,
        add_save_timestamp=True,
        drop_duplicates=drop_duplicates,
    )

    if only_in is not None:
        csvcmp.only_in(
            ParallelInputArgs(
                left_input=[only_in[0]],
                right_input=[only_in[1]],
                left_trans_funcs=[],
                right_trans_funcs=[],
                data_save_file_extensions=[],
                output_transformed_rows=False,
            ),
            options=options,
        )

    if intersection is not None:
        csvcmp.intersection(
            ParallelInputArgs(
                left_input=[intersection[0]],
                right_input=[intersection[1]],
                left_trans_funcs=[],
                right_trans_funcs=[],
                data_save_file_extensions=[],
                output_transformed_rows=False,
            ),
            options=options,
        )

