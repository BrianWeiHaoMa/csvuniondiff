from collections.abc import Callable
import json
import os
import sys
import logging

import pandas as pd


class CommandOptions:
    def __init__(
            self,
            align_columns: bool | None = None,
            columns_to_use: list[str] | None = None,
            fill_null: bool | None = None,
            drop_null: bool | None = None,
            match_rows: bool | None = None,
            enable_printing: bool | None = None,
            add_save_timestamp: bool | None = None,
        ):
        self.align_columns = align_columns
        self.columns_to_use = columns_to_use
        self.fill_null = fill_null
        self.drop_null = drop_null
        self.match_rows = match_rows
        self.enable_printing = enable_printing
        self.add_save_timestamp = add_save_timestamp


class ParallelInputParser:
    def __init__(
            self, 
            left_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            right_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            input_dir: str,
            options: CommandOptions = CommandOptions(),
            left_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame] | Callable[[object], object]] | None = None,
            right_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame] | Callable[[object], object]] | None = None,
            data_save_file_extensions: list[str] | None = None,
            use_input_dir_path: bool = True,
        ):
        if type(left_input) != list:
            left_input = [left_input]
            right_input = [right_input]

        if data_save_file_extensions is None:
            data_save_file_extensions = ["csv"] * len(left_input)

        if left_trans_funcs is None:
            left_trans_funcs = [lambda x: x] * len(left_input)
        
        if right_trans_funcs is None:
            right_trans_funcs = [lambda x: x] * len(right_input)

        self.check_data(
            left_input=left_input,
            right_input=right_input,
            left_trans_funcs=left_trans_funcs,
            right_trans_funcs=right_trans_funcs,
            data_save_file_extensions=data_save_file_extensions,
        )

        self.left_input = left_input
        self.right_input = right_input
        self.input_dir = input_dir
        self.options = options
        self.left_trans_funcs = left_trans_funcs
        self.right_trans_funcs = right_trans_funcs
        self.data_save_file_extensions = data_save_file_extensions
        self.use_input_dir_path = use_input_dir_path

    def check_data(
            self,
            left_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            right_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            left_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame] | Callable[[object], object]] | None = None,
            right_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame] | Callable[[object], object]] | None = None,
            data_save_file_extensions: list[str] | None = None,
        ) -> None:
        if len(left_input) != len(right_input):
            raise ValueError(f"The number of elements in left_input and right_input should be the same ({len(left_input)} != {len(right_input)})")

        if len(data_save_file_extensions) != len(left_input):
            raise ValueError(f"The number of elements in data_save_file_extensions should be the same as the number of elements in left_input ({len(data_save_file_extensions)} != {len(left_input)})")
        
        if len(left_trans_funcs) != len(left_input):
            raise ValueError(f"The number of elements in left_trans_funcs should be the same as the number of elements in left_input ({len(left_trans_funcs)} != {len(left_input)})")
        
        if len(right_trans_funcs) != len(right_input):
            raise ValueError(f"The number of elements in right_trans_funcs should be the same as the number of elements in right_input ({len(right_trans_funcs)} != {len(right_input)})")
        
    def get_names(
            self,
            left_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            right_input: list[str | pd.DataFrame] | str | pd.DataFrame,
        ) -> tuple[list[str], list[str]]:
        left_names = [
            self._get_file_name(input_) 
            if type(input_) == str 
            else f"df_{i}_{input_.shape}" for i, input_ in enumerate(left_input)
        ]
        right_names = [
            self._get_file_name(input_) 
            if type(input_) == str 
            else f"df_{i}_{input_.shape}" for i, input_ in enumerate(right_input)
        ]
        return left_names, right_names
    
    def change_inputs_to_dfs(
            self, 
            first_input: list[str | pd.DataFrame], 
            use_input_dir_path: bool,
            drop_null: bool,
            fill_null: bool,
            input_dir_path_override: str = None,
            **kwargs,
        ) -> tuple[list[pd.DataFrame], ...]:
        all_inputs = [first_input] + [input_ for input_ in kwargs.values()]

        for input_ in all_inputs:
            for i, val in enumerate(input_):
                if type(val) == str:
                    if use_input_dir_path:
                        if input_dir_path_override is None:
                            input_dir_path = self.input_dir
                        else:
                            input_dir_path = input_dir_path_override
                        val = os.path.join(input_dir_path, val)

                    if val.endswith(".csv"):
                        input_[i] = pd.read_csv(val)    
                    elif val.endswith(".xlsx"):
                        input_[i] = pd.read_excel(val)
                    else:
                        raise ValueError(f"File type not supported: {val}")
                    
                    if drop_null:
                        input_[i] = input_[i].dropna()
                    if fill_null:
                        input_[i] = input_[i].fillna("NULL")
                    
        return tuple(all_inputs)

    def parse_input(
            self
        ) -> tuple:        
        left_names, right_names = self.get_names(
            self.left_input, 
            self.right_input,
        )

        left_dfs, right_dfs = self.change_inputs_to_dfs(
            self.left_input, 
            right_input=self.right_input,
            use_input_dir_path=self.use_input_dir_path,
            drop_null=self.options.drop_null,
            fill_null=self.options.fill_null,
        )
        
        return (
            left_dfs, 
            right_dfs, 
            self.left_trans_funcs,
            self.right_trans_funcs,
            left_names, 
            right_names, 
            self.data_save_file_extensions,
        )

    def _get_file_name(
            self, 
            file_path: str,
        ) -> str:
        file_name = os.path.basename(file_path)
        return file_name


class CsvCmp:
    def __init__(
            self, 
            input_dir: str = "./",
            output_dir: str | None = "./csvcmp-data/",
        ):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def only_in(
            self, 
            left_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            right_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            options: CommandOptions = CommandOptions(),
            left_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame] | Callable[[object], object]] | None = None,
            right_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame] | Callable[[object], object]] | None = None,
            data_save_file_extensions: list[str] | None = None,
            use_input_dir_path: bool = True,
            output_transformed_rows: bool = False,
        ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]: 
        LOGGER, data_save_dir_path = self._setup(
            options=options,
            command=self.only_in,
        )

        parallel_input_parser = ParallelInputParser(
            left_input=left_input,
            right_input=right_input,
            input_dir=self.input_dir,
            options=options,
            left_trans_funcs=left_trans_funcs,
            right_trans_funcs=right_trans_funcs,
            data_save_file_extensions=data_save_file_extensions,
            use_input_dir_path=use_input_dir_path,
        )

        (
            left_dfs, 
            right_dfs, 
            left_trans_funcs,
            right_trans_funcs,
            left_file_names, 
            right_file_names, 
            data_save_file_extensions,
        ) = parallel_input_parser.parse_input()

        def _get_left_and_right_only_ind(
                left_df: pd.DataFrame, 
                right_df: pd.DataFrame,
            ) -> tuple[pd.Index, pd.Index]:
            columns = list(left_df.columns)

            left_df_ind = left_df.reset_index(names="_left_index")
            right_df_ind = right_df.reset_index(names="_right_index")

            outer_df = pd.merge(
                left_df_ind, 
                right_df_ind, 
                how="outer", 
                on=columns, 
                indicator=True,
            )

            left_index_series = outer_df[
                outer_df["_merge"] == "left_only"
            ]["_left_index"]
            right_index_series = outer_df[
                outer_df["_merge"] == "right_only"
            ]["_right_index"]
            
            return pd.Index(left_index_series), pd.Index(right_index_series)

        left_only_results = []
        right_only_results = []

        for i in range(len(left_dfs)):
            left_df = left_dfs[i] 
            right_df = right_dfs[i] 
            left_trans_func = left_trans_funcs[i]
            right_trans_func = right_trans_funcs[i]
            left_file_name = left_file_names[i]
            right_file_name = right_file_names[i]
            data_save_file_extension = data_save_file_extensions[i]

            left_df_trans = left_trans_func(left_df)
            right_df_trans = right_trans_func(right_df)

            left_data_save_file_name = (
                f"{i}_only_in_" 
                + self._get_file_name_no_extension(left_file_name)
                + f".{data_save_file_extension}"
            )
            right_data_save_file_name = (
                f"{i}_only_in_" 
                + self._get_file_name_no_extension(right_file_name)
                + f".{data_save_file_extension}"
            )

            LOGGER.info(f"For index {i}\n")

            if options.match_rows:
                left_value_counts = left_df_trans.value_counts()
                right_value_counts = right_df_trans.value_counts()

                shared = left_value_counts.index.intersection(right_value_counts.index)

                shared_left_sub_right = left_value_counts[shared] - right_value_counts[shared]
                shared_right_sub_left = right_value_counts[shared] - left_value_counts[shared]

                left_extra = shared_left_sub_right[shared_left_sub_right > 0]
                right_extra = shared_right_sub_left[shared_right_sub_left > 0]
            
                left_to_add_indices = pd.Index([], dtype=int)
                for ind, count in list(pd.DataFrame(left_extra).itertuples(index=True)):
                    ind = left_df_trans[left_df_trans.isin(ind)].dropna().tail(count).index
                    left_to_add_indices = left_to_add_indices.append(ind)

                right_to_add_indices = pd.Index([], dtype=int)
                for ind, count in list(pd.DataFrame(right_extra).itertuples(index=True)):
                    ind = right_df_trans[right_df_trans.isin(ind)].dropna().tail(count).index
                    right_to_add_indices = right_to_add_indices.append(ind)

                left_only_ind, right_only_ind = _get_left_and_right_only_ind(left_df_trans, right_df_trans)

                left_only_final_ind = left_only_ind.append(left_to_add_indices)
                right_only_final_ind = right_only_ind.append(right_to_add_indices)
            else:
                left_only_final_ind, right_only_final_ind = _get_left_and_right_only_ind(left_df_trans, right_df_trans)

            if output_transformed_rows:
                left_only_final = left_df.iloc[left_only_final_ind].sort_index()
                right_only_final = right_df.iloc[right_only_final_ind].sort_index()
            else:
                left_only_final = left_df_trans.iloc[left_only_final_ind].sort_index()
                right_only_final = right_df_trans.iloc[right_only_final_ind].sort_index()

            def _only_in_format(
                name: str, 
                df: pd.DataFrame,
            ) -> str:
                return f"Only in {name} {df.shape}:\n{df.head(50)}\n"

            LOGGER.info(_only_in_format(left_file_name, left_only_final))
            LOGGER.info(_only_in_format(right_file_name, right_only_final))
            LOGGER.info("")

            if self.output_dir is not None:
                self._output_to_file(
                    output_dir_path=data_save_dir_path,
                    file_name=left_data_save_file_name,
                    df=left_only_final, 
                    logger=LOGGER,
                )
                self._output_to_file(
                    output_dir_path=data_save_dir_path,
                    file_name=right_data_save_file_name,
                    df=right_only_final, 
                    logger=LOGGER,
                )

            left_only_results.append(left_only_final)
            right_only_results.append(right_only_final)
        return left_only_results, right_only_results
    
    def _setup(
        self,
        options: CommandOptions,
        command: Callable,
    ) -> tuple[logging.Logger, str | None]:
        if self.output_dir is not None:
            data_save_dir_path = self._make_output_dir(
                command.__name__, 
                add_timestamp=options.add_save_timestamp,
            )
        else: 
            data_save_dir_path = None

        LOGGER = self._get_logger(
            enable_printing=options.enable_printing,
            output_dir=data_save_dir_path,
            output_file_name=f"{command.__name__}.log",
            logger_name=command.__name__,
        )
        return LOGGER, data_save_dir_path
    
    def _get_logger(
            self,
            enable_printing: bool,
            output_dir: str | None,
            output_file_name: str,
            logger_name: str,
        ) -> logging.Logger:
        logger = logging.Logger(logger_name, logging.DEBUG)

        if enable_printing:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(logging.DEBUG)
            logger.addHandler(stream_handler)
        
        if output_dir is not None:
            file_handler = logging.FileHandler(os.path.join(output_dir, output_file_name))
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        
        return logger

    def _get_file_extension(
            self, 
            file_path: str,
        ) -> str:
        file_name = os.path.basename(file_path)
        arr = file_name.split(".")
        if len(arr) <= 1 or arr[-1] == "":
            raise ValueError(f"File path {file_path} does not have an extension")
        return arr[-1]
    
    def _get_file_name_no_extension(
            self, 
            file_path: str,
        ) -> str:
        file_name = os.path.basename(file_path)
        file_name_parts = file_name.split(".")[:-1]
        return ".".join(file_name_parts)
    
    def _make_output_dir(
        self,
        save_sub_folder: str,
        add_timestamp: bool = True,
    ) -> str:
        if add_timestamp:
            data_save_dir_path = os.path.join(
                self.output_dir, 
                f"{save_sub_folder}{os.sep}{pd.Timestamp.now().strftime('%Y-%m-%d-%H%M%S')}",
            )
        else:
            data_save_dir_path = os.path.join(self.output_dir, f"{save_sub_folder}{os.sep}results")

        os.makedirs(data_save_dir_path, exist_ok=True)
        return data_save_dir_path
        
    def _output_to_file(
        self, 
        output_dir_path: str,
        file_name: str,
        df: pd.DataFrame,
        logger: logging.Logger,
    ) -> None:
        full_file_path = os.path.join(output_dir_path, file_name)

        logger.info(f"Outputting to {full_file_path}")
        if full_file_path.endswith(".csv"):
            df.to_csv(full_file_path, index=False)
        elif full_file_path.endswith(".xlsx"):
            df.to_excel(full_file_path, index=False)
        else:
            raise ValueError(f"File type not supported: {full_file_path}")

    def _get_function_info_string(
            self, 
            func_name: str, 
            all_locals: dict,
        ) -> str:
        all_locals.pop("self")
        for key, val in all_locals.items():
            all_locals[key] = str(val)
        return f"{func_name}()\n{json.dumps(all_locals, indent=4)}\n"


if __name__ == "__main__":
    obj = CsvCmp("./src/tests/test-data/testset-1/")
    obj.only_in(
        left_input=["test1.csv"],
        right_input=["test2.csv"],
        options=CommandOptions(
            match_rows=False, 
            enable_printing=False, 
            add_save_timestamp=True,
        ),
    )