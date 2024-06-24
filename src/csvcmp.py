from collections.abc import Callable
import json
import os
import sys
import logging

import pandas as pd
import numpy as np


def change_inputs_to_dfs(
        first_input: list[str | pd.DataFrame], 
        drop_null: bool = False,
        fill_null: str | None = "NULL",
        input_dir: str | None = None,
        **kwargs,
    ) -> tuple[list[pd.DataFrame], ...]:
    all_inputs = [first_input] + [input_ for input_ in kwargs.values()]

    for input_ in all_inputs:
        for i, val in enumerate(input_):
            if type(val) == str:
                if input_dir:
                    val = os.path.join(input_dir, val)

                if val.endswith(".csv"):
                    input_[i] = pd.read_csv(val)    
                elif val.endswith(".xlsx"):
                    input_[i] = pd.read_excel(val)
                elif val.endswith(".json"):
                    input_[i] = pd.read_json(val)
                elif val.endswith(".xml"):
                    input_[i] = pd.read_xml(val)
                elif val.endswith(".html"):
                    input_[i] = pd.read_html(val)[0]
                else:
                    raise ValueError(f"File type not supported: {val}")
                
                if drop_null:
                    input_[i] = input_[i].dropna()
                if fill_null is not None:
                    input_[i] = input_[i].fillna(fill_null)
                
    return tuple(all_inputs)


class CommandOptions:
    def __init__(
            self,
            align_columns: bool | None = None,
            columns_to_use: list[str] | None = None,
            columns_to_ignore: list[str] | None = None,
            fill_null: bool | None = None,
            drop_null: bool | None = None,
            match_rows: bool | None = None,
            enable_printing: bool | None = None,
            add_save_timestamp: bool | None = None,
            drop_duplicates: bool | None = None,
        ):
        self.align_columns = align_columns
        self.columns_to_use = columns_to_use
        self.columns_to_ignore = columns_to_ignore
        self.fill_null = fill_null
        self.drop_null = drop_null
        self.match_rows = match_rows
        self.enable_printing = enable_printing
        self.add_save_timestamp = add_save_timestamp
        self.drop_duplicates = drop_duplicates

        if columns_to_ignore is not None and columns_to_use is not None:
            raise ValueError("Cannot have both columns_to_use and columns_to_ignore set")

    def __str__(self) -> str:
        return str(self.__dict__)


class ParallelInput:
    def __init__(
            self, 
            left_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            right_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            input_dir: str,
            options: CommandOptions = CommandOptions(),
            left_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame] | Callable[[object], object]] | None = None,
            right_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame] | Callable[[object], object]] | None = None,
            data_save_file_extensions: list[str] | None = None,
        ):
        if type(left_input) != list:
            left_input = [left_input]
            right_input = [right_input]
        if len(left_input) != len(right_input):
            raise ValueError(f"The number of elements in left_input and right_input should be the same ({len(left_input)} != {len(right_input)})")

        if data_save_file_extensions is None:
            data_save_file_extensions = ["csv"] * len(left_input)
        if len(data_save_file_extensions) != len(left_input):
            raise ValueError(f"The number of elements in data_save_file_extensions should be the same as the number of elements in left_input ({len(data_save_file_extensions)} != {len(left_input)})")

        if left_trans_funcs is None:
            left_trans_funcs = [lambda x: x] * len(left_input)
        if len(left_trans_funcs) != len(left_input):
            raise ValueError(f"The number of elements in left_trans_funcs should be the same as the number of elements in left_input ({len(left_trans_funcs)} != {len(left_input)})")
        
        if right_trans_funcs is None:
            right_trans_funcs = [lambda x: x] * len(right_input)
        if len(right_trans_funcs) != len(right_input):
            raise ValueError(f"The number of elements in right_trans_funcs should be the same as the number of elements in right_input ({len(right_trans_funcs)} != {len(right_input)})")
        
        self.length = len(left_input)
        self.index = 0

        self.options = options
        
        self.left_names = self.get_names(left_input)
        self.right_names = self.get_names(right_input)

        self.left_dfs, self.right_dfs = change_inputs_to_dfs(
            first_input=left_input, 
            right_input=right_input,
            drop_null=options.drop_null,
            fill_null=options.fill_null,
            input_dir=input_dir,
        )

        self.left_dfs_trans = [left_trans_funcs[i](self.left_dfs[i]) for i in range(self.length)]
        self.right_dfs_trans = [right_trans_funcs[i](self.right_dfs[i]) for i in range(self.length)]

        self.data_save_file_extensions = data_save_file_extensions
        
    def get_names(
            self,
            inputs: list[str | pd.DataFrame] | str | pd.DataFrame,
        ) -> list[str]:
        names = [
            self.get_file_name(input_) 
            if type(input_) == str 
            else f"df_{i}_{input_.shape}" for i, input_ in enumerate(inputs)
        ]
        return names

    def get_file_name(
            self, 
            file_path: str,
        ) -> str:
        file_name = os.path.basename(file_path)
        return file_name
    
    def align_columns(
            self,
            left_df: pd.DataFrame,
            right_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        common_columns = left_df.columns.intersection(right_df.columns, sort=True)
        left_only_columns = left_df.columns.difference(common_columns)
        right_only_columns = right_df.columns.difference(common_columns)
        left_df = left_df[common_columns.append(left_only_columns)]
        right_df = right_df[common_columns.append(right_only_columns)]
        # left_df = left_df.reindex(columns=common_columns.append(left_only_columns))
        # right_df = right_df.reindex(columns=common_columns.append(right_only_columns))
        return left_df, right_df
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < self.length:
            left_df = self.left_dfs[self.index]
            right_df = self.right_dfs[self.index]
            left_df_trans = self.left_dfs_trans[self.index]
            right_df_trans = self.right_dfs_trans[self.index]
            left_name = self.left_names[self.index]
            right_name = self.right_names[self.index]
            data_save_file_extension = self.data_save_file_extensions[self.index]

            if self.options.columns_to_use is not None:
                columns_to_use = pd.Index(self.options.columns_to_use)
            elif self.options.columns_to_ignore is not None:
                columns_to_ignore = pd.Index(self.options.columns_to_ignore)
                
                left_columns_to_use = left_df_trans.columns.difference(columns_to_ignore)
                right_columns_to_use = right_df_trans.columns.difference(columns_to_ignore)
                if left_columns_to_use.difference(right_columns_to_use).size > 0:
                    raise ValueError(f"Final left and right dfs do not both have the given columns")
                
                columns_to_use = left_columns_to_use
            else:
                columns_to_use = left_df_trans.columns.intersection(right_df_trans.columns)
            
            if self.options.align_columns:
                left_df, right_df = self.align_columns(left_df, right_df)
                left_df_trans, right_df_trans = self.align_columns(left_df_trans, right_df_trans)

            old_index = self.index
            self.index += 1
                
            return (
                old_index,
                left_df,
                right_df,
                left_df_trans,
                right_df_trans,
                columns_to_use,
                left_name,
                right_name,
                data_save_file_extension,
            )
        raise StopIteration        


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
            output_transformed_rows: bool = False,
        ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]: 
        LOGGER, data_save_dir_path = self._setup(
            options=options,
            command=self.only_in,
            local_vars=locals(),
        )

        def _get_left_and_right_only_ind(
                left_df: pd.DataFrame, 
                right_df: pd.DataFrame,
                columns_to_use: pd.Index,
            ) -> tuple[pd.Index, pd.Index]:
            left_df_ind = left_df.reset_index(names="_left_index")
            right_df_ind = right_df.reset_index(names="_right_index")

            outer_df = pd.merge(
                left_df_ind, 
                right_df_ind, 
                how="outer", 
                on=columns_to_use.tolist(), 
                indicator=True,
            )

            left_index_series = outer_df[outer_df["_merge"] == "left_only"]["_left_index"].astype(int)
            right_index_series = outer_df[outer_df["_merge"] == "right_only"]["_right_index"].astype(int)
            
            return pd.Index(left_index_series), pd.Index(right_index_series)

        left_only_results = []
        right_only_results = []

        for (
            i,
            left_df,
            right_df,
            left_df_trans,
            right_df_trans,
            columns_to_use,
            left_name,
            right_name,
            data_save_file_extension,
        ) in ParallelInput(
            left_input=left_input,
            right_input=right_input,
            input_dir=self.input_dir,
            options=options,
            left_trans_funcs=left_trans_funcs,
            right_trans_funcs=right_trans_funcs,
            data_save_file_extensions=data_save_file_extensions,
        ):
            left_data_save_file_name = (
                f"{i}_only_in_" 
                + self._get_file_name_no_extension(left_name)
                + f".{data_save_file_extension}"
            )
            right_data_save_file_name = (
                f"{i}_only_in_" 
                + self._get_file_name_no_extension(right_name)
                + f".{data_save_file_extension}"
            )

            LOGGER.info(f"For index {i}\n")

            if options.match_rows:
                left_value_counts = left_df_trans.value_counts(subset=columns_to_use.to_list())
                right_value_counts = right_df_trans.value_counts(subset=columns_to_use.to_list())

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

                left_only_ind, right_only_ind = _get_left_and_right_only_ind(left_df_trans, right_df_trans, columns_to_use)

                left_only_final_ind = left_only_ind.append(left_to_add_indices)
                right_only_final_ind = right_only_ind.append(right_to_add_indices)
            else:
                left_only_final_ind, right_only_final_ind = _get_left_and_right_only_ind(left_df_trans, right_df_trans, columns_to_use)

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
            
            if options.drop_duplicates:
                left_only_final = left_only_final.drop_duplicates(keep="first")
                right_only_final = right_only_final.drop_duplicates(keep="first")

            LOGGER.info(_only_in_format(left_name, left_only_final))
            LOGGER.info(_only_in_format(right_name, right_only_final))
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
    
    def intersection(
            self, 
            left_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            right_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            options: CommandOptions = CommandOptions(),
            left_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame] | Callable[[object], object]] | None = None,
            right_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame] | Callable[[object], object]] | None = None,
            data_save_file_extensions: list[str] | None = None,
            output_transformed_rows: bool = False,
        ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]: 
        LOGGER, data_save_dir_path = self._setup(
            options=options,
            command=self.intersection,
            local_vars=locals(),
        )

        left_results = []
        right_results = []

        for (
            i,
            left_df,
            right_df,
            left_df_trans,
            right_df_trans,
            columns_to_use,
            left_name,
            right_name,
            data_save_file_extension,
        ) in ParallelInput(
            left_input=left_input,
            right_input=right_input,
            input_dir=self.input_dir,
            options=options,
            left_trans_funcs=left_trans_funcs,
            right_trans_funcs=right_trans_funcs,
            data_save_file_extensions=data_save_file_extensions,
        ):
            left_data_save_file_name = (
                f"{i}_intersecting_" 
                + self._get_file_name_no_extension(left_name)
                + f".{data_save_file_extension}"
            )
            right_data_save_file_name = (
                f"{i}_intersecting_" 
                + self._get_file_name_no_extension(right_name)
                + f".{data_save_file_extension}"
            )

            LOGGER.info(f"For index {i}\n")

            if options.match_rows:
                left_value_counts = left_df_trans.value_counts(sort=False, subset=columns_to_use.to_list())
                right_value_counts = right_df_trans.value_counts(sort=False, subset=columns_to_use.to_list())

                shared = left_value_counts.index.intersection(right_value_counts.index)

                min_of_shared = np.minimum(left_value_counts[shared], right_value_counts[shared])
            
                left_final_ind = pd.Index([], dtype=int)
                for ind, count in list(pd.DataFrame(min_of_shared).itertuples(index=True)):
                    ind = left_df_trans[left_df_trans.isin(ind)].dropna().head(count).index
                    left_final_ind = left_final_ind.append(ind)

                right_final_ind = pd.Index([], dtype=int)
                for ind, count in list(pd.DataFrame(min_of_shared).itertuples(index=True)):
                    ind = right_df_trans[right_df_trans.isin(ind)].dropna().head(count).index
                    right_final_ind = right_final_ind.append(ind)
            else:
                left_df_trans_ind = left_df_trans.reset_index(names="_left_index")
                right_df_trans_ind = right_df_trans.reset_index(names="_right_index")

                merged_df = pd.merge(
                    left_df_trans_ind, 
                    right_df_trans_ind, 
                    how="inner", 
                    on=columns_to_use.to_list(),
                )

                left_final_ind = pd.Index(merged_df["_left_index"].drop_duplicates())
                right_final_ind = pd.Index(merged_df["_right_index"].drop_duplicates())
                
            if output_transformed_rows:
                left_final = left_df.iloc[left_final_ind].sort_index()
                right_final = right_df.iloc[right_final_ind].sort_index()
            else:
                left_final = left_df_trans.iloc[left_final_ind].sort_index()
                right_final = right_df_trans.iloc[right_final_ind].sort_index()

            def _intersection_format(
                name: str, 
                df: pd.DataFrame,
            ) -> str:
                return f"Intersecting rows from {name} {df.shape}:\n{df.head(50)}\n"
            
            if options.drop_duplicates:
                left_final = left_final.drop_duplicates(keep="first")
                right_final = right_final.drop_duplicates(keep="first")

            LOGGER.info(_intersection_format(left_name, left_final))
            LOGGER.info(_intersection_format(right_name, right_final))
            LOGGER.info(f"")

            if self.output_dir is not None:
                self._output_to_file(
                    output_dir_path=data_save_dir_path,
                    file_name=left_data_save_file_name,
                    df=left_final, 
                    logger=LOGGER,
                )
                self._output_to_file(
                    output_dir_path=data_save_dir_path,
                    file_name=right_data_save_file_name,
                    df=right_final, 
                    logger=LOGGER,
                )

            left_results.append(left_final)
            right_results.append(right_final)
        return left_results, right_results

    def _setup(
        self,
        options: CommandOptions,
        command: Callable,
        local_vars: dict,
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

        LOGGER.info(
            self._get_function_info_string(
                command.__name__, 
                local_vars,
            )
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
        elif full_file_path.endswith(".json"):
            df.to_json(full_file_path, orient="records", indent=4, index=False)
        elif full_file_path.endswith(".xml"):
            df.to_xml(full_file_path, index=False)
        elif full_file_path.endswith(".html"):
            df.to_html(full_file_path, index=False)
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
    # obj = CsvCmp("./src/tests/test-data/only-in/testset-1/")
    # obj.only_in(
    #     left_input=["test1.csv"],
    #     right_input=["test2.csv"],
    #     options=CommandOptions(
    #         match_rows=False, 
    #         enable_printing=False, 
    #         add_save_timestamp=True,
    #     ),
    # )

    # obj = CsvCmp("./src/tests/test-data/intersection/testset-1/")
    # obj.intersection(
    #     left_input=["test1.csv"],
    #     right_input=["test2.csv"],
    #     options=CommandOptions(
    #         match_rows=False, 
    #         enable_printing=True, 
    #         add_save_timestamp=True,
    #         drop_duplicates=True,
    #     ),
    # )
    
    obj = CsvCmp("./src/tests/test-data/random/")
    obj.only_in(
        left_input=["test2.csv"],
        right_input=["test3.csv"],
        options=CommandOptions(
            match_rows=True, 
            enable_printing=True, 
            add_save_timestamp=True,
            align_columns=True,
        ),
    )