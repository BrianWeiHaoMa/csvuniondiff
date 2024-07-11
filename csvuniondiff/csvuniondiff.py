from collections.abc import Callable
import os
import sys
import logging

import pandas as pd
import numpy as np


def change_inputs_to_dfs(
        first_input: list[str | pd.DataFrame], 
        drop_null: bool = False,
        fill_null: str | None = None,
        input_dir: str = os.getcwd(),
        **kwargs,
    ) -> list[pd.DataFrame] | tuple[list[pd.DataFrame], ...]:
    """A useful function for converting a list of strings to dataframes. 
    The strings should be file paths to files with extensions csv, xlsx, json, xml, or html.
    Dataframes originally in the input list will be left as is. Add extra lists as kwargs
    to convert more than one list.

    Args:
        first_input (list[str  |  pd.DataFrame]): A list of strings or dataframes.
        drop_null (bool, optional): Drop rows with null values. Defaults to False.
        fill_null (str | None, optional): Fill rows with a string. Defaults to None which doesn't fill.
        input_dir (str, optional): A directory path that will be appended to all file paths in the list. Defaults to os.getcwd().

    Raises:
        ValueError: Unsupported file extension.

    Returns:
        list[pd.DataFrame] | tuple[list[pd.DataFrame], ...]: The resulting list(s) of dataframes.
    """
    all_inputs = [first_input] + [input_ for input_ in kwargs.values()]

    for input_ in all_inputs:
        for i, val in enumerate(input_):
            if type(val) is str:
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
                
    if len(all_inputs) <= 1:
        return all_inputs[0]
    return tuple(all_inputs)

def _pretty_format_dict(
        func,
    ):
    def wrapper(*args, **kwargs):
        dont_show_vals = [
            None,
            False,
            [],
        ]
        data = func(*args, **kwargs)
        res = ""
        for key, val in data.items():
            if val not in dont_show_vals:
                res += f"{key}: {val}\n"
        return res
    return wrapper


class CommandOptions:
    def __init__(
            self,
            align_columns: bool = False,
            use_columns: list[str] | None = None,
            ignore_columns: list[str] | None = None,
            fill_null: str | None = None,
            drop_null: bool = False,
            match_rows: bool = True,
            enable_printing: bool = True,
            add_save_timestamp: bool = False,
            drop_duplicates: bool = False,
            keep_columns: list[str] | None = None,
            use_common_columns: bool = False,
            print_prepared: bool = False,
            print_transformed: bool = False,
        ):
        """Options that are globally applicable CsvUnionDiff commands.

        drop_null (bool, optional): Drop rows with null values. Defaults to False.
        fill_null (str | None, optional): Fill rows with a string. Defaults to None which doesn't fill.

        Args:
            align_columns (bool, optional): Align the maximal set of common columns sorted on the left. Defaults to False.
            use_columns (list[str] | None, optional): The columns to use for comparison. Defaults to None.
            ignore_columns (list[str] | None, optional): The columns to ignore for comparison. Defaults to None.
            fill_null (str | None, optional): Fill rows with a string. Defaults to None.
            drop_null (bool, optional): Drop rows with null values. Defaults to False.
            match_rows (bool, optional): Use the match rows algorithm. Defaults to True.
            enable_printing (bool, optional): Print to stdout. Defaults to True.
            add_save_timestamp (bool, optional): Add a timestamp subfolder for output files. Defaults to False.
            drop_duplicates (bool, optional): Drop duplicate rows. Defaults to False.
            keep_columns (list[str] | None, optional): Only keep these columns in the final result. Defaults to None.
            use_common_columns (bool, optional): Use the maximal set of common columns for comparison. Defaults to False.
            print_prepared (bool, optional): Print the prepared dataframes (before comparison). Defaults to False.
            print_transformed (bool, optional): Print the transformed dataframes (before comparison). Defaults to False.
        """
        self.align_columns = align_columns
        self.use_columns = use_columns
        self.ignore_columns = ignore_columns
        self.fill_null = fill_null
        self.drop_null = drop_null
        self.match_rows = match_rows
        self.enable_printing = enable_printing
        self.add_save_timestamp = add_save_timestamp
        self.drop_duplicates = drop_duplicates
        self.keep_columns = keep_columns
        self.use_common_columns = use_common_columns
        self.print_prepared = print_prepared
        self.print_transformed = print_transformed

        self.check()

    def check(self):
        if self.use_columns is not None and self.ignore_columns is not None:
            raise ValueError("Only one of use_columns and ignore_columns should be used")
        
        if self.use_columns is not None and self.use_common_columns:
            raise ValueError("Only one of use_columns and use_common_columns should be used")
        
        if self.ignore_columns is not None and self.use_common_columns:
            raise ValueError("Only one of ignore_columns and use_common_columns should be used")

    @_pretty_format_dict
    def __str__(self) -> dict:
        return self.__dict__


class ParallelInputArgs:
    def __init__(
        self, 
        left_input: list[str | pd.DataFrame],
        right_input: list[str | pd.DataFrame],
        left_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame]] = [],
        right_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame]] = [],
        data_save_file_extensions: list[str] = [],
        return_transformed_rows: bool = False,
    ):
        """Parallel input arguments for CsvUnionDiff commands where
        the left and right input lists are compared in parallel. And a result list
        of the same size as left_input and right_input is returned.

        Args:
            left_input (list[str  |  pd.DataFrame]): A list of strings or dataframes to be compared with. The strings should be file paths to files with extensions csv, xlsx, json, xml, or html.
            right_input (list[str  |  pd.DataFrame]): Same as left_input.
            left_trans_funcs (list[Callable[[pd.DataFrame], pd.DataFrame]], optional): Transformation functions for elements in left_input. Defaults to [].
            right_trans_funcs (list[Callable[[pd.DataFrame], pd.DataFrame]], optional): Transformation functions for elements in right_input. Defaults to [].
            data_save_file_extensions (list[str], optional): The extension to save. Defaults to [] which will just be csv.
            return_transformed_rows (bool, optional): Return the rows after transformation. Defaults to False.
        """
        self.left_input = left_input
        self.right_input = right_input
        self.left_trans_funcs = left_trans_funcs
        self.right_trans_funcs = right_trans_funcs
        self.data_save_file_extensions = data_save_file_extensions
        self.return_transformed_rows = return_transformed_rows

    @_pretty_format_dict
    def __str__(self):
        return self.__dict__
    

class ParallelInputRes:
    def __init__(
            self,
            index: int,
            left_df: pd.DataFrame,
            right_df: pd.DataFrame,
            left_df_trans: pd.DataFrame,
            right_df_trans: pd.DataFrame,
            columns_to_use: pd.Index,
            left_name: str,
            right_name: str,
            data_save_file_extension: str,
    ):
        self.index = index
        self.left_df = left_df
        self.right_df = right_df
        self.left_df_trans = left_df_trans
        self.right_df_trans = right_df_trans
        self.columns_to_use = columns_to_use
        self.left_name = left_name
        self.right_name = right_name
        self.data_save_file_extension = data_save_file_extension


class ParallelInput:
    def __init__(
            self, 
            data: ParallelInputArgs,
            input_dir: str,
            options: CommandOptions = CommandOptions(),
        ):
        left_input = data.left_input
        right_input = data.right_input
        left_trans_funcs = data.left_trans_funcs
        right_trans_funcs = data.right_trans_funcs
        data_save_file_extensions = data.data_save_file_extensions

        if len(left_input) != len(right_input):
            raise ValueError(f"The number of elements in left_input and right_input should be the same ({len(left_input)} != {len(right_input)})")

        if data_save_file_extensions == []:
            data_save_file_extensions = ["csv"] * len(left_input)
        if len(data_save_file_extensions) != len(left_input):
            raise ValueError(f"The number of elements in data_save_file_extensions should be the same as the number of elements in left_input ({len(data_save_file_extensions)} != {len(left_input)})")

        if left_trans_funcs == []:
            left_f: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x
            left_trans_funcs = [left_f] * len(left_input)
        if len(left_trans_funcs) != len(left_input):
            raise ValueError(f"The number of elements in left_trans_funcs should be the same as the number of elements in left_input ({len(left_trans_funcs)} != {len(left_input)})")
        
        if right_trans_funcs == []:
            right_f: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x
            right_trans_funcs = [right_f] * len(right_input)
        if len(right_trans_funcs) != len(right_input):
            raise ValueError(f"The number of elements in right_trans_funcs should be the same as the number of elements in right_input ({len(right_trans_funcs)} != {len(right_input)})")
        
        self.length = len(left_input)
        self.index = 0

        self.options = options
        
        self.left_names = self.get_names(left_input)
        self.right_names = self.get_names(right_input)

        dfs_res = change_inputs_to_dfs(
            first_input=left_input, 
            right_input=right_input,
            drop_null=options.drop_null,
            fill_null=options.fill_null,
            input_dir=input_dir,
        )

        if type(dfs_res) is tuple:
            self.left_dfs, self.right_dfs = dfs_res

        self.left_dfs_trans = [left_trans_funcs[i](self.left_dfs[i].copy()) for i in range(self.length)]
        self.right_dfs_trans = [right_trans_funcs[i](self.right_dfs[i].copy()) for i in range(self.length)]

        self.data_save_file_extensions = data_save_file_extensions
        
    def get_names(
            self,
            inputs: list[str | pd.DataFrame],
        ) -> list[str]:
        names = []
        for i, input_ in enumerate(inputs):
            if type(input_) is pd.DataFrame:
                names.append(f"df_{i}_{input_.shape}")
            elif type(input_) is str:
                names.append(self.get_file_name(input_))
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
        left_only_columns = left_df.columns.difference(common_columns, sort=False)
        right_only_columns = right_df.columns.difference(common_columns, sort=False)
        left_df = left_df[common_columns.append(left_only_columns)]
        right_df = right_df[common_columns.append(right_only_columns)]
        return left_df, right_df
    
    def __len__(self):
        return self.length

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

            if self.options.use_columns is not None:
                columns_to_use = pd.Index(self.options.use_columns)
            elif self.options.ignore_columns is not None:
                columns_to_ignore = pd.Index(self.options.ignore_columns)

                left_columns_to_use = left_df_trans.columns.difference(columns_to_ignore)
                right_columns_to_use = right_df_trans.columns.difference(columns_to_ignore)
                if left_columns_to_use.difference(right_columns_to_use).size > 0:
                    raise ValueError(f"left and right columns to use aren't the same after ignoring columns")
                
                columns_to_use = left_columns_to_use
            elif self.options.use_common_columns:
                columns_to_use = left_df_trans.columns.intersection(right_df_trans.columns)
            else:
                if left_df_trans.columns.to_list() != right_df_trans.columns.to_list():
                    raise ValueError(f"left and right columns aren't the same")
                
                columns_to_use = left_df_trans.columns
            
            if self.options.align_columns:
                left_df, right_df = self.align_columns(left_df, right_df)
                left_df_trans, right_df_trans = self.align_columns(left_df_trans, right_df_trans)

            old_index = self.index
            self.index += 1
                
            return ParallelInputRes(
                old_index,
                left_df,
                right_df,
                left_df_trans,
                right_df_trans,
                columns_to_use,
                left_name,
                right_name,
                data_save_file_extension.lstrip(".") if data_save_file_extension is not None else None,
            )
        raise StopIteration        


class CsvUnionDiff:
    def __init__(
            self, 
            input_dir: str = os.getcwd(),
            output_dir: str | None = os.path.join(os.getcwd(), "csvcmp-data"),
        ):
        """A class for comparing CSV-like files.

        Args:
            input_dir (str, optional): The directory path of the inputs which will be prepended to any input files. Defaults to os.getcwd().
            output_dir (str | None, optional): The directory path of the outputs or None if no output files are needed. Defaults to os.path.join(os.getcwd(), "csvcmp-data").
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

    def diff(
            self, 
            args: ParallelInputArgs,
            options: CommandOptions = CommandOptions(),
        ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]: 
        """Compare two sets of dataframes in parallel and return the rows that are only in the left and right dataframes.
        """
        LOGGER, data_save_dir_path = self._setup(
            options=options,
            command=self.diff,
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

            left_index_series = outer_df[outer_df["_merge"] == "left_only"]["_left_index"]
            right_index_series = outer_df[outer_df["_merge"] == "right_only"]["_right_index"]
            
            return pd.Index(left_index_series), pd.Index(right_index_series)

        left_only_results = []
        right_only_results = []

        parallel_input = ParallelInput(
            args,
            input_dir=self.input_dir,
            options=options,
        )

        for p_data in parallel_input:
            left_data_save_file_name = (
                f"({p_data.index}left)_only_in_" 
                + self._get_file_name_no_extension(p_data.left_name)
                + f".{p_data.data_save_file_extension}"
            )
            right_data_save_file_name = (
                f"({p_data.index}right)_only_in_" 
                + self._get_file_name_no_extension(p_data.right_name)
                + f".{p_data.data_save_file_extension}"
            )

            if len(parallel_input) > 1:
                LOGGER.info(f"For input pair {p_data.index}")

            if options.match_rows:
                left_value_counts = p_data.left_df_trans.value_counts(subset=p_data.columns_to_use.to_list())
                right_value_counts = p_data.right_df_trans.value_counts(subset=p_data.columns_to_use.to_list())

                shared = left_value_counts.index.intersection(right_value_counts.index)

                shared_left_sub_right = left_value_counts[shared] - right_value_counts[shared]
                shared_right_sub_left = right_value_counts[shared] - left_value_counts[shared]

                left_extra = shared_left_sub_right[shared_left_sub_right > 0]
                right_extra = shared_right_sub_left[shared_right_sub_left > 0]
            
                left_df_tmp = p_data.left_df_trans[p_data.columns_to_use.to_list()] 
                left_to_add_indices = pd.Index([], dtype=int)
                for ind, count in list(pd.DataFrame(left_extra).itertuples(index=True)):
                    ind = left_df_tmp[left_df_tmp.isin(ind)].dropna().tail(count).index
                    left_to_add_indices = left_to_add_indices.append(ind)

                right_df_tmp = p_data.right_df_trans[p_data.columns_to_use.to_list()]
                right_to_add_indices = pd.Index([], dtype=int)
                for ind, count in list(pd.DataFrame(right_extra).itertuples(index=True)):
                    ind = right_df_tmp[right_df_tmp.isin(ind)].dropna().tail(count).index
                    right_to_add_indices = right_to_add_indices.append(ind)

                left_only_ind, right_only_ind = _get_left_and_right_only_ind(p_data.left_df_trans, p_data.right_df_trans, p_data.columns_to_use)

                left_only_final_ind = left_only_ind.append(left_to_add_indices)
                right_only_final_ind = right_only_ind.append(right_to_add_indices)
            else:
                left_only_final_ind, right_only_final_ind = _get_left_and_right_only_ind(p_data.left_df_trans, p_data.right_df_trans, p_data.columns_to_use)

            if args.return_transformed_rows:
                left_only_final = p_data.left_df_trans.iloc[left_only_final_ind].sort_index()
                right_only_final = p_data.right_df_trans.iloc[right_only_final_ind].sort_index()
            else:
                left_only_final = p_data.left_df.iloc[left_only_final_ind].sort_index()
                right_only_final = p_data.right_df.iloc[right_only_final_ind].sort_index()

            if options.drop_duplicates:
                left_only_final = left_only_final.drop_duplicates(keep="first")
                right_only_final = right_only_final.drop_duplicates(keep="first")

            if options.keep_columns is not None:
                left_columns = pd.Index(options.keep_columns).intersection(left_only_final.columns)
                right_columns = pd.Index(options.keep_columns).intersection(right_only_final.columns)
                left_only_final = left_only_final[left_columns]
                right_only_final = right_only_final[right_columns]

            if options.print_prepared:
                LOGGER.info(self._get_prepared_df_string(p_data.left_name, p_data.left_df))
                LOGGER.info(self._get_prepared_df_string(p_data.right_name, p_data.right_df))

            if options.print_transformed:
                LOGGER.info(self._get_transformed_df_string(p_data.left_name, p_data.left_df_trans))
                LOGGER.info(self._get_transformed_df_string(p_data.right_name, p_data.right_df_trans))

            def _diff_format(
                name: str, 
                df: pd.DataFrame,
            ) -> str:
                return f"Only in {name} {df.shape}:\n{df}\n"

            LOGGER.info(_diff_format(p_data.left_name, left_only_final))
            LOGGER.info(_diff_format(p_data.right_name, right_only_final))

            if data_save_dir_path is not None:
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
            LOGGER.info(f"")

            left_only_results.append(left_only_final)
            right_only_results.append(right_only_final)
        return left_only_results, right_only_results
    
    def union(
            self, 
            args: ParallelInputArgs,
            options: CommandOptions = CommandOptions(),
        ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]: 
        """Compare two sets of dataframes in parallel and return the rows that are in both the left and right dataframes.
        """
        LOGGER, data_save_dir_path = self._setup(
            options=options,
            command=self.union,
            local_vars=locals(),
        )

        left_results = []
        right_results = []

        parallel_input = ParallelInput(
            args,
            input_dir=self.input_dir,
            options=options,
        )

        for p_data in parallel_input:
            left_data_save_file_name = (
                f"({p_data.index}left)_intersecting_" 
                + self._get_file_name_no_extension(p_data.left_name)
                + f".{p_data.data_save_file_extension}"
            )
            right_data_save_file_name = (
                f"({p_data.index}right)_intersecting_" 
                + self._get_file_name_no_extension(p_data.right_name)
                + f".{p_data.data_save_file_extension}"
            )

            if len(parallel_input) > 1:
                LOGGER.info(f"For input pair {p_data.index}")

            if options.match_rows:
                left_value_counts = p_data.left_df_trans.value_counts(sort=False, subset=p_data.columns_to_use.to_list())
                right_value_counts = p_data.right_df_trans.value_counts(sort=False, subset=p_data.columns_to_use.to_list())

                shared = left_value_counts.index.intersection(right_value_counts.index)

                min_of_shared = np.minimum(left_value_counts[shared], right_value_counts[shared])
            
                left_df_tmp = p_data.left_df_trans[p_data.columns_to_use.to_list()]
                left_final_ind = pd.Index([], dtype=int)
                for ind, count in list(pd.DataFrame(min_of_shared).itertuples(index=True)):
                    ind = left_df_tmp[left_df_tmp.isin(ind)].dropna().head(count).index
                    left_final_ind = left_final_ind.append(ind)

                right_df_tmp = p_data.right_df_trans[p_data.columns_to_use.to_list()]
                right_final_ind = pd.Index([], dtype=int)
                for ind, count in list(pd.DataFrame(min_of_shared).itertuples(index=True)):
                    ind = right_df_tmp[right_df_tmp.isin(ind)].dropna().head(count).index
                    right_final_ind = right_final_ind.append(ind)
            else:
                left_df_trans_ind = p_data.left_df_trans.reset_index(names="_left_index")
                right_df_trans_ind = p_data.right_df_trans.reset_index(names="_right_index")

                merged_df = pd.merge(
                    left_df_trans_ind, 
                    right_df_trans_ind, 
                    how="inner", 
                    on=p_data.columns_to_use.to_list(),
                )

                left_final_ind = pd.Index(merged_df["_left_index"].drop_duplicates())
                right_final_ind = pd.Index(merged_df["_right_index"].drop_duplicates())
                
            if args.return_transformed_rows:
                left_final = p_data.left_df_trans.iloc[left_final_ind].sort_index()
                right_final = p_data.right_df_trans.iloc[right_final_ind].sort_index()
            else:
                left_final = p_data.left_df.iloc[left_final_ind].sort_index()
                right_final = p_data.right_df.iloc[right_final_ind].sort_index()
            
            if options.drop_duplicates:
                left_final = left_final.drop_duplicates(keep="first")
                right_final = right_final.drop_duplicates(keep="first")

            if options.keep_columns is not None:
                left_columns = pd.Index(options.keep_columns).intersection(left_final.columns)
                right_columns = pd.Index(options.keep_columns).intersection(right_final.columns)
                left_final = left_final[left_columns]
                right_final = right_final[right_columns]

            if options.print_prepared:
                LOGGER.info(self._get_prepared_df_string(p_data.left_name, p_data.left_df))
                LOGGER.info(self._get_prepared_df_string(p_data.right_name, p_data.right_df))

            if options.print_transformed:
                LOGGER.info(self._get_transformed_df_string(p_data.left_name, p_data.left_df_trans))
                LOGGER.info(self._get_transformed_df_string(p_data.right_name, p_data.right_df_trans))

            def _union_format(
                name: str, 
                df: pd.DataFrame,
            ) -> str:
                return f"Intersecting rows from {name} {df.shape}:\n{df}\n"

            LOGGER.info(_union_format(p_data.left_name, left_final))
            LOGGER.info(_union_format(p_data.right_name, right_final))

            if data_save_dir_path is not None:
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
            LOGGER.info(f"")

            left_results.append(left_final)
            right_results.append(right_final)
        return left_results, right_results

    def _setup(
        self,
        options: CommandOptions,
        command: Callable,
        local_vars: dict,
    ) -> tuple[logging.Logger, str | None]:
        current_time = pd.Timestamp.now()

        if options.add_save_timestamp:
            inner_folder = current_time.strftime("%Y-%m-%d-%H%M%S")
        else:
            inner_folder = None

        data_save_dir_path = self._make_output_dir(
            save_sub_folder=command.__name__, 
            timestamp_str=inner_folder,
        )

        LOGGER = self._get_logger(
            enable_printing=options.enable_printing,
            output_dir=data_save_dir_path,
            output_file_name=f"{command.__name__}.log",
            logger_name=command.__name__,
        )

        LOGGER.info(f"Timestamp: {current_time}\n")
        LOGGER.info(f"Input directory: {self.input_dir}\n")

        LOGGER.info(
            self._get_function_info_string(
                command.__name__, 
                local_vars,
            )
        )

        return LOGGER, data_save_dir_path
    
    def _get_logger(
            self,
            enable_printing: bool | None,
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
            file_handler = logging.FileHandler(os.path.join(output_dir, output_file_name), mode="w")
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
        timestamp_str: str | None,
    ) -> str | None:
        if self.output_dir is not None:
            if timestamp_str is not None:
                data_save_dir_path = os.path.join(
                    self.output_dir, 
                    f"{save_sub_folder}{os.sep}{timestamp_str}",
                )
            else:
                data_save_dir_path = os.path.join(self.output_dir, f"{save_sub_folder}{os.sep}results")

            os.makedirs(data_save_dir_path, exist_ok=True)
            return data_save_dir_path
        return None
        
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

        middle = ""
        for key, val in all_locals.items():
            string = f"{key}\n{"-" * len(key)}\n{str(val)}\n"
            middle += string

        indent = 4
        lines = middle.split("\n")
        shifted_lines = [' ' * indent + line for line in lines]
        middle = "\n".join(shifted_lines)
        
        final_string = f"{func_name}(\n{middle}".rstrip() + "\n)\n"
        return final_string
    
    def _get_prepared_df_string(
            self, 
            name: str,
            df: pd.DataFrame,
        ) -> str:
        return f"Prepared df for {name} {df.shape}:\n{df}\n"
    
    def _get_transformed_df_string(
            self, 
            name: str,
            df: pd.DataFrame,
        ) -> str:
        return f"Transformed df for {name} {df.shape}:\n{df}\n"
