from collections.abc import Callable
import json
import os

import pandas as pd


class CsvCmp:
    def __init__(
            self, 
            input_dir_path: str = "./",
            data_save_dir_path: str = "./csvcmp-data/",
        ):
        self.input_dir_path = input_dir_path
        self.data_save_dir_path = data_save_dir_path

    def compare_row_existence(
            self, 
            left_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            right_input: list[str | pd.DataFrame] | str | pd.DataFrame,
            left_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
            right_trans_funcs: list[Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
            data_save_file_extensions: list[str] | None = None,
            ignore_null_rows: bool = False,
            match_rows: bool = True,
            use_input_dir_path: bool = True,
            output_transformed_rows: bool = False,
        ): 
        print(
            self._get_function_info_string(
                self.compare_row_existence.__name__, 
                locals(),
            )
        )

        if type(left_input) != list:
            left_input = [left_input]
            right_input = [right_input]

        if len(left_input) != len(right_input):
            raise ValueError("The number of elements in left_input and right_input should be the same")
        
        if data_save_file_extensions is None:
            data_save_file_extensions = ["csv"] * len(left_input)
        elif len(data_save_file_extensions) != len(left_input):
            raise ValueError("The number of elements in data_save_file_extensions should be the same as the number of elements in left_input")
        
        if left_trans_funcs is None:
            left_trans_funcs = [lambda x: x] * len(left_input)
        elif len(left_trans_funcs) != len(left_input):
            raise ValueError("The number of elements in left_trans_funcs should be the same as the number of elements in left_input")
        
        if right_trans_funcs is None:
            right_trans_funcs = [lambda x: x] * len(right_input)
        elif len(right_trans_funcs) != len(right_input):
            raise ValueError("The number of elements in right_trans_funcs should be the same as the number of elements in right_input")

        left_file_names = [
            self._get_file_name(input_) if type(input_) == str 
            else f"df_{i}_{input_.shape}"
            for i, input_ in enumerate(left_input)
        ]
        right_file_names = [
            self._get_file_name(input_) if type(input_) == str 
            else f"df_{i}_{input_.shape}"
            for i, input_ in enumerate(right_input)
        ]

        left_dfs, right_dfs = self._change_inputs_to_dfs(
            left_input, 
            right_input=right_input,
            use_input_dir_path=use_input_dir_path,
            ignore_null_rows=ignore_null_rows,
        )

        def _only_in_format(
                name: str, 
                df: pd.DataFrame,
            ) -> str:
            return f"Only in {name} {df.shape}:\n{df.head(20)}\n\n"
        
        def _get_left_and_right_only_indices(
                left_df: pd.DataFrame, 
                right_df: pd.DataFrame,
            ) -> tuple[pd.Series, pd.Series]:
            columns = list(left_df.columns)
            left_df_tmp = left_df.reset_index(names="_left_index")
            right_df_tmp = right_df.reset_index(names="_right_index")
            outer_df = pd.merge(
                left_df_tmp, 
                right_df_tmp, 
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
            return left_index_series, right_index_series

        for i, (
            left_df, 
            right_df, 
            left_trans_func,
            # right_trans_func,
            left_file_name, 
            right_file_name, 
            data_save_file_extension,
        ) in enumerate(
            zip(
                left_dfs, 
                right_dfs, 
                left_trans_funcs,
                # right_trans_funcs,
                left_file_names, 
                right_file_names, 
                data_save_file_extensions,
            )
        ):
            left_data_save_file_name = (
                f"only_in_" +
                self._get_file_name_no_extension(left_file_name)
                + f".{data_save_file_extension}"
            )
            right_data_save_file_name = (
                f"only_in_" +
                self._get_file_name_no_extension(right_file_name)
                + f".{data_save_file_extension}"
            )

        #     print(f"For index {i}\n")

        #     # Assume that the columns of left_df and right_df are exactly the same.
        #     if match_rows:
        #         left_value_counts = left_df.value_counts()
        #         right_value_counts = right_df.value_counts()

        #         shared = left_value_counts.index.intersection(right_value_counts.index)

        #         shared_left_sub_right = left_value_counts[shared] - right_value_counts[shared]
        #         shared_right_sub_left = right_value_counts[shared] - left_value_counts[shared]

        #         left_extra = shared_left_sub_right[shared_left_sub_right > 0]
        #         right_extra = shared_right_sub_left[shared_right_sub_left > 0]
                
        #         left_to_add_indices = []
        #         for ind, count in list(pd.DataFrame(left_extra).itertuples(index=True)):
        #             left_to_add_indices.append(left_df[left_df.isin(ind)].dropna().head(count))

        #         right_to_add_indices = []
        #         for ind, count in list(pd.DataFrame(right_extra).itertuples(index=True)):
        #             right_to_add_indices.append(right_df[right_df.isin(ind)].dropna().head(count))

        #         left_only, right_only = _get_left_and_right_only(left_df, right_df)

        #         left_only_final = pd.concat([left_only] + left_to_add_indices).sort_index()
        #         right_only_final = pd.concat([right_only] + right_to_add_indices).sort_index()
        #     else:
        #         left_only_final, right_only_final = _get_left_and_right_only(left_df, right_df)

        #     result_string += _only_in_format(left_file_name, left_only_final)
        #     result_string += _only_in_format(right_file_name, right_only_final)
            
        #     result_string += "\n"

        #     output_dir_path = self._make_output_dir(self.compare_row_existence.__name__)
        #     self._output_to_file(
        #         output_dir_path=output_dir_path,
        #         file_name=left_data_save_file_name,
        #         df=left_only_final, 
        #     )
        #     self._output_to_file(
        #         output_dir_path=output_dir_path,
        #         file_name=right_data_save_file_name,
        #         df=right_only_final, 
        #     )
        # print(result_string)

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
    
    def _get_file_name(
            self, 
            file_path: str
        ) -> str:
        file_name = os.path.basename(file_path)
        return file_name
    
    def _make_output_dir(
        self,
        save_sub_folder: str,
        add_timestamp: bool = True,
    ) -> str:
        if add_timestamp:
            data_save_dir_path = os.path.join(
                self.data_save_dir_path, 
                f"{save_sub_folder}_{pd.Timestamp.now().strftime('%Y-%m-%d-%H%M%S')}",
            )
        else:
            data_save_dir_path = os.path.join(self.data_save_dir_path, save_sub_folder)

        os.makedirs(data_save_dir_path, exist_ok=True)
        return data_save_dir_path
        
    def _output_to_file(
        self, 
        output_dir_path: str,
        file_name: str,
        df: pd.DataFrame,
    ) -> None:
        full_file_path = os.path.join(output_dir_path, file_name)
        if full_file_path.endswith(".csv"):
            df.to_csv(full_file_path, index=False)
        elif full_file_path.endswith(".xlsx"):
            df.to_excel(full_file_path, index=False)
        else:
            raise ValueError(f"File type not supported: {full_file_path}")

    def _get_function_info_string(
            self, 
            func_name: str, 
            all_locals: str
        ) -> str:
        all_locals.pop("self")
        return f"{func_name}()\n{json.dumps(all_locals, indent=4)}\n"
    
    def _remove_file_extension(
            self, 
            file_name: str
        ) -> str:
        return file_name.split(".")[0]

    def _change_inputs_to_dfs(
            self, 
            first_input: list[str | pd.DataFrame], 
            use_input_dir_path: bool,
            ignore_null_rows: bool,
            **kwargs,
        ) -> tuple[list[pd.DataFrame], ...]:
        all_inputs = ([first_input] + [_ for _ in kwargs.values()]).copy()

        for input_ in all_inputs:
            for i, val in enumerate(input_):
                if type(val) == str:
                    if use_input_dir_path:
                        val = os.path.join(self.input_dir_path, val)

                    if val.endswith(".csv"):
                        input_[i] = pd.read_csv(val)    
                    elif val.endswith(".xlsx"):
                        input_[i] = pd.read_excel(val)
                    else:
                        raise ValueError(f"File type not supported: {val}")
                    
                    if ignore_null_rows:
                        input_[i] = input_[i].dropna()
                    else:
                        input_[i] = input_[i].fillna("NULL")
                    
        return tuple(all_inputs)
    
    def _get_index_after_comparing(
            self, 
            left_df: pd.DataFrame, 
            right_df: pd.DataFrame,
            compare_func: callable,
            left_transform_func: callable = lambda x: x,
            right_transform_func: callable = lambda x: x,
        ) -> pd.Index:
        left_df = left_transform_func(left_df)
        right_df = right_transform_func(right_df)
        comp_res_df = compare_func(left_df, right_df)

        return comp_res_df.index
    

obj = CsvCmp("./test-data/testset-1/")
obj.compare_row_existence(["test1.csv"], ["test2.csv"], match_rows=True)
