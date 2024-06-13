from typing import List, Tuple
import json
import os

import pandas as pd


class CsvCmp:
    def __init__(
            self, 
            input_dir_path: str,
            data_save_dir_path: str = "./csvcmp-data/"
        ):
        self.input_dir_path = input_dir_path
        self.data_save_dir_path = data_save_dir_path

    def compare_row_existence(
            self, 
            left_input: List[str | pd.DataFrame] | str | pd.DataFrame,
            right_input: List[str | pd.DataFrame] | str | pd.DataFrame,
            ignore_null_rows: bool = False,
            match_rows: bool = True,
            use_input_dir_path: bool = True
        ):
        if type(left_input) != list:
            left_input = [left_input]
            right_input = [right_input]

        if len(left_input) != len(right_input):
            raise ValueError("The number of elements in left_input and right_input should be the same")

        all_locals = locals()
        all_locals.pop("self")
        result_string = self._get_function_info_string(self.compare_row_existence.__name__, all_locals)

        left_names = [
            self._get_file_name(_) if type(_) == str 
            else "df_" + str(_.shape()) 
            for _ in left_input
        ]
        right_names = [
            self._get_file_name(_) if type(_) == str 
            else "df_" + str(_.shape()) 
            for _ in right_input
        ]

        left_dfs, right_dfs = self._change_inputs_to_dfs(
            left_input, 
            right_input=right_input,
            use_input_dir_path=use_input_dir_path,
            ignore_null_rows=ignore_null_rows,
        )

        def _only_in_format(name: str, df: pd.DataFrame) -> str:
            return f"Only in {name} {df.shape}:\n{df.head(20)}\n\n"
        
        def _get_left_and_right_only(left_df: pd.DataFrame, right_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
            columns = list(left_df.columns)
            left_df_tmp = left_df.reset_index(names="_left_index")
            right_df_tmp = right_df.reset_index(names="_right_index")
            outer_df = pd.merge(left_df_tmp, right_df_tmp, how="outer", on=list(columns), indicator=True)

            left_index_series = outer_df[outer_df["_merge"] == "left_only"]["_left_index"]
            right_index_series = outer_df[outer_df["_merge"] == "right_only"]["_right_index"]
            left_only_final = left_df.iloc[left_index_series]
            right_only_final = right_df.iloc[right_index_series]

            return left_only_final, right_only_final

        for left_df, right_df, left_name, right_name, i in zip(
            left_dfs, 
            right_dfs, 
            left_names, 
            right_names, 
            range(len(left_dfs))
        ):
            result_string += f"For index {i}\n\n"
            
            # Assume that the columns of left_df and right_df are exactly the same.
            if match_rows:
                left_value_counts = left_df.value_counts()
                right_value_counts = right_df.value_counts()

                shared = left_value_counts.index.intersection(right_value_counts.index)

                shared_left_sub_right = left_value_counts[shared] - right_value_counts[shared]
                shared_right_sub_left = right_value_counts[shared] - left_value_counts[shared]

                left_extra = shared_left_sub_right[shared_left_sub_right > 0]
                right_extra = shared_right_sub_left[shared_right_sub_left > 0]
                
                left_to_add = []
                for ind, count in list(pd.DataFrame(left_extra).itertuples(index=True)):
                    left_to_add.append(left_df[left_df.isin(ind)].dropna().head(count))

                right_to_add = []
                for ind, count in list(pd.DataFrame(right_extra).itertuples(index=True)):
                    right_to_add.append(right_df[right_df.isin(ind)].dropna().head(count))

                left_only, right_only = _get_left_and_right_only(left_df, right_df)

                left_only_final = pd.concat([left_only] + left_to_add).sort_index()
                right_only_final = pd.concat([right_only] + right_to_add).sort_index()
                print(right_only_final)
                exit(1)
            else:
                left_only_final, right_only_final = _get_left_and_right_only(left_df, right_df)

            result_string += _only_in_format(left_name, left_only_final)
            result_string += _only_in_format(right_name, right_only_final)
            
            result_string += "\n"
            # self._output_to_file(
            #     file_sub_path=f"{self.compare_row_existence.__name__}/{left_name}_only.csv", 
            #     df=left_only_final, 
            #     data_save_dir_path=self.data_save_dir_path
            # )
            # self._output_to_file(
            #     file_sub_path=f"{self.compare_row_existence.__name__}/{right_name}_only.csv", 
            #     df=right_only_final, 
            #     data_save_dir_path=self.data_save_dir_path
            # )
        print(result_string)

    def _get_function_info_string(
            self, 
            func_name: str, 
            all_locals: str
        ) -> str:
        return f"{func_name}()\n{json.dumps(all_locals, indent=4)}\n"
    
    def _get_file_name(
            self, 
            file_path: str
        ) -> str:
        return file_path.split("/")[-1]
    
    def _remove_file_extension(
            self, 
            file_name: str
        ) -> str:
        return file_name.split(".")[0]
      
    def _output_to_file(
            self, 
            file_sub_path: str, 
            df: pd.DataFrame,
            data_save_dir_path: str,
            add_timestamp: bool = True
        ) -> None:
        if add_timestamp:
            file_sub_path = file_sub_path.split(".")
            file_sub_path = file_sub_path[0] + "_" + str(pd.Timestamp.now()) + "." + file_sub_path[1]

        os.makedirs(data_save_dir_path, exist_ok=True)

        full_file_path = os.path.join(data_save_dir_path, file_sub_path)
        if full_file_path.endswith(".csv"):
            df.to_csv(full_file_path, index=False)
        elif full_file_path.endswith(".xlsx"):
            df.to_excel(full_file_path, index=False)
        else:
            raise ValueError(f"File type not supported: {full_file_path}")

    def _change_inputs_to_dfs(
            self, 
            first_input: List[str | pd.DataFrame], 
            use_input_dir_path: bool,
            ignore_null_rows: bool,
            **kwargs
        ) -> Tuple[List[pd.DataFrame], ...]:
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
