# CSVUnionDiff
CSVUnionDiff is an open-source library for comparing CSV-like files through union and difference operations. 

- [CSVUnionDiff](#csvuniondiff)
  - [Features](#features)
  - [Installation and usage](#installation-and-usage)
  - [Examples](#examples)
    - [Command-line](#command-line)
    - [Programming](#programming)
  - [Possible use cases](#possible-use-cases)
    - [Command-line](#command-line-1)
    - [Programming](#programming-1)
  - [Match rows algorithm explanation](#match-rows-algorithm-explanation)

## Features
- A convenient command-line tool for quickly comparing files.
- A robust python package for comparing files in a programmatic way.
- Incorporates pandas to allow for various input and output types.
- A union operation to get the common rows between files.
- A diff operation to get unique rows between files.
- A match rows option which forces comparisons to be carried in a [specific useful way](#match-rows-explanation).

## Installation and usage
To install through command-line, use 
```
python -m pip install csvuniondiff
```
To view available options for the command-line tool, use
```
csvuniondiff -h
```
To use the package in python, do
```
from csvuniondiff import ...
```
where ```...``` can be replaced with whatever is available from the package.

## Examples
### Command-line
Currently supported command line options are:
```
options:
  -h, --help            show this help message and exit
  --version             print the version of this package
  --diff DIFF DIFF      use the diff command, takes 2 files as arguments
  --union UNION UNION   use the union command, takes 2 files as arguments
  -a, --align-columns   aligns common columns on the left sorted
  -c [USE_COLUMNS ...], --use-columns [USE_COLUMNS ...]
                        only use these columns for comparison
  --ignore-columns [IGNORE_COLUMNS ...]
                        do not use these columns for comparison
  -f [FILL_NULL], --fill-null [FILL_NULL]
                        fills null option value so that they can be compared, default is 'NULL'
  -d, --drop-null       drop rows with nulls
  -D, --drop-duplicates
                        drop duplicate rows
  -i INPUT_DIR, --input-dir INPUT_DIR
                        use this directory path as the base for the path to the files
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        save outputs from the script to this directory
  -m, --match-rows      use the match rows algorithm for comparison
  -k [KEEP_COLUMNS ...], --keep-columns [KEEP_COLUMNS ...]
                        only keep these columns in the final result
  -C, --use-common-columns
                        use the maximal set of common columns for comparison
  --dont-add-timestamp  don't add a timestamp directory when outputting files
  --disable-printing    disable printing to stdout
  --print-prepared      print the prepared df before comparison
  --save-file-extension SAVE_FILE_EXTENSION
                        the extension for output files (csv, xlsx, json, xml, or html)
  -r, --row-counts      use the counts of each unique row in the final result instead
```

1. 
    **test2.csv**:
    |       | column7 | column1 | column2 | column3  | column4  | column5  | column6  |
    |-------|---------|---------|---------|----------|----------|----------|----------|
    | 0     | value7  | value1  | value2  | value3   | value4   | value5   | value6   |
    | 1     | value14 | value8  | value9  | value10  | value11  | value12  | value13  |
    | 2     | value21 | value15 | value16 | value17  | value18  | value19  | value20  |

    **test4.csv**:
    |       | column4 | column3 | column2 | column1 |
    |-------|---------|---------|---------|---------|
    | 0     |         | value3  | value2  | value1  |
    | 1     | value9  | value8  |         | value6  |
    | 2     | value14 |         |         | value11 |

    **Input**
    ```
    csvuniondiff
    --input-dir csvuniondiff/tests/test-data/random/ 
    --union test2.csv test4.csv 
    --match-rows            # use the match rows algorithm
    --fill-null value4      # fills nulls with 'value4'
    --align-columns         # align common columns sorted on the left
    --use-common-columns    # compare using all common columns
    ```

    **Output**
    ```
    Timestamp: 2024-07-10 14:27:45.402980

    Input directory: csvuniondiff/tests/test-data/random/

    union(
        args
        ----
        left_input: ['test2.csv']
        right_input: ['test4.csv']
        data_save_file_extensions: ['csv']

        options
        -------
        align_columns: True
        fill_null: value4
        match_rows: True
        enable_printing: True
        add_save_timestamp: True
        use_common_columns: True
    )

    Intersecting rows from test2.csv (1, 7):
    column1 column2 column3 column4 column7 column5 column6
    0  value1  value2  value3  value4  value7  value5  value6

    Intersecting rows from test4.csv (1, 4):
    column1 column2 column3 column4
    0  value1  value2  value3  value4
    ```
2. 
    Look [here](#match-rows-explanation) for input files.
   
    **Input**
    ```
    csvuniondiff
    --input-dir csvuniondiff/tests/test-data/diff/testset-1/ 
    --diff csv1.csv csv2.csv
    ```

    **Output**
    ```
    Timestamp: 2024-07-10 12:00:06.554911

    Input directory: csvuniondiff/tests/test-data/diff/testset-1/

    diff(
        args
        ----
        left_input: ['csv1.csv']
        right_input: ['csv2.csv']
        data_save_file_extensions: ['csv']

        options
        -------
        enable_printing: True
        add_save_timestamp: True
    )

    Only in csv1.csv (5, 3):
                Name  Age                      Email
    3  Michael Wilson   32  michaelwilson@example.com
    4  Michael Wilson   32  michaelwilson@example.com
    5    Bob Thompson   35    bobthompson@example.com
    6     Emily Davis   27     emilydavis@example.com
    7  Michael Wilson   32  michaelwilson@example.com

    Only in csv2.csv (3, 3):
                    Name  Age                       Email
    6       John Smith__1   35       johnsmith@example.com
    7  Michael Johnson__1   32  michaeljohnson@example.com
    8      Emily Davis__1   27      emilydavis@example.com
    ```

### Programming
1. 
    **test1.csv**
    | Index | Name             | Age | Email                    |
    |-------|------------------|-----|--------------------------|
    | 0     | John Doe         | 25  | johndoe@example.com      |
    | 1     | Jane Smith       | 30  | janesmith@example.com    |
    | 2     | Mark Johnson     | 40  | markjohnson@example.com  |
    | 3     | Emily Davis      | 35  | emilydavis@example.com   |
    | 4     | Michael Brown    | 28  | michaelbrown@example.com |
    | 5     | Sarah Wilson     | 32  | sarahwilson@example.com  |
    | 6     | David Thompson   | 45  | davidthompson@example.com|
    | 7     | Jessica Martinez | 27  | jessicamartinez@example.com|
    | 8     | Christopher Lee  | 33  | christopherlee@example.com|
    | 9     | Laura Taylor     | 29  | laurataylor@example.com  |
    
    **test2.csv**
    | Index | Name             | Email                     | Age |
    |-------|------------------|---------------------------|-----|
    | 0     | John Doe         | johndoe25@example.com     | 25  |
    | 1     | Jane Smith       | janesmith30@example.com   | 30  |
    | 2     | Mark Johnson     | markjohnson40@example.com | 40  |
    | 3     | Emily Davis      | emilydavis35@example.com  | 35  |
    | 4     | Jessica Martinez | jessicamartinez27@example.com | 27 |
    | 5     | Christopher Lee  | christopherlee33@example.com | 33 |
    | 6     | Laura Taylor     | laurataylor29@example.com | 29  |
    | 7     | Brian Harris     | brianharris33@example.com | 33  |

    **Input**
    ```
    import pandas as pd
    from csvuniondiff.csvuniondiff import (
        CsvUnionDiff,
        ParallelInputArgs,
        CommandOptions,
    )

    obj = CsvUnionDiff(
        "./csvuniondiff/tests/test-data/diff/testset-2/",
        None,
    )

    def left_df_trans(df: pd.DataFrame) -> pd.DataFrame:
        def email_trans(row):
            arr = row["Email"].split("@")
            return arr[0] + str(row["Age"]) + "@" + arr[1]
        
        df["Email"] = df.apply(email_trans, axis=1)

        df = df[["Name", "Email", "Age"]]
        return df

    left_dfs, right_dfs = obj.diff(
        args=ParallelInputArgs(
            ["test1.csv"], 
            ["test2.csv"], 
            left_trans_funcs=[left_df_trans],
            right_trans_funcs=[lambda x: x],
            return_transformed_rows=False, # selects the rows from original table
        ),
        options=CommandOptions(
            match_rows=True,
            enable_printing=True
        ),
    )

    left_df = left_dfs[0]
    right_df = right_dfs[0] # use the results somewhere
    ```

    **Output**
    ```
    Timestamp: 2024-07-11 11:37:38.748144

    Input directory: ./csvuniondiff/tests/test-data/diff/testset-2/

    diff(
        args
        ----
        left_input: ['test1.csv']
        right_input: ['test2.csv']
        left_trans_funcs: [<function left_df_trans at 0x000001E40ACBA340>]
        right_trans_funcs: [<function <lambda> at 0x000001E4259B3E20>]

        options
        -------
        match_rows: True
        enable_printing: True
    )

    Only in test1.csv (3, 3):
                Name  Age                      Email
    4   Michael Brown   28   michaelbrown@example.com
    5    Sarah Wilson   32    sarahwilson@example.com
    6  David Thompson   45  davidthompson@example.com

    Only in test2.csv (1, 3):
            Name                      Email  Age
    7  Brian Harris  brianharris33@example.com   33
    ```
2. 
    **Input**
    ```
    from csvuniondiff.csvuniondiff import (
        CsvUnionDiff,
        ParallelInputArgs,
        CommandOptions,
    )

    obj = CsvUnionDiff(
        input_dir="./csvuniondiff/tests/test-data/diff/testset-1/",
        output_dir=None,

    )

    left_dfs, right_dfs = obj.diff(
        args=ParallelInputArgs(
            left_input=["test1.csv"],
            right_input=["test2.csv"],
        ),
        options=CommandOptions(
            enable_printing=True,
            add_save_timestamp=True,
        )
    )

    only_in_test1, only_in_test2 = left_dfs[0], right_dfs[0] # Use the dataframe results somewhere
    ```

    **Output**
    ```
    Timestamp: 2024-07-10 13:23:35.239955

    Input directory: ./csvuniondiff/tests/test-data/diff/testset-1/

    diff(
        args
        ----
        left_input: ['test1.csv']
        right_input: ['test2.csv']

        options
        -------
        match_rows: True
        enable_printing: True
        add_save_timestamp: True
    )

    Only in test1.csv (5, 3):
                Name  Age                      Email
    3  Michael Wilson   32  michaelwilson@example.com
    4  Michael Wilson   32  michaelwilson@example.com
    5    Bob Thompson   35    bobthompson@example.com
    6     Emily Davis   27     emilydavis@example.com
    7  Michael Wilson   32  michaelwilson@example.com

    Only in test2.csv (5, 3):
                    Name  Age                       Email
    1            John Doe   25         johndoe@example.com
    3          Jane Smith   30       janesmith@example.com
    6       John Smith__1   35       johnsmith@example.com
    7  Michael Johnson__1   32  michaeljohnson@example.com
    8      Emily Davis__1   27      emilydavis@example.com
    ```

## Possible use cases
### Command-line
1.  
    A personal usecase of mine is to **cross-check SQL results with an expected CSV/Excel file** (perhaps one that was created manually). 
    I would use my SQL management tool to generate the CSV file from my query, then call ```csvuniondiff --diff my.csv expected.csv --match-rows``` 
    to see the differences and the magnitude of the differences. I could also call ```csvuniondiff --union my.csv expected.csv --match-rows``` 
    to see what rows my SQL query is getting right.
2.  
    You want to compare 2 CSV files but some aspect covered by this tool makes it impossible to (for example NULL values, unaligned columns, or you
    want to only compare a subset of columns etc.) and you want to do it fast.

### Programming
1.  
    I had a case where I needed to check for the existence of rows with certain values in specific columns across many Excel files. 
    I can make a dataframe with the columns and values that I am looking for:
    |    | Name            | Age | Email                     |
    |----|-----------------|-----|---------------------------|
    | 0  | John Doe        | 25  | johndoe@example.com       |

    I can put all of the Excel files in a directory and then run the union command with the above CSV against 
    the target CSV's in the directory.
2.  
    The files are slightly different but could be transformed to be compared.
3.  
    You don't want to personally code out difference and union operations with match rows and stdout output.

## Match rows algorithm explanation
To explain the match rows option, let's consider the following CSV tables:

<table>
<tr><th>csv1</th><th>csv2</th></tr>
<tr><td>

|    | Name            | Age | Email                     |
|----|-----------------|-----|---------------------------|
| 0  | John Doe        | 25  | johndoe@example.com       |
| 1  | Jane Smith      | 30  | janesmith@example.com     |
| 2  | Alice Johnson   | 28  | alicejohnson@example.com  |
| 3  | Michael Wilson  | 32  | michaelwilson@example.com |
| 4  | Michael Wilson  | 32  | michaelwilson@example.com |
| 5  | Bob Thompson    | 35  | bobthompson@example.com   |
| 6  | Emily Davis     | 27  | emilydavis@example.com    |
| 7  | Michael Wilson  | 32  | michaelwilson@example.com |
| 8  | Sarah Brown     | 29  | sarahbrown@example.com    |
</td><td>

|     | Name              | Age | Email                   |
|-----|-------------------|-----|-------------------------|
| 0   | John Doe          | 25  | johndoe@example.com     |
| 1   | John Doe          | 25  | johndoe@example.com     |
| 2   | Jane Smith        | 30  | janesmith@example.com   |
| 3   | Jane Smith        | 30  | janesmith@example.com   |
| 4   | Alice Johnson     | 28  | alicejohnson@example.com|
| 5   | Sarah Brown       | 29  | sarahbrown@example.com  |
| 6   | John Smith__1     | 35  | johnsmith@example.com   |
| 7   | Michael Johnson__1| 32  | michaeljohnson@example.com|
| 8   | Emily Davis__1    | 27  | emilydavis@example.com  |
</td></tr> 
</table>

When matching rows in the diff operation
1.  The first John Doe in both files is matched, so the second John Doe in **csv2** is only in **csv2**.
2.  The first Jane Smith in both files is matched, so the second Jane Smith in **csv2** is only in **csv2**.
3.  Both files have exactly 1 Alice Johnson and Sarah Brown so they are both matched and neither are only in **csv1** or **csv2**.
4.  The remaining rows are all unique between the two files so they are only in **csv1** or **csv2**, respectively.

Therefore, with the match rows option, the results of the diff operation will be:

<table>
<tr><th>only in csv1</th><th>only in csv2</th></tr>
<tr><td>

|     | Name            | Age | Email                     |
|-----|-----------------|-----|---------------------------|
| 3   | Michael Wilson  | 32  | michaelwilson@example.com |
| 4   | Michael Wilson  | 32  | michaelwilson@example.com |
| 5   | Bob Thompson    | 35  | bobthompson@example.com   |
| 6   | Emily Davis     | 27  | emilydavis@example.com    |
| 7   | Michael Wilson  | 32  | michaelwilson@example.com |
</td><td>

|     | Name              | Age | Email                      |
|-----|-------------------|-----|----------------------------|
| 1   | John Doe          | 25  | johndoe@example.com        |
| 3   | Jane Smith        | 30  | janesmith@example.com      |
| 6   | John Smith__1     | 35  | johnsmith@example.com      |
| 7   | Michael Johnson__1| 32  | michaeljohnson@example.com |
| 8   | Emily Davis__1    | 27  | emilydavis@example.com     |
</td></tr>
</table>

Using the union operation with match rows instead with **csv1** and **csv2**, we get:

<table>
<tr><th>intersecting from csv1</th><th>intersecting from csv2</th></tr>
<tr><td>

|       | Name           | Age | Email                     |
|-------|----------------|-----|---------------------------|
| 0     | John Doe       | 25  | johndoe@example.com       |
| 1     | Jane Smith     | 30  | janesmith@example.com     |
| 2     | Alice Johnson  | 28  | alicejohnson@example.com  |
| 8     | Sarah Brown    | 29  | sarahbrown@example.com    |
</td><td>

| Index | Name           | Age | Email                     |
|-------|----------------|-----|---------------------------|
| 0     | John Doe       | 25  | johndoe@example.com       |
| 2     | Jane Smith     | 30  | janesmith@example.com     |
| 4     | Alice Johnson  | 28  | alicejohnson@example.com  |
| 5     | Sarah Brown    | 29  | sarahbrown@example.com    |
</td></tr>
</table>

