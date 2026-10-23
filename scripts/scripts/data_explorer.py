class DataExplorer:

    def __init__(self, data):
        self.df = data.to_pandas() if hasattr(data, "to_pandas") else data.copy()

    def show_columns(self):
        print("\nColumns:")
        print(self.df.columns.tolist())

    def show_data_types(self):
        print("\nData Types:")
        print(self.df.dtypes)

    def check_missing_values(self):
        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def check_duplicates(self):
        print("\nDuplicate Rows:")
        print(self.df.duplicated().sum())

    def check_empty_values(self):
        print("\nEmpty Values:")

        for column in self.df.columns:
            empty_values = (
                self.df[column]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

            print(column, "->", empty_values)

    def check_inconsistencies(self):
        print("\nInconsistencies:")

        for column in self.df.columns:

            non_string_values = (
                ~self.df[column].dropna()
                .apply(lambda x: isinstance(x, str))
            ).sum()

            print(
                column,
                "-> non-string values:",
                non_string_values
            )

    def show_shape(self):
        print("\nDataset Shape:")
        print("Rows:", len(self.df))
        print("Columns:", len(self.df.columns))