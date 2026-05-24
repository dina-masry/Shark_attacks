from src.data_cleaning import run_pipeline
 
if __name__ == "__main__":
    df_clean = run_pipeline(
        filepath="data/raw/attacks.csv",
        output_path="data/processed/attacks_clean.csv",
    )
    print(df_clean.head())
    print(df_clean.dtypes)
    print(df_clean.duplicated())
    print(df_clean.isnull().sum())
    print(df_clean['age'].describe())
    print(df_clean['year'].describe())
    print(df_clean['sex'].value_counts())
    print(df_clean['fatal_y_n'].value_counts())

