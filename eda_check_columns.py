from pathlib import Path
import pandas as pd

def check_columns(folder_name):

    folder_path = Path(f"C:\\CMI-DataScience-Project\\{folder_name}")
    col_list = []
    total_rows = 0

    for file in folder_path.glob("*.csv"):
        df = pd.read_csv(file
                         #, nrows=0
                         )
        col_names = df.columns.to_list()
        col_list.append(", ".join(col_names))
        total_rows = len(df) + total_rows

    ddf = pd.DataFrame(col_list,columns=["cols"])
    ddf.to_csv(Path(fr"C:\CMI-DataScience-Project\column_check\check_{folder_name}_columns.csv"),index=False)
    print(f"Total rows in csvs from the {folder_name} folder: {total_rows}")


#check_columns("atp_doubles")
#check_columns("atp_futures")
#check_columns("atp_matches")
#check_columns("atp_qual_chall")