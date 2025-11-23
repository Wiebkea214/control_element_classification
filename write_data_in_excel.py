import os

import pandas as pd

#################################################

def init_excel(headers, path):

    # Create new or empty old file
    df = pd.DataFrame(columns=headers)
    df.to_excel(path, index=False)


def write_excel(headers, data, path):

    # Data to append to file
    df_add = pd.DataFrame([data], columns=headers)

    if os.path.exists(path):
        df = pd.read_excel(path)
        df = pd.concat([df, df_add], ignore_index=True)
    else:
        df = df_add

    # Write back to file
    df.to_excel(path, index=False)

