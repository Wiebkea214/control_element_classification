import os

import pandas as pd
from langchain_core.documents import Document

####################### Receiving information #####################

def get_FTS(path_fts):

    col_testcase = 'a_TestCase'
    col_main = ''
    col_teststeps = 'a_TestSteps'
    col_expresults = 'a_ExpectedTestResult'

    dict_fts = {}

    # Load excel file
    print(f"\n--- Reading FTS in {path_fts} ---")
    if os.path.exists(path_fts):
        reader = pd.read_excel(path_fts, engine='openpyxl')
        reader = reader.fillna("")
    else:
        print(f"!!! File {path_fts} not found")
        return

    # dynamically find column with test step content
    col_main = next((col for col in reader.columns if not col.startswith('a_') and 'ID' not in col), None)
    if not col_main:
        raise ValueError("!!! No main column found")

    print(f"\n--- Writing Test Steps in dictionary ---")
    for i, line in reader.iterrows():
        line_testcase = line[col_testcase]
        line_exp_result = line[col_expresults]
        line_main = line[col_main]
        line_teststeps = line[col_teststeps]

        if line_testcase and line_teststeps and line_main and len(line_teststeps)>5:
            line_main = line_main.strip().split() # removes spaces/tabs on beginning/end and splits line at whitespace
            key = line_main[0]
            val1 = line[col_teststeps].lower().strip()
            val2 = line[col_expresults].lower().strip()
            dict_fts[key] = f"{val1}. {val2}."
        else: pass

    print(f"--- Finished reading and storing FTS ---")

    return dict_fts


def get_BMV(path_bmv):

    col_name = 'POSNR'
    col_element_type = 'BENENNUNG1'
    col_description1 = 'BENENNUNG2'
    col_description2 = 'BENENNUNG3'
    col_description3 = 'BENENNUNG4'
    col_description4 = 'BENENNUNG5'
    col_description5 = 'BENENNUNG6'
    col_cab = 'CAB'
    col_location = 'LOCATION'

    dict_bmv = {}
    docs_bmv = []

    print(f"\n--- Reading bmv in {path_bmv} ---")

    reader = pd.read_excel(path_bmv, engine='openpyxl')
    reader = reader.fillna("")

    for i, line in reader.iterrows():
        key_id = line[col_name]
        val_type = str(line[col_element_type]).lower().strip()
        val1 = str(line[col_description1]).lower().strip()
        val_loc = line[col_location]

        if key_id and val_type and val1 and val_loc:
            val2 = str(line[col_description2]).lower().strip()
            val3 = str(line[col_description3]).lower().strip()
            val4 = str(line[col_description4]).lower().strip()
            val5 = str(line[col_description5]).lower().strip()
            val_cab = str(line[col_cab]).lower().strip()

            ### write in dict ###
            print(f"\n--- Writing bmv in dictionary ---")
            dict_bmv[key_id] = [val_type, val1, val2, val3, val_cab, val_loc]

            ### write in sts_doc ###
            print(f"\n--- Writing bmv in Documents ---")
            docs_bmv.append(Document(
                page_content=f"{key_id} is a {val_type} in {val_cab}. {val1}. {val2}. {val3}. {val4}. {val5}.",
                metadata={"id": key_id, "cab": val_cab, "location_num": int(val_loc)},
            ))
        else: pass

    print(f"--- Finished reading and storing bmv ---")

    return dict_bmv, docs_bmv # Should be written in txt file