#係数調整実験の結果をcsvでまとめる
import os
import pandas as pd
from collections import defaultdict
import grah_utils


# classification_df = pd.read_csv("../experience_results/0619_selfdis_coeff_ex/coeff_0p5/comb_0.50,0.50,0.50,0.50,0.50,0.50/IEMO/classification_report.csv")
# print(classification_df.loc["support"])
# exit()



base_path = "../experience_results/0703_selfdis_coeff_ex"

#history_csv(base_path)
grah_utils.coeffects_result2csv_0709(base_path)