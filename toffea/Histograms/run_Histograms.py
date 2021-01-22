# If you want only partial QCD files to be processed (to reduce file size), please use this file to run ML.py
import os
from coffea import util
import coffea.processor as processor
QCD_samples = ["QCD_Pt_300to470,QCD_Pt_470to600,QCD_Pt_600to800,QCD_Pt_800to1000,QCD_Pt_1000to1400","QCD_Pt_1400to1800","QCD_Pt_1800to2400","QCD_Pt_2400to3200","QCD_Pt_3200toInf"]
for sample in QCD_samples:
    os.system(f"python Histograms.py -d {sample} -y 2017 -m -o 2017_QCD ")
output_list = [f for f in os.listdir("./")]
print(output_list)
merge_file = processor.dict_accumulator()
for f in output_list:
    if "2017_QCD" in f:
        merge_file.add(util.load(f))
util.save(merge_file, "DataHistograms_2017QCD.coffea")
