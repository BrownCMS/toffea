'''
Module for loading filelists
'''
import os
from toffea.filelists.samplenames import samples, subsamples, res1tores2_samples, zprime3g_samples
import pickle

with open(f"{os.path.dirname(__file__) or '.'}/filelists.pkl", "rb") as f:
	filelist = pickle.load(f)

if __name__ == "__main__":
	print("*** Input file configuration ***")
	for year in ["2016", "2017", "2018"]:
		print("\n*** {} ***".format(year))
		for subsample in sorted(filelist[year].keys()):
			print("{} : {} : {} files".format(year, subsample, len(filelist[year][subsample])))