'''
Module for loading filelists
'''
from toffea.filelists.samplenames import samples, subsamples, res1tores2_samples, zprime3g_samples
import pickle

with open("filelists.pkl", "rb") as f:
	filelists = pickle.load(f)

if __name__ == "__main__":
	print("*** Input file configuration ***")
	with open("filelists.pkl")
	for year in ["2016", "2017", "2018"]:
		print("\n*** {} ***".format(year))
		for subsample in sorted(filelist[year].keys()):
			print("{} : {} : {} files".format(year, subsample, len(filelist[year][subsample])))