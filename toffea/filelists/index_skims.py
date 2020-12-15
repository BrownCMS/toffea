'''
This script makes text file lists of the input skims on BRUX
'''
from glob import glob
from toffea.filelists.samplenames import samples, subsamples, res1tores2_samples, zprime3g_samples
import pickle

basedir = "/home/dryu/store/DijetSkim"

filelist = {
	"2016": {}, 
	"2017": {}, 
	"2018": {}
}

# JetHT
jetht_version = "v2_0_1"
for year in ["2016", "2017", "2018"]:
	for jetht_period in samples[year]["JetHT"]:
		filelist[year][jetht_period] = glob(f"{basedir}/{jetht_version}/JetHT{year}/JetHT/{jetht_period}/*/*/nanoskim*root")

# Single muon
singlemuon_version = "v2_0_1"
for year in ["2016", "2017", "2018"]:
	for singlemuon_period in samples[year]["SingleMuon"]:
		filelist[year][singlemuon_period] = glob(f"{basedir}/{singlemuon_version}/SingleMuon{year}/SingleMuon/{singlemuon_period}/*/*/nanoskim*root")

# QCD
# /home/dryu/store/DijetSkim/v2_0_4/QCD_Pt_2017/QCD_Pt_470to600_TuneCP5_13TeV_pythia8/QCD_Pt_470to600_ext1/201211_221851/0000
qcd_version = "v2_0_4"
for year in ["2016", "2017", "2018"]:
	for qcd_slice in samples[year]["QCD"]:
		glob_pattern = f"{basedir}/{qcd_version}/QCD_Pt_{year}/{qcd_slice}*/*/*/*/nanoskim*root"
		filelist[year][qcd_slice] = glob(glob_pattern)
		if len(filelist[year][qcd_slice]) == 0:
			print("WARNING : Found no files for pattern {}".format(glob_pattern))


# Signal Res1ToRes2*
# /home/dryu/store/DijetSkim/v2_0_4/Res1ToRes2GluTo3Glu_2017/
#   Res1ToRes2GluTo3Glu_M1-8000_R-0p9_TuneCP5_13TeV-madgraph-pythia8/Res1ToRes2GluTo3Glu_M1-8000_R-0p9/201211_215140/0000
res1res2_version = "v2_0_4"
for year in ["2017"]:
	for sample in res1tores2_samples:
		glob_pattern = f"{basedir}/{res1res2_version}/Res1ToRes2GluTo3Glu_{year}/{sample}*/*/*/*/nanoskim*root"
		filelist[year][sample] = glob(glob_pattern)
		if len(filelist[year][sample]) == 0:
			print("WARNING : Found no files for pattern {}".format(glob_pattern))

# Signal Z' to ggg
# /home/dryu/store/DijetSkim/v2_0_4/ZprimeTo3Gluon_2018/ZprimeTo3Gluon_TuneCUETP8M1_13TeV_pythia8/ZprimeTo3Gluon_scan_2018/201211_232950/0000
zprime3g_version = "v2_0_4"
for year in ["2017"]:
	for sample in res1tores2_samples:
		glob_pattern = f"{basedir}/{zprime3g_version}/Res1ToRes2GluTo3Glu_{year}/{sample}*/*/*/*/nanoskim*root"
		filelist[year][sample] = glob(glob_pattern)
		if len(filelist[year][sample]) == 0:
			print("WARNING : Found no files for pattern {}".format(glob_pattern))

with open("filelists.pkl", "wb") as f:
	pickle.dump(filelist, f)

for year in ["2016", "2017", "2018"]:
	print("\n*** {} ***".format(year))
	for subsample in sorted(filelist[year].keys()):
		print("{} : {} : {} files".format(year, subsample, len(filelist[year][subsample])))