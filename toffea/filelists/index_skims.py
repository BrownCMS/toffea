'''
This script makes text file lists of the input skims on BRUX
'''
import os
from pprint import pprint
from glob import glob
from toffea.filelists.samplenames import samples, subsamples, res1tores2_samples, res1tores2_decay_mode, zprime3g_samples
import pickle
import ROOT
from collections import defaultdict
import copy

basedir = "/isilon/hadoop/store/user/dryu/DijetSkim"

filelist = {
	"2016": {}, 
	"2017": {}, 
	"2018": {}
}

# JetHT
jetht_version = "v2_0_7"
for year in ["2016", "2017", "2018"]:
	for jetht_period in samples[year]["JetHT"]:
		filelist[year][jetht_period] = glob(f"{basedir}/{jetht_version}/JetHT{year}/JetHT/{jetht_period}/*/*/nanoskim*root")

# Single muon
singlemuon_version = "v2_0_7"
for year in ["2016", "2017", "2018"]:
	for singlemuon_period in samples[year]["SingleMuon"]:
		filelist[year][singlemuon_period] = glob(f"{basedir}/{singlemuon_version}/SingleMuon{year}/SingleMuon/{singlemuon_period}/*/*/nanoskim*root")

# QCD
# /home/dryu/store/DijetSkim/v2_0_4/QCD_Pt_2017/QCD_Pt_470to600_TuneCP5_13TeV_pythia8/QCD_Pt_470to600_ext1/201211_221851/0000
qcd_version = "v2_0_7"
for year in ["2016", "2017", "2018"]:
	for qcd_slice in samples[year]["QCD"]:
		glob_pattern = f"{basedir}/{qcd_version}/QCD_Pt_{year}/{qcd_slice}*/*/*/*/nanoskim*root"
		filelist[year][qcd_slice] = glob(glob_pattern)
		if len(filelist[year][qcd_slice]) == 0:
			print("WARNING : Found no files for pattern {}".format(glob_pattern))


# Signal Res1ToRes2*
# /home/dryu/store/DijetSkim/v2_0_4/Res1ToRes2GluTo3Glu_2017/
#   Res1ToRes2GluTo3Glu_M1-8000_R-0p9_TuneCP5_13TeV-madgraph-pythia8/Res1ToRes2GluTo3Glu_M1-8000_R-0p9/201211_215140/0000
res1res2_version = "v2_0_7"
for year in ["2017"]:
	for decay_mode in res1tores2_decay_mode:
		for sample in res1tores2_samples:
			if sample.find(decay_mode) < 0:
				continue
			glob_pattern = f"{basedir}/{res1res2_version}/{decay_mode}*{year}/{sample}*/*/*/*/nanoskim*root"
			filelist[year][sample] = glob(glob_pattern)
			if len(filelist[year][sample]) == 0:
				print("WARNING : Found no files for pattern {}".format(glob_pattern))

# Signal Z' to ggg
# /home/dryu/store/DijetSkim/v2_0_4/ZprimeTo3Gluon_2018/ZprimeTo3Gluon_TuneCUETP8M1_13TeV_pythia8/ZprimeTo3Gluon_scan_2018/201211_232950/0000
zprime3g_version = "v2_0_7"
for year in ["2017"]:
	for sample in zprime3g_samples:
		glob_pattern = f"{basedir}/{zprime3g_version}/ZprimeTo3Gluon_{year}/{sample}*/*/*/*/nanoskim*root"
		filelist[year][sample] = glob(glob_pattern)
		if len(filelist[year][sample]) == 0:
			print("WARNING : Found no files for pattern {}".format(glob_pattern))

# Require that the histogram file is present
bad_skims = []
for year in ["2016", "2017", "2018"]:
	for subsample in sorted(filelist[year].keys()):
		for nanoskim_path in copy.deepcopy(filelist[year][subsample]): # Cannot remove elements from a list while looping over same list
			if not os.path.isfile(nanoskim_path.replace("nanoskim", "hists")):
				print("WARNING : Didn't find histogram file {} corresponding to nanoskim file {}".format(nanoskim_path.replace("nanoskim", "hists"), nanoskim_path))
				bad_skims.append(nanoskim_path)
				filelist[year][subsample].remove(nanoskim_path)
if len(bad_skims) >= 1:
	print("\nWARNING : Removed some nanoskims from the index because their histogram file was not found.")
	pprint(bad_skims)


# Record number of events in the original NanoAOD files, before the skim
total_events = {}
triggered_events = {}
selected_events = {}
for year in ["2016", "2017", "2018"]:
	total_events[year] = {}
	triggered_events[year] = {}
	selected_events[year] = {}

	for subsample in sorted(filelist[year].keys()):
		total_events[year][subsample] = 0
		triggered_events[year][subsample] = 0
		selected_events[year][subsample] = 0

		for fpath in [x.replace("nanoskim", "hists") for x in filelist[year][subsample]]:
			histfile = ROOT.TFile(fpath, "READ")
			total_events[year][subsample] += histfile.Get("h_ProcessedEvents").GetBinContent(1)
			triggered_events[year][subsample] += histfile.Get("h_TriggeredEvents").GetBinContent(1)
			selected_events[year][subsample] += histfile.Get("h_TriggeredEvents").GetBinContent(1)

			histfile.Close()

# Data only: record number of events passing each trigger
def zzero():
	return 0
trigger_pass = {}
for year in ["2016", "2017", "2018"]:
	trigger_pass[year] = {}
	for subsample in sorted(filelist[year].keys()):
		if not ("JetHT" in subsample or "SingleMuon" in subsample):
			continue

		trigger_pass[year][subsample] = defaultdict(zzero)
		for fpath in [x.replace("nanoskim", "hists") for x in filelist[year][subsample]]:
			histfile = ROOT.TFile(fpath, "READ")
			trigger_hist = histfile.Get("h_TriggerPass")
			for bin in range(1, trigger_hist.GetNbinsX() + 1):
				trigger_name = trigger_hist.GetXaxis().GetBinLabel(bin)
				trigger_pass[year][subsample][trigger_name] += trigger_hist.GetBinContent(bin)
			histfile.Close()


# Save stuff
with open("filelists.pkl", "wb") as f:
	pickle.dump(filelist, f)

skim_metadata = {
	"total_events": total_events, 
	"triggered_events": triggered_events, 
	"selected_events": selected_events, 
	"trigger_pass": trigger_pass,
}
with open("skim_metadata.pkl", "wb") as f:
	pickle.dump(skim_metadata, f)

# Print stuff
for year in ["2016", "2017", "2018"]:
	print("\n*** {} ***".format(year))
	for subsample in sorted(filelist[year].keys()):
		print("{} : {} : {} files \n".format(year, subsample, len(filelist[year][subsample])))

pprint(skim_metadata)