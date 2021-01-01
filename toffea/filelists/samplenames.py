'''
Module for organizing samples
- Years are handled independently.
- "samples" are top-level processes. Rule of thumb: correspond to an entry on a histogram.
- "subsamples" are parts of samples, like QCD_pt bins or JetHT_Run2018<period>
'''
from pprint import pprint

# Signal: Res1ToRes2
res1tores2_samples = []
res1tores2_decay_mode = []
for decay_mode in ["Res1ToRes2QTo3Q", "Res1ToRes2GluTo3Glu"]:
	res1tores2_decay_mode.append(decay_mode)
	for signal_mass in [500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]:
		for signal_R in ["0p1", "0p2", "0p3", "0p5", "0p7", "0p9"]:
			res1tores2_samples.append("{}_M1-{}_R-{}".format(decay_mode, signal_mass, signal_R))

# Signal: ZPrimeTo3Gluons
signal_masses = [500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000]
zprime3g_samples = []
for signal_mass in signal_masses:
	zprime3g_samples.append(f"ZprimeTo3Gluon_M{signal_mass}")

samples = {
	"2016": {
		"JetHT": [f"JetHT_Run2016{x}" for x in ["B2", "C", "D", "E", "F", "G", "H"]], 
		"SingleMuon": [f"SingleMuon_Run2016{x}" for x in ["B2", "C", "D", "E", "F", "G", "H"]], 
		"QCD": ["QCD_Pt_300to470", "QCD_Pt_470to600", "QCD_Pt_600to800", "QCD_Pt_800to1000", "QCD_Pt_1000to1400", "QCD_Pt_1400to1800", "QCD_Pt_1800to2400", "QCD_Pt_2400to3200", "QCD_Pt_3200toInf"],
	}, 
	"2017": {
		"JetHT": [f"JetHT_Run2017{x}" for x in ["B", "C", "D", "E", "F"]], 
		"SingleMuon": [f"SingleMuon_Run2017{x}" for x in ["B", "C", "D", "E", "F"]], 
		"QCD": ["QCD_Pt_300to470", "QCD_Pt_470to600", "QCD_Pt_600to800", "QCD_Pt_800to1000", "QCD_Pt_1000to1400", "QCD_Pt_1400to1800", "QCD_Pt_1800to2400", "QCD_Pt_2400to3200", "QCD_Pt_3200toInf"],
	}, 
	"2018": {
		"JetHT": [f"JetHT_Run2018{x}" for x in ["B", "C", "D"]], 
		"SingleMuon": [f"SingleMuon_Run2018{x}" for x in ["B", "C", "D"]], 
		"QCD": ["QCD_Pt_300to470", "QCD_Pt_470to600", "QCD_Pt_600to800", "QCD_Pt_800to1000", "QCD_Pt_1000to1400", "QCD_Pt_1400to1800", "QCD_Pt_1800to2400", "QCD_Pt_2400to3200", "QCD_Pt_3200toInf"],
	}
}

# Add signal samples to dictionary
for year in ["2016", "2017", "2018"]:
	for sample_name in res1tores2_samples + zprime3g_samples:
		samples[year][sample_name] = [sample_name]

# Subsamples
subsamples = {}
for year in ["2016", "2017", "2018"]:
	subsamples[year] = []
	for sample, subsample_list in samples[year].items():
		subsamples[year].extend(subsample_list)

if __name__ == "__main__":
	print("*** Printing sample configuration ***")
	print("\n\nSamples:")
	pprint(samples)
	#print("\n***********************\n\nSubsamples:")
	#pprint(subsamples)

