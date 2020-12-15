# filelists
This submodule contains the infrastructure for managing the input files (nanoAOD skims). The various samples are organized by "year" (2016, 2017, or 2018), "sample" (top level processes, corresponding to an entry in a histogram) and "subsample" (low level processes, corresponding to a CMS dataset/one CRAB submission). Some specific examples:

- Data: the samples are "JetHT" and "SingleMuon". The subsamples are the run periods, like "JetHT_Run2016C".
- QCD MC: the sample name is "QCD", and the subsamples are the pT slices, "QCD_Pt_470to600". 
- Signal:
   - Res1ToRes2* samples: one sample and subsample for each signal point (there is one dataset)

- `index_skims.py`: Makes a big dictionary of `year : subsa`