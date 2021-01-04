#! /usr/bin/env python
from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import os
import sys
import math
import concurrent.futures
import gzip
import pickle
import json
import time
#import numexpr
import array
from functools import partial
import re

import uproot
import numpy as np
from IPython.display import display
from coffea import hist
from coffea import lookup_tools
from coffea import util
import coffea.processor as processor
from coffea.nanoevents.schemas import NanoAODSchema
import awkward1 as awk
import copy
from coffea.analysis_objects import JaggedCandidateArray
from coffea.analysis_tools import PackedSelection
from toffea.common.binning import dijet_binning


class HackSchema(NanoAODSchema):
    def __init__(self, base_form):
        base_form["contents"].pop("Muon_fsrPhotonIdx", None)
        for key in list(base_form["contents"]):
            if "_genPartIdx" in key:
                base_form["contents"].pop(key, None)
        super().__init__(base_form)

class TrijetHistogramMaker(processor.ProcessorABC):
    def __init__(self, isMC):
        self._isMC = isMC

        # Histograms
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        selection_axis = hist.Cat("selection", "Selection name")


        self._accumulator = processor.dict_accumulator()
        self._accumulator["total_events"] = processor.defaultdict_accumulator(int)

        # Define histograms here
        self._accumulator["mjjj"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin("mjjj", r"$M_{jjj}$ [GeV]", dijet_binning))
        self._accumulator["m_ij"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin("m_01", "$m_{01}$ [GeV]", dijet_binning),
                                                hist.Bin("m_12", "$m_{12}$ [GeV]", dijet_binning),
                                                hist.Bin("m_20", "$m_{20}$ [GeV]", dijet_binning))
        self._accumulator["dR_ij"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin("dR_01", "$\\Delta R_{01}$ [GeV]", 100, 0., 4),
                                                hist.Bin("dR_12", "$\\Delta R_{12}$ [GeV]", 100, 0., 4),
                                                hist.Bin("dR_20", "$\\Delta R_{20}$ [GeV]", 100, 0., 4))
        self._accumulator["dEta_ij"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin("dEta_01", "$\\Delta \\eta_{01}$ [GeV]", 100, 0., 2),
                                                hist.Bin("dEta_12", "$\\Delta \\eta_{12}$ [GeV]", 100, 0., 2),
                                                hist.Bin("dEta_20", "$\\Delta \\eta_{20}$ [GeV]", 100, 0., 2))
        self._accumulator["moverM_ij"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin("moverM_01", "$m_{01}/M_{jjj}$", 100, 0, 1),
                                                hist.Bin("moverM_12", "$m_{12}/M_{jjj}$", 100, 0, 1),
                                                hist.Bin("moverM_20", "$m_{20}/M_{jjj}$", 100, 0, 1))
                                                    
        for jet in [0, 1, 2]:
            self._accumulator[f"pt{jet}"] = hist.Hist("Events", 
                                                    dataset_axis, 
                                                    selection_axis, 
                                                    hist.Bin(f"pt{jet}", r"$p^{T}_{jet}$ [GeV]".format(T="T", jet=jet), dijet_binning))
            self._accumulator[f"eta{jet}"] = hist.Hist("Events", 
                                                    dataset_axis, 
                                                    selection_axis, 
                                                    hist.Bin(f"eta{jet}", f"$\\eta_{jet}$", 100, -3, 3))
            self._accumulator[f"ptoverM{jet}"] = hist.Hist("Events", 
                                                    dataset_axis, 
                                                    selection_axis, 
                                                    hist.Bin(f"ptoverM{jet}", r"$p^{T}_{jet}/M_{{jjj}}$".format(T="T", jet=jet, jjj="jjj"), 100, 0, 2.5))
                                                

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):

        output = self._accumulator.identity()
        dataset_name = events.metadata['dataset']
        output["total_events"][dataset_name] += events.__len__()
        
        # HLT selection
        HLT_mask = []
        if year == "2016":
            if "SingleMuon" in dataset_name: #this does not work, as the name of file which is under processing is unknown
                if "2016B2" in dataset_name:
                    HLT_mask = events.HLT.IsoMu24 | events.HLT.IsoTkMu24 | events.HLT.Mu50
                else:
                    HLT_mask = events.HLT.IsoMu24 | events.HLT.IsoTkMu24 | events.HLT.Mu50 | events.HLT.TkMu50
            else: #https://twiki.cern.ch/twiki/bin/view/CMS/HLTPathsRunIIList
                if "2016B2" in dataset_name:
                    HLT_mask = events.HLT.PFHT800 | events.HLT.PFHT900 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
                elif "2016H" in dataset_name:
                    HLT_mask = events.HLT.PFHT900 | events.HLT.AK8PFJet450 | events.HLT.AK8PFJet500 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
                else:
                    HLT_mask = events.HLT.PFHT800 | events.HLT.PFHT900 | events.HLT.AK8PFJet450 | events.HLT.AK8PFJet500 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
        if year == "2017":
            if "SingleMuon" in dataset_name:
                if "2017B" in dataset_name:
                    HLT_mask = events.HLT.IsoMu27 | events.HLT.Mu50
                else:
                    HLT_mask = events.HLT.IsoMu27 | events.HLT.Mu50 | events.HLT.OldMu100 | events.HLT.TkMu100
            else:
                HLT_mask = events.HLT.PFHT1050 | events.HLT.AK8PFJet500 | events.HLT.AK8PFJet550 | events.HLT.CaloJet500_NoJetID | events.HLT.CaloJet550_NoJetID | events.HLT.PFJet500
        if year == "2018":
            if "SingleMuon" in dataset_name:
                HLT_mask = events.HLT.IsoMu24 | events.HLT.Mu50 | events.HLT.OldMu100 | events.HLT.TkMu100
            else:
                HLT_mask = events.HLT.PFHT1050 | events.HLT.AK8PFJet500 | events.HLT.AK8PFJet550 | events.HLT.CaloJet500_NoJetID | events.HLT.CaloJet550_NoJetID | events.HLT.PFJet500
        
        # Require 3 jets
        jet_mask = (events.Jet.pt > 30.) & (abs(events.Jet.eta) < 2.5) & (events.Jet.isTight)
        event_mask = (awk.sum(jet_mask, axis=1) >= 3)
        event_mask = event_mask & HLT_mask
        events_3j = events[event_mask]
        
        # Reduce jet mask to only events with 3 good jets
        jet_mask = jet_mask[event_mask]

        # Array of the jets to consider for trijet resonance
        selected_jets = events_3j.Jet[jet_mask][:, :3]

        # Pairs of jets
        #pairs = awk.argcombinations(selected_jets, 2)
        #jet_i, jet_j = awk.unzip(pairs)
        pairs = [(0, 1), (1, 2), (2, 0)]
        jet_i, jet_j = zip(*pairs) # Returns [0, 1, 2] , [1, 2, 0]
        
        m_ij = (selected_jets[:, jet_i] + selected_jets[:, jet_j]).mass
        dR_ij = selected_jets[:, jet_i].delta_r(selected_jets[:, jet_j])
        dEta_ij = abs(selected_jets[:, jet_i].eta - selected_jets[:, jet_j].eta)
        
        max_dR   = awk.max(dR_ij, axis=1)
        max_dEta = awk.max(dEta_ij, axis=1)
        min_dR   = awk.min(dR_ij, axis=1)
        min_dEta = awk.min(dEta_ij, axis=1)
        min_pT   = awk.min(selected_jets.pt, axis=1)

        #m01 = (selected_jets[:, 0] + selected_jets[:, 1]).mass
        #m12 = (selected_jets[:, 1] + selected_jets[:, 2]).mass
        #m20 = (selected_jets[:, 2] + selected_jets[:, 0]).mass
        #dR01 = (selected_jets[:, 0].delta_r(selected_jets[:, 1]))
        #dR12 = (selected_jets[:, 1].delta_r(selected_jets[:, 2]))
        #dR20 = (selected_jets[:, 2].delta_r(selected_jets[:, 0]))
        #dEta01 = abs(selected_jets[:, 0].eta - selected_jets[:, 1].eta)
        #dEta12 = abs(selected_jets[:, 1].eta - selected_jets[:, 2].eta)
        #dEta20 = abs(selected_jets[:, 2].eta - selected_jets[:, 0].eta)

        m3j = selected_jets.sum().mass #(selected_jets[:, 0] + selected_jets[:, 1] + selected_jets[:, 2]).mass
        
        pt_i_overM = selected_jets.pt / m3j
        m_01_overM = m_ij[:,0] / m3j
        m_12_overM = m_ij[:,1] / m3j
        m_20_overM = m_ij[:,2] / m3j
        
        # Event selection - pre-selection
        selections = {}
        selection_items = {}
        selections["pre-selection"] = PackedSelection()
        selection_items["pre-selection"] = []
        selections["pre-selection"].add("MaxDEta", max_dEta < 1.3)
        selection_items["pre-selection"].append("MaxDEta")
        selections["pre-selection"].add("MinDR", min_dR > 0.4)
        selection_items["pre-selection"].append("MinDR")
        selections["pre-selection"].add("MinJetPt", min_pT > 50.)
        selection_items["pre-selection"].append("MinJetPt")
        
        # Event selection - pre-selection & HLT_trigger
        selections["JetHLT - presel"] = PackedSelection()
        selection_items["JetHLT - presel"] = []
        if year == "2016":
            JetHLT_mask = []
            if "2016B2" in dataset_name:
                JetHLT_mask = events.HLT.PFHT800 | events.HLT.PFHT900 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
            elif "2016H" in dataset_name:
                JetHLT_mask = events.HLT.PFHT900 | events.HLT.AK8PFJet450 | events.HLT.AK8PFJet500 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
            else:
                JetHLT_mask = events.HLT.PFHT800 | events.HLT.PFHT900 | events.HLT.AK8PFJet450 | events.HLT.AK8PFJet500 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
            selections["JetHLT - presel"].add("JetHLT_fired", JetHLT_mask[event_mask])
            selection_items["JetHLT - presel"].append("JetHLT_fired")
        if year == "2017":
            JetHLT_mask = events.HLT.PFHT1050 | events.HLT.AK8PFJet500 | events.HLT.AK8PFJet550 | events.HLT.CaloJet500_NoJetID | events.HLT.CaloJet550_NoJetID | events.HLT.PFJet500
            selections["JetHLT - presel"].add("JetHLT_fired", JetHLT_mask[event_mask])
            selection_items["JetHLT - presel"].append("JetHLT_fired")
        if year == "2018":
            JetHLT_mask = events.HLT.PFHT1050 | events.HLT.AK8PFJet500 | events.HLT.AK8PFJet550 | events.HLT.CaloJet500_NoJetID | events.HLT.CaloJet550_NoJetID | events.HLT.PFJet500
            selections["JetHLT - presel"].add("JetHLT_fired", JetHLT_mask[event_mask])
            selection_items["JetHLT - presel"].append("JetHLT_fired")
        selections["JetHLT - presel"].add("MaxDEta", max_dEta < 1.3)
        selection_items["JetHLT - presel"].append("MaxDEta")
        selections["JetHLT - presel"].add("MinDR", min_dR > 0.4)
        selection_items["JetHLT - presel"].append("MinDR")
        selections["JetHLT - presel"].add("MinJetPt", min_pT > 50.)
        selection_items["JetHLT - presel"].append("MinJetPt")
        
        # Fill histograms
        for selection_name, selection in selections.items():
            output["mjjj"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                mjjj=m3j[selection.require(**{name: True for name in selection_items[selection_name]})]
                               )
                               
            output["m_ij"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                m_01=m_ij[:,0][selection.require(**{name: True for name in selection_items[selection_name]})],
                                m_12=m_ij[:,1][selection.require(**{name: True for name in selection_items[selection_name]})],
                                m_20=m_ij[:,2][selection.require(**{name: True for name in selection_items[selection_name]})]
                                )
                                
            output["dR_ij"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                dR_01=dR_ij[:,0][selection.require(**{name: True for name in selection_items[selection_name]})],
                                dR_12=dR_ij[:,1][selection.require(**{name: True for name in selection_items[selection_name]})],
                                dR_20=dR_ij[:,2][selection.require(**{name: True for name in selection_items[selection_name]})]
                                )
                                
            output["dEta_ij"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                dEta_01=dEta_ij[:,0][selection.require(**{name: True for name in selection_items[selection_name]})],
                                dEta_12=dEta_ij[:,1][selection.require(**{name: True for name in selection_items[selection_name]})],
                                dEta_20=dEta_ij[:,2][selection.require(**{name: True for name in selection_items[selection_name]})]
                                )
                                
            output["moverM_ij"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                moverM_01=m_01_overM[selection.require(**{name: True for name in selection_items[selection_name]})],
                                moverM_12=m_12_overM[selection.require(**{name: True for name in selection_items[selection_name]})],
                                moverM_20=m_20_overM[selection.require(**{name: True for name in selection_items[selection_name]})]
                                )
                                
            output["pt0"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                pt0=selected_jets[:, 0][selection.require(**{name: True for name in selection_items[selection_name]})].pt
                                )
                                
            output["pt1"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                pt1=selected_jets[:, 1][selection.require(**{name: True for name in selection_items[selection_name]})].pt
                                )
                                
            output["pt2"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                pt2=selected_jets[:, 2][selection.require(**{name: True for name in selection_items[selection_name]})].pt
                                )
                                
            output["eta0"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                eta0=selected_jets[:, 0][selection.require(**{name: True for name in selection_items[selection_name]})].eta
                                )
                                
            output["eta1"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                eta1=selected_jets[:, 1][selection.require(**{name: True for name in selection_items[selection_name]})].eta
                                )
                                
            output["eta2"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                eta2=selected_jets[:, 2][selection.require(**{name: True for name in selection_items[selection_name]})].eta
                                )
            output["ptoverM0"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                ptoverM0=pt_i_overM[:, 0][selection.require(**{name: True for name in selection_items[selection_name]})]
                                )
            output["ptoverM1"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                ptoverM1=pt_i_overM[:, 1][selection.require(**{name: True for name in selection_items[selection_name]})]
                                )
            output["ptoverM2"].fill(dataset=dataset_name, 
                                selection=selection_name, 
                                ptoverM2=pt_i_overM[:, 2][selection.require(**{name: True for name in selection_items[selection_name]})]
                                )

        return output

    def postprocess(self, accumulator):
            return accumulator

if __name__ == "__main__":

    from toffea.filelists.filelists import filelist
    from toffea.filelists.samplenames import samples
    import argparse
    parser = argparse.ArgumentParser(description="Make histograms for Trijet data")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--subsamples", "-d", type=str, help="List of subsamples to run (comma-separated)")
    input_group.add_argument("--allsamples", "-a", type=str, help="Run all subsamples of the given sample (comma-separated)")
    input_group.add_argument("--test", "-t", action="store_true", help="Run a small test job")
    parser.add_argument("--quicktest", "-q", action="store_true", help="Run a small test job on selected dataset")
    parser.add_argument("--year", "-y", type=str, help="Year: 2016, 2017, or 2018")
    parser.add_argument("--isMC", "-m", action="store_true", help="Set run over MC instead of collision data")
    parser.add_argument("--workers", "-w", type=int, default=32, help="Number of workers")
    parser.add_argument("--save_tag", "-o", type=str, help="Save tag for output file")
    #parser.add_argument("--nopbar", action="store_true", help="Disable progress bar (do this on condor)")
    parser.add_argument("--condor", action="store_true", help="Flag for running on condor")
    args = parser.parse_args()
    
    samples2process = []

    if args.test:
        year = "2017"
        samples2process = ["Res1ToRes2GluTo3Glu_M1-1000_R-0p5"]
        isMC = True
        save_tag = "test"
    elif args.allsamples:
        year = args.year
        allsamples = args.allsamples.split(",")
        isMC = args.isMC
        save_tag = args.save_tag
        for item in allsamples:
            if "QCD" in item:
                samples2process += samples[year]["QCD"]
            if "SingleMuon" in item:
                samples2process += samples[year]["SingleMuon"]
            if "Res1ToRes2GluTo3Glu" in item:               
                samples2process += samples[year]["Res1ToRes2GluTo3Glu"]
            if "Res1ToRes2QTo3Q" in item:
                samples2process += samples[year]["Res1ToRes2QTo3Q"]
            if "ZprimeTo3Gluon" in item:
                samples2process += samples[year]["ZprimeTo3Gluon"]
            break
    elif args.subsamples:
        year = args.year
        samples2process = args.subsamples.split(",")
        isMC = args.isMC
        save_tag = args.save_tag
    
    print("Please check samples to process: ", samples2process)

    # Make dictionary of subsample : [files to run]
    subsample_files = {}
    for subsample_name in samples2process:
        if not subsample_name in filelist[year]:
            raise ValueError(f"Dataset {subsample_name} not in dictionary.")
    for subsample_name in samples2process:
        subsample_files[subsample_name] = filelist[year][subsample_name]

        if args.quicktest or args.test:
            subsample_files[subsample_name] = subsample_files[subsample_name][:3]

        if args.condor:
            # Copy input files to worker node... seems to fail sporadically when reading remote input files :(
            local_filelist = []
            for remote_file in subsample_files[subsample_name]:
                print(remote_file)
                retry_counter = 0
                expected_path = f"{os.path.expandvars('$_CONDOR_SCRATCH_DIR')}/{os.path.basename(remote_file)}"
                print(expected_path)
                while retry_counter < 5 and not (os.path.isfile(expected_path) and os.path.getsize(expected_path) > 1.e6):
                    if retry_counter >= 1:
                        time.sleep(10)
                    os.system(f"cp {remote_file} $_CONDOR_SCRATCH_DIR")
                    retry_counter += 1
                if not (os.path.isfile(expected_path) and os.path.getsize(expected_path) > 1.e6):
                    raise RuntimeError("FATAL : Failed to copy file {}".format(remote_file))
                os.system("ls -lrth $_CONDOR_SCRATCH_DIR")
                local_filelist.append(f"{os.path.expandvars('$_CONDOR_SCRATCH_DIR')}/{os.path.basename(remote_file)}")
            subsample_files[subsample_name] = local_filelist

    ts_start = time.time()

    output = processor.run_uproot_job(subsample_files,
                                        treename='Events',
                                        processor_instance=TrijetHistogramMaker(isMC=isMC),
                                        executor=processor.futures_executor,
                                        chunksize=50000,
                                        executor_args={
                                            'workers': args.workers, 
                                            'flatten': False, 
                                            'status':not args.condor, 
                                            "schema": HackSchema},
                                        # maxchunks=1,
                                        )
    util.save(output, f"DataHistograms_{save_tag}.coffea")

    # Performance benchmarking and cutflows
    ts_end = time.time()
    total_events = 0
    subsample_nevents = {}
    for k, v in output['total_events'].items():
        if k in subsample_nevents:
            subsample_nevents[k] += v
        else:
            subsample_nevents[k] = v
        total_events += v

    print("Total time: {} seconds".format(ts_end - ts_start))
    print("Total rate: {} Hz".format(total_events / (ts_end - ts_start)))

    if args.condor:
        os.system("mkdir -pv $_CONDOR_SCRATCH_DIR/hide")
        os.system("mv $_CONDOR_SCRATCH_DIR/*Run2018*root $_CONDOR_SCRATCH_DIR/hide")