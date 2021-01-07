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
        for pt_cut in range(100,1000,50):
            self._accumulator[f"N_min_pT_cut{pt_cut}".format(pt_cut)] = processor.defaultdict_accumulator(int)

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
            if "SingleMuon" in dataset_name:
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
        pairs = [(0, 1), (1, 2), (2, 0)]
        jet_i, jet_j = zip(*pairs) # Returns [0, 1, 2] , [1, 2, 0]
        
        m_ij = (selected_jets[:, jet_i] + selected_jets[:, jet_j]).mass
        dR_ij = selected_jets[:, jet_i].delta_r(selected_jets[:, jet_j])
        dEta_ij = abs(selected_jets[:, jet_i].eta - selected_jets[:, jet_j].eta)
        
        dR_0_12 = selected_jets[:, 0].delta_r(selected_jets[:, 1] + selected_jets[:, 2])
        dEta_0_12 = abs(selected_jets[:, 0].eta - (selected_jets[:, 1] + selected_jets[:, 2]).eta)
        dPhi_0_12 = abs(selected_jets[:, 0].phi - (selected_jets[:, 1] + selected_jets[:, 2]).phi)
        dPt_0_12 = selected_jets[:, 0].pt - (selected_jets[:, 1] + selected_jets[:, 2]).pt
        
        max_dR   = awk.max(dR_ij, axis=1)
        max_dEta = awk.max(dEta_ij, axis=1)
        min_dR   = awk.min(dR_ij, axis=1)
        min_dEta = awk.min(dEta_ij, axis=1)
        min_pT   = awk.min(selected_jets.pt, axis=1)

        m3j = selected_jets.sum().mass
        
        pt_i_overM = selected_jets.pt / m3j
        m_01_overM = m_ij[:,0] / m3j
        m_12_overM = m_ij[:,1] / m3j
        m_20_overM = m_ij[:,2] / m3j
        
        # Event selection - pre-selection & HLT_trigger
        selections = {}
        selection_items = {}

        selections["JetHLT"] = PackedSelection()
        selection_items["JetHLT"] = []
        if year == "2016":
            JetHLT_mask = []
            if "2016B2" in dataset_name:
                JetHLT_mask = events.HLT.PFHT800 | events.HLT.PFHT900 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
            elif "2016H" in dataset_name:
                JetHLT_mask = events.HLT.PFHT900 | events.HLT.AK8PFJet450 | events.HLT.AK8PFJet500 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
            else:
                JetHLT_mask = events.HLT.PFHT800 | events.HLT.PFHT900 | events.HLT.AK8PFJet450 | events.HLT.AK8PFJet500 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
            selections["JetHLT"].add("JetHLT_fired", JetHLT_mask[event_mask])
            selection_items["JetHLT"].append("JetHLT_fired")
        if year == "2017":
            JetHLT_mask = events.HLT.PFHT1050 | events.HLT.AK8PFJet500 | events.HLT.AK8PFJet550 | events.HLT.CaloJet500_NoJetID | events.HLT.CaloJet550_NoJetID | events.HLT.PFJet500
            selections["JetHLT"].add("JetHLT_fired", JetHLT_mask[event_mask])
            selection_items["JetHLT"].append("JetHLT_fired")
        if year == "2018":
            JetHLT_mask = events.HLT.PFHT1050 | events.HLT.AK8PFJet500 | events.HLT.AK8PFJet550 | events.HLT.CaloJet500_NoJetID | events.HLT.CaloJet550_NoJetID | events.HLT.PFJet500
            selections["JetHLT"].add("JetHLT_fired", JetHLT_mask[event_mask])
            selection_items["JetHLT"].append("JetHLT_fired")
        # selections["JetHLT"].add("MaxDEta", max_dEta < 1.3)
        # selection_items["JetHLT"].append("MaxDEta")
        # selections["JetHLT"].add("MinDR", min_dR > 0.4)
        # selection_items["JetHLT"].append("MinDR")
        # selections["JetHLT"].add("MinJetPt", min_pT > 50.)
        # selection_items["JetHLT"].append("MinJetPt")
        
        # Implement different cuts
        for pt_cut in range(100,1000,50):
            selections[f"min_pT_cut{pt_cut}".format(pt_cut)] = selections["JetHLT"]
            selections[f"min_pT_cut{pt_cut}".format(pt_cut)].add("MinJetPt_cut", min_pT > pt_cut)
            selection_items[f"min_pT_cut{pt_cut}".format(pt_cut)] = selection_items["JetHLT"]
            selection_items[f"min_pT_cut{pt_cut}".format(pt_cut)].append("MinJetPt_cut")
            sel_mask = selections["JetHLT"].require(**{name: True for name in selection_items["JetHLT"]})
            for selection_name, selection in selections.items():
                if "JetHLT" in selection_name:
                    continue
                print(selection_name)
                sel_mask = sel_mask & selection.require(**{name: True for name in selection_items[selection_name]})
            output[f"N_min_pT_cut{pt_cut}".format(pt_cut)][dataset_name] += events_3j[sel_mask].__len__()

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