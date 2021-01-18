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
        for pt_cut in range(30,1150,5):
            self._accumulator[f"N_min_pT_cut{pt_cut}".format(pt_cut)] = processor.defaultdict_accumulator(int)
        for eta_cut in np.arange(0,2.5,0.05):
            self._accumulator[f"N_max_eta_cut{eta_cut}".format(eta_cut)] = processor.defaultdict_accumulator(int)
        for dEta_max_cut in np.arange(0,5,0.1):
            self._accumulator[f"N_dEta_jj_max_cut{dEta_max_cut}".format(dEta_max_cut)] = processor.defaultdict_accumulator(int)
        for dR_min_cut in np.arange(0,5,0.1):
            self._accumulator[f"N_dR_jj_min_cut{dR_min_cut}".format(dR_min_cut)] = processor.defaultdict_accumulator(int)

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
        
        max_dR   = awk.max(dR_ij, axis=1)
        max_dEta = awk.max(dEta_ij, axis=1)
        min_dR   = awk.min(dR_ij, axis=1)
        min_dEta = awk.min(dEta_ij, axis=1)
        min_pT   = awk.min(selected_jets.pt, axis=1)
        max_eta  = abs(awk.max(selected_jets.eta, axis=1))
        
        jet_k = [2, 0, 1]
        dR_i_jk = selected_jets[:, jet_i].delta_r(selected_jets[:, jet_j] + selected_jets[:, jet_k])
        dEta_i_jk = abs(selected_jets[:, jet_i].eta - (selected_jets[:, jet_j] + selected_jets[:, jet_k]).eta)
        dPhi_i_jk = abs(selected_jets[:, jet_i].phi - (selected_jets[:, jet_j] + selected_jets[:, jet_k]).phi)
        dPt_i_jk = abs(selected_jets[:, jet_i].pt - (selected_jets[:, jet_j] + selected_jets[:, jet_k]).pt)
        
        max_dPhi_jjj = awk.max(dPhi_i_jk, axis=1)

        m3j = selected_jets.sum().mass
        
        pt_i_overM = selected_jets.pt / m3j
        max_pt_overM = awk.max(pt_i_overM, axis=1)
        min_pt_overM = awk.min(pt_i_overM, axis=1)
        m_01_overM = m_ij[:,0] / m3j
        m_12_overM = m_ij[:,1] / m3j
        m_20_overM = m_ij[:,2] / m3j
        
        for pt_cut in range(30,1150,5):
            cut_name = f"min_pT_cut{pt_cut}".format(pt_cut)
            selection = PackedSelection()
            selection.add("MinJetPt_cut", min_pT > pt_cut)
            sel_mask = selection.require(**{name: True for name in selection.names})
            output[f"N_min_pT_cut{pt_cut}".format(pt_cut)][dataset_name] += events_3j[sel_mask].__len__()
            
        for eta_cut in np.arange(0,2.5,0.05):
            cut_name = f"max_eta_cut{eta_cut}".format(eta_cut)
            selection = PackedSelection()
            selection.add("MaxJetEta_cut", max_eta < eta_cut)
            sel_mask = selection.require(**{name: True for name in selection.names})
            output[f"N_max_eta_cut{eta_cut}".format(eta_cut)][dataset_name] += events_3j[sel_mask].__len__()
            
        for dEta_max_cut in np.arange(0,5,0.1):
            cut_name = f"dEta_max_cut{dEta_max_cut}".format(dEta_max_cut)
            selection = PackedSelection()
            selection.add("MaxJJdEta_cut", max_dEta < dEta_max_cut)
            sel_mask = selection.require(**{name: True for name in selection.names})
            output[f"N_dEta_jj_max_cut{dEta_max_cut}".format(dEta_max_cut)][dataset_name] += events_3j[sel_mask].__len__()
            
        for dR_min_cut in np.arange(0,5,0.1):
            cut_name = f"dR_min_cut{dR_min_cut}".format(dR_min_cut)
            selection = PackedSelection()
            selection.add("MinJJdR_cut", min_dR > dR_min_cut)
            sel_mask = selection.require(**{name: True for name in selection.names})
            output[f"N_dR_jj_min_cut{dR_min_cut}".format(dR_min_cut)][dataset_name] += events_3j[sel_mask].__len__()
            
        #min cut for the variable dPhi_jjj_max
        # for dPhi_jjj_max_min_cut in range(0,6,0.1):
            # cut_name = f"dPhi_jjj_max_min_cut{dPhi_jjj_max_min_cut}".format(dPhi_jjj_max_min_cut)
            # selections[cut_name] = PackedSelection()
            # selections[cut_name].add("j_jj_dPhi_max_cut", min_dR > dPhi_jjj_max_min_cut)
            # selection_items[cut_name] = []
            # selection_items[cut_name].append("j_jj_dPhi_max_cut")
            # sel_mask = HLT_mask & selections[cut_name].require(**{name: True for name in selection_items[cut_name]})
            # output[f"N_dPhi_jjj_max_min_cut{dPhi_jjj_max_min_cut}".format(dPhi_jjj_max_min_cut)][dataset_name] += events_3j[sel_mask].__len__()
            
        # for dPhi_jjj_min_max_cut in range(0,6,0.1):
            # cut_name = f"dPhi_jjj_max_min_cut{dPhi_jjj_max_min_cut}".format(dPhi_jjj_max_min_cut)
            # selections[cut_name] = PackedSelection()
            # selections[cut_name].add("j_jj_dPhi_max_cut", min_dR > dPhi_jjj_max_min_cut)
            # selection_items[cut_name] = []
            # selection_items[cut_name].append("j_jj_dPhi_max_cut")
            # sel_mask = HLT_mask & selections[cut_name].require(**{name: True for name in selection_items[cut_name]})
            # output[f"N_dPhi_jjj_max_min_cut{dPhi_jjj_max_min_cut}".format(dPhi_jjj_max_min_cut)][dataset_name] += events_3j[sel_mask].__len__()

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
        # Drop some abundant QCD MC files
        if "QCD_Pt_600to800" in subsample_name:
            subsample_files[subsample_name] = filelist[year][subsample_name][:13]
        elif "QCD_Pt_800to1000" in subsample_name:
            subsample_files[subsample_name] = filelist[year][subsample_name][:3]
        elif "QCD_Pt_300to470" in subsample_name:
            subsample_files[subsample_name] = filelist[year][subsample_name]
        elif "QCD_Pt_470to600" in subsample_name:
            subsample_files[subsample_name] = filelist[year][subsample_name]
        elif "QCD_Pt_" in subsample_name:
            subsample_files[subsample_name] = filelist[year][subsample_name][:1]
        else:
            subsample_files[subsample_name] = filelist[year][subsample_name]

        if args.quicktest or args.test:
            subsample_files[subsample_name] = subsample_files[subsample_name][:1]

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
            
    for key in subsample_files.keys():
        size = len(subsample_files[key])
        print(f"For {key}, {size} file(s) will be processed")

    ts_start = time.time()

    output = processor.run_uproot_job(subsample_files,
                                        treename='Events',
                                        processor_instance=TrijetHistogramMaker(isMC=isMC),
                                        executor=processor.futures_executor,
                                        chunksize=250000,
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