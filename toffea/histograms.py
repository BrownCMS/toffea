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
from coffea import hist
from coffea import lookup_tools
from coffea import util
import coffea.processor as processor
from coffea.nanoevents.schemas import NanoAODSchema
import awkward1 as awk
import copy
from coffea.analysis_objects import JaggedCandidateArray
from toffea.common.binning import dijet_binning

np.set_printoptions(threshold=np.inf)

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
                                                hist.Bin("mjjj", r"$m_{jjj}$ [GeV]", dijet_binning), 
                                                )

        for pair in [(0, 1), (1, 2), (2, 0)]:
            self._accumulator[f"m{pair[0]}{pair[1]}"] = hist.Hist("Events", 
                                                    dataset_axis, 
                                                    selection_axis, 
                                                    hist.Bin(f"m{pair[0]}{pair[1]}", f"$m_{{{pair[0]}{pair[1]}}}$ [GeV]$", dijet_binning))
            self._accumulator[f"dR{pair[0]}{pair[1]}"] = hist.Hist("Events", 
                                                    dataset_axis, 
                                                    selection_axis, 
                                                    hist.Bin(f"dR{pair[0]}{pair[1]}", f"$\\Delta R_{{{pair[0]}{pair[1]}}}$ [GeV]$", 75, 0., 7.5))
            self._accumulator[f"dEta{pair[0]}{pair[1]}"] = hist.Hist("Events", 
                                                    dataset_axis, 
                                                    selection_axis, 
                                                    hist.Bin(f"dEta{pair[0]}{pair[1]}", f"$\\Delta \\eta_{{{pair[0]}{pair[1]}}}$ [GeV]$", 75, 0., 7.5))


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self._accumulator.identity()
        print(events.keys())
        dataset_name = events.metadata['dataset']
        output["total_events"][dataset_name] += events.size

        # Require 3 jets
        #events_3j = events[events.nJet >= 3]
        jet_mask = (events.Jet.pt > 30.) & (abs(events.Jet.eta) < 2.5) & (events.Jet.isTight)
        event_mask = awk.sum(jet_mask, axis=1) >= 3
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

        m_ij = (events_3j.Jet[:, jet_i] + events_3j.Jet[:, jet_j]).mass
        dR_ij = events_3j.Jet[:, jet_i].delta_r(events_3j.Jet[:, jet_j])
        dEta_ij = abs(events_3j.Jet[:, jet_i].eta - events_3j.Jet[:, jet_j].eta)

        max_dR   = awk.max(dR_ij, axis=1)
        max_dEta = awk.max(dEta_ij, axis=1)
        min_dR   = awk.min(dR_ij, axis=1)
        min_dEta = awk.min(dEta_ij, axis=1)
        min_pT = awk.min()

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

        # Event selection
        selections = {}
        selections["sr"] = PackedSelection()
        selections["sr"].add("MaxDEta", max_dEta < 1.3)
        selections["sr"].add("MinDR", min_dR > 0.4)
        selections["sr"].add("MinJetPt", awk.min(selected_jets.pt) > 50.)
         

        return output

    def postprocess(self, accumulator):
            return accumulator

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Make histograms for B FFR data")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--subsamples", "-d", type=str, help="List of subsamples to run (comma-separated")
    input_group.add_argument("--quicktest", "-q", action="store_true", help="Run a small test job")
    parser.add_argument("--year", "-y", type=str, help="Year: 2016, 2017, or 2018")
    parser.add_argument("--isMC", "-m", action="store_true", help="Set run over MC instead of collision data")
    parser.add_argument("--workers", "-w", type=int, default=16, help="Number of workers")
    parser.add_argument("--save_tag", "-s", type=str, help="Save tag for output file")
    #parser.add_argument("--nopbar", action="store_true", help="Disable progress bar (do this on condor)")
    parser.add_argument("--condor", action="store_true", help="Flag for running on condor")
    args = parser.parse_args()

    if args.quicktest:
        year = "2017"
        subsamples = ["Res1ToRes2GluTo3Glu_M1-3000_R-0p5"]
        isMC = True
    else:
        year = args.year
        subsamples = args.subsamples.split(",")
        isMC = args.isMC

    from toffea.filelists.filelists import filelist

    # Make dictionary of subsample : [files to run]
    subsample_files = {}
    for subsample_name in subsamples:
        if not subsample_name in filelist[year]:
            raise ValueError(f"Dataset {subsample_name} not in dictionary.")
    for subsample_name in subsamples:
        #with open(filelist[year][subsample_name], 'r') as filelist_txt:
        #    subsample_files[subsample_name] = [x.strip() for x in filelist_txt.readlines()]
        subsample_files[subsample_name] = filelist[year][subsample_name]

        if args.quicktest:
            subsample_files[subsample_name] = subsample_files[subsample_name][:3]

        if args.condor:
            # Copy input files to worker node... seems to fail sporadically when reading remote input files :(
            local_filelist = []
            for remote_file in subsample_files[subsample_name]:
                retry_counter = 0
                expected_path = f"{os.path.expandvars('$_CONDOR_SCRATCH_DIR')}/{os.path.basename(remote_file)}"
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
    print(subsample_files)


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
    util.save(output, f"DataHistograms_{args.save_tag}.coffea")

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