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
import numexpr
import array
from functools import partial
import re

import uproot
import numpy as np
from coffea import hist
from coffea import lookup_tools
from coffea import util
import coffea.processor as processor
import awkward
import copy
from coffea.analysis_objects import JaggedCandidateArray
from toffea.common.binning import dijet_binning

np.set_printoptions(threshold=np.inf)

class DataProcessor(processor.ProcessorABC):
  def __init__(self):
    # Histograms
    dataset_axis = hist.Cat("dataset", "Primary dataset")
    selection_axis = hist.Cat("selection", "Selection name")

    self._accumulator = processor.dict_accumulator()
    self._accumulator["nevents"] = processor.defaultdict_accumulator(int)

    self._accumulator["mjjj"] = hist.Hist("Events", 
                                          dataset_axis, 
                                          selection_axis, 
                                          hist.Bin("mjjj", r"$m_{jjj}$ [GeV]", dijet_binning),
                                        )


  @property
  def accumulator(self):
    return self._accumulator

  def process(self, df):
    output = self._accumulator.identity()
    dataset_name = df['dataset']
    match_subjob = self._re_subjob.search(dataset_name)
    if match_subjob:
      dataset_name.replace(match_subjob.group("subjob_tag"), "")
    output["nevents"][dataset_name] += df.size


    return output

  def postprocess(self, accumulator):
      return accumulator

if __name__ == "__main__":

  import argparse
  parser = argparse.ArgumentParser(description="Make histograms for B FFR data")
  parser.add_argument("--datasets", "-d", type=str, help="List of datasets to run (comma-separated")
  parser.add_argument("--workers", "-w", type=int, default=16, help="Number of workers")
  parser.add_argument("--quicktest", "-q", action="store_true", help="Run a small test job")
  parser.add_argument("--save_tag", "-s", type=str, help="Save tag for output file")
  #parser.add_argument("--nopbar", action="store_true", help="Disable progress bar (do this on condor)")
  parser.add_argument("--condor", action="store_true", help="Flag for running on condor")
  args = parser.parse_args()

  if args.quicktest:
    datasets = ["Run2018D_part2_subjob0"]
  else:
    datasets = args.datasets.split(",")

  from data_index import in_txt

  dataset_files = {}
  for dataset_name in datasets:
    if not dataset_name in in_txt:
      raise ValueError(f"Dataset {dataset_name} not in dictionary.")

    with open(in_txt[dataset_name], 'r') as filelist:
      dataset_files[dataset_name] = [x.strip() for x in filelist.readlines()]

    if args.quicktest:
      dataset_files[dataset_name] = dataset_files[dataset_name][:3]

    if args.condor:
      # Copy input files to worker node... seems to fail sporadically when reading remote :(
      local_filelist = []
      for remote_file in dataset_files[dataset_name]:
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
      dataset_files[dataset_name] = local_filelist

  ts_start = time.time()
  print(dataset_files)
  output = processor.run_uproot_job(dataset_files,
                                treename='Events',
                                processor_instance=DataProcessor(),
                                executor=processor.futures_executor,
                                executor_args={'workers': args.workers, 'flatten': False, 'status':not args.condor},
                                chunksize=50000,
                                # maxchunks=1,
                            )
  util.save(output, f"DataHistograms_{args.save_tag}.coffea")

  # Performance benchmarking and cutflows
  ts_end = time.time()
  total_events = 0
  dataset_nevents = {}
  for k, v in output['nevents'].items():
    if k in dataset_nevents:
      dataset_nevents[k] += v
    else:
      dataset_nevents[k] = v
    total_events += v

  print("Total time: {} seconds".format(ts_end - ts_start))
  print("Total rate: {} Hz".format(total_events / (ts_end - ts_start)))

  if args.condor:
    os.system("mkdir -pv $_CONDOR_SCRATCH_DIR/hide")
    os.system("mv $_CONDOR_SCRATCH_DIR/*Run2018*root $_CONDOR_SCRATCH_DIR/hide")