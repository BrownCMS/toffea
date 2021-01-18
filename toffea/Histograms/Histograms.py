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
                                                hist.Bin("dR_01", "$\\Delta R_{01}$", 100, 0., 10),
                                                hist.Bin("dR_12", "$\\Delta R_{12}$", 100, 0., 10),
                                                hist.Bin("dR_20", "$\\Delta R_{20}$", 100, 0., 10))
        self._accumulator["dEta_ij"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin("dEta_01", "$\\Delta \\eta_{01}$", 100, 0., 5),
                                                hist.Bin("dEta_12", "$\\Delta \\eta_{12}$", 100, 0., 5),
                                                hist.Bin("dEta_20", "$\\Delta \\eta_{20}$", 100, 0., 5))
        self._accumulator["moverM_ij"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin("moverM_01", "$m_{01}/M_{jjj}$", 100, 0, 1),
                                                hist.Bin("moverM_12", "$m_{12}/M_{jjj}$", 100, 0, 1),
                                                hist.Bin("moverM_20", "$m_{20}/M_{jjj}$", 100, 0, 1))                          
        self._accumulator[f"pt_i"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"pt_0", "$p^{T}_{0}$ [GeV]", dijet_binning),
                                                hist.Bin(f"pt_1", "$p^{T}_{1}$ [GeV]", dijet_binning),
                                                hist.Bin(f"pt_2", "$p^{T}_{2}$ [GeV]", dijet_binning))
        self._accumulator[f"eta_i"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"eta_0", "$\\eta_{0}$", 100, -3, 3),
                                                hist.Bin(f"eta_1", "$\\eta_{1}$", 100, -3, 3),
                                                hist.Bin(f"eta_2", "$\\eta_{2}$", 100, -3, 3))
        self._accumulator[f"ptoverM_i"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"ptoverM_0", "$p^{T}_{0}/M_{{jjj}}$", 100, 0, 2.5),
                                                hist.Bin(f"ptoverM_1", "$p^{T}_{1}/M_{{jjj}}$", 100, 0, 2.5),
                                                hist.Bin(f"ptoverM_2", "$p^{T}_{2}/M_{{jjj}}$", 100, 0, 2.5))
                                                    
        self._accumulator[f"dR_i_jk"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"dR_0_12", "$\\Delta R_{0, 12}$", 100, 0., 10),
                                                hist.Bin(f"dR_1_20", "$\\Delta R_{1, 20}$", 100, 0., 10),
                                                hist.Bin(f"dR_2_01", "$\\Delta R_{2, 01}$", 100, 0., 10))
        self._accumulator[f"dEta_i_jk"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"dEta_0_12", "$\\Delta \\eta_{0, 12}$", 100, 0., 8),
                                                hist.Bin(f"dEta_1_20", "$\\Delta \\eta_{1, 20}$", 100, 0., 8),
                                                hist.Bin(f"dEta_2_01", "$\\Delta \\eta_{2, 01}$", 100, 0., 8))
        self._accumulator[f"dPhi_i_jk"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"dPhi_0_12", "$\\Delta \\phi_{0, 12}$", 100, 0., 6.5),
                                                hist.Bin(f"dPhi_1_20", "$\\Delta \\phi_{1, 20}$", 100, 0., 6.5),
                                                hist.Bin(f"dPhi_2_01", "$\\Delta \\phi_{2, 01}$", 100, 0., 6.5))
        # dPt_binning = dijet_binning.copy()
        # dPt_binning.reverse()
        # dPt_binning = [i * -1 for i in dPt_binning][:-1]
        # dPt_binning = dPt_binning+ dijet_binning
        self._accumulator[f"dPtoverM_i_jk"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"dPtoverM_0_12", "$\\Delta p_{T}^{0, 12} / M_{jjj}$", 100, 0, 3),
                                                hist.Bin(f"dPtoverM_1_20", "$\\Delta p^{T}_{1, 20} / M_{jjj}$", 100, 0, 3),
                                                hist.Bin(f"dPtoverM_2_01", "$\\Delta p^{T}_{2, 01} / M_{jjj}$", 100, 0, 3))
                                                
        self._accumulator[f"max_dR"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"max_dR", "$\\Delta R_{max}$", 100, 0, 10))
        self._accumulator[f"max_dEta"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"max_dEta", "$\\Delta \\eta_{max}$", 100, 0, 8))
        self._accumulator[f"min_dR"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"min_dR", "$\\Delta R_{min}$", 100, 0, 5))
        self._accumulator[f"min_dEta"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"min_dEta", "$\\Delta \\eta_{min}$", 100, 0, 5))
        self._accumulator[f"min_pt"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"min_pt", "$p^{T}_{min}$ [GeV]", dijet_binning))  
        self._accumulator[f"max_eta"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"max_eta", "$|\\eta|_{max}$", 100, 0, 2.5))  
        self._accumulator[f"max_dR_j_jj"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"max_dR_j_jj", "$\\Delta R^{max}_{j, jj}$", 100, 0, 12))
        self._accumulator[f"max_dEta_j_jj"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"max_dEta_j_jj", "$\\Delta \\eta^{max}_{j, jj}$", 100, 0, 10))
        self._accumulator[f"max_dPhi_j_jj"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"max_dPhi_j_jj", "$\\Delta \\phi^{max}_{j, jj}$", 100, 0, 6.5))
        self._accumulator[f"min_dR_j_jj"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"min_dR_j_jj", "$\\Delta R^{min}_{j, jj}$", 100, 0, 8))
        self._accumulator[f"min_dEta_j_jj"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"min_dEta_j_jj", "$\\Delta \\eta^{min}_{j, jj}$", 100, 0, 5))
        self._accumulator[f"min_dPhi_j_jj"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"min_dPhi_j_jj", "$\\Delta \\phi^{min}_{j, jj}$", 100, 0, 6.5))
        self._accumulator[f"max_dPtoverM_j_jj"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"max_dPtoverM_j_jj", "max $\\Delta p_{T}^{j, jj} / M_{jjj}$", 100, 0, 4))
        self._accumulator[f"min_dPtoverM_j_jj"] = hist.Hist("Events", 
                                                dataset_axis, 
                                                selection_axis, 
                                                hist.Bin(f"min_dPtoverM_j_jj", "min $\\Delta p_{T}^{j, jj} / M_{jjj}$", 100, 0, 4))

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
        
        jet_k = [2, 0, 1]
        dR_i_jk = selected_jets[:, jet_i].delta_r(selected_jets[:, jet_j] + selected_jets[:, jet_k])
        dEta_i_jk = abs(selected_jets[:, jet_i].eta - (selected_jets[:, jet_j] + selected_jets[:, jet_k]).eta)
        dPhi_i_jk = abs(selected_jets[:, jet_i].phi - (selected_jets[:, jet_j] + selected_jets[:, jet_k]).phi)

        m3j = selected_jets.sum().mass
        
        pt_i_overM = selected_jets.pt / m3j
        m_01_overM = m_ij[:,0] / m3j
        m_12_overM = m_ij[:,1] / m3j
        m_20_overM = m_ij[:,2] / m3j
        dPtoverM_0_12 = abs(selected_jets[:, 0].pt - (selected_jets[:, 1] + selected_jets[:, 2]).pt) / m3j
        dPtoverM_1_20 = abs(selected_jets[:, 1].pt - (selected_jets[:, 2] + selected_jets[:, 0]).pt) / m3j
        dPtoverM_2_01 = abs(selected_jets[:, 2].pt - (selected_jets[:, 0] + selected_jets[:, 1]).pt) / m3j
        
        # Event selection masks
        selection_masks = {}
        # Pre-selection
        selection = PackedSelection()
        selection.add("Dummy", m3j > 000)
        sel_mask = selection.require(**{name: True for name in selection.names})
        selection_masks["Pre-selection"] = sel_mask
        
        # HLT_trigger (this is already done at the beginning)
        # if year == "2016":
            # JetHLT_mask = []
            # if "2016B2" in dataset_name:
                # JetHLT_mask = events.HLT.PFHT800 | events.HLT.PFHT900 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
            # elif "2016H" in dataset_name:
                # JetHLT_mask = events.HLT.PFHT900 | events.HLT.AK8PFJet450 | events.HLT.AK8PFJet500 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
            # else:
                # JetHLT_mask = events.HLT.PFHT800 | events.HLT.PFHT900 | events.HLT.AK8PFJet450 | events.HLT.AK8PFJet500 | events.HLT.PFJet500 | events.HLT.CaloJet500_NoJetID
            # selection_masks["JetHLT"] = JetHLT_mask[event_mask]
        # if year == "2017":
            # JetHLT_mask = events.HLT.PFHT1050 | events.HLT.AK8PFJet500 | events.HLT.AK8PFJet550 | events.HLT.CaloJet500_NoJetID | events.HLT.CaloJet550_NoJetID | events.HLT.PFJet500
            # selection_masks["JetHLT"] = JetHLT_mask[event_mask]
        # if year == "2018":
            # JetHLT_mask = events.HLT.PFHT1050 | events.HLT.AK8PFJet500 | events.HLT.AK8PFJet550 | events.HLT.CaloJet500_NoJetID | events.HLT.CaloJet550_NoJetID | events.HLT.PFJet500
            # selection_masks["JetHLT"] = JetHLT_mask[event_mask]
            
        # Fill histograms
        for selection, selection_mask in selection_masks.items():
                output["mjjj"].fill(dataset=dataset_name, 
                                selection=selection, 
                                mjjj=m3j[selection_mask]
                               )
                               
                output["m_ij"].fill(dataset=dataset_name, 
                                selection=selection, 
                                m_01=m_ij[:,0][selection_mask],
                                m_12=m_ij[:,1][selection_mask],
                                m_20=m_ij[:,2][selection_mask]
                                )
                                
                output["dR_ij"].fill(dataset=dataset_name, 
                                selection=selection, 
                                dR_01=dR_ij[:,0][selection_mask],
                                dR_12=dR_ij[:,1][selection_mask],
                                dR_20=dR_ij[:,2][selection_mask]
                                )
                                
                output["dEta_ij"].fill(dataset=dataset_name, 
                                selection=selection, 
                                dEta_01=dEta_ij[:,0][selection_mask],
                                dEta_12=dEta_ij[:,1][selection_mask],
                                dEta_20=dEta_ij[:,2][selection_mask]
                                )
                                
                output["moverM_ij"].fill(dataset=dataset_name, 
                                selection=selection, 
                                moverM_01=m_01_overM[selection_mask],
                                moverM_12=m_12_overM[selection_mask],
                                moverM_20=m_20_overM[selection_mask]
                                )
                                
                output["pt_i"].fill(dataset=dataset_name, 
                                selection=selection, 
                                pt_0=selected_jets[:, 0][selection_mask].pt,
                                pt_1=selected_jets[:, 1][selection_mask].pt,
                                pt_2=selected_jets[:, 2][selection_mask].pt
                                )
                                
                output["eta_i"].fill(dataset=dataset_name, 
                                selection=selection, 
                                eta_0=selected_jets[:, 0][selection_mask].eta,
                                eta_1=selected_jets[:, 1][selection_mask].eta,
                                eta_2=selected_jets[:, 2][selection_mask].eta
                                )
                                
                output["ptoverM_i"].fill(dataset=dataset_name, 
                                selection=selection, 
                                ptoverM_0=pt_i_overM[:, 0][selection_mask],
                                ptoverM_1=pt_i_overM[:, 1][selection_mask],
                                ptoverM_2=pt_i_overM[:, 2][selection_mask]
                                )
                                
                output["dR_i_jk"].fill(dataset=dataset_name, 
                                selection=selection, 
                                dR_0_12=dR_i_jk[:, 0][selection_mask],
                                dR_1_20=dR_i_jk[:, 1][selection_mask],
                                dR_2_01=dR_i_jk[:, 2][selection_mask]
                                )
                                
                output["dEta_i_jk"].fill(dataset=dataset_name, 
                                selection=selection, 
                                dEta_0_12=dEta_i_jk[:, 0][selection_mask],
                                dEta_1_20=dEta_i_jk[:, 1][selection_mask],
                                dEta_2_01=dEta_i_jk[:, 2][selection_mask]
                                )
                                
                output["dPhi_i_jk"].fill(dataset=dataset_name, 
                                selection=selection, 
                                dPhi_0_12=dPhi_i_jk[:, 0][selection_mask],
                                dPhi_1_20=dPhi_i_jk[:, 1][selection_mask],
                                dPhi_2_01=dPhi_i_jk[:, 2][selection_mask]
                                )
                                
                output["dPtoverM_i_jk"].fill(dataset=dataset_name, 
                                selection=selection, 
                                dPtoverM_0_12=dPtoverM_0_12[selection_mask],
                                dPtoverM_1_20=dPtoverM_1_20[selection_mask],
                                dPtoverM_2_01=dPtoverM_2_01[selection_mask]
                                )
                                
                dR_ij_2fill   = dR_ij[selection_mask]
                dEta_ij_2fill = dEta_ij[selection_mask]
                dR_i_jk_2fill   = dR_i_jk[selection_mask]
                dEta_i_jk_2fill = dEta_i_jk[selection_mask]
                dPhi_i_jk_2fill = dPhi_i_jk[selection_mask]
                dPtoverM_0_12_2fill = dPtoverM_0_12[selection_mask]
                dPtoverM_1_20_2fill = dPtoverM_1_20[selection_mask]
                dPtoverM_2_01_2fill = dPtoverM_2_01[selection_mask]
                selected_jets_2fill = selected_jets[selection_mask]
                max_dR_2fill   = awk.max(dR_ij_2fill, axis=1)
                max_dEta_2fill = awk.max(dEta_ij_2fill, axis=1)
                min_dR_2fill   = awk.min(dR_ij_2fill, axis=1)
                min_dEta_2fill = awk.min(dEta_ij_2fill, axis=1)
                min_pt_2fill   = awk.min(selected_jets_2fill.pt, axis=1)
                max_eta_2fill  = awk.max(abs(selected_jets_2fill.eta), axis=1)
                max_dR_i_jk_2fill = awk.max(dR_i_jk_2fill, axis=1)
                min_dR_i_jk_2fill = awk.min(dR_i_jk_2fill, axis=1)
                max_dEta_i_jk_2fill = awk.max(dEta_i_jk_2fill, axis=1)
                min_dEta_i_jk_2fill = awk.min(dEta_i_jk_2fill, axis=1)
                max_dPhi_i_jk_2fill = awk.max(dPhi_i_jk_2fill, axis=1)
                min_dPhi_i_jk_2fill = awk.min(dPhi_i_jk_2fill, axis=1)
                max_dPtoverM_i_jk_2fill = []
                min_dPtoverM_i_jk_2fill = []
                for pair in zip(dPtoverM_0_12_2fill, dPtoverM_1_20_2fill, dPtoverM_2_01_2fill):
                    max_dPtoverM_i_jk_2fill.append(max(pair))
                    min_dPtoverM_i_jk_2fill.append(min(pair))
                max_dPtoverM_i_jk_2fill = np.array(max_dPtoverM_i_jk_2fill)
                min_dPtoverM_i_jk_2fill = np.array(min_dPtoverM_i_jk_2fill)
                max_dR_2fill = awk.fill_none(max_dR_2fill, -99)
                max_dEta_2fill = awk.fill_none(max_dEta_2fill, -99)
                min_dR_2fill = awk.fill_none(min_dR_2fill, -99)
                min_dEta_2fill = awk.fill_none(min_dEta_2fill, -99)
                min_pt_2fill = awk.fill_none(min_pt_2fill, -99)
                max_eta_2fill = awk.fill_none(max_eta_2fill, -99)
                max_dR_i_jk_2fill = awk.fill_none(max_dR_i_jk_2fill, -99)
                min_dR_i_jk_2fill = awk.fill_none(min_dR_i_jk_2fill, -99)
                max_dEta_i_jk_2fill = awk.fill_none(max_dEta_i_jk_2fill, -99)
                min_dEta_i_jk_2fill = awk.fill_none(min_dEta_i_jk_2fill, -99)
                max_dPhi_i_jk_2fill = awk.fill_none(max_dPhi_i_jk_2fill, -99)
                min_dPhi_i_jk_2fill = awk.fill_none(min_dPhi_i_jk_2fill, -99)
                                
                output["max_dR"].fill(dataset=dataset_name, selection=selection, max_dR=max_dR_2fill)
                                
                output["max_dEta"].fill(dataset=dataset_name, selection=selection, max_dEta=max_dEta_2fill)
                                
                output["min_dR"].fill(dataset=dataset_name, selection=selection, min_dR=min_dR_2fill)
                                
                output["min_dEta"].fill(dataset=dataset_name, selection=selection, min_dEta=min_dEta_2fill)
                                
                output["min_pt"].fill(dataset=dataset_name, selection=selection, min_pt=min_pt_2fill)
                
                output["max_eta"].fill(dataset=dataset_name, selection=selection, max_eta=max_eta_2fill)
                
                output["max_dR_j_jj"].fill(dataset=dataset_name, selection=selection, max_dR_j_jj=max_dR_i_jk_2fill)
                
                output["max_dEta_j_jj"].fill(dataset=dataset_name, selection=selection, max_dEta_j_jj=max_dEta_i_jk_2fill)
                
                output["max_dPhi_j_jj"].fill(dataset=dataset_name, selection=selection, max_dPhi_j_jj=max_dPhi_i_jk_2fill)
                
                output["max_dPtoverM_j_jj"].fill(dataset=dataset_name, selection=selection, max_dPtoverM_j_jj=max_dPtoverM_i_jk_2fill)
                
                output["min_dR_j_jj"].fill(dataset=dataset_name, selection=selection, min_dR_j_jj=min_dR_i_jk_2fill)
                
                output["min_dEta_j_jj"].fill(dataset=dataset_name, selection=selection, min_dEta_j_jj=min_dEta_i_jk_2fill)
                
                output["min_dPhi_j_jj"].fill(dataset=dataset_name, selection=selection, min_dPhi_j_jj=min_dPhi_i_jk_2fill)
                
                output["min_dPtoverM_j_jj"].fill(dataset=dataset_name, selection=selection, min_dPtoverM_j_jj=min_dPtoverM_i_jk_2fill)

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