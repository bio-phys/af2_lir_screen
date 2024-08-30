#Python functions to analyze the output from a fragment AlphaFold2 scan via Alphapulldown with the goal of identifying LIRs and otehr SLiMs
#Version 2024-08-30, Jan Stuke

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.distances as distances
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis.rms import rmsd
import mdtraj as mdt
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.PDBParser import PDBParser
import pymol2
import math
import pickle
import os
import json

#Parameters
msms_executable='/PATH/TO/MSMS/msms.x86_64Linux2.2.6.1' #For occlusion by residue depth calculation

#Functions

class seq_scan_system:

    @staticmethod
    def load_seq(seq_file):
        """
        seq_file (str): name of fasta file
        """
        with open(seq_file, "r") as f:
            seq = ""
            for line in f:
                if line[0] == ">":
                    header = line
                    name = line[1:].split(" ")[0]
                else:
                    seq = seq + line.replace("\n","")
                    while seq[-1]==" ": #Remove terminal white space from sequence
                        seq=seq[:-1]
        return seq, name
    
    def __init__(self, fasta_path, bait_fasta, cand_fasta, name_add="", prefer_add=True, cand_start_residue_num=1, cand_chain_index=1, n_baits=1, output_dir="./", msa_type="frag_msa", frag_len="any", verbose=False):
        """
        fasta_path (str): path to fasta files
        bait_fasta (str): name of fasta with bait sequence
        cand_fasta (str): name of fasta with candidate sequence
        name_add (str): addition to name some fragments may have
        prefer_addition (bool): if True, fragments with the name addition will be preferred over those without IF their residue range is identical. If False, it will be the other way around.
        cand_start_residue_num (int): resnumber of first residue listed in fasta file
        cand_chain_index (int): index of candidate chain in structure files
        n_baits (int): number of baits used in every prediction
        output_dir (str): path to output directory
        msa_type (str): "frag_msa" (individual msa for every fragment) or "one_msa" (shared msa from full-length sequence)
        frag_len (str or int): "any" will take fragments of any length in the fragment_main_dir. With an int it will only take fragments of that length.
        verbose (bool): en/disable verbose output
        """
        if fasta_path[-1] != "/":
            fasta_path=fasta_path+"/"
        self.bait_fasta=fasta_path+bait_fasta
        self.cand_fasta=fasta_path+cand_fasta
        bait_seq, bait_name=self.load_seq(self.bait_fasta)
        cand_seq, cand_name=self.load_seq(self.cand_fasta)
        self.name=str(n_baits*str(bait_name+"_and_")+cand_name).replace("|","_").replace("\n","")
        self.seq=cand_seq
        self.len_bait_seq=len(bait_seq)
        #Determine starting index of candidate chain
        self.cand_chain_index=cand_chain_index
        self.start_index=self.cand_chain_index*self.len_bait_seq
        self.start_residue_num=cand_start_residue_num
        self.n_baits=n_baits
        self.name_add=name_add
        self.prefer_addition=prefer_add
        self.pLDDT={}
        self.pLDDT_by_max_pLDDT={} 
        self.pLDDT_by_min_minPAE={}
        self.minPAE={}
        self.minPAE_by_max_pLDDT={}
        self.minPAE_by_min_minPAE={}
        self.peaks=[]
        self.msa_type=msa_type
        self.frag_len=frag_len
        self.verbose=verbose
    
    def find_fragments(self, fragment_main_dir, name_addition_optional=True): #If name_addition_optional, the function will still prefer the fragment with or without addition (self.name_add) based on self.prefer_addition if available
        """
        fragment_main_dir (str): path to alphapulldown output for the respective fragments
        name_addition_optional (bool): If False, will only consider fragments with the addition. If True, it will also consider those without.
        """    
        self.fragments=[]
        contains=self.name
        if self.name_add in contains:
            contains_addition=contains #Name with addition
            contains=str(self.name.replace(str(self.name_add), "")) #Name without addition
        else:
            contains_addition=self.name+self.name_add
        if self.verbose==True:
            print(f"Looking for fragments with the name base {contains} and a length of {str(self.frag_len)} residues.")
        else:
            pass
        for file in os.listdir(fragment_main_dir):
            try:
                if self.msa_type=="frag_msa":
                    residues=file.split("-")[-2:]
                    connector=str("-")
                elif self.msa_type=="one_msa":
                    residues=(file.split("_")[-1]).split("-")
                    connector=str("_")
                else:
                    raise Exception(f"Unknown msa_type {self.msa_type}.")
                #Check frag_len
                if self.frag_len=="any":
                    pass
                elif int(self.frag_len)==int(int(residues[1])-int(residues[0])+1):
                    pass
                else:
                    continue
                if contains != contains_addition:
                    if self.prefer_addition==True:
                        if contains_addition == str(file).replace(f"{connector}{str(residues[0])}-{str(residues[1])}",""):
                            self.fragments.append(file)
                            try:
                                self.fragments.remove(str(contains+f"{connector}{str(residues[0])}-{str(residues[1])}"))
                            except:
                                pass
                        elif contains == str(file).replace(f"{connector}{str(residues[0])}-{str(residues[1])}","") and name_addition_optional == True and str(contains_addition+f"{connector}{str(residues[0])}-{str(residues[1])}") not in self.fragments:
                            self.fragments.append(file)
                        else:
                            pass
                    elif self.prefer_addition==False and name_addition_optional==True:
                        if contains == str(file).replace(f"{connector}{str(residues[0])}-{str(residues[1])}",""):
                            self.fragments.append(file)
                            try:
                                self.fragments.remove(str(contains_addition+f"{connector}{str(residues[0])}-{str(residues[1])}"))
                            except:
                                pass
                        elif contains_addition == str(file).replace(f"{connector}{str(residues[0])}-{str(residues[1])}","") and str(contains+f"{connector}{str(residues[0])}-{str(residues[1])}") not in self.fragments:
                            self.fragments.append(file)
                        else:
                            pass
                    else:
                        pass
                else:
                    if contains == str(file).replace(f"{connector}{str(residues[0])}-{str(residues[1])}",""):
                        self.fragments.append(file)
                    else:
                        pass
            except:
                pass
        if self.verbose==True:
            print(f"Found {str(len(self.fragments))} fragments.")
        else:
            pass
 
    def read_fragments(self, fragment_main_dir, av_over_for_minPAE=1):
        """
        fragment_main_dir (str): path to alphapulldown output for the respective fragments
        av_over_for_minPAE (int): use average of the av_over_for_minPAE lowest PAE values for every residue of the fragment to calculate the minPAE
        """
        if fragment_main_dir[-1] != "/":
            fragment_main_dir=fragment_main_dir+"/"
        failed_fragments=[]
        for fragment in self.fragments:
            fragment_unmod_name=str(fragment)
            try:
                if self.msa_type=="frag_msa":
                    residues=fragment.split("-")[-2:]
                    connector="-"
                elif self.msa_type=="one_msa":
                    residues=(fragment.split("_")[-1]).split("-")
                    connector="_"
                else:
                    raise Exception(f"Unknown msa_type {self.msa_type}.")
                if fragment[-1] != "/":
                    fragment=fragment+"/"
                with open(str(fragment_main_dir+fragment+"ranking_debug.json"), "r") as f:
                    top_model=((json.load(f))["order"])[0]
                with open(str(fragment_main_dir+fragment+"result_"+top_model+".pkl"), "rb") as df:
                    data=pickle.load(df)
                    frag_pLDDT=(data["plddt"])[self.start_index:int(self.start_index+1+int(residues[1])-int(residues[0]))]
                    frag_PAE=(data["predicted_aligned_error"])[self.start_index:int(self.start_index+1+int(residues[1])-int(residues[0]))]
                for frag_resnum in range(0, int(residues[1])+1-int(residues[0]), 1):
                    res_pLDDT=frag_pLDDT[frag_resnum]
                    if self.start_index==0:
                        res_minPAE=np.mean(sorted((frag_PAE[frag_resnum])[int(residues[1])+1-int(residues[0]):])[:int(av_over_for_minPAE)], axis=None)
                    elif self.cand_chain_index==self.n_baits:
                        res_minPAE=np.mean(sorted((frag_PAE[frag_resnum])[:int(self.len_bait_seq*self.n_baits)])[:int(av_over_for_minPAE)], axis=None)
                    else:
                        raise Exception("Not supoorted yet.")
                    resnum=int(residues[0])+int(frag_resnum)
                    try:
                        (self.pLDDT[resnum])[fragment[:-1]]=res_pLDDT #Remove the "/" from the end of the fragment name
                    except:
                        self.pLDDT[resnum]={}
                        (self.pLDDT[resnum])[fragment[:-1]]=res_pLDDT
                    try:
                        (self.minPAE[resnum])[fragment[:-1]]=res_minPAE
                    except:
                        self.minPAE[resnum]={}
                        (self.minPAE[resnum])[fragment[:-1]]=res_minPAE
            except:
                failed_fragments.append(fragment_unmod_name) #Removing the fragment directly leads to bugs.
                if self.verbose==True:
                    print(f"Could not load fragment {str(fragment_unmod_name)} either because of incomplete data or because the file names are not compatible.")
        for failed_fragment in failed_fragments:
            self.fragments.remove(failed_fragment)

    def evaluate_structure_of_fragment(self, fragment_main_dir):
        """
        LEGACY, NEVER IMPLEMENTED
        """     
        pass
                                          
    def calc_minmax(self):
        for key in self.pLDDT.keys():
            max_pLDDT_key=([key2 for key2 in self.pLDDT[key].keys() if (self.pLDDT[key])[key2] == max([(self.pLDDT[key])[key2_] for key2_ in self.pLDDT[key].keys()])])[0]
            self.pLDDT_by_max_pLDDT[key]=(float((self.pLDDT[key])[max_pLDDT_key]), str(max_pLDDT_key)) #(float(Value), str(fragment)) 
            self.minPAE_by_max_pLDDT[key]=(float((self.minPAE[key])[max_pLDDT_key]), str(max_pLDDT_key))
        for key in self.minPAE.keys():
            min_minPAE_key=([key2 for key2 in self.minPAE[key].keys() if (self.minPAE[key])[key2] == min([(self.minPAE[key])[key2_] for key2_ in self.minPAE[key].keys()])])[0]
            self.pLDDT_by_min_minPAE[key]=(float((self.pLDDT[key])[min_minPAE_key]), str(min_minPAE_key)) #(float(Value), str(fragment)) 
            self.minPAE_by_min_minPAE[key]=(float((self.minPAE[key])[min_minPAE_key]), str(min_minPAE_key))
        
    def find_peaks(self, find_mode="fragment", min_len=4, max_len=None, min_pLDDT=75.0, max_minPAE=4.0, screen_by="min_minPAE", remove_overlap="never"):
        """
        find_mode (str): find_mode "fragment" scans all fragments individually for peaks, find_mode "sequence" scans over the max/min values for each residue for all fragments (allowing mixing and overlap between fragments). This method should be used with a conservative max_len (e.g. 6)
        min_len (int): minimum length of peaks
        max_len (int): maximum length of peaks
        min_pLDDT (float): minimum pLDDT cutoff
        max_minPAE (float): maximum minPAE cutoff
        screen_by (str): either "min_minPAE" or "max_pLDDTT"
        remove_overlap (str): is only relevant for find_mode "fragment": if peaks from different fragments fully ("full") or partially ("partially") overlap in their residue numbers, all but one will be removed 
        """
        if find_mode=="fragment":
            
            #Find raw peaks
            raw_peaks=[] 
            for fragment in self.fragments:
                if fragment[-1]!="/":
                    fragment+="/"
                if self.msa_type=="frag_msa":
                    residue_range=(fragment[:-1]).split("-")[-2:]
                elif self.msa_type=="one_msa":
                    residue_range=((fragment[:-1]).split("_")[-1]).split("-")
                else:
                    raise Exception(f"Unknown msa_type {self.msa_type}.")
                residue_indices=[i for i in range(0, int(residue_range[-1])-int(residue_range[-2])+1, 1)]
                mode="find"
                if min_len==1:
                    residue_indices_to_search=residue_indices
                else:
                    residue_indices_to_search=residue_indices[:-int(min_len-1)]
                for residue_index in residue_indices_to_search:
                    resnum=residue_index+int(residue_range[-2])
                    av_pLDDT=np.mean([(self.pLDDT[key])[fragment[:-1]] for key in range(resnum, resnum+min_len, 1)], axis=None)
                    av_minPAE=np.mean([(self.minPAE[key])[fragment[:-1]] for key in range(resnum, resnum+min_len, 1)], axis=None)
                    if mode=="find":
                        if av_pLDDT >= min_pLDDT and av_minPAE <= max_minPAE:
                            raw_peak={"Start": int(resnum), "End": int(resnum+min_len-1), "fragment": str(fragment[:-1])}
                            mode="extend"
                            if residue_index+1==len(residue_indices_to_search):
                                raw_peak["End"]=int(resnum+min_len-1)
                                raw_peaks.append(raw_peak)
                                mode="find"
                            else:
                                pass
                        else:
                            continue
                    elif mode=="extend":
                        if av_pLDDT >= min_pLDDT and av_minPAE <= max_minPAE:
                            raw_peak["End"]=int(resnum+min_len-1)
                            if raw_peak["End"]==int(fragment[:-1].split("-")[-1]): #That works for both msa_types
                                raw_peaks.append(raw_peak)
                                mode="find"
                            else:
                                pass
                        else:
                            raw_peaks.append(raw_peak)
                            mode="find"
                            continue

            #Prune peaks to max_len    
            if max_len==None:
                pass
            else:
                for raw_peak in raw_peaks:
                    while raw_peak["End"]-raw_peak["Start"]+1 > max_len:
                        current_start_res=raw_peak["Start"]
                        current_end_res=raw_peak["End"]
                        if screen_by == "min_minPAE":
                            if (self.minPAE[current_start_res])[(raw_peak["fragment"])] >= (self.minPAE[current_end_res])[(raw_peak["fragment"])]:
                                current_start_res=current_start_res+1
                            else:
                                current_end_res=current_end_res-1
                        elif screen_by=="max_pLDDT":
                            if (self.pLDDT[current_start_res])[(raw_peak["fragment"])] >= (self.pLDDT[current_end_res])[(raw_peak["fragment"])]:
                                current_start_res=current_start_res+1
                            else:
                                current_end_res=current_end_res-1
                        else:
                            raise Exception("Unkown 'screen_by' value")
                        raw_peak["Start"]=current_start_res
                        raw_peak["End"]=current_end_res
 
            
            #Remove (completely/partially) overlapping peaks (or don't)
            if remove_overlap=="never":
                for raw_peak in raw_peaks:
                    av_pLDDT=np.mean([(self.pLDDT[key])[(raw_peak["fragment"])] for key in range(raw_peak["Start"], raw_peak["End"]+1 , 1)], axis=None)
                    av_minPAE=np.mean([(self.minPAE[key])[(raw_peak["fragment"])] for key in range(raw_peak["Start"], raw_peak["End"]+1 , 1)], axis=None)
                    peak={"Start": int(raw_peak["Start"]), "End": int(raw_peak["End"]), "av_pLDDT": float(av_pLDDT), "av_minPAE": float(av_minPAE), "fragment": str((raw_peak["fragment"]))}
                    try:
                        peak["Start"]
                        peak["End"]
                        peak["av_pLDDT"]
                        peak["av_minPAE"]
                        peak["fragment"]
                        self.peaks.append(peak)
                    except:
                        print("NOTE: Removed incomplete Peak.")
                        pass

            else: 
                for raw_peak in raw_peaks:
                    better_peak_exists=False
                    for raw_peak_to_compete_against in raw_peaks:
                        if raw_peak_to_compete_against==raw_peak:
                            continue
                        else:
                            #Check overlap
                            overlap_type=None
                            if raw_peak["End"] < raw_peak_to_compete_against["Start"] or raw_peak_to_compete_against["End"] < raw_peak["Start"]: #no overlap
                                continue
                            elif raw_peak["End"] == raw_peak_to_compete_against["End"] or raw_peak["Start"] == raw_peak_to_compete_against["Start"]: #complete overlap type 1
                                overlap_type="full"
                            elif (raw_peak["Start"] < raw_peak_to_compete_against["Start"] and raw_peak["End"] < raw_peak_to_compete_against["End"]) or (raw_peak["Start"] > raw_peak_to_compete_against["Start"] and raw_peak["End"] > raw_peak_to_compete_against["End"]):
                                overlap_type="full"
                            else:
                                overlap_type="partial"
                            #Decide whether to go into overlap removal
                            if (remove_overlap=="full" and overlap_type=="full") or remove_overlap=="partial": 
                                pass
                            else:
                                continue
                            #Prefer longer peak:
                            if raw_peak["End"]-raw_peak["Start"] > raw_peak_to_compete_against["End"]-raw_peak_to_compete_against["Start"]:
                                continue
                            elif raw_peak["End"]-raw_peak["Start"] < raw_peak_to_compete_against["End"]-raw_peak_to_compete_against["Start"]:
                                better_peak_exists=True
                                break
                            else:
                                pass
                            #Prefer peak with higher "sort_by_score"
                            if screen_by=="by_pLDDT":
                                raw_peak_score=np.mean([(self.pLDDT[key])[(raw_peak["fragment"])] for key in range(raw_peak["Start"], raw_peak["End"]+1, 1)], axis=None)
                                raw_peak_to_compete_against_score=np.mean([(self.pLDDT[key])[(raw_peak_to_compete_against["fragment"])] for key in range(raw_peak_to_compete_against["Start"], raw_peak_to_compete_against["End"]+1, 1)], axis=None)
                                if raw_peak_score > raw_peak_to_compete_against_score:
                                    continue
                                elif raw_peak_score < raw_peak_to_compete_against_score:
                                    better_peak_exists=True
                                    break
                                else:
                                    raw_peak_score=np.mean([(self.minPAE[key])[(raw_peak["fragment"])] for key in range(raw_peak["Start"], raw_peak["End"]+1, 1)], axis=None)
                                    raw_peak_to_compete_against_score=np.mean([(self.minPAE[key])[(raw_peak_to_compete_against["fragment"])] for key in range(raw_peak_to_compete_against["Start"], raw_peak_to_compete_against["End"]+1, 1)], axis=None)
                                    if raw_peak_score <= raw_peak_to_compete_against_score:
                                        continue 
                                    else:
                                        better_peak_exists=True
                                        break
                            elif screen_by=="min_minPAE":
                                raw_peak_score=np.mean([(self.minPAE[key])[(raw_peak["fragment"])] for key in range(raw_peak["Start"], raw_peak["End"]+1, 1)], axis=None)
                                raw_peak_to_compete_against_score=np.mean([(self.minPAE[key])[(raw_peak_to_compete_against["fragment"])] for key in range(raw_peak_to_compete_against["Start"], raw_peak_to_compete_against["End"]+1, 1)], axis=None)
                                if raw_peak_score < raw_peak_to_compete_against_score:
                                    continue
                                elif raw_peak_score > raw_peak_to_compete_against_score:
                                    better_peak_exists=True
                                    break
                                else:
                                    raw_peak_score=np.mean([(self.pLDDT[key])[(raw_peak["fragment"])] for key in range(raw_peak["Start"], raw_peak["End"]+1, 1)], axis=None)
                                    raw_peak_to_compete_against_score=np.mean([(self.pLDDT[key])[(raw_peak_to_compete_against["fragment"])] for key in range(raw_peak_to_compete_against["Start"], raw_peak_to_compete_against["End"]+1, 1)], axis=None)
                                    if raw_peak_score >= raw_peak_to_compete_against_score:
                                        continue
                                    else:
                                        better_peak_exists=True
                                        break
                            else:
                                raise Exception("Unkown 'screen_by' value")
                    if better_peak_exists==False:
                        av_pLDDT=np.mean([(self.pLDDT[key])[(raw_peak["fragment"])] for key in range(raw_peak["Start"], raw_peak["End"]+1 , 1)], axis=None)
                        av_minPAE=np.mean([(self.minPAE[key])[(raw_peak["fragment"])] for key in range(raw_peak["Start"], raw_peak["End"]+1 , 1)], axis=None)
                        peak={"Start": int(raw_peak["Start"]), "End": int(raw_peak["End"]), "av_pLDDT": float(av_pLDDT), "av_minPAE": float(av_minPAE), "fragment": str((raw_peak["fragment"]))}
                        try:
                            peak["Start"]
                            peak["End"]
                            peak["av_pLDDT"]
                            peak["av_minPAE"]
                            peak["fragment"]
                            self.peaks.append(peak)
                        except:
                            print("NOTE: Removed incomplete Peak.")
                            pass
                    else:
                        continue

                        
        elif find_mode=="sequence":
         
            #Find potential peaks
            residue_indices=[i for i in range(0, len(self.pLDDT), 1)]
            raw_peaks=[]
            mode="find"
            if min_len==1:
                residue_indices_to_search=residue_indices
            else:
                residue_indices_to_search=residue_indices[:-int(min_len-1)]
            for residue_index in residue_indices_to_search:
                resnum=residue_index+self.start_residue_num
                if screen_by=="min_minPAE":
                    av_pLDDT=np.mean([(self.pLDDT_by_min_minPAE[key])[0] for key in range(resnum, resnum+min_len, 1)], axis=None)
                    av_minPAE=np.mean([(self.minPAE_by_min_minPAE[key])[0] for key in range(resnum, resnum+min_len, 1)], axis=None)
                    if mode=="find":
                        if av_pLDDT >= min_pLDDT and av_minPAE <= max_minPAE:
                            raw_peak={"Start": int(resnum), "End": int(resnum+min_len-1)}
                            mode="extend"
                            if residue_index+1==len(residue_indices_to_search):
                                raw_peak["End"]=int(resnum+min_len-1)
                                raw_peaks.append(raw_peak)
                                mode="find"
                            else:
                                pass
                        else:
                            continue
                    elif mode=="extend":
                        if av_pLDDT >= min_pLDDT and av_minPAE <= max_minPAE:
                            raw_peak["End"]=int(resnum+min_len-1)
                            if raw_peak["End"]==self.start_residue_num+len(self.pLDDT)-1:
                                raw_peaks.append(raw_peak)
                                mode="find"
                            else:
                                pass
                        else:
                            raw_peaks.append(raw_peak)
                            mode="find"
                            continue
                elif screen_by=="max_pLDDT":
                    av_pLDDT=np.mean([(self.pLDDT_by_max_pLDDT[key])[0] for key in range(resnum, resnum+min_len, 1)], axis=None)
                    av_minPAE=np.mean([(self.minPAE_by_max_pLDDT[key])[0] for key in range(resnum, resnum+min_len, 1)], axis=None)
                    if mode=="find":
                        if av_pLDDT >= min_pLDDT and av_minPAE <= max_minPAE:
                            raw_peak={"Start": int(resnum), "End": int(resnum+min_len-1)}
                            mode="extend"
                            if residue_index+1==len(residue_indices_to_search):
                                raw_peak["End"]=int(resnum+min_len-1)
                                raw_peaks.append(raw_peak)
                                mode="find"
                            else:
                                pass
                        else:
                            continue
                    elif mode=="extend":
                        if av_pLDDT >= min_pLDDT and av_minPAE <= max_minPAE:
                            raw_peak["End"]=int(resnum+min_len-1)
                            if raw_peak["End"]==self.start_residue_num+len(self.pLDDT)-1:
                                raw_peaks.append(raw_peak)
                                mode="find"
                            else:
                                pass
                        else:
                            raw_peaks.append(raw_peak)
                            mode="find"
                            continue
                else:
                    raise Exception("Unkown 'screen_by' value")
                    
            #Check if the peaks occur within single fragments
            for raw_peak in raw_peaks:
                potential_fragments=[]
                while raw_peak["End"]-raw_peak["Start"]+1 > max_len:
                    current_start_res=raw_peak["Start"]
                    current_end_res=raw_peak["End"]
                    if screen_by == "min_minPAE":
                        if self.minPAE_by_min_minPAE[current_start_res] >= self.minPAE_by_min_minPAE[current_end_res]:
                            current_start_res=current_start_res+1
                        else:
                            current_end_res=current_end_res-1
                    elif screen_by=="max_pLDDT":
                        if self.pLDDT_by_max_pLDDT[current_start_res] <= self.pLDDT_by_max_pLDDT[current_end_res]:
                            current_start_res=current_start_res+1
                        else:
                            current_end_res=current_end_res-1
                    else:
                        raise Exception("Unkown 'screen_by' value")
                    raw_peak["Start"]=current_start_res
                    raw_peak["End"]=current_end_res
                while potential_fragments==[]:
                    for fragment in self.fragments:
                        if self.msa_type=="frag_msa":
                            residue_range=fragment.split("-")[-2:]
                        elif self.msa_type=="one_msa":
                            residue_range=(fragment.split("_")[-1]).split("-")
                        else:
                            raise Exception(f"Unknown msa_type {self.msa_type}.")
                        if (int(residue_range[-2]) <= raw_peak["Start"] and int(residue_range[-1]) >= raw_peak["End"]):
                            potential_fragments.append(fragment)
                        else:
                            pass
                    if potential_fragments != []:
                        break
                    else:
                        current_start_res=raw_peak["Start"]
                        current_end_res=raw_peak["End"]
                        if screen_by == "min_minPAE":
                            if self.minPAE_by_min_minPAE[current_start_res] >= self.minPAE_by_min_minPAE[current_end_res]:
                                current_start_res=current_start_res+1
                            else:
                                current_end_res=current_end_res-1
                        elif screen_by=="max_pLDDT":
                            if self.pLDDT_by_max_pLDDT[current_start_res] <= self.pLDDT_by_max_pLDDT[current_end_res]:
                                current_start_res=current_start_res+1
                            else:
                                current_end_res=current_end_res-1
                        else:
                            raise Exception("Unkown 'screen_by' value")
                        raw_peak["Start"]=current_start_res
                        raw_peak["End"]=current_end_res
                        
                    if raw_peak["End"]-raw_peak["Start"]+1 < min_len:
                        break
                    else:
                        continue
                peak={}
                for potential_fragment in potential_fragments:
                    av_pLDDT=np.mean([(self.pLDDT[key])[potential_fragment] for key in range(raw_peak["Start"], raw_peak["End"]+1 , 1)], axis=None)
                    av_minPAE=np.mean([(self.minPAE[key])[potential_fragment] for key in range(raw_peak["Start"], raw_peak["End"]+1 , 1)], axis=None)
                    if av_pLDDT >= min_pLDDT and av_minPAE <= max_minPAE:
                        if peak=={}:
                            peak={"Start": int(raw_peak["Start"]), "End": int(raw_peak["End"]), "av_pLDDT": float(av_pLDDT), "av_minPAE": float(av_minPAE), "fragment": str(potential_fragment)}
                        else:
                            if screen_by=="min_minPAE":
                                if av_minPAE < peak["av_minPAE"]:
                                    peak={"Start": int(raw_peak["Start"]), "End": int(raw_peak["End"]), "av_pLDDT": float(av_pLDDT), "av_minPAE": float(av_minPAE), "fragment": str(potential_fragment)}
                                else:
                                    continue
                            elif screen_by=="max_pLDDT":
                                if av_pLDDT > peak["av_pLDDT"]:
                                    peak={"Start": int(raw_peak["Start"]), "End": int(raw_peak["End"]), "av_pLDDT": float(av_pLDDT), "av_minPAE": float(av_minPAE), "fragment": str(potential_fragment)}
                                else:
                                    continue
                            else:
                                raise Exception("Unkown 'screen_by' value")
                    else:
                        continue
                try:
                    peak["Start"]
                    peak["End"]
                    peak["av_pLDDT"]
                    peak["av_minPAE"]
                    peak["fragment"]
                    self.peaks.append(peak)
                except:
                    print("NOTE: Removed incomplete peak.")
                    pass
        
        else:
            print(f"The find_mode '{find_mode}' is not supported. Please use 'sequence' or 'fragment'.")    

        print(f"Found {str(len(self.peaks))} peaks.")

    def classify_peaks(self, fragment_main_dir, peaks_to_use="all", LC3="LC3B", chain_names=["B","C"], dssp_cutoff=-0.5, req_n_hbonds=2, n_contacts_threshold=5, d_contact_cutoff=5, high_conf_only=True, add_H=True): #dssp cutoff in kcal/mol, d_contact_cutoff in A between CAs
        """
        fragment_main_dir (str): path to alphapulldown output for the respective fragments
        peaks_to_use (str): "all" or "corelir" (use predefined residue window)
        LC3 (str): name of bait protein
        chain_names (list of str): chain identifiers for all PDB files
        dssp_cutoff (float): dssp cutoff for h-bonds in kcal/mol
        req_n_hbonds (int): required h-bonds for canonical LIR motif
        n_contacts_treshold (int): number of C_alpha contacts required for an interface
        d_contact_cutoff (int or float): distance cutoff between C_alphas in Angstrom
        high_conf_only (bool): if True only consider high confidence residues in all interactions 
        add_H (bool): add backbone amide protons for h-bond calculations
        """
        if self.verbose==True:
            print("\nClassifying peaks.\n")
        try:
            os.mkdir("./temp/")
        except:
            pass
        #Standard (self.peaks) or custom peaks
        if peaks_to_use=="all": #Classify all standard peaks
            peaks=self.peaks
        elif peaks_to_use=="corelir": #Classify the corelir pseudopeaks
            peaks=self.corelir_scores
        else:
            peaks=peaks_to_use #Classify a custom list of peaks
        #Iterate over all peaks
        for peak in peaks:
            if self.verbose==True:
                print(f"Currently processing {peak}.")
            #Load structure
            if fragment_main_dir[-1] != "/":
                fragment_main_dir+="/"
            if add_H==True:
                with pymol2.PyMOL() as pymol:
                    pymol.cmd.load(str(fragment_main_dir)+str(peak["fragment"]+"/ranked_0.pdb"), 'current_fragment')
                    pymol.cmd.h_add("backbone and not name CA")
                    pymol.cmd.save("./temp/current_fragment.pdb")   
                u_frag = mda.Universe("./temp/current_fragment.pdb")
            else:
                u_frag = mda.Universe(str(fragment_main_dir)+str(peak["fragment"]+"/ranked_0.pdb"))
            if self.msa_type=="frag_msa":
                residues=str(peak["fragment"]).split("-")[-2:]
            elif self.msa_type=="one_msa":
                residues=(str(peak["fragment"]).split("_")[-1]).split("-")
            else:
                raise Exception(f"Unknown msa_type {self.msa_type}.")
            ###                     ###
            ### Other bait proteins ###
            ###                     ###
            if str(LC3)=="other":
                peak["type"]="Other"            
                    #Write out the sequence of the peak
                seq=''
                for i in range(peak["Start"], peak["End"]+1,1):
                    seq+=self.seq[int(i)-int(self.start_residue_num)]
                peak["sequence"]=str(peak["Start"])+"-"+seq+"-"+str(peak["End"])

            ###                        ###
            ### LC3-like bait proteins ###
            ###                        ###
            if LC3 in ["LC3B", "Atg8", "Atg8CL", "Atg8E", "Atg8A", "GABARAP"]:   
                #Check conditions for type of interactions
                    #1) Is it a (non)canonical LC3-LiR interaction?
                        #A) Any residues in HP1?
                distances_HP1 = {"LC3B": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 108 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 52 and name CA"], "dists": [9.75, 5.25]}, #dist in A
                                 "GABARAP": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 104 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 49 and name CA"], "dists": [9.75, 5.25]},
                                 "Atg8": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 104 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 49 and name CA"], "dists": [9.75, 5.25]},
                                 "Atg8CL": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 105 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 50 and name CA"], "dists": [9.75, 5.25]},
                                 "Atg8A": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 106 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 51 and name CA"], "dists": [9.75, 5.25]},
                                 "Atg8E": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 106 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 51 and name CA"], "dists": [9.75, 5.25]}
                                }
                residue_in_HP1 = None
                i=-1
                for residue in u_frag.select_atoms("chainID "+str(chain_names[self.cand_chain_index])).residues:
                    if high_conf_only == True:
                        if int(residues[0]) + 1 + i < peak["Start"] or int(residues[0]) + 1 + i > peak["End"]: #Only include residues from the actual peak if high_conf_only==True
                            i=i+1
                            continue
                        else:
                            i=i+1
                            pass
                    in_HP1 = False
                    for index, ref_atom in enumerate((distances_HP1[LC3])["atoms"]):
                        try:
                            #For alphapulldown data without relaxing the model, this is essentially heavy atom contacts to CA only, since the output structure does not contain H-atomstw
                            dmat = distances.distance_array(residue.atoms.positions, u_frag.select_atoms(ref_atom).positions, box=self.u.dimensions)
                        except:
                            dmat = distances.distance_array(residue.atoms.positions, u_frag.select_atoms(ref_atom).positions) #In case no box is defined
                        if np.amin(dmat) <= float(((distances_HP1[LC3])["dists"])[index]):
                            in_HP1 = True
                        else:
                            in_HP1 = False
                            break
                    if in_HP1 == True:
                        residue_in_HP1 = {"name": str(residue.resname), "resnum": int(residues[0])+i}
                        break
                    else:
                        continue
                        #B) Any residues in HP2?
                distances_HP2 = {"LC3B": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 67 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 54 and name CA"], "dists": [9.50, 4.50]}, #dist in A
                                 "GABARAP": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 64 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 51 and name CA"], "dists": [9.50, 4.50]},
                                 "Atg8": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 64 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 51 and name CA"], "dists": [9.50, 4.50]},
                                 "Atg8CL": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 65 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 52 and name CA"], "dists": [9.50, 4.50]},
                                 "Atg8A": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 66 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 53 and name CA"], "dists": [9.50, 4.50]},
                                 "Atg8E": {"atoms": ["chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 66 and name CA", "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 53 and name CA"], "dists": [9.50, 4.50]}
                                }
                residue_in_HP2 = None
                i=-1
                for residue in u_frag.select_atoms("chainID "+str(chain_names[self.cand_chain_index])).residues:
                    if high_conf_only == True:
                        if int(residues[0]) + 1 + i < peak["Start"] or int(residues[0]) + 1 + i > peak["End"]: #Only include residues from the actual peak if high_conf_only==True
                            i=i+1
                            continue
                        else:
                            i=i+1
                            pass
                    in_HP2 = False
                    for index, ref_atom in enumerate((distances_HP2[LC3])["atoms"]):
                        try: 
                            dmat = distances.distance_array(residue.atoms.positions, u_frag.select_atoms(ref_atom).positions, box=self.u.dimensions)
                        except:
                            dmat = distances.distance_array(residue.atoms.positions, u_frag.select_atoms(ref_atom).positions) #In case no box is defined
                        if np.amin(dmat) <= float(((distances_HP2[LC3])["dists"])[index]):
                            in_HP2 = True
                        else:
                            in_HP2 = False
                            break
                    if in_HP2 == True:
                        residue_in_HP2 = {"name": str(residue.resname), "resnum": int(residues[0])+i}
                        break
                    else:
                        continue
                        #C) Number of H-bonds with beta-sheet
                LC3_betasheet_residues = {"LC3B": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 50:55",
                                          "GABARAP": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 47:52",
                                          "Atg8": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 47:52",
                                          "Atg8CL": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 48:53",
                                          "Atg8A": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 49:54",
                                          "Atg8E": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 49:54"
                                         }
                n_hbonds=0
                i=-1
                for residue in u_frag.select_atoms("chainID "+str(chain_names[self.cand_chain_index])).residues:
                    if high_conf_only == True:
                        if int(residues[0]) + 1 + i < peak["Start"] or int(residues[0]) + 1 + i > peak["End"]: #Only include residues from the actual peak if high_conf_only==True
                            i=i+1
                            continue
                        else:
                            i=i+1
                            pass
                    for residue_LC3 in u_frag.select_atoms(LC3_betasheet_residues[LC3]).residues:
                        try:
                            donor_atomgroup=residue.atoms
                            acceptor_atomgroup=residue_LC3.atoms
                            #Use DSSP definition of an H-bond (https://doi.org/10.1002/bip.360221211)
                            #Select donor and acceptor atoms
                            donor_N = donor_atomgroup.select_atoms("name N")
                            donor_H = donor_atomgroup.select_atoms("name HN H H01 H02 H03")
                            acceptor_O = acceptor_atomgroup.select_atoms("name O")
                            acceptor_C = acceptor_atomgroup.select_atoms("name C")
                            #Calculate N-C, N-O, H-O, and H-C vector
                            NC_vec = donor_N.positions[0]-acceptor_C.positions[0]
                            NO_vec = donor_N.positions[0]-acceptor_O.positions[0]
                            HC_vec = donor_H.positions[0]-acceptor_C.positions[0]
                            HO_vec = donor_H.positions[0]-acceptor_O.positions[0]
                            #Compute DSSP energy
                            dssp_E = 0.42*0.20*332.0*(1/np.linalg.norm(NO_vec)+1/np.linalg.norm(HC_vec)-1/np.linalg.norm(HO_vec)-1/np.linalg.norm(NC_vec)) #kcal/mol
                            #Check criterium
                            if dssp_E <= dssp_cutoff:
                                n_hbonds+=1
                            else:
                                pass
                        except: #A common exception would be a proline in the list of potential donor residues, whcih does not have an backbone amid H
                            pass
                        try:
                            donor_atomgroup=residue_LC3.atoms
                            acceptor_atomgroup=residue.atoms
                            #Use DSSP definition of an H-bond (https://doi.org/10.1002/bip.360221211)
                            #Select donor and acceptor atoms
                            donor_N = donor_atomgroup.select_atoms("name N")
                            donor_H = donor_atomgroup.select_atoms("name HN H H01 H02 H03")
                            acceptor_O = acceptor_atomgroup.select_atoms("name O")
                            acceptor_C = acceptor_atomgroup.select_atoms("name C")
                            #Calculate N-C, N-O, H-O, and H-C vector
                            NC_vec = donor_N.positions[0]-acceptor_C.positions[0]
                            NO_vec = donor_N.positions[0]-acceptor_O.positions[0]
                            HC_vec = donor_H.positions[0]-acceptor_C.positions[0]
                            HO_vec = donor_H.positions[0]-acceptor_O.positions[0]
                            #Compute DSSP energy
                            dssp_E = 0.42*0.20*332.0*(1/np.linalg.norm(NO_vec)+1/np.linalg.norm(HC_vec)-1/np.linalg.norm(HO_vec)-1/np.linalg.norm(NC_vec)) #kcal/mol
                            #Check criterium
                            if dssp_E <= dssp_cutoff:
                                n_hbonds+=1
                            else:
                                pass
                        except:
                            pass
                        #D) Are the residues bound to HP1 and HP2 part of a canonical LiR sequence?
                canonical_LiR=False
                for x in range(0,1,1): #One iteration loop for breaking if one check fails
                    HP1_canonical_residues=["TYR", "TRP", "PHE"]
                    try:
                        if residue_in_HP1["name"].upper() in HP1_canonical_residues:
                            canonical_LiR=True
                        else:
                            canonical_LiR=False
                            break
                    except:
                        canonical_LiR=False
                        break
                    HP2_canonical_residues=["ILE", "LEU", "VAL"]
                    try:
                        if residue_in_HP2["name"].upper() in HP2_canonical_residues:
                            canonical_LiR=True
                        else:
                            canonical_LiR=False
                            break
                    except:
                        canonical_LiR=False
                        break
                    try:
                        if int(residue_in_HP2["resnum"])-int(residue_in_HP1["resnum"]) == 3:
                            canonical_LiR=True
                        else:
                            canonical_LiR=False
                            break
                    except:
                        canonical_LiR=False
                        break
                    #Write out the sequence of the peak and indicate HP1 and HP2 if existing
                seq=''
                for i in range(peak["Start"], peak["End"]+1,1):
                    seq+=self.seq[int(i)-int(self.start_residue_num)]
                    try:
                        if i == residue_in_HP1["resnum"]:
                            seq+="(HP1)"
                            continue
                    except:
                        pass
                    try:
                        if i == residue_in_HP2["resnum"]:
                            seq+="(HP2)"
                            continue
                    except:
                        pass
                peak["sequence"]=str(peak["Start"])+"-"+seq+"-"+str(peak["End"])
                    # Evaluation -> If all are true: canonical, elif at least one is true: non-canonical, else: go to 2)
                canonical_LiR_checks=[]
                        #Check A) HP1
                if residue_in_HP1 != None:
                    canonical_LiR_checks.append(int(1))
                        #Check B) HP2
                if residue_in_HP2 != None:
                    canonical_LiR_checks.append(int(2))
                        #Check C) H-bonds
                if int(n_hbonds) >= int(req_n_hbonds):
                    canonical_LiR_checks.append(int(3))
                        #Check D) 
                if canonical_LiR == True:
                    canonical_LiR_checks.append(int(4))
                if self.verbose==True:
                    print(f"Canonical LiR checks passed {str(canonical_LiR_checks)} (1: HP1, 2: HP2, 3: H-bonds, 4: Can. seq.).") 
                        #Evaluation
                if len(canonical_LiR_checks) == 4:
                    peak["type"] = "Canonical"
                    continue
                elif len(canonical_LiR_checks) != 0:
                    peak["type"] = "Non-canonical"
                    continue
                else:
                    pass
                    #2) Other binding mode proximal to LiR site: number of heavy atom contacts close to LiR binding site
                LC3_LiR_site_residues = {"LC3B": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 50:55 and name CA",
                                         "GABARAP": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 47:52 and name CA",
                                         "Atg8": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 47:52 and name CA",
                                         "Atg8CL": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 48:53 and name CA",
                                         "Atg8A": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 49:54 and name CA",
                                         "Atg8E": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 49:54 and name CA"
                                        }
                if high_conf_only == True:
                    fragment_residues="chainID "+str(chain_names[self.cand_chain_index])+" and name CA and resid " + str(int(peak["Start"])-int(residues[0])+1)+":"+str(int(peak["End"])-int(residues[0])+1) 
                else:
                    fragment_residues="chainID "+str(chain_names[self.cand_chain_index])+" and name CA"
                try: 
                    dmat = distances.distance_array(u_frag.select_atoms(fragment_residues).positions, u_frag.select_atoms(LC3_LiR_site_residues[LC3]).positions, box=self.u.dimensions)
                except:
                    dmat = distances.distance_array(u_frag.select_atoms(fragment_residues).positions, u_frag.select_atoms(LC3_LiR_site_residues[LC3]).positions) #In case no box is defined
                LC3_LiR_site_n_contacts=np.sum(np.where(dmat[:,:] <= d_contact_cutoff, np.int8(1), np.int8(0)), axis=None)
                if LC3_LiR_site_n_contacts >= n_contacts_threshold:
                    peak["type"]="Other at LiR-site"
                    continue
                    #3) Other binding mode proximal to Ub-like binding site: number of heavy atom contacts close to Ub-like binding site
                LC3_UBQlike_site_residues = {"LC3B": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 79:82 and name CA",
                                             "GABARAP": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 76:79 and name CA",
                                             "Atg8": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 76:79 and name CA",
                                             "Atg8CL": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 77:80 and name CA",
                                             "Atg8A": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 78:81 and name CA",
                                             "Atg8E": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 78:81 and name CA"
                                            }
                if high_conf_only == True:
                    fragment_residues="chainID "+str(chain_names[self.cand_chain_index])+" and name CA and resid " + str(int(peak["Start"])-int(residues[0])+1)+":"+str(int(peak["End"])-int(residues[0])+1) 
                else:
                    fragment_residues="chainID "+str(chain_names[self.cand_chain_index])+" and name CA"
                try: 
                    dmat = distances.distance_array(u_frag.select_atoms(fragment_residues).positions, u_frag.select_atoms(LC3_UBQlike_site_residues[LC3]).positions, box=self.u.dimensions)
                except:
                    dmat = distances.distance_array(u_frag.select_atoms(fragment_residues).positions, u_frag.select_atoms(LC3_UBQlike_site_residues[LC3]).positions) #In case no box is defined
                LC3_UBQlike_site_n_contacts=np.sum(np.where(dmat[:,:] <= d_contact_cutoff, np.int8(1), np.int8(0)), axis=None)
                if LC3_UBQlike_site_n_contacts >= n_contacts_threshold:
                    peak["type"]="UBQ-like site"
                    continue
                    #4) Other (currently everything not in 1-3 goes into 4)
                peak["type"]="Other"
            ###                 ###
            ### SUMO-like baits ###
            ###                 ###
            elif LC3 in ["SUMO1", "SUMO2"]:
                #A) Number of H-bonds with beta-sheet
                LC3_betasheet_residues = {"SUMO1": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 33:38",
                                          "SUMO2": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 29:34"
                                         }
                n_hbonds=0
                i=-1
                for residue in u_frag.select_atoms("chainID "+str(chain_names[self.cand_chain_index])).residues:
                    if high_conf_only == True:
                        if int(residues[0]) + 1 + i < peak["Start"] or int(residues[0]) + 1 + i > peak["End"]: #Only include residues from the actual peak if high_conf_only==True
                            i=i+1
                            continue
                        else:
                            i=i+1
                            pass
                    for residue_LC3 in u_frag.select_atoms(LC3_betasheet_residues[LC3]).residues:
                        try:
                            donor_atomgroup=residue.atoms
                            acceptor_atomgroup=residue_LC3.atoms
                            #Use DSSP definition of an H-bond (https://doi.org/10.1002/bip.360221211)
                            #Select donor and acceptor atoms
                            donor_N = donor_atomgroup.select_atoms("name N")
                            donor_H = donor_atomgroup.select_atoms("name HN H H01 H02 H03")
                            acceptor_O = acceptor_atomgroup.select_atoms("name O")
                            acceptor_C = acceptor_atomgroup.select_atoms("name C")
                            #Calculate N-C, N-O, H-O, and H-C vector
                            NC_vec = donor_N.positions[0]-acceptor_C.positions[0]
                            NO_vec = donor_N.positions[0]-acceptor_O.positions[0]
                            HC_vec = donor_H.positions[0]-acceptor_C.positions[0]
                            HO_vec = donor_H.positions[0]-acceptor_O.positions[0]
                            #Compute DSSP energy
                            dssp_E = 0.42*0.20*332.0*(1/np.linalg.norm(NO_vec)+1/np.linalg.norm(HC_vec)-1/np.linalg.norm(HO_vec)-1/np.linalg.norm(NC_vec)) #kcal/mol
                            #Check criterium
                            if dssp_E <= dssp_cutoff:
                                n_hbonds+=1
                            else:
                                pass
                        except: #A common exception would be a proline in the list of potential donor residues, whcih does not have an backbone amid H
                            pass
                        try:
                            donor_atomgroup=residue_LC3.atoms
                            acceptor_atomgroup=residue.atoms
                            #Use DSSP definition of an H-bond (https://doi.org/10.1002/bip.360221211)
                            #Select donor and acceptor atoms
                            donor_N = donor_atomgroup.select_atoms("name N")
                            donor_H = donor_atomgroup.select_atoms("name HN H H01 H02 H03")
                            acceptor_O = acceptor_atomgroup.select_atoms("name O")
                            acceptor_C = acceptor_atomgroup.select_atoms("name C")
                            #Calculate N-C, N-O, H-O, and H-C vector
                            NC_vec = donor_N.positions[0]-acceptor_C.positions[0]
                            NO_vec = donor_N.positions[0]-acceptor_O.positions[0]
                            HC_vec = donor_H.positions[0]-acceptor_C.positions[0]
                            HO_vec = donor_H.positions[0]-acceptor_O.positions[0]
                            #Compute DSSP energy
                            dssp_E = 0.42*0.20*332.0*(1/np.linalg.norm(NO_vec)+1/np.linalg.norm(HC_vec)-1/np.linalg.norm(HO_vec)-1/np.linalg.norm(NC_vec)) #kcal/mol
                            #Check criterium
                            if dssp_E <= dssp_cutoff:
                                n_hbonds+=1
                            else:
                                pass
                        except:
                            pass
                #Write out the sequence of the peak
                seq=''
                for i in range(peak["Start"], peak["End"]+1,1):
                    seq+=self.seq[int(i)-int(self.start_residue_num)]
                peak["sequence"]=str(peak["Start"])+"-"+seq+"-"+str(peak["End"])
                #See if there are enough h-bonds to consider it SIM binding
                if n_hbonds>=req_n_hbonds:
                    peak["type"]="SIM"
                else:
                    peak["type"]="other"
            ###                 ###
            ### FIP200-like baits ###
            ###                 ###
            elif LC3 in ["FIP200-Claw"]:
                #A) Number of H-bonds with beta-sheet
                LC3_betasheet_residues = {"FIP200-Claw": "chainID "+str(chain_names[1-self.cand_chain_index])+" and resid 77:82"
                                         }
                n_hbonds=0
                i=-1
                for residue in u_frag.select_atoms("chainID "+str(chain_names[self.cand_chain_index])).residues:
                    if high_conf_only == True:
                        if int(residues[0]) + 1 + i < peak["Start"] or int(residues[0]) + 1 + i > peak["End"]: #Only include residues from the actual peak if high_conf_only==True
                            i=i+1
                            continue
                        else:
                            i=i+1
                            pass
                    for residue_LC3 in u_frag.select_atoms(LC3_betasheet_residues[LC3]).residues:
                        try:
                            donor_atomgroup=residue.atoms
                            acceptor_atomgroup=residue_LC3.atoms
                            #Use DSSP definition of an H-bond (https://doi.org/10.1002/bip.360221211)
                            #Select donor and acceptor atoms
                            donor_N = donor_atomgroup.select_atoms("name N")
                            donor_H = donor_atomgroup.select_atoms("name HN H H01 H02 H03")
                            acceptor_O = acceptor_atomgroup.select_atoms("name O")
                            acceptor_C = acceptor_atomgroup.select_atoms("name C")
                            #Calculate N-C, N-O, H-O, and H-C vector
                            NC_vec = donor_N.positions[0]-acceptor_C.positions[0]
                            NO_vec = donor_N.positions[0]-acceptor_O.positions[0]
                            HC_vec = donor_H.positions[0]-acceptor_C.positions[0]
                            HO_vec = donor_H.positions[0]-acceptor_O.positions[0]
                            #Compute DSSP energy
                            dssp_E = 0.42*0.20*332.0*(1/np.linalg.norm(NO_vec)+1/np.linalg.norm(HC_vec)-1/np.linalg.norm(HO_vec)-1/np.linalg.norm(NC_vec)) #kcal/mol
                            #Check criterium
                            if dssp_E <= dssp_cutoff:
                                n_hbonds+=1
                            else:
                                pass
                        except: #A common exception would be a proline in the list of potential donor residues, whcih does not have an backbone amid H
                            pass
                        try:
                            donor_atomgroup=residue_LC3.atoms
                            acceptor_atomgroup=residue.atoms
                            #Use DSSP definition of an H-bond (https://doi.org/10.1002/bip.360221211)
                            #Select donor and acceptor atoms
                            donor_N = donor_atomgroup.select_atoms("name N")
                            donor_H = donor_atomgroup.select_atoms("name HN H H01 H02 H03")
                            acceptor_O = acceptor_atomgroup.select_atoms("name O")
                            acceptor_C = acceptor_atomgroup.select_atoms("name C")
                            #Calculate N-C, N-O, H-O, and H-C vector
                            NC_vec = donor_N.positions[0]-acceptor_C.positions[0]
                            NO_vec = donor_N.positions[0]-acceptor_O.positions[0]
                            HC_vec = donor_H.positions[0]-acceptor_C.positions[0]
                            HO_vec = donor_H.positions[0]-acceptor_O.positions[0]
                            #Compute DSSP energy
                            dssp_E = 0.42*0.20*332.0*(1/np.linalg.norm(NO_vec)+1/np.linalg.norm(HC_vec)-1/np.linalg.norm(HO_vec)-1/np.linalg.norm(NC_vec)) #kcal/mol
                            #Check criterium
                            if dssp_E <= dssp_cutoff:
                                n_hbonds+=1
                            else:
                                pass
                        except:
                            pass
                #Write out the sequence of the peak
                seq=''
                for i in range(peak["Start"], peak["End"]+1,1):
                    seq+=self.seq[int(i)-int(self.start_residue_num)]
                peak["sequence"]=str(peak["Start"])+"-"+seq+"-"+str(peak["End"])
                #See if there are enough h-bonds to consider it SIM binding
                if n_hbonds>=req_n_hbonds:
                    peak["type"]="FIR"
                else:
                    peak["type"]="other" 
        pass
    
    def plot_minmax(self, plot_by="min_minPAE", max_pLDDT_plot=100.0, max_minPAE_plot=45.0):
        """
        plot_by (str): "min_minPAE" or "max_pLDDT"
        max_pLDDT_plot (float): upper pLDDT axis limit
        max_minPAE (float): upper minPAE axis limit
        """ 
        #by max pLDDT
        if plot_by=="max_pLDDT":
            #pLDDT
            plot_pLDDT=np.array([[key, (self.pLDDT_by_max_pLDDT[key])[0]] for key in self.pLDDT_by_max_pLDDT.keys()])
            plot_pLDDT=plot_pLDDT[plot_pLDDT[:,0].argsort(kind="mergesort")]
            #minPAE
            plot_minPAE=np.array([[key, (self.minPAE_by_max_pLDDT[key])[0]] for key in self.minPAE_by_max_pLDDT.keys()])
            plot_minPAE=plot_minPAE[plot_minPAE[:,0].argsort(kind="mergesort")]
        #by min minPAE
        elif plot_by=="min_minPAE":
            #pLDDT
            plot_pLDDT=np.array([[key, (self.pLDDT_by_min_minPAE[key])[0]] for key in self.pLDDT_by_min_minPAE.keys()])
            plot_pLDDT=plot_pLDDT[plot_pLDDT[:,0].argsort(kind="mergesort")]
            #minPAE
            plot_minPAE=np.array([[key, (self.minPAE_by_min_minPAE[key])[0]] for key in self.minPAE_by_min_minPAE.keys()])
            plot_minPAE=plot_minPAE[plot_minPAE[:,0].argsort(kind="mergesort")]
        #sorting variable not known, raise exception:
        else:
            raise Exception("Unkown 'plot_by' value")
        
        #Plotting
        
        fig, axs = plt.subplots(2,1,sharex=True, figsize=(12/2.54, 12/2.54))
        fig.subplots_adjust(hspace=0)
        
        axs[0].plot(plot_pLDDT[:,0], plot_pLDDT[:,1], color="purple")
        axs[0].legend(fontsize=10, frameon=False)
        axs[0].set_ylim([0.0,max_pLDDT_plot])
        axs[0].set_ylabel("pLDDT")
        axs[0].grid(True)
        
        axs[1].plot(plot_minPAE[:,0], plot_minPAE[:,1], color="purple")
        axs[1].legend(fontsize=10, frameon=False)
        axs[1].set_xlabel("Residue Number")
        axs[1].set_ylim([0.0, max_minPAE_plot])
        axs[1].set_ylabel(r"minPAE [$\AA$]")
        axs[1].grid(True)
        
                #Peaks
        for peak in self.peaks:
            axs[0].fill([peak["Start"], peak["End"], peak["End"], peak["Start"]], [0.0, 0.0, max_pLDDT_plot, max_pLDDT_plot], color="grey", alpha=0.5)
            axs[1].fill([peak["Start"], peak["End"], peak["End"], peak["Start"]], [0.0, 0.0, max_minPAE_plot, max_minPAE_plot], color="grey", alpha=0.5)
        
        if self.prefer_addition==True: 
            if self.name_add in self.name:
                plot_name=self.name+f"_{str(self.frag_len)}mers_pLDDT_and_minPAE_over_seq.png"
            else:
                plot_name=self.name+self.name_add+f"_{str(self.frag_len)}mers_pLDDT_and_minPAE_over_seq.png"
        else:
            plot_name=self.name.replace(self.name_add,"")+f"_{str(self.frag_len)}mers_pLDDT_and_minPAE_over_seq.png"

        plt.savefig(plot_name, dpi=600)
        plt.show(block=False)
    
    def write_peaks_to_csv(self, overwrite=False):
        """
        overwrite (bool): overwrite existing file if True
        """  
        #File name
        if self.prefer_addition==True:
            if self.name_add in self.name:
                csv_name=self.name+f"_{str(self.frag_len)}mers_peaks.csv"
            else:
                csv_name=self.name+self.name_add+f"_{str(self.frag_len)}mers_peaks.csv"
        else:
            csv_name=self.name.replace(self.name_add,"")+f"_{str(self.frag_len)}mers_peaks.csv"

            #Check if file already exists:
        if os.path.isfile(str(os.getcwd())+"/"+csv_name)==False:
            pass
        else:
            if overwrite==True:
                print(f"NOTE: Overwriting existing file '{csv_name}'.")
            else:
                add_to_file_name=1
                while os.path.isfile(str(os.getcwd())+"/"+csv_name) == True:
                    start_of_appendix=-5-len(str(add_to_file_name-1))
                    if str(csv_name[start_of_appendix:-4])==str(f"_{str(add_to_file_name-1)}"):
                        csv_name=csv_name[:start_of_appendix]+f"_{str(add_to_file_name)}.csv"
                    else:
                        csv_name=csv_name[:-4]+f"_{str(add_to_file_name)}.csv"
                    add_to_file_name+=1
                print(f"NOTE: Chosen file name already exists and you have chosen not to overwrite it. Will write data to f'{csv_name}' instead.")  
        
        #Write self.peaks to file
            #Ordered keys for output
        csv_keys=["Start", "End", "sequence", "type", "fragment", "av_pLDDT", "av_minPAE"]
        
        #Transform peak dictionary into a sorted list of lines for the csv file [sorted by i) Binding site Start Resnum. ii) Fragment Start Resnum.]
        peak_list=[]
        for peak in self.peaks:
            #Check peaks for completeness and setting missing values to None
            for csv_key in csv_keys: 
                try:
                    peak[csv_key]
                except:
                    peak[csv_key]=None
            peak_list.append([peak[key] for key in csv_keys])
        if self.msa_type=="frag_msa":
            peak_list_sorted=sorted(peak_list, key=lambda x: (int(x[int(csv_keys.index("Start"))]), int(x[int(csv_keys.index("fragment"))].split("-")[-2])))
        elif self.msa_type=="one_msa":
            peak_list_sorted=sorted(peak_list, key=lambda x: (int(x[int(csv_keys.index("Start"))]), int((x[int(csv_keys.index("fragment"))].split("_")[-1]).split("-")[-2])))
        else:
            raise Exception(f"Unknown msa_type {self.msa_type}.") 
        
        with open(csv_name, 'w') as f:
            if len(peak_list_sorted)==0:
                f.write(str(csv_keys).replace("[","").replace("]","").replace("'","").replace('"',''))
            else:
                f.write(str(csv_keys).replace("[","").replace("]","").replace("'","").replace('"','')+"\n")
            counter=1
            for peak_data in peak_list_sorted:
                if counter < len(peak_list_sorted):
                    f.write(str(peak_data).replace("[","").replace("]","")+"\n")
                else:
                    f.write(str(peak_data).replace("[","").replace("]",""))
                counter+=1

    def calc_bait_rmsd(self, fragment_main_dir, LC3="LC3B", chain_names=["B","C"], treshold=0.0001, max_cycles=25, overwrite=False):
        """
        fragment_main_dir (str): path to alphapulldown output for the respective fragments
        LC3 (str): name of LC3 protein
        chain_names (str): names of chains in PDB
        treshold (float): treshold for iterative alignemnt in Angstrom
        max_cycles (int): maximum iterations for iterative alignment
        overwrite (bool): overwrite previously calculated RMSDs / RMSFs
        """  
        #Create dictionary to store results
        
        try:
            self.bait_rmsds
            if overwrite==True:
                print("Overwriting existing Bait RMSDs.")
                self.bait_rmsds={}
            else:
                print("Found already existing Bait RMSDs. Checking if extra entries can be added.") #E.g. if a previous run crashed. Will check for each fragment if it is already present in the dictionary and will only add those which are not.
        except:
            self.bait_rmsds={}
        
        try:
            self.bait_rmsfs
            if overwrite==True:
                print("Overwriting existing Bait RMSFs.")
                self.bait_rmsfs={}
            else:
                print("Found already existing Bait RMSFs. Checking if extra entries can be added.") #E.g. if a previous run crashed. Will check for each fragment if it is already present in the dictionary and will only add those which are not.
        except:
            self.bait_rmsfs={}
        
        #Create dictionary of input files
        
        fragment_structures={}
        if fragment_main_dir[-1] != "/":
            fragment_main_dir=fragment_main_dir+"/"
        for fragment in self.fragments:
            if fragment[-1] != "/":
                fragment=fragment+"/"
            model_to_use="ranked_0.pdb"
            fragment_structures[str(fragment[:-1])]=str(fragment_main_dir+fragment+"ranked_0.pdb")
        
        #Generate a temporary PDB for each input file, that only contains the Bait chain. This is necessary to load all bait+fragment complexes as frames of one trajectory. Otherwise their topology (due to the different fragment chains) would be different. 
       
            #Create temp directory
        try:
            os.mkdir("./temp/")
        except:
            pass 
         
        not_yet_ordered_baits=[]
        
        for fragment in fragment_structures.keys():
            u_bait_and_frag = mda.Universe(fragment_structures[fragment])
            sel_bait=u_bait_and_frag.select_atoms("chainID "+str(chain_names[1-self.cand_chain_index]))
            bait_temp_file=f"./temp/bait_for_{fragment}.pdb"
            sel_bait.write(bait_temp_file)
            if self.msa_type=="frag_msa":
                residue_range=fragment.split("-")[-2:]
            elif self.msa_type=="one_msa":
                residue_range=(fragment.split("_")[-1]).split("-")
            else:
                raise Exception(f"Unknown msa_type {self.msa_type}.")

            not_yet_ordered_baits.append([int(residue_range[-2]),str(fragment), str(bait_temp_file)]) # [Residue ID of first residue in fragment, fragment name, file with only bait structure for respective fragment]
        
        not_yet_ordered_baits=np.array(not_yet_ordered_baits, dtype='object')

        #Sort array by column 0 (first residue in respective fragment)
        
        ordered_baits=not_yet_ordered_baits[np.argsort(not_yet_ordered_baits[:,0], axis=0)]
        
        #Selections for alignment and RMSD calculation
        
        LC3_align_on = {"LC3B": "backbone and not resid 1:5 116:125", #Exclude flexible termini
                        "GABARAP": "backbone and not resid 1:3 111:117",
                        "Atg8": "backbone and not resid 1:3 111:117",
                        "Atg8CL": "backbone and not resid 1:4 112:119",
                        "Atg8E": "backbone and not resid 1:5 113:122",
                        "Atg8A": "backbone and not resid 1:5 113:122",
                        "SUMO1": "backbone and resid 22:97",
                        "SUMO2": "backbone and resid 18:93",
                        "FIP200-Claw": "backbone and not resid 1:18 53:66 104:108",
                        "other": "backbone"
                       }
        
        #Create Universe
        u_all_baits = mda.Universe(ordered_baits[0, 2], *(ordered_baits[:, 2]), all_coordinates=False, in_memory=True)
        initial_frame = mda.Universe(ordered_baits[0, 2], in_memory=True)
        protein = u_all_baits.select_atoms(LC3_align_on[LC3])
        protein_ref = initial_frame.select_atoms(LC3_align_on[LC3])

        #Define loop parameters:

        new_reference = initial_frame
        counter = 0
        new_RMSD = 10000000000 #A #High value to avoid abortion before first loop
        delta_RMSD = 10000000000 #A #High value to avoid abortion before first loop
        if delta_RMSD <= treshold:
            print("WARNING: You seem to be having set a strangely high treshold. Are you sure this is what you wanted to do?")
        #Main loop

        while delta_RMSD > treshold and counter <= max_cycles:

            #First alignment on initial structure
            old_reference = new_reference
            prealigner = align.AlignTraj(u_all_baits, old_reference, select=LC3_align_on[LC3], in_memory=True).run()

            #Get average coordinates for reference structure

            reference_coordinates = u_all_baits.trajectory.timeseries(asel=protein).mean(axis=1)

            #Create reference structure

            new_reference = mda.Merge(protein).load_new(reference_coordinates[:,None,:], order="afc")

            #Compute RMSD between old and new reference

            old_RMSD = new_RMSD
            new_RMSD = rmsd(old_reference.select_atoms(LC3_align_on[LC3]).positions, new_reference.select_atoms(LC3_align_on[LC3]).positions, center=False, superposition=False)
            delta_RMSD = abs(old_RMSD - new_RMSD)
            #Count cycles

            counter = counter + 1

        #Echo number of cycles and circumstances of exiting
        if counter <= max_cycles:
            print(f"Finished alignment after {counter} cycles with an RMSD change of {delta_RMSD} which is below the chosen treshold of {treshold}.") 
        elif counter > max_cycles:
            print(f"Aborted alignment after {counter} cycles with an RMSD change of {delta_RMSD} which is higher than the chosen treshold of {treshold}.") 
        #Align trajectroy onto reference

        aligner = align.AlignTraj(u_all_baits, new_reference, select=LC3_align_on[LC3], in_memory=True).run()

        #Calculate C-alpha RMSF

        calphas = u_all_baits.select_atoms("name CA")
        rmsfer = RMSF(calphas, verbose=True).run()
        
        for index, resnum in enumerate(calphas.resnums):
            self.bait_rmsfs[resnum]=rmsfer.rmsf[index]
        
        #Calculate backbone RMSD for every bait structure with respect ot the converged reference
        
        counter=0
        
        for ts in u_all_baits.trajectory:
            bait_rmsd=rmsd(new_reference.select_atoms(LC3_align_on[LC3]).positions, u_all_baits.select_atoms(LC3_align_on[LC3]).positions, center=False, superposition=False)
            self.bait_rmsds[ordered_baits[counter, 1]]=bait_rmsd
            counter+=1
            
    def plot_rmsd(self):
        fig, ax = plt.subplots(1,1,figsize=(8/2.54, 4/2.54))
        if self.msa_type=="frag_msa":
            order_values_and_keys=np.array(sorted([[(int(key.split("-")[-1])-int(key.split("-")[-2]))/2+int(key.split("-")[-2]), str(key)] for key in self.bait_rmsds.keys()]), dtype='object') #Values are resnumbers of the fragment center (can be non-integer)
        elif self.msa_type=="one_msa":
            order_values_and_keys=np.array(sorted([[(int((key.split("_")[-1]).split("-")[-1])-int((key.split("_")[-1]).split("-")[-2]))/2+int((key.split("_")[-1]).split("-")[-2]), str(key)] for key in self.bait_rmsds.keys()]), dtype='object') #Values are resnumbers of the fragment center (can be non-integer)
        else:
            raise Exception(f"Unknown msa_type {msa_type}.")
        ax.plot(order_values_and_keys[:,0], [self.bait_rmsds[key] for key in order_values_and_keys[:,1]], alpha=0.8, linestyle=None)
        ax.set_ylabel(r"Bait BB RMSD [$\AA$]", fontsize=8)
        ax.tick_params(axis='both', which='both', labelsize=6)
        ax.set_xlabel("Fragment Center Resnum.", fontsize=8)
        ax.grid(True, alpha=0.5)
        if self.prefer_addition==True:
            if self.name_add in self.name:
                plot_name=self.name+f"_{str(self.frag_len)}mers_bait_rmsd.png"
            else:
                plot_name=self.name+self.name_add+f"_{str(self.frag_len)}mers_bait_rmsd.png"
        else:
            plot_name=self.name.replace(self.name_add,"")+f"_{str(self.frag_len)}mers_bait_rmsd.png"
        plt.tight_layout()
        plt.savefig(plot_name, dpi=600)
        plt.show(block=False)
        
    def plot_rmsf(self):
        fig, ax = plt.subplots(1,1,figsize=(8/2.54, 4/2.54))
        ax.plot(sorted(self.bait_rmsfs.keys()), [self.bait_rmsfs[key] for key in sorted(self.bait_rmsfs.keys())], alpha=0.8)
        ax.set_ylabel(r"C$_\alpha$ RMSF [$\AA$]", fontsize=8)
        ax.tick_params(axis='both', which='both', labelsize=6)
        ax.set_xlabel("Bait Resnum.", fontsize=8)
        ax.grid(True, alpha=0.5)
        if self.prefer_addition==True:
            if self.name_add in self.name:
                plot_name=self.name+f"_{str(self.frag_len)}mers_bait_rmsf.png"
            else:
                plot_name=self.name+self.name_add+f"_{str(self.frag_len)}mers_bait_rmsf.png"
        else:
            plot_name=self.name.replace(self.name_add,"")+f"_{str(self.frag_len)}mers_bait_rmsf.png"
        plt.tight_layout()
        plt.savefig(plot_name, dpi=600)
        plt.show(block=False)
    
    def get_frag_lens(self):
        #Create dic to store the length of each fragment
        try:
            self.frag_lens
        except:
            self.frag_lens={}
        #Iterate over all fragments:
        for fragment in self.fragments:
            if self.msa_type=="frag_msa":
                residues=fragment.split("-")[-2:]
            elif self.msa_type=="one_msa":
                residues=(fragment.split("_")[-1]).split("-")
            else:
                raise Exception(f"Unknown msa_type {self.msa_type}.")
            if fragment[-1] == "/":
                fragment=fragment[:-1]
            self.frag_lens[fragment]=1+int(residues[1])-int(residues[0])        
        
    def get_corelir_scores(self, start_resnum=None, lir_len=None, sort_by="min_minPAE"):
        """
        start_resnum (int): residue number of first LIR residue
        lir_len (int): length of the LIR
        sort_by (str): "max_pLDDT" or "min_minPAE"
        """
        #Create dic to store data
        try:
            self.corelir_scores
        except:
            self.corelir_scores=[]
        #get av_pLDDT and min_PAE score
        if start_resnum==None and lir_len==None:
            #if the region is not specified, look for peaks in the fragment and take the best one
            for fragment in self.fragments:
                if self.msa_type=="frag_msa":
                    residues=fragment.split("-")[-2:]
                elif self.msa_type=="one_msa":
                    residues=(fragment.split("_")[-1]).split("-")
                else:
                    raise Exception(f"Unknown msa_type {self.msa_type}.")
                if fragment[-1] == "/":
                    fragment=fragment[:-1] 
                candidates=[peak for peak in self.peaks if peak["fragment"]==str(fragment)]
                if candidates==[]:
                    self.corelir_scores.append({"Start": int(residues[0]), "End": int(residues[1]), "fragment": str(fragment), "av_pLDDT": None, "av_minPAE": None})
                    continue
                elif len(candidates)==1:
                    best_peak=candidates[0]
                else:
                    if sort_by=="min_minPAE":
                        key1="av_minPAE"
                        key2="av_pLDDT"
                        sign1=int(1)
                        sign2=int(-1)
                    elif sort_by=="max_pLDDT":
                        key1="av_pLDDT"
                        key2="av_minPAE"
                        sign1=int(-1)
                        sign2=int(1)
                    best_peak=sorted(candidates, key=lambda x: (sign1*int(x[key1]), sign2*int(x[key2])))[0]
                self.corelir_scores.append({"Start": int(residues[0]), "End": int(residues[1]), "fragment": str(fragment), "av_pLDDT": best_peak["av_pLDDT"], "av_minPAE": best_peak["av_minPAE"]})
                     
        elif start_resnum!=None and lir_len!=None:
            for fragment in self.fragments:
                if self.msa_type=="frag_msa":
                    residues=fragment.split("-")[-2:]
                elif self.msa_type=="one_msa":
                    residues=(fragment.split("_")[-1]).split("-")
                else:
                    raise Exception(f"Unknown msa_type {self.msa_type}.")
                if fragment[-1] == "/":
                    fragment=fragment[:-1]
                if int(residues[0]) > int(start_resnum) or int(residues[1]) < int(start_resnum)+int(lir_len)-1:
                    self.corelir_scores.append({"Start": int(residues[0]), "End": int(residues[1]), "fragment": str(fragment), "av_pLDDT": None, "av_minPAE": None})
                    continue  
                av_pLDDT=np.mean([(self.pLDDT[key])[fragment] for key in range(int(start_resnum), int(start_resnum)+int(lir_len) , 1)], axis=None)
                av_minPAE=np.mean([(self.minPAE[key])[fragment] for key in range(int(start_resnum), int(start_resnum)+int(lir_len) , 1)], axis=None)
                self.corelir_scores.append({"Start": int(residues[0]), "End": int(residues[1]), "fragment": str(fragment), "av_pLDDT": av_pLDDT, "av_minPAE": av_minPAE}) 
        else:
            raise Exception(f"Non-valid combination of start_resnum ({str(start_resnum)})and lir_len ({str(lir_len)}). Only allowed combinations are either both are 'None' or both have an integer value.")
    

    def plot_corelir_scores(self, indicate_msa_at_len=15, max_pLDDT_plot=100.0, max_minPAE_plot=45.0, len_min=None, len_max=None):
        """
        indicate_msa_at_len (int): minimum fragment length for the AkphaFold MSA
        max_pLDDT_plot (float): upper pLDDT axis limit
        max_minPAE (float): upper minPAE axis limit
        len_min (int): minimum fragment length
        len_max (int): maximum fragment length
        """
        #Check if data to plot is available and run respective functions if not.
        try:
            self.frag_lens
        except:
            self.get_frag_lens()
        try:
            self.corelir_scores
        except:
            self.get_corelir_scores()
        if self.corelir_scores==[]:
            print("Cannot plot corelir_scores because there are none.")
        #Prepare data for plotting
        corelir_dic={}
        for corelir in self.corelir_scores:
            corelir_dic[(corelir["fragment"])]=corelir     
        sorted_scores=np.array([[self.frag_lens[fragment], (corelir_dic[fragment])["av_pLDDT"], (corelir_dic[fragment])["av_minPAE"]] if fragment[-1] != "/" else
                                [self.frag_lens[fragment[:-1]], (corelir_dic[fragment[:-1]])["av_pLDDT"], (corelir_dic[fragment[:-1]])["av_minPAE"]]
                                for fragment in self.fragments
                               ]
                              )
        try:
            sorted_scores=sorted_scores[sorted_scores[:,0].argsort(kind="mergesort")]
        except:
            print("WARNING: Sorting of values failed.")

        fig, axs = plt.subplots(2,1,sharex=True, figsize=(8/2.54, 8/2.54))
        fig.subplots_adjust(hspace=0)
        
        axs[0].plot(sorted_scores[:,0], sorted_scores[:,1], color="purple")
        axs[0].legend(fontsize=10, frameon=False)
        axs[0].set_xlim([len_min, len_max])
        axs[0].set_ylim([0.0,max_pLDDT_plot])
        axs[0].set_ylabel("pLDDT")
        axs[0].grid(True)
        
        axs[1].plot(sorted_scores[:,0], sorted_scores[:,2], color="purple")
        axs[1].legend(fontsize=10, frameon=False)
        axs[1].set_xlabel("Fragment length")
        axs[1].set_xlim([len_min, len_max])
        axs[1].set_ylim([0.0, max_minPAE_plot])
        axs[1].set_ylabel(r"minPAE [$\AA$]")
        axs[1].grid(True)
        
                #MSA indication
        if indicate_msa_at_len != None:
            axs[0].plot([indicate_msa_at_len,indicate_msa_at_len],[0,max_pLDDT_plot], color="black", linestyle=":")
            axs[1].plot([indicate_msa_at_len,indicate_msa_at_len],[0,max_minPAE_plot], color="black", linestyle=":")
        
        if self.name_add in self.name:
            plot_name=self.name+"_pLDDT_and_minPAE_against_len.png"
        else:
            plot_name=self.name+self.name_add+"_pLDDT_and_minPAE_against_len.png"
        
        plt.savefig(plot_name, dpi=1200)
        plt.show(block=False) 

#Functions

def get_occlusion_secstr(ref_structure, start_res_num=1, alphafold2=True, af2_treshold=70.0, use_simple=False):
    """
    ref_structure (str): pdb file name
    start_res_num (int): residue number of first residue in pdb
    alphafold2 (bool): is the ref_structure an af prediction?
    af2_treshold (float): minimum pLDDT value for sec. str. assignment. Below will be considered "unstructured"
    use_simple (bool): if True use simple Sec. Str. assignemnts, if False the more detailed ones
    """

    #Secondary structure assignment

    #Map Value to letter

    #Simple

        #Alpha Helix
        #Beta Sheet
        #Coil

    simple_secstr_dic = {"Alpha": ("H",0),
                         "Beta": ("E",1),
                         "Coil": ("C",2),
                         "Error": ("NA", np.nan)
                         }

    #Detailed

        #Alpha Helix
        #Isolated Beta-bridge
        #Extended strand / Beta Ladder
        #3_10 Helix
        #Pi Helix
        #Hydrogen bonded turn
        #Bend 
        #Loop

    detailed_secstr_dic = {"Alpha": ("H",0),
                           "IsolatedBeta": ("B",3),
                           "ExtendedBeta": ("E",4),
                           "3_10": ("G",1),
                           "Pi": ("I",2),
                           "Turn": ("T",5),
                           "Bend": ("S",6),
                           "Loop": (" ",7),
                           "Error": ("NA", np.nan)
                          }

    if use_simple == True:
        use_secstr_dic = simple_secstr_dic
    elif use_simple == False:
        use_secstr_dic = detailed_secstr_dic
    
    traj = mdt.load(ref_structure, top=ref_structure)
        
    secstr = mdt.compute_dssp(traj, simplified=use_simple)
    
    secstr_matrix = np.zeros_like(secstr, dtype=np.float64)
                
    for secstr_element in use_secstr_dic:
        secstr_matrix[secstr == (use_secstr_dic[secstr_element])[0]] = (use_secstr_dic[secstr_element])[1]
    
    secstr_matrix=secstr_matrix.flatten()
    
    occlusion_dic={}
    
    if alphafold2==True:
        u=mda.Universe(ref_structure)
        pLDDTs=u.select_atoms("name CA").tempfactors
        for index, secstr in enumerate(secstr_matrix):
            resnum=index+start_res_num
            if pLDDTs[index]<af2_treshold: #Residues with low confidence are considered not occluded
                occlusion_dic[resnum]=0
            else:
                if (use_simple==False and secstr <= 2) or (use_simple==True and secstr <= 0): #Residues with helical- or sheet-like secondary structure are considered occluded, but indicated separately
                    occlusion_dic[resnum]=2
                elif (use_simple==False and secstr <= 4) or (use_simple==True and secstr <= 1):
                    occlusion_dic[resnum]=1
                else:
                    occlusion_dic[resnum]=0
    else:
        for index, secstr in enumerate(secstr_matrix):
            resnum=index+start_res_num
            if (use_simple==False and secstr <= 2) or (use_simple==True and secstr <= 0):
                occlusion_dic[resnum]=2
            elif (use_simple==False and secstr <= 4) or (use_simple==True and secstr <= 1):
                occlusion_dic[resnum]=1
            else:
                occlusion_dic[resnum]=0

    return occlusion_dic 

def get_occlusion_depth(ref_structure, start_res_num=1):
    """
    ref_structure (str): pdb file name
    start_res_num (int): residue number of first residue in pdb
    """
        
    model = PDBParser().get_structure("model", ref_structure)[0]
    res_depth = ResidueDepth(model, msms_exec=msms_executable)
    
    occlusion_dic={}
    
    for residue in res_depth.keys():
        resnum=start_res_num+int((residue[1])[1])-1
        depth=(res_depth[residue])[0]
        occlusion_dic[resnum]=depth
    
    return occlusion_dic
    
    
def plot_summary(instance1, instance2, instance1_name="State_A", instance2_name="State_B", occlusion_data=None, occlusion_type=None, plot_by="min_minPAE", max_pLDDT_plot=100.0, max_minPAE_plot=35.0):
    """
    instance1 (instance): an instance of the seq_scan_system class
    instance2 (instance): another instance of the seq_scan_system class
    instance1_name (str): name of instance1 for output
    instance2_name (str): name of instance2 for output
    occlusion_data (dic of str): (optional) keys "ResDepth" and "SecStr"
    occlusion_type (str or None): "both", "ResDepth", or "SecStr" (or None)
    max_pLDDT_plot (float): upper pLDDT axis limit
    max_minPAE (float): upper minPAE axis limit
    """

    #by max pLDDT
    if plot_by=="max_pLDDT":
        #Instance 1
        #pLDDT
        plot_pLDDT_1=np.array([[key, (instance1.pLDDT_by_max_pLDDT[key])[0]] for key in instance1.pLDDT_by_max_pLDDT.keys()])
        plot_pLDDT_1=plot_pLDDT_1[plot_pLDDT_1[:,0].argsort(kind="mergesort")]
        #minPAE
        plot_minPAE_1=np.array([[key, (instance_1.minPAE_by_max_pLDDT[key])[0]] for key in instance1.minPAE_by_max_pLDDT.keys()])
        plot_minPAE_1=plot_minPAE_1[plot_minPAE_1[:,0].argsort(kind="mergesort")]
        #Instance 2
        #pLDDT
        plot_pLDDT_2=np.array([[key, (instance2.pLDDT_by_max_pLDDT[key])[0]] for key in instance2.pLDDT_by_max_pLDDT.keys()])
        plot_pLDDT_2=plot_pLDDT_2[plot_pLDDT_2[:,0].argsort(kind="mergesort")]
        #minPAE
        plot_minPAE_2=np.array([[key, (instance_2.minPAE_by_max_pLDDT[key])[0]] for key in instance2.minPAE_by_max_pLDDT.keys()])
        plot_minPAE_2=plot_minPAE_2[plot_minPAE_2[:,0].argsort(kind="mergesort")]
    #by min minPAE
    elif plot_by=="min_minPAE":
        #Instance 1
        #pLDDT
        plot_pLDDT_1=np.array([[key, (instance1.pLDDT_by_min_minPAE[key])[0]] for key in instance1.pLDDT_by_min_minPAE.keys()])
        plot_pLDDT_1=plot_pLDDT_1[plot_pLDDT_1[:,0].argsort(kind="mergesort")]
        #minPAE
        plot_minPAE_1=np.array([[key, (instance1.minPAE_by_min_minPAE[key])[0]] for key in instance1.minPAE_by_min_minPAE.keys()])
        plot_minPAE_1=plot_minPAE_1[plot_minPAE_1[:,0].argsort(kind="mergesort")]
        #Instance 2
        #pLDDT
        plot_pLDDT_2=np.array([[key, (instance2.pLDDT_by_min_minPAE[key])[0]] for key in instance2.pLDDT_by_min_minPAE.keys()])
        plot_pLDDT_2=plot_pLDDT_2[plot_pLDDT_2[:,0].argsort(kind="mergesort")]
        #minPAE
        plot_minPAE_2=np.array([[key, (instance2.minPAE_by_min_minPAE[key])[0]] for key in instance2.minPAE_by_min_minPAE.keys()])
        plot_minPAE_2=plot_minPAE_2[plot_minPAE_2[:,0].argsort(kind="mergesort")]
    #sorting variable not known, raise exception:
    else:
        raise Exception("Unkown 'plot_by' value")
     
    #Colors
    colors={"instance1": "blue",
            "instance2": "red"
            }      
    
    #Fonsizes
    legend_fontsize=8
    label_fontsize=8
    tick_fontsize=6
 
    #Plotting    
    if occlusion_type==None:
        fig, axs = plt.subplots(2,1,sharex=True, figsize=(10/2.54, 4/2.54), height_ratios=[2,2])
    elif occlusion_type=="both":
        fig, axs = plt.subplots(4,1,sharex=True, figsize=(10/2.54, 6/2.54), height_ratios=[2,2,1,1])
    else:
        fig, axs = plt.subplots(3,1,sharex=True, figsize=(10/2.54, 5/2.54), height_ratios=[2,2,1])
    fig.subplots_adjust(hspace=0)
        
    axs[0].plot(plot_pLDDT_1[:,0], plot_pLDDT_1[:,1], color=colors["instance1"], alpha=0.8)
    axs[0].plot(plot_pLDDT_2[:,0], plot_pLDDT_2[:,1], color=colors["instance2"], alpha=0.8)
    axs[0].legend(fontsize=legend_fontsize, frameon=False)
    axs[0].set_ylim([0.0,max_pLDDT_plot])
    axs[0].set_ylabel("pLDDT", fontsize=label_fontsize)
    axs[0].tick_params(axis='both', which='both', labelsize=tick_fontsize)
    axs[0].grid(True, alpha=0.5)
        
    axs[1].plot(plot_minPAE_1[:,0], plot_minPAE_1[:,1], color=colors["instance1"], alpha=0.8)
    axs[1].plot(plot_minPAE_2[:,0], plot_minPAE_2[:,1], color=colors["instance2"], alpha=0.8)
    axs[1].legend(fontsize=legend_fontsize, frameon=False)
    axs[1].set_ylim([0.0, max_minPAE_plot])
    axs[1].set_ylabel(r"minPAE [$\AA$]", fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='both', labelsize=tick_fontsize)
    axs[1].grid(True, alpha=0.5)
        
    #Filling the space between the lines
        
    for residue_index, residue in enumerate(plot_pLDDT_1[:,0]):
        #pLDDT
        if plot_pLDDT_1[residue_index,1] >= plot_pLDDT_2[residue_index,1]:
             axs[0].fill([residue-0.5, residue+0.5, residue+0.5, residue-0.5], [plot_pLDDT_2[residue_index,1], plot_pLDDT_2[residue_index,1], plot_pLDDT_1[residue_index,1], plot_pLDDT_1[residue_index,1]], color=colors["instance1"], alpha=0.5, linewidth=0)
        else:
            axs[0].fill([residue-0.5, residue+0.5, residue+0.5, residue-0.5], [plot_pLDDT_1[residue_index,1], plot_pLDDT_1[residue_index,1], plot_pLDDT_2[residue_index,1], plot_pLDDT_2[residue_index,1]], color=colors["instance2"], alpha=0.5, linewidth=0)
        #minPAE
        if plot_minPAE_1[residue_index,1] >= plot_minPAE_2[residue_index,1]:
            axs[1].fill([residue-0.5, residue+0.5, residue+0.5, residue-0.5], [plot_minPAE_2[residue_index,1], plot_minPAE_2[residue_index,1], plot_minPAE_1[residue_index,1], plot_minPAE_1[residue_index,1]], color=colors["instance2"], alpha=0.5, linewidth=0)
        else:
            axs[1].fill([residue-0.5, residue+0.5, residue+0.5, residue-0.5], [plot_minPAE_1[residue_index,1], plot_minPAE_1[residue_index,1], plot_minPAE_2[residue_index,1], plot_minPAE_2[residue_index,1]], color=colors["instance1"], alpha=0.5, linewidth=0)
        
    #Peaks
    
    #Peak Types
    
    peak_colors = {"Canonical": "green", "SIM": "green", "FIR": "green", #Canonical LiR, SIM, and FIR are mutually exclusive 
                   "Non-canonical": "yellow",
                   "Other at LiR-site": "brown",
                   "UBQ-like site": "purple",
                   "FG-NUP": "pink",
                   "Other": "grey"
                  }
    
    #Plot Peaks
    
    for peak in instance1.peaks:
        try:
            if peak["type"] not in peak_colors.keys():
                peak["type"]="Other"
        except:
            peak["type"]="Other"
        for residue in range(peak["Start"], peak["End"]+1,1):
            residue_index=([residue_index_ for residue_index_, residue_ in enumerate(plot_pLDDT_1[:,0]) if residue_==residue])[0]
            axs[0].fill([residue-0.5, residue+0.5, residue+0.5, residue-0.5], [0.0, 0.0, min(plot_pLDDT_1[residue_index, 1], plot_pLDDT_2[residue_index, 1]), min(plot_pLDDT_1[residue_index, 1], plot_pLDDT_2[residue_index, 1])], color=peak_colors[(peak["type"])], alpha=0.5, linewidth=0)
            axs[1].fill([residue-0.5, residue+0.5, residue+0.5, residue-0.5], [max(plot_minPAE_1[residue_index, 1], plot_minPAE_2[residue_index, 1]), max(plot_minPAE_1[residue_index, 1], plot_minPAE_2[residue_index, 1]), max_minPAE_plot, max_minPAE_plot], color=peak_colors[(peak["type"])], alpha=0.5, linewidth=0)
    for peak in instance2.peaks:
        try:
            if peak["type"] not in peak_colors.keys():
                peak["type"]="Other"
        except:
            peak["type"]="Other"
        for residue in range(peak["Start"], peak["End"]+1,1):
            residue_index=([residue_index_ for residue_index_, residue_ in enumerate(plot_pLDDT_1[:,0]) if residue_==residue])[0]
            axs[0].fill([residue-0.5, residue+0.5, residue+0.5, residue-0.5], [max(plot_pLDDT_1[residue_index, 1], plot_pLDDT_2[residue_index, 1]), max(plot_pLDDT_1[residue_index, 1], plot_pLDDT_2[residue_index, 1]), max_pLDDT_plot, max_pLDDT_plot], color=peak_colors[(peak["type"])], alpha=0.5, linewidth=0)
            axs[1].fill([residue-0.5, residue+0.5, residue+0.5, residue-0.5], [0.0, 0.0, min(plot_minPAE_1[residue_index, 1], plot_minPAE_2[residue_index, 1]), min(plot_minPAE_1[residue_index, 1], plot_minPAE_2[residue_index, 1])], color=peak_colors[(peak["type"])], alpha=0.5, linewidth=0)
    
    
    if occlusion_type==None:
        axs[1].set_xlabel("Residue Number", fontsize=label_fontsize)
    elif occlusion_type=="both":
        #Occlusion by ResDepth
        try:
            occlusion_data_resdepth=occlusion_data["ResDepth"]
            plot_occlusion=np.array([[key, occlusion_data_resdepth[key]] for key in occlusion_data_resdepth.keys()])
            plot_occlusion=plot_occlusion[plot_occlusion[:,0].argsort(kind="mergesort")]
            axs[2].plot(plot_occlusion[:,0], 0.1*plot_occlusion[:,1], color="black", alpha=0.8)
            axs[2].set_ylim([0.0, max(1.0,0.1*1.05*max(plot_occlusion[:,1]))])
            axs[2].set_yticks([0.0,np.around(0.5*max(1.0,0.1*1.05*max(plot_occlusion[:,1])),1)])
            axs[2].set_ylabel("r.d.\n[nm]", fontsize=label_fontsize, rotation=90)
            axs[2].grid(True, alpha=0.5)
        except:
            print("WARNING: Could not find Res. Depth data.")
            pass
        axs[2].tick_params(axis='both', which='both', labelsize=tick_fontsize)
        try:
            occlusion_data_secstr=occlusion_data["SecStr"]
            plot_occlusion=np.array([[key, occlusion_data_secstr[key]] for key in occlusion_data_secstr.keys()])
            plot_occlusion=plot_occlusion[plot_occlusion[:,0].argsort(kind="mergesort")]
            for residue_index, residue in enumerate(plot_occlusion[:,0]):
                if plot_occlusion[residue_index,1]==2:
                    oc_color="black"
                elif plot_occlusion[residue_index,1]==1:
                    oc_color="grey"
                else:
                    oc_color="white"
                axs[3].fill([residue-0.5, residue+0.5, residue+0.5, residue-0.5], [0.0, 0.0, 1.0, 1.0], color=oc_color, alpha=0.8, linewidth=0)
            axs[3].set_ylim([0.0, 1.0])
            axs[3].set_yticks([])
            axs[3].set_ylabel("Sec.\nStr.", fontsize=label_fontsize, rotation=90)
            axs[3].grid(True, alpha=0.5)
        except:
            print("WARNING: Could not find Sec. Str. data.")
            pass
        axs[3].tick_params(axis='both', which='both', labelsize=tick_fontsize)
        axs[3].set_xlabel("Residue Number", fontsize=label_fontsize)
    else:
        #Occlusion
        if occlusion_type == "SecStr":
            plot_occlusion=np.array([[key, occlusion_data[key]] for key in occlusion_data.keys()])
            plot_occlusion=plot_occlusion[plot_occlusion[:,0].argsort(kind="mergesort")]
            for residue_index, residue in enumerate(plot_occlusion[:,0]):
                if plot_occlusion[residue_index,1]==2: #Helices
                    oc_color="black"
                elif plot_occlusion[residue_index,1]==1: #beta sheets
                    oc_color="grey"
                else:
                    oc_color="white"
                axs[2].fill([residue-0.5, residue+0.5, residue+0.5, residue-0.5], [0.0, 0.0, 1.0, 1.0], color=oc_color, alpha=0.8, linewidth=0)
            axs[2].set_ylim([0.0, 1.0])
            axs[2].set_yticks([])
            axs[2].set_ylabel("Sec.\nStr.", fontsize=label_fontsize, rotation=90)
        elif occlusion_type == "ResDepth":
            plot_occlusion=np.array([[key, occlusion_data[key]] for key in occlusion_data.keys()])
            plot_occlusion=plot_occlusion[plot_occlusion[:,0].argsort(kind="mergesort")]
            axs[2].plot(plot_occlusion[:,0], 0.1*plot_occlusion[:,1], color="black", alpha=0.8)
            axs[2].set_ylim([0.0, max(1.0,0.1*1.05*max(plot_occlusion[:,1]))])
            axs[2].set_yticks([0.0,np.around(0.5*max(1.0,0.1*1.05*max(plot_occlusion[:,1])),1)])
            axs[2].set_ylabel("r.d.\n[nm]", fontsize=label_fontsize, rotation=90)
            axs[2].grid(True, alpha=0.5)
        else:
            print("WARNING: Occlusion Type not recognized.")
            plot_occlusion=np.array([[key, occlusion_data[key]] for key in occlusion_data.keys()])
            plot_occlusion=plot_occlusion[plot_occlusion[:,0].argsort(kind="mergesort")]
            axs[2].plot(plot_occlusion[:,0], 0.1*plot_occlusion[:,1], color="black", alpha=0.8)
            axs[2].set_ylim([0.0, 0.1*1.05*max(plot_occlusion[:,1])])
            axs[2].set_ylabel("Occ.", fontsize=label_fontsize, rotation=90)
        axs[2].tick_params(axis='both', which='both', labelsize=tick_fontsize)
        axs[2].set_xlabel("Residue Number", fontsize=label_fontsize)
    
    plot_name=str(instance1_name)+"_"+str(instance1.frag_len)+"mers-"+str(instance2_name)+"_"+str(instance2.frag_len)+"mers_"+str(occlusion_type)+"_summary_over_seq.png"

    plt.savefig(plot_name, dpi=1200, bbox_inches="tight")
    plt.show(block=False)
    
def print_diff(instance1, instance2, print_type="pLDDT", print_range="all"):
    """
    instance1 (instance): an instance of the seq_scan_system class
    instance2 (instance): another instance of the seq_scan_system class
    print_type (str): "pLDDT" or "minPAE"
    print_range (str or tuple): "all" or range
    """
 
    if print_type=="pLDDT":
        diff_pLDDT_tab=np.array([[key, instance1.pLDDT_av[key]-instance2.pLDDT_av[key], instance1.pLDDT_max[key]-instance2.pLDDT_max[key]] for key in instance1.pLDDT_av.keys()])
        diff_pLDDT_tab=diff_pLDDT_tab[diff_pLDDT_tab[:,0].argsort(kind="mergesort")]
        if print_range=="all":
            for line in diff_pLDDT_tab:
                print(line)
        else:
            for line in diff_pLDDT_tab[int(print_range[0]):int(print_range[1])]:
                print(line)
    elif print_type=="minPAE":
        diff_minPAE_av=np.array([[key, instance1.minPAE_av[key]-instance2.minPAE_av[key], instance1.minPAE_min[key]-instance2.minPAE_min[key]] for key in instance1.minPAE_av.keys()])
        diff_minPAE_av=diff_minPAE_av[diff_minPAE_av[:,0].argsort(kind="mergesort")]
        if print_range=="all":
            for line in diff_minPAE_tab:
                print(line)
        else:
            for line in diff_minPAE_tab[int(print_range[0]):int(print_range[1])]:
                print(line)
    else:
        print("Type not recognized. Choose 'pLDDT' or 'minPAE'")

def plot_corelir_summary(instance1, instance2, instance1_name="State_A", instance2_name="State_B", indicate_msa_at_len=15, max_pLDDT_plot=100.0, max_minPAE_plot=45.0):
    """
    instance1 (instance): an instance of the seq_scan_system class
    instance2 (instance): another instance of the seq_scan_system class
    instance1_name (str): name of instance1 for output
    instance2_name (str): name of instance2 for output
    indicate_msa_at_len (int): minimum fragment length required for AlphaFold2 to calculate and MSA
    max_pLDDT_plot (float): upper pLDDT axis limit
    max_minPAE (float): upper minPAE axis limit
    """
    
    #Instance 1
    corelir_dic1={}
    for corelir in instance1.corelir_scores:
        corelir_dic1[(corelir["fragment"])]=corelir 
    sorted_scores1=np.array([[instance1.frag_lens[fragment], (corelir_dic1[fragment])["av_pLDDT"], (corelir_dic1[fragment])["av_minPAE"]] if fragment[-1] != "/" else
                             [instance1.frag_lens[fragment[:-1]], (corelir_dic1[fragment[:-1]])["av_pLDDT"], (corelir_dic1[fragment[:-1]])["av_minPAE"]]
                             for fragment in instance1.fragments
                            ]
                           ) 
    try:
        sorted_scores1=sorted_scores1[sorted_scores1[:,0].argsort(kind="mergesort")]
    except:
        print("WARNING: Sorting of corelir scores failed.")
    #Instance 2
    corelir_dic2={}
    for corelir in instance2.corelir_scores:
        corelir_dic2[(corelir["fragment"])]=corelir
    sorted_scores2=np.array([[instance2.frag_lens[fragment], (corelir_dic2[fragment])["av_pLDDT"], (corelir_dic2[fragment])["av_minPAE"]] if fragment[-1] != "/" else
                             [instance2.frag_lens[fragment[:-1]], (corelir_dic2[fragment[:-1]])["av_pLDDT"], (corelir_dic2[fragment[:-1]])["av_minPAE"]]
                             for fragment in instance2.fragments
                            ]
                           )
    try:
        sorted_scores2=sorted_scores2[sorted_scores2[:,0].argsort(kind="mergesort")]
    except:
        print("WARNING: Sorting of corelir scores failed.")
    #Colors
    colors={"instance1": "blue",
            "instance2": "red"
            }      
    
    #Fontsizes
    legend_fontsize=8
    label_fontsize=8
    tick_fontsize=6
    #Plotting    
    fig, axs = plt.subplots(2,1,sharex=True, figsize=(8/2.54, 4/2.54), height_ratios=[1,1])
    fig.subplots_adjust(hspace=0)
        #pLDDT
    axs[0].plot(sorted_scores1[:,0], sorted_scores1[:,1], color=colors["instance1"], alpha=0.8)
    axs[0].plot(sorted_scores2[:,0], sorted_scores2[:,1], color=colors["instance2"], alpha=0.8)
    axs[0].legend(fontsize=legend_fontsize, frameon=False)
    axs[0].set_ylim([0.0,max_pLDDT_plot])
    axs[0].set_ylabel("pLDDT", fontsize=label_fontsize)
    axs[0].tick_params(axis='both', which='both', labelsize=tick_fontsize)
    axs[0].grid(True, alpha=0.5)
        #minPAE
    axs[1].plot(sorted_scores1[:,0], sorted_scores1[:,2], color=colors["instance1"], alpha=0.8)
    axs[1].plot(sorted_scores2[:,0], sorted_scores2[:,2], color=colors["instance2"], alpha=0.8)
    axs[1].legend(fontsize=legend_fontsize, frameon=False)
    axs[1].set_ylim([0.0, max_minPAE_plot])
    axs[1].set_ylabel(r"minPAE [$\AA$]", fontsize=label_fontsize)
    axs[1].tick_params(axis='both', which='both', labelsize=tick_fontsize)
    axs[1].grid(True, alpha=0.5)
        
    #Filling the space between the lines
        #Fill parameters (may alter to obtain a smooth looking fill)
    fill_step=0.1 
        #Create fill lists
    expanded_lens=[fill_step*len_ for len_ in range(int(1/fill_step*float(min(sorted_scores1[0,0], sorted_scores2[0,0]))), int(1/fill_step*float(max(sorted_scores1[-1,0], sorted_scores2[-1,0]))+1.0), 1)]
    expanded_scores1=[]
    for len_ in expanded_lens:
        try:
            if len_ % 1.0 ==0:
                len_=int(len_)
            curr_index=np.where(sorted_scores1[:,0]==len_)[0][0]
            next_index=curr_index+1
            expanded_scores1.append([float(len_), sorted_scores1[curr_index,1],sorted_scores1[curr_index,2]])
        except:
            try:
                fit_pLDDT=(sorted_scores1[curr_index,1]-sorted_scores1[next_index,1])/(sorted_scores1[curr_index,0]-sorted_scores1[next_index,0])*len_+(sorted_scores1[curr_index,1]-((sorted_scores1[curr_index,1]-sorted_scores1[next_index,1])/(sorted_scores1[curr_index,0]-sorted_scores1[next_index,0])*sorted_scores1[curr_index,0]))
                fit_minPAE=(sorted_scores1[curr_index,2]-sorted_scores1[next_index,2])/(sorted_scores1[curr_index,0]-sorted_scores1[next_index,0])*len_+(sorted_scores1[curr_index,2]-((sorted_scores1[curr_index,2]-sorted_scores1[next_index,2])/(sorted_scores1[curr_index,0]-sorted_scores1[next_index,0])*sorted_scores1[curr_index,0]))
                expanded_scores1.append([len_, fit_pLDDT, fit_minPAE])
            except:
                pass
    expanded_scores1=np.array(expanded_scores1)
    expanded_scores2=[]
    for len_ in expanded_lens:
        try:
            curr_index=np.where(sorted_scores2[:,0]==len_)[0][0]
            next_index=curr_index+1
            expanded_scores2.append([len_, sorted_scores2[curr_index,1],sorted_scores2[curr_index,2]])
        except:
            try:
                fit_pLDDT=(sorted_scores2[curr_index,1]-sorted_scores2[next_index,1])/(sorted_scores2[curr_index,0]-sorted_scores2[next_index,0])*len_+(sorted_scores2[curr_index,1]-((sorted_scores2[curr_index,1]-sorted_scores2[next_index,1])/(sorted_scores2[curr_index,0]-sorted_scores2[next_index,0])*sorted_scores2[curr_index,0]))
                fit_minPAE=(sorted_scores2[curr_index,2]-sorted_scores2[next_index,2])/(sorted_scores2[curr_index,0]-sorted_scores2[next_index,0])*len_+(sorted_scores2[curr_index,2]-((sorted_scores2[curr_index,2]-sorted_scores2[next_index,2])/(sorted_scores2[curr_index,0]-sorted_scores2[next_index,0])*sorted_scores2[curr_index,0]))
                expanded_scores2.append([len_, fit_pLDDT, fit_minPAE])
            except:
                pass
    expanded_scores2=np.array(expanded_scores2)

    for len_index1, len_value1 in enumerate(expanded_scores1[:,0]): #Iterate over data points of instance 1
        for len_index2, len_value2 in enumerate(expanded_scores2[:,0]): #Find corresponding data points in instance 2
            if len_value1==len_value2:
                #pLDDT
                if expanded_scores1[len_index1,1] >= expanded_scores2[len_index2,1]:
                    axs[0].fill([len_value1-0.5*fill_step, len_value1+0.5*fill_step, len_value1+0.5*fill_step, len_value1-0.5*fill_step], [expanded_scores2[len_index2,1], expanded_scores2[len_index2,1], expanded_scores1[len_index1,1], expanded_scores1[len_index1,1]], color=colors["instance1"], alpha=0.5, linewidth=0)
                else:
                    axs[0].fill([len_value1-0.5*fill_step, len_value1+0.5*fill_step, len_value1+0.5*fill_step, len_value1-0.5*fill_step], [expanded_scores1[len_index1,1], expanded_scores1[len_index1,1], expanded_scores2[len_index2,1], expanded_scores2[len_index2,1]], color=colors["instance2"], alpha=0.5, linewidth=0)
                #minPAE
                if expanded_scores1[len_index1,2] >= expanded_scores2[len_index2,2]:
                    axs[1].fill([len_value1-0.5*fill_step, len_value1+0.5*fill_step, len_value1+0.5*fill_step, len_value1-0.5*fill_step], [expanded_scores2[len_index2,2], expanded_scores2[len_index2,2], expanded_scores1[len_index1,2], expanded_scores1[len_index1,2]], color=colors["instance2"], alpha=0.5, linewidth=0)
                else:
                    axs[1].fill([len_value1-0.5*fill_step, len_value1+0.5*fill_step, len_value1+0.5*fill_step, len_value1-0.5*fill_step], [expanded_scores1[len_index1,2], expanded_scores1[len_index1,2], expanded_scores2[len_index2,2], expanded_scores2[len_index2,2]], color=colors["instance1"], alpha=0.5, linewidth=0)
            else:
                pass
    #Peaks
    
    #Peak Types
    
    peak_colors = {"Canonical": "green", "SIM": "green", "FIR": "green", #Canonical LIR, SIM and FIR are mutually exclusive
                   "Non-canonical": "yellow",
                   "Other at LiR-site": "brown",
                   "UBQ-like site": "purple",
                   "FG-NUP": "pink",
                   "Other": "grey"
                  }
    
    #Plot Peaks
     
    for peak in instance1.corelir_scores:
        try:
            if peak["type"] not in peak_colors.keys():
                peak["type"]="Other"
        except:
            peak["type"]="Other"
        len_value=instance1.frag_lens[(peak["fragment"])]
        try:
            len_index1=([len_index_ for len_index_, len_ in enumerate(sorted_scores1[:,0]) if len_==len_value])[0] #Len value in data from instance1?
            len_index2=([len_index_ for len_index_, len_ in enumerate(sorted_scores2[:,0]) if len_==len_value])[0] #Len value in data from instance2?
        except:
            continue
        axs[0].fill([len_value-0.5, len_value+0.5, len_value+0.5, len_value-0.5], [0.0, 0.0, min(sorted_scores1[len_index1, 1], sorted_scores2[len_index2, 1]), min(sorted_scores1[len_index1, 1], sorted_scores2[len_index2, 1])], color=peak_colors[(peak["type"])], alpha=0.5, linewidth=0)
        axs[1].fill([len_value-0.5, len_value+0.5, len_value+0.5, len_value-0.5], [max(sorted_scores1[len_index1, 2], sorted_scores2[len_index2, 2]), max(sorted_scores1[len_index1, 2], sorted_scores2[len_index2, 2]), max_minPAE_plot, max_minPAE_plot], color=peak_colors[(peak["type"])], alpha=0.5, linewidth=0)
    for peak in instance2.corelir_scores:
        try:
            if peak["type"] not in peak_colors.keys():
                peak["type"]="Other"
        except:
            peak["type"]="Other"
        len_value=instance2.frag_lens[(peak["fragment"])]
        try:
            len_index1=([len_index_ for len_index_, len_ in enumerate(sorted_scores1[:,0]) if len_==len_value])[0] #Len value in data from instance1?
            len_index2=([len_index_ for len_index_, len_ in enumerate(sorted_scores2[:,0]) if len_==len_value])[0] #Len value in data from instance2?
        except:
            continue
        axs[0].fill([len_value-0.5, len_value+0.5, len_value+0.5, len_value-0.5], [max(sorted_scores1[len_index1, 1], sorted_scores2[len_index2, 1]), max(sorted_scores1[len_index1, 1], sorted_scores2[len_index2, 1]), max_pLDDT_plot, max_pLDDT_plot], color=peak_colors[(peak["type"])], alpha=0.5, linewidth=0)
        axs[1].fill([len_value-0.5, len_value+0.5, len_value+0.5, len_value-0.5], [0.0, 0.0, min(sorted_scores1[len_index1, 2], sorted_scores2[len_index2, 2]), min(sorted_scores1[len_index1, 2], sorted_scores2[len_index2, 2])], color=peak_colors[(peak["type"])], alpha=0.5, linewidth=0)
    
    axs[1].set_xlabel("Fragment Length", fontsize=label_fontsize)
    
    if indicate_msa_at_len != None:
        axs[0].plot([indicate_msa_at_len,indicate_msa_at_len],[0,max_pLDDT_plot], color="black", linestyle=":")
        axs[1].plot([indicate_msa_at_len,indicate_msa_at_len],[0,max_minPAE_plot], color="black", linestyle=":")
    
    plot_name=str(instance1_name)+"-"+str(instance2_name)+"_summary_over_len.png"
    plt.savefig(plot_name, dpi=1200, bbox_inches="tight")
    plt.show(block=False)
