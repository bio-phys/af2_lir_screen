#Python functions to summarize the output from the "seq_scan_analysis" class

#    This script is part of af2_lir_screen.
#    Copyright (C) 2024  Jan Stuke
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#Version 2 of the seq_scan_summary.py by Jan F. M. Stuke, 18.01.2025
#Notable changes:
# 1) Transposed PAE values: Instead of minimum PAE in the bait residues, we now consider minimum PAE in the candidate residues
# 2) Specific non-canonical LIR classes based on previous experimental evidence
# 3) New and more easily interpretable scoring system

#Imports

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import os
import sys
import ast

#Functions
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

def read_features_from_csv(file_name, frag_type, keep_original_frag=False):
    """
    file_name (str): name of csv file containing list of fragments
    frag_type (str): identifier of fragments in csv file, indicating e.g. fragment length and mutations / PTMs
    keep_original_frag (bool): If true, will add the fragment name to the refined features of each peak. The fragment name is only necessary for some functions and not supported in others. Hence, only ask for it, if you need it.
    """
    #Read raw features
    raw_feature_indices={"Start": None, "End": None, "sequence": None, "type": None, "fragment": None, "av_pLDDT": None, "min_len_pLDDT": None, "av_minPAE": None, "min_len_minPAE": None}
    raw_features_for_peaks=[]
    with open(file_name, "r") as f:
        for index, line in enumerate(f):
            elements=[element.replace(" ","").replace("\n","") for element in line.split(",")]
            if index==0:
                for raw_feature in raw_feature_indices.keys():
                    try:
                        raw_feature_indices[raw_feature]=elements.index(raw_feature)
                    except:
                        raise Exception(f"CSV file is missing a column with feature {raw_feature}.")
            else:
                peak_raw_features={}
                for raw_feature in raw_feature_indices.keys():
                    peak_raw_features[raw_feature]=elements[(raw_feature_indices[raw_feature])]
                raw_features_for_peaks.append(peak_raw_features)
    #Transform into refined features
    refined_features_for_peaks=[]
    for raw_features_for_peak in raw_features_for_peaks:
        refined_features_for_peak={}
        #Type
        refined_features_for_peak["type"]=raw_features_for_peak["type"].replace("'","").replace('"','')
        #min_seq (will be updated later when comparing peaks with overlapping range)
        refined_features_for_peak["min_seq"]=raw_features_for_peak["sequence"].split("-")[1]
        #Range
        refined_features_for_peak["range"]=(int(raw_features_for_peak["Start"]), int(raw_features_for_peak["End"]))
        #min_len_pLDDT and min_len_minPAE
        refined_features_for_peak["min_len_pLDDTs"]={frag_type: [float(raw_features_for_peak["min_len_pLDDT"])]}
        refined_features_for_peak["min_len_minPAEs"]={frag_type: [float(raw_features_for_peak["min_len_minPAE"])]}
        #HP), HP1 and HP2
        seq=raw_features_for_peak["sequence"].split("-")[1]
        hp_counter=0 #For every hydrophobic pocket we have to reduce index by 5
        for index, aa in enumerate(seq):
            try:
                if seq[index+1:index+6]=="(HP0)":
                    refined_features_for_peak["HP0"]=(str(aa), int(raw_features_for_peak["Start"])+index-hp_counter*5) #Amino acid name and residue number as tuple                
                elif seq[index+1:index+6]=="(HP1)":
                    refined_features_for_peak["HP1"]=(str(aa), int(raw_features_for_peak["Start"])+index-hp_counter*5) #Amino acid name and residue number as tuple
                elif seq[index+1:index+6]=="(HP2)":
                    refined_features_for_peak["HP2"]=(str(aa), int(raw_features_for_peak["Start"])+index-hp_counter*5) #Amino acid name and residue number as tuple
                else:
                    continue
                hp_counter+=1 
            except:
                break
        try:
            refined_features_for_peak["HP0"]
        except:
            refined_features_for_peak["HP0"]=None        
        try:
            refined_features_for_peak["HP1"]
        except:
            refined_features_for_peak["HP1"]=None
        try:
            refined_features_for_peak["HP2"]
        except:
            refined_features_for_peak["HP2"]=None
        #Occurences (set to 1 here, will be used later to combine identical / overlapping peaks)
        refined_features_for_peak["occurences"]={}
        (refined_features_for_peak["occurences"])[frag_type]=1
        #Fragment name only necessary for some functions and not supported in others. Hence, only ask for it, if you need it.
        if keep_original_frag==True:
            refined_features_for_peak["fragment"]=str(raw_features_for_peak["fragment"].replace("'","").replace('"',''))
        #Add to output
        refined_features_for_peaks.append(refined_features_for_peak)
    return refined_features_for_peaks

def summarize_peaks(input_peak_list, frag_types, min_motif_len=3):
    """
    input_peak_list (list of dic): list of all peaks to be summarized
    frag_types (list of str): list of all frag types to be included in the analysis. Occurences will be counted for all types separately. 
    min_motif_len (int): minimum length of a peak motif shared by multiple fragments 
    """
    unique_peaks=[]
    for peak in input_peak_list:
        if unique_peaks==[]:
            unique_peaks.append(peak)
            continue
        peak_is_unique=True
        for unique_peak in unique_peaks:
            #Straight forward comparisons
            if peak["HP0"]==unique_peak["HP0"] and peak["HP1"]==unique_peak["HP1"] and peak["HP2"]==unique_peak["HP2"] and peak["type"]==unique_peak["type"]:
                #Not so straight forward comparisons (Redundant if HP), HP1 or HP2 is not None. Important otherwise 
                #Main test: Is the range overlapping? Create a fragment from the higher start to the lower end value and check whether that fragment exists in both peaks
                min_overlap_range=(max(int(peak["range"][0]), int(unique_peak["range"][0])),min(int(peak["range"][1]), int(unique_peak["range"][1])))
                if (min_overlap_range[0] >= peak["range"][0] and min_overlap_range[0] <= peak["range"][1] and min_overlap_range[1] >= peak["range"][0] and min_overlap_range[1] <= peak["range"][1] and
                    min_overlap_range[0] >= unique_peak["range"][0] and min_overlap_range[0] <= unique_peak["range"][1] and min_overlap_range[1] >= unique_peak["range"][0] and min_overlap_range[1] <= unique_peak["range"][1]) and min_overlap_range[1]-min_overlap_range[0] >= min_motif_len-1:
                        truncate_min_seq=(min_overlap_range[0]-unique_peak["range"][0], unique_peak["range"][1]-min_overlap_range[1])
                        min_seq=unique_peak["min_seq"][truncate_min_seq[0]:len(unique_peak["min_seq"])-truncate_min_seq[1]]
                        unique_peak["range"]=min_overlap_range
                        unique_peak["min_seq"]=min_seq
                        try:
                            (unique_peak["occurences"])[(list(peak["occurences"].keys())[0])]=(unique_peak["occurences"])[(list(peak["occurences"].keys())[0])]+1
                        except:
                            (unique_peak["occurences"])[(list(peak["occurences"].keys())[0])]=1
                        try:
                            (unique_peak["min_len_pLDDTs"])[(list(peak["min_len_pLDDTs"].keys())[0])].append(((peak["min_len_pLDDTs"])[(list(peak["min_len_pLDDTs"].keys())[0])])[0])
                        except:
                            (unique_peak["min_len_pLDDTs"])[(list(peak["min_len_pLDDTs"].keys())[0])]=(peak["min_len_pLDDTs"])[(list(peak["min_len_pLDDTs"].keys())[0])]
                        try:
                            (unique_peak["min_len_minPAEs"])[(list(peak["min_len_minPAEs"].keys())[0])].append(((peak["min_len_minPAEs"])[(list(peak["min_len_minPAEs"].keys())[0])])[0])
                        except:
                            (unique_peak["min_len_minPAEs"])[(list(peak["min_len_minPAEs"].keys())[0])]=(peak["min_len_minPAEs"])[(list(peak["min_len_minPAEs"].keys())[0])]
                        peak_is_unique=False
                        break
                else:
                    continue
            else:
                continue
        if peak_is_unique==True:
            unique_peaks.append(peak)
        else:
            continue
    #Add an entry for every frag type that does not contain this peak to occurences (it's easier to do here than to sanitize the inputs for plotting)
    for peak in unique_peaks:
        for frag_type in frag_types:
            try:
                (peak["occurences"])[frag_type]
            except:
                (peak["occurences"])[frag_type]=0
            try:
                (peak["min_len_pLDDTs"])[frag_type]
            except:
                (peak["min_len_pLDDTs"])[frag_type]=[]
            try:
                (peak["min_len_minPAEs"])[frag_type]
            except:
                (peak["min_len_minPAEs"])[frag_type]=[]
    return unique_peaks

def find_fragments(frag_name, fragment_main_dir, frag_name_add="", name_addition_optional=True, prefer_addition=True, msa_type="frag_msa", frag_len="any"): 
    """
    frag_name (str): name stem of fragments
    fragment_main_dir (str): path to alphapulldown output for the respective fragments
    frag_name_add (str): in case some fragments have an addition to the name stem
    name_addition_optional (bool): If False, will only consider fragments with the addition. If True, it will also consider those without.
    prefer_addition (bool): if True, fragments with the name addition will be preferred over those without IF their residue range is identical. If False, it will be the other way around.
    msa_type (str): "frag_msa" or "one_msa" . This is because in the alphapulldown output the residue range is connected by either "-" or "_" at the end of the name string
    frag_len (str or int): "any" will take fragments of any length in the fragment_main_dir. With an int it will only take fragments of that length.
    """
    fragments=[]
    if frag_name_add in frag_name:
        contains_addition=frag_name #Name with addition
        frag_name=str(frag_name.replace(str(frag_name_add), "")) #Name without addition
    else:
        contains_addition=frag_name+frag_name_add
    for file in os.listdir(fragment_main_dir):
        try:
            if msa_type=="frag_msa":
                residues=file.split("-")[-2:]
                connector="-"
            elif msa_type=="one_msa":
                residues=(file.split("_")[-1]).split("-")
                connector="_"
            else:
                raise Exception(f"Unknown msa_type {msa_type}.")
                #Check frag_len
            if frag_len=="any":
                pass
            elif int(frag_len)==int(int(residues[1])-int(residues[0])+1):
                pass
            else:
                continue
            if frag_name != contains_addition:
                if prefer_addition==True:
                    if contains_addition == str(file).replace(f"{connector}{str(residues[0])}-{str(residues[1])}",""):
                        fragments.append(file)
                        try:
                            fragments.remove(str(frag_name+f"{connector}{str(residues[0])}-{str(residues[1])}"))
                        except:
                            pass
                    elif frag_name == str(file).replace(f"{connector}{str(residues[0])}-{str(residues[1])}","") and name_addition_optional == True and str(contains_addition+f"-{str(residues[0])}-{str(residues[1])}") not in fragments:
                        fragments.append(file)
                    else:
                        pass
                elif prefer_addition==False and name_addition_optional==True:
                    if frag_name == str(file).replace(f"{connector}{str(residues[0])}-{str(residues[1])}",""):
                        fragments.append(file)
                        try:
                            fragments.remove(str(contains_addition+f"{connector}{str(residues[0])}-{str(residues[1])}"))
                        except:
                            pass
                    elif contains_addition == str(file).replace(f"{connector}{str(residues[0])}-{str(residues[1])}","") and str(frag_name+f"{connector}{str(residues[0])}-{str(residues[1])}") not in fragments:
                        fragments.append(file)
                    else:
                        pass
            else:
                if frag_name == str(file).replace(f"{connector}{str(residues[0])}-{str(residues[1])}",""):
                    fragments.append(file)
                else:
                    pass
        except:
            pass
    return fragments

def get_closest_phosmim(input_peak_list, frag_types, frag_names, frag_name_adds, frag_dirs, frag_lens, mod_types=["WT", "ST"], mod_fastas={"WT": "WT.fasta", "ST": "ST.fasta"}, mod_seq_starts_at_res={"WT": 1, "ST": 1}, min_motif_len=3, output_name="output"):
    """
    input_peak_list (list of dic): list of all peaks to be summarized
    frag_types (list of str): list of all frag types to be included in the analysis. Occurences will be counted for all types separately. 
    frag_names (list of str): name stem of fragments
    frag_name_adds (list of tuples [str, bool] ): in case some fragments have an addition to the name stem (str), that should or should not be preferred over the fragment with only the stem (bool). NOTE: The promiscuity of the frag_name system is not a problem here, as long as you prefer the one with the name_add (else you will not detect any differences.)
    frag_dirs (list of str): path to alphapulldown output for the respective fragments
    frag_len (list of str/int): "any" will take fragments of any length in the fragment_main_dir. With an int it will only take fragments of that length.
    mod_types (list of str): list of all mod types to be compared in the analysis.
    mod_fastas (dic with mod_types as keys and str as entries): names of fastas associated with different mod types
    mod_seq_starts_at_res (dic with mod_types as keys and int as entries): residue number (usually starts at 1) of the first residue in the respective fasta
    min_motif_len (int): minimum length of a peak motif shared by multiple fragments 
    output_name (str): name of output file
    """
    #Get all unmodfied and modified fragments
    all_fragments={}
    for mod_type in mod_types:
        all_fragments[mod_type]=[]
        for frag_type in frag_types:
            if mod_type not in frag_type:
                continue
            else:
                type_frags=find_fragments(frag_name=frag_names[frag_type], fragment_main_dir=frag_dirs[frag_type], frag_name_add=(frag_name_adds[frag_type])[0], prefer_addition=(frag_name_adds[frag_type])[1], frag_len=frag_lens[frag_type])
                type_frags_with_ranges=[(fragment, int(fragment.split("-")[-2]), (int(fragment.split("-")[-1])), str(mod_type)) for fragment in type_frags]
                all_fragments[mod_type]+=type_frags_with_ranges
    #Get positions of phosphomimetics
    seqs={}
    for mod_type in mod_types:
        seqs[mod_type]=load_seq(mod_fastas[mod_type])[0]
    mod_pos_list=[]
    for i, aa in enumerate(seqs[mod_types[0]]):
        for mod_type in mod_types[1:]:
            if i+int(mod_seq_starts_at_res[mod_types[0]])-int(mod_seq_starts_at_res[mod_type]) >= 0:
                pot_mod_aa=(seqs[mod_type])[i+int(mod_seq_starts_at_res[mod_types[0]])-int(mod_seq_starts_at_res[mod_type])]
            else:
                pot_mod_aa=None #If there is no aa at this position it means the modified sequence was truncated. We consider this to be a form of modification
            if aa==pot_mod_aa:
                continue
            else:
                mod_pos_list.append(i+int(mod_seq_starts_at_res[mod_types[0]]))
                break
    #Iterate over all peaks
    mod_dist_list=[]
    used_peak_for_comparison=[] #Keep track of peaks used for comparison so that they are not counte twice, e.g., change from WT to ST and change from ST to WT
    for peak in input_peak_list:
        if peak in used_peak_for_comparison:
            continue
        #Check if there is a modified position in the range of the fragment
        frag_start, frag_end = int(peak["fragment"].split("-")[-2]), (int(peak["fragment"].split("-")[-1]))
        contains_mod_pos=[]
        for mod_pos in mod_pos_list:
            if mod_pos >= frag_start and mod_pos <= frag_end:
                contains_mod_pos.append(mod_pos)
        if contains_mod_pos==[]:
            continue
        #Determine the mod_type of the peak
        for mod_type in mod_types:
            for fragment in all_fragments[mod_type]:
                if fragment[0]==peak["fragment"]:
                    peak_mod_type=mod_type
        #Loop over all fragments to check for corresponding fragment names in other mod types
        comp_cand_frag_names=[]
        for mod_type in mod_types:
            if mod_type==peak_mod_type:
                continue
            else:
                pass
            for fragment in all_fragments[mod_type]:
                if fragment[1]==frag_start and fragment[2]==frag_end:
                    comp_cand_frag_names.append(fragment[0])
        #Loop over all peaks for the candidate frag names
        successful_comp_frags=[]
        mod_lead_to_change=True
        for second_peak in input_peak_list:
            if second_peak==peak:
                continue
            if second_peak["fragment"] in comp_cand_frag_names and second_peak["fragment"] not in successful_comp_frags:
                no_second_peak_found=False
                #Straight forward comparisons
                if peak["HP0"]==second_peak["HP0"] and peak["HP1"]==second_peak["HP1"] and peak["HP2"]==second_peak["HP2"] and peak["type"]==second_peak["type"]:
                    #Not so straight forward comparisons (Redundant if HP), HP1 or HP2 is not None. Important otherwise 
                    #Main test: Is the range overlapping? Create a fragment from the higher start to the lower end value and check whether that fragment exists in both peaks
                    min_overlap_range=(max(int(peak["range"][0]), int(second_peak["range"][0])),min(int(peak["range"][1]), int(second_peak["range"][1])))
                    if (min_overlap_range[0] >= peak["range"][0] and min_overlap_range[0] <= peak["range"][1] and min_overlap_range[1] >= peak["range"][0] and min_overlap_range[1] <= peak["range"][1] and
                        min_overlap_range[0] >= second_peak["range"][0] and min_overlap_range[0] <= second_peak["range"][1] and min_overlap_range[1] >= second_peak["range"][0] and min_overlap_range[1] <= second_peak["range"][1]) and min_overlap_range[1]-min_overlap_range[0] >= min_motif_len-1:
                        successful_comp_frags.append(second_peak["fragment"])
                        mod_lead_to_change=False
                        #Modification did not introduce any major changes
                    else:
                        mod_lead_to_change=True
                else:
                    mod_lead_to_change=True
                #Modified peak is 
                used_peak_for_comparison.append(second_peak) #Don't use this peak in the main loop
            else:
                continue
        if mod_lead_to_change==False:
            continue
        else:
            #Find closest modification
            closest_mod_dist=None
            for mod_pos in contains_mod_pos:
                if mod_pos >= (peak["range"])[0] and mod_pos <= (peak["range"])[1]:
                    closest_mod_dist=0
                    break
                new_min_dist=min(abs((peak["range"])[0]-mod_pos), abs((peak["range"])[1]-mod_pos))
                if closest_mod_dist==None or new_min_dist<closest_mod_dist:
                    closest_mod_dist=new_min_dist
            mod_dist_list.append(closest_mod_dist)
    #Construct ouput file name and save mod_dist_list as .txt file
    if output_name[-4:]==".csv":
        output_name=output_name[:-4]
    output_name=output_name+"_mod_with_effect_dist_to_motif.csv"
    with open(output_name, "w") as f:
        f.write(",".join([str(i) for i in mod_dist_list]))
    return mod_dist_list
                    
def plot_mod_dist(files, output="output"):
    """
    files (list of str): list of all input files to use
    output (str): name of output file
    """
    mod_dists=[]
    for file in files:
        with open(file, "r") as f:
            values=[int(i) for line in f for i in line.split(",")]
        mod_dists+=values
    if mod_dists==[]:
        print("No modification distances were loaded. Perhaps the input files are empty?")
        return
    fig=plt.figure(figsize=(5/2.54,5/2.54))
    plt.hist(mod_dists, bins=[i-0.5 for i in range(0, max(mod_dists)+2,1)], density=True, color="red", linewidth=0.5, edgecolor="black")
    plt.xlabel("distance from motif", fontsize=8)
    plt.xticks(fontsize=8)
    plt.ylabel("PDF", fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig(f"{str(output)}.pdf", dpi=1200, bbox_inches="tight")
    plt.show()

def add_relative_occurences(unique_peaks, frag_types, frag_names, frag_name_adds, frag_dirs, frag_lens):
    """
    unique_peaks (list of dic): list of all unique peaks to consider
    frag_types (list of str): fragment types to consider
    frag_names (dic of str): keys are frag_types
    frag_name_adds (dic of tuple): keys are frag_types
    frag_dirs (dic of str): keys are frag_types
    frag_lens (dic of int): keys are frag_types
    """
    unique_ranges_dic={}
    for frag_type in frag_types:
        all_fragments=find_fragments(frag_name=frag_names[frag_type], fragment_main_dir=frag_dirs[frag_type], frag_name_add=(frag_name_adds[frag_type])[0], prefer_addition=(frag_name_adds[frag_type])[1], frag_len=frag_lens[frag_type])
        all_ranges=[(int(fragment.split("-")[-2]), (int(fragment.split("-")[-1]))) for fragment in all_fragments]
        unique_ranges_dic[frag_type]=set(all_ranges)
    updated_unique_peaks=[]
    for unique_peak in unique_peaks:
        unique_peak["rel. occ."]={}
        unique_peak["pot. occ."]={}
        for frag_type in frag_types:
            max_possible_occ=0
            for unique_range in unique_ranges_dic[frag_type]:
                if (unique_peak["range"])[0] >= unique_range[0] and (unique_peak["range"])[1] <= unique_range[1]:
                    max_possible_occ+=1
            if (unique_peak["occurences"])[frag_type]==0: #Avoid division by 0 for peaks that cannot occur for a certain fragment length
                (unique_peak["rel. occ."])[frag_type]=0
            else:
                (unique_peak["rel. occ."])[frag_type]=(unique_peak["occurences"])[frag_type]/max_possible_occ
            #Store maximum possible occurences for alter scoring
            (unique_peak["pot. occ."])[frag_type]=max_possible_occ
        updated_unique_peaks.append(unique_peak)
    return updated_unique_peaks

def compute_scores(unique_peaks, mod_types=["WT", "ST"], no_peak_pLDDT=75.0, no_peak_minPAE=8.0):
    """
    unique_peaks (list of dic): list of unique peaks updated with the occurence count
    mod_types (list of str): modification types to consider
    no_peak_pLDDT (float): pLDDT score to be used for unassigned residues (should correspond to the cutoff used in the initial screen)
    no_peak_pLDDT (float): minPAE score to be used for unassigned residues (should correspond to the cutoff used in the initial screen)
    """
    for unique_peak in unique_peaks:
        unique_peak["best core-motif pLDDT"]={}
        unique_peak["best core-motif minPAE"]={}
        unique_peak["total fraction"]={}
        unique_peak["weighted core-motif pLDDT"]={}
        unique_peak["weighted core-motif minPAE"]={}
        unique_peak["smart fraction"]={}
        unique_peak["length-weighted fraction"]={}
        for mod_type in mod_types:
        #1) best min_len pLDDT score (Injection of no_peak_pLDDT score guarantes value that is not None)
            (unique_peak["best core-motif pLDDT"])[mod_type]=max([_i for _j in [(unique_peak["min_len_pLDDTs"])[_key] if mod_type in _key else [no_peak_pLDDT] for _key in unique_peak["min_len_pLDDTs"].keys()] for _i in _j])
        #2) best min_len minPAE score (Injection of no_peak_minPAE score guarantes value that is not None)
            (unique_peak["best core-motif minPAE"])[mod_type]=min([__i for __j in [(unique_peak["min_len_minPAEs"])[_key] if mod_type in _key else [no_peak_minPAE] for _key in unique_peak["min_len_minPAEs"].keys()] for __i in __j])
        #3) total fraction of fragments with candidate LIR
            (unique_peak["total fraction"])[mod_type]=sum([(unique_peak["occurences"])[_key] if mod_type in _key else 0 for _key in unique_peak["occurences"].keys()])/sum([(unique_peak["pot. occ."])[_key] if mod_type in _key else 0 for _key in unique_peak["pot. occ."].keys()])
        #4) weighted min_len pLDDT score
            #4A) We need the number of fragments without this peak to add the penalty scores. We also use this penalty count in 5)
            number_of_penalties=0
            for _key in list(unique_peak["pot. occ."].keys()):
                if mod_type in _key:
                    number_of_penalties+=((unique_peak["pot. occ."])[_key]-(unique_peak["occurences"])[_key])
            #4B) Actual score
            (unique_peak["weighted core-motif pLDDT"])[mod_type]=(sum([sum((unique_peak["min_len_pLDDTs"])[_key]) if mod_type in _key else 0 for _key in unique_peak["min_len_pLDDTs"].keys()])+number_of_penalties*no_peak_pLDDT)/sum([(unique_peak["pot. occ."])[_key] if mod_type in _key else 0 for _key in unique_peak["pot. occ."].keys()])
        #5) weighted min_len minPAE score
            (unique_peak["weighted core-motif minPAE"])[mod_type]=(sum([sum((unique_peak["min_len_minPAEs"])[_key]) if mod_type in _key else 0 for _key in unique_peak["min_len_minPAEs"].keys()])+number_of_penalties*no_peak_minPAE)/sum([(unique_peak["pot. occ."])[_key] if mod_type in _key else 0 for _key in unique_peak["pot. occ."].keys()])
        #6) (total fraction of fragments + total fraction of fragment_types) / 2
            number_of_frag_types=0
            number_of_frag_types_with_hit=0
            for _key in list(unique_peak["pot. occ."].keys()):
                if mod_type in _key:
                    number_of_frag_types+=1
                    if (unique_peak["occurences"])[_key]!=0:
                        number_of_frag_types_with_hit+=1
            (unique_peak["smart fraction"])[mod_type]=((unique_peak["total fraction"])[mod_type]+float(number_of_frag_types_with_hit/number_of_frag_types))/2
        #7) fraction of fragments with candidate LIR weigthed by length
            frag_with_lir_sum=0
            frag_total_sum=0
            for _key in list(unique_peak["pot. occ."].keys()):
                if mod_type in _key:
                    _len=int(_key.split("_")[1])
                    frag_with_lir_sum+=(unique_peak["occurences"])[_key]*_len
                    frag_total_sum+=(unique_peak["pot. occ."])[_key]*_len
            if frag_total_sum==0:
                (unique_peak["length-weighted fraction"])[mod_type]=0
            else:
                (unique_peak["length-weighted fraction"])[mod_type]=frag_with_lir_sum/frag_total_sum
                

def add_res_depth(unique_peaks, res_depth_file, no_res_depth_value=0.142):
    """
    unique_peaks (list of dic): list of unique peaks updated with the occurence count
    res_depth_file (str): name of file storing the residue depth values for every residue. Needs to be a .csv (with , separation) with the columns 'resnum' and 'res. depth [nm]'
    no_res_depth_value (float): reside depth value to use for residues with unassigned residue depth [in nm]
    """
    #1) Load residue depth information
    res_depths={}
    expected_keys_position={"resnum": None, " res. depth [nm]": None}
    with open(res_depth_file, "r") as f:
        for index, line in enumerate(f):
            elements=line.replace("\n","").split(",")
            if index==0:
                for jndex, element in enumerate(elements):
                    if element.replace("'","").replace('"','') in list(expected_keys_position.keys()):
                        expected_keys_position[element.replace("'","").replace('"','')]=jndex
            else:
                resnum, res_depth=int(elements[expected_keys_position["resnum"]]), elements[expected_keys_position[" res. depth [nm]"]]
                res_depths[resnum]=res_depth
    #2) Average res. depth. of motif
    for unique_peak in unique_peaks:
        motif_res_depths=[]
        for resnum in range((unique_peak["range"])[0],(unique_peak["range"])[1]+1,1):
            res_depth=res_depths[resnum]
            if str(res_depth).replace(" ","").replace("'","").replace('"','').replace("\n","")=="None" or res_depth==None:
                res_depth=no_res_depth_value #Add minimal value if the residue depth at this position is None. None is not useful when computing averages
            motif_res_depths.append(float(res_depth))
        av_depth=np.mean(motif_res_depths)
        unique_peak["av. res. depth [nm]"]=av_depth
            
            
def write_unique_peaks_to_csv(unique_peaks, csv_name, overwrite=False):
    """
    unique_peaks (list of dic): list of unique peaks updated with the occurence count
    csv_name (str): name of ouput csv file
    overwrite (bool): If True the script will overwrite an existing csv file with the name csv_name. If False it will add a counter at the end in that case.
    """
        #Check file ending:
    if str(csv_name)[-4:]!=".csv":
        csv_name=str(csv_name)+".csv"
        #Check if file already exists:
    if os.path.isfile(str(os.getcwd())+"/"+csv_name)==False:
        pass
    else:
        if overwrite==True:
            print(f"NOTE: Overwriting existing file '{csv_name}'.")
        else:
            add_to_file_name=1
            while os.path.isfile(str(os.getcwd())+"/"+csv_name) == True:
                if add_to_file_name==1:
                    csv_name=csv_name[:-4]+f"_{str(add_to_file_name)}.csv"
                else:
                    csv_name=csv_name[:-6]+f"_{str(add_to_file_name)}.csv"
                add_to_file_name+=1
            print(f"NOTE: Chosen file name already exists and you have chosen not to overwrite it. Will write data to f'{csv_name}' instead.")

    #Write self.peaks to file
        #Ordered keys for output
    csv_keys=["occurences", "pot. occ.",  "rel. occ.", "range", "min_seq", "type", "HP0", "HP1", "HP2", "min_len_pLDDTs", "min_len_minPAEs", 
              "best core-motif pLDDT", "best core-motif minPAE", "total fraction", "weighted core-motif pLDDT", "weighted core-motif minPAE", "smart fraction", "length-weighted fraction", "av. res. depth [nm]"]

    #Transform peak dictionary into a sorted list of lines for the csv file [sorted by i) Number of occurences. ii) min_seq start resnum.]
    peak_list=[]
    for peak in unique_peaks:
        #Check peaks for completeness and setting missing values to None
        for csv_key in csv_keys:
            try:
                peak[csv_key]
            except:
                peak[csv_key]=None
        peak_list.append([peak[key] for key in csv_keys])
    try:
        peak_list_sorted=sorted(peak_list, key=lambda x: (-int(sum([(x[int(csv_keys.index("occurences"))])[key_] for key_ in x[int(csv_keys.index("occurences"))].keys()])), int((x[int(csv_keys.index("range"))])[0])))
    except:
        print("Sorting of peak list failed.")
        peak_list_sorted=peak_list
    with open(csv_name, 'w') as f:
        key_string=""
        for csv_key in csv_keys:
            key_string+=str(csv_key)+";"
        key_string=key_string[:-1]
        if len(peak_list_sorted)==0:
            f.write(key_string)
        else:
            f.write(key_string+"\n")
        counter=1
        for peak_data in peak_list_sorted:
            output_string=""
            for entry in peak_data:
                output_string+=str(entry)+";"
            output_string=output_string[:-1]
            if counter < len(peak_list_sorted):
                f.write(output_string+"\n")
            else:
                f.write(output_string)
            counter+=1

                
def plot_peak_summary(unique_peaks, frag_types, output_name, use_n_first_peaks=False, sort_by="self", sorters=[(0,None)], plotwidth=6):
    """
    unique_peaks (list of dic): list of unique peaks updated with the occurence count
    frag_types (list of str): fragment types to consider
    output_name (str): name of output file
    use_n_first_peaks (bool or int or list of int): If False, will show all peaks. If int will show the first int peaks after sorting. If list of ints, will show the peaks with the indices listed, e.g. [0,1,4,8] will show the first, the second, the fifth, and the ninth peak.
    sort_by (str): always sort by absolute ("abs") occurences, alwyas sort by relative ("rel") occurences, or sort peaks for each plot accordingly ("self")
    sorters (list of tuple): tuples define the range in the list of frag_types that should be used for each sorting priority. Sorting will then commence according to the list, with elements with index 0 being first priority, elements iwth index 1 second, etc...
    plodwidth (int or float): in cm
    """
    #Plot the absolute and relative occurences for each LiR
    output_add={"occurences": "_absocc", "rel. occ.": "_relocc"}
    y_label={"occurences": 'Number of fragments', "rel. occ.": 'Fraction of fragments'}
    for counting_type in ["occurences", "rel. occ."]:
        #Find the first n peaks, based on absolute / relative occurences, with the option to prefer earlier specified fragment types over later named fragment types, e.g. 52mers are compared first, then 36mers, then 16mers
        if sort_by=="self":
            peak_list_sorted=sorted(unique_peaks, key=lambda x: [-float(sum([(x[counting_type])[key_] for key_ in frag_types[sorter[0]:sorter[1]]])) for sorter in sorters])
        elif sort_by=="abs":
            peak_list_sorted=sorted(unique_peaks, key=lambda x: [-float(sum([(x["occurences"])[key_] for key_ in frag_types[sorter[0]:sorter[1]]])) for sorter in sorters])
        elif sort_by=="rel":
            peak_list_sorted=sorted(unique_peaks, key=lambda x: [-float(sum([(x["rel. occ."])[key_] for key_ in frag_types[sorter[0]:sorter[1]]])) for sorter in sorters])
        else:
            raise Exception("Unknown 'sort_by' value.")
        if use_n_first_peaks==False:
            use_peaks=peak_list_sorted
            output_peak_info=""
        elif type(use_n_first_peaks) is list:
            indices=[int(i) for i in use_n_first_peaks]
            indices_str=[str(i) for i in use_n_first_peaks]
            use_peaks=np.array(peak_list_sorted)[np.array(indices)]
            output_peak_info="_peaks_shown_"+"_".join(indices_str)
        else:
            use_peaks=peak_list_sorted[:int(use_n_first_peaks)]
            output_peak_info=""

        labels=[]
        max_len=7 #Maximum number of letters before part of the label is ommitted
        for use_peak in use_peaks:
            label=[]
            for index, aa in enumerate(use_peak["min_seq"]):
                try:
                    if (use_peak["min_seq"])[index+1:index+6]=="(HP0)" or (use_peak["min_seq"])[index+1:index+6]=="(HP1)" or (use_peak["min_seq"])[index+1:index+6]=="(HP2)":
                        label.append(r"$\bf{"+str(aa)+"}$")
                    else:
                        label.append(str(aa))
                except:
                    label=append(str(aa))
            cleaned_label=[]
            i=0
            for j in range(0, len(label),1):
                if i >= len(label):
                    break
                if label[i]=="(":
                    i+=4 #Skip over (HP0), (HP1), and (HP2)
                else:
                    cleaned_label.append(label[i])
                i+=1
            if len(cleaned_label) > max_len:
                cleaned_label=cleaned_label[:2]+["..."]+cleaned_label[-2:] 
            label_string="".join(cleaned_label)
            if use_peak["type"]=="DEWDE-LIR":
                full_label=r""+str((use_peak["range"])[0])+"-"+label_string+"-"+str((use_peak["range"])[1])+"\n"+str("[DE]W[DE]-LIR")
            elif use_peak["type"]=="low-conf-LIR":
                full_label=r""+str((use_peak["range"])[0])+"-"+label_string+"-"+str((use_peak["range"])[1])+"\n"+str("lcLIR")
            else:
                full_label=r""+str((use_peak["range"])[0])+"-"+label_string+"-"+str((use_peak["range"])[1])+"\n"+str(use_peak["type"])
            labels.append(full_label)
        x = np.arange(len(labels))  
        width = 0.72/len(frag_types)
        fig, ax = plt.subplots(figsize=(float(plotwidth)/2.54,6.0/2.54))
        #print([key for key in mpl.colormaps.keys()]) #Easy way to check available colors if you want to change them
        colors=mpl.colormaps["Paired"].colors
        all_data=[]
        for index, frag_type in enumerate(frag_types):
            data=[(use_peak[counting_type])[frag_type] if (use_peak[counting_type])[frag_type]!=None else 0 for use_peak in use_peaks]
            all_data=all_data+data
            ax.bar(x - width*((float(len(frag_types))/2-0.5)-float(index)), data, width, label=str(frag_type), edgecolor="black", color=colors[index])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(y_label[counting_type], fontsize=8)
        try:
            max_y=max(max(all_data),1.0) #Works for rel. occ. (1.0 is max), and abs. occ. (1.0 is min)
        except:
            max_y=1.0 #if all_data is empty
        ax.set_yticks(np.arange(0, max_y*1.24, max_y*0.25))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.tick_params(axis='both', which='both', labelsize=6)
        ax.legend(fontsize=6, ncol=3, frameon=True, columnspacing=0.5, labelspacing=0.5, shadow=False, handlelength=0.7, loc="upper center")
        ax.set_ylim([0,max_y*1.5])
        fig.tight_layout()
        if output_name[-4:] in [".png", ".jpg", ".pdf", ".tga"]:
            output=output_name[:-4]+output_peak_info+output_add[counting_type]+output_name[-4:]
        else:
            output=output_name[:-4]+output_peak_info+output_add[counting_type]+".pdf"
        plt.savefig(output, dpi=1200, bbox_inches="tight")
        plt.show(block=False)
    
def load_unique_peaks_from_csv(file_name):
    """
    file_name (str): name of input file
    """
    unique_peaks=[]
    with open(file_name, "r") as f:
        for i, line in enumerate(f):
            elements=[element.replace("\n","") for element in line.split(";")]
            if i==0:
                features=elements
            else:
                unique_peak={}
                for j, feature in enumerate(features):
                    unique_peak[feature]=elements[j]
                unique_peaks.append(unique_peak)
    return unique_peaks

def plot_summary_scores(data_set_names, data_sets, data_set_colors, set_excl_peaks_to_worst_score_instead_of_removal_per_dataset=False, sub_types=[("WT", "solid"), ("ST", "dashed")], scores=[("best core-motif pLDDT", 75, 100), ("best core-motif minPAE", 8, 0), ("total fraction", 0, 1), ("weighted core-motif pLDDT", 75, 100,), ("weighted core-motif minPAE", 8, 0,), ("smart fraction", 0, 1), ("length-weighted fraction", 0, 1)], use_peak_types=[], exclude_peak_types=[], output_name="output_summary_plot", number_of_bins=100, limit_av_depth_to=None, hide_labels=False): #We only use exclude_peak_types if use_peak_types is empty
    """
    data_set_names (list of str):
    data_sets (list of peak dictionaries):
    data_set_colors (list of str):
    set_excl_peaks_to_worst_score_instead_of_removal_per_dataset (bool): If True, will use worst scores for peaks not making the cutoff instead of removing them. You should use False if you already filtered your data_set and apply a different treatment there.
    sub_types (list of tuples (str, str)): names and linestyles for sub_types of peaks
    scores (list of tuples (str, int or float, int or float)): each tuple contains the name of the score, the worst score, and the best score (in that order)
    use_peak_types (list of str): peak types to be used in the summary
    exclude_peak_types (list of str): peak types to not be used in the summary. Is only used if use_peak_types is an empty list
    number_of_bins (int): number of bins for the cumulative histogram
    limit_av_depth_to (float): upper cutoff (in nm) for residue depth
    hide_labels (bool): If True, will not show labels
    """
    if isinstance(set_excl_peaks_to_worst_score_instead_of_removal_per_dataset, list):
        pass
    else:
        set_all_to=set_excl_peaks_to_worst_score_instead_of_removal_per_dataset
        set_excl_peaks_to_worst_score_instead_of_removal_per_dataset=[set_all_to for _ in  data_set_names]
    for score in scores:
        fig=plt.figure(figsize=(5/2.54,5/2.54))
        plt.xlim([float(score[1]), float(score[2])])
        plt.ylim([-0.1, 1.1])
        try:
            x_label_dic={"length-weighted fraction": "LO score"}
            plt.xlabel(x_label_dic[str(score[0])], fontsize=8)
        except:
            plt.xlabel(score[0], fontsize=8)
        plt.xticks(fontsize=8)
        plt.ylabel("CDF", fontsize=8)
        plt.yticks(fontsize=8)
        for i, data_set in enumerate(data_sets):            
            for j, sub_type in enumerate(sub_types):
                values=[]
                for peak in data_set:
                    if use_peak_types!=[]:
                        if str(peak["type"]).replace("'","").replace('"','') in use_peak_types:
                            if limit_av_depth_to==None:
                                values.append(float(ast.literal_eval(str(peak[score[0]]))[sub_type[0]]))
                            elif peak["av. res. depth [nm]"]!=None and str(peak["av. res. depth [nm]"]).replace("'","").replace(" ","").replace('"','').replace("\n","").lower()!=str("none"):
                                if float(peak["av. res. depth [nm]"])<=limit_av_depth_to:
                                    values.append(float(ast.literal_eval(str(peak[score[0]]))[sub_type[0]]))
                                else:
                                    if set_excl_peaks_to_worst_score_instead_of_removal_per_dataset[i]==True:
                                        values.append(score[1])
                            else:
                                values.append(float(ast.literal_eval(str(peak[score[0]]))[sub_type[0]]))
                        else:
                            if set_excl_peaks_to_worst_score_instead_of_removal_per_dataset[i]==True:
                                values.append(score[1])
                    else:
                        if str(peak["type"]).replace("'","").replace('"','') not in exclude_peak_types:
                            if limit_av_depth_to==None:
                                values.append(float(ast.literal_eval(str(peak[score[0]]))[sub_type[0]]))
                            elif peak["av. res. depth [nm]"]!=None and str(peak["av. res. depth [nm]"]).replace("'","").replace(" ","").replace('"','').replace("\n","").lower()!=str("none"):
                                if float(peak["av. res. depth [nm]"])<=limit_av_depth_to:
                                    values.append(float(ast.literal_eval(str(peak[score[0]]))[sub_type[0]]))
                                else:
                                    if set_excl_peaks_to_worst_score_instead_of_removal_per_dataset[i]==True:
                                        values.append(score[1])
                            else:
                                values.append(float(ast.literal_eval(str(peak[score[0]]))[sub_type[0]]))
                        else:
                            if set_excl_peaks_to_worst_score_instead_of_removal_per_dataset[i]==True:
                                values.append(score[1])
                #im=sns.histplot(values, zorder=2, binrange=(min(score[1], score[2]), max(score[1], score[2])), stat="proportion", bins=number_of_bins, alpha=0.8, color=data_set_colors[i], cumulative=True, fill=False, element="step", linestyle=sub_type[1], label=f"{str(data_set_names[i])}_{str(sub_type[0])}")
                if score[1]>score[2]:
                    im=plt.hist(values, zorder=2, range=(min(score[1], score[2], min(values)), max(score[1], score[2], max(values))), density=True, bins=number_of_bins, alpha=0.8, color=data_set_colors[i], cumulative=-1, histtype="step", linestyle=sub_type[1], label=f"{str(data_set_names[i])}_{str(sub_type[0])}")
                elif score[2]>score[1]:
                    im=plt.hist(values, zorder=2, range=(min(score[1], score[2], min(values)), max(score[1], score[2], max(values))), density=True, bins=number_of_bins, alpha=0.8, color=data_set_colors[i], cumulative=True, histtype="step", linestyle=sub_type[1], label=f"{str(data_set_names[i])}_{str(sub_type[0])}")
                else:
                    raise Exception(f"Upper and lower boundaries for score {str(score[0])} identical, so plotting it seems like a waste of pixels.")
            print(f"Dataset {str(data_set_names[i])} contained {str(len(values))} peaks.")
        if hide_labels==False:
            plt.legend(fontsize=8)
        plt.grid(alpha=0.5)
        #Construct name of output file
        if output_name[-4:] in [".png", ".jpg", ".pdf", ".svg"]:
            name=output_name[:-4]
        else:
            name=output_name
        if use_peak_types!=[]:
            name+="_include"
            for peak_type in use_peak_types:
                name+="_"+str(peak_type)
        elif exclude_peak_types!=[]:
            name+="_exclude"
            for peak_type in exclude_peak_types:
                name+="_"+str(peak_type)
        else:
            pass
        if limit_av_depth_to!=None:
            name+=f"_av_depth_lim_{str(limit_av_depth_to)}"
        name+="_"+str(score[0])+".pdf"
        name=name.replace(" ","_").replace(' ','')
        print(f"Writing file {str(name)}.")
        plt.savefig(name, bbox_inches="tight", dpi=1200)
        plt.show()

def search_exp_lirs(systems, peak_list, func_lirs, non_func_lirs, output="summary_lir_systems", score=("length-weighted fraction", 0, 1), min_overlap=1, mod_types=["WT", "ST"], use_peak_types=[], exclude_peak_types=[], limit_av_depth_to=None):
    """
    systems (list of str): names of systems 
    peak_list (dic of list): dic (with systems as keys) of lists of unique peaks updated with scores
    func_lirs (dic of list of tuples of ints): residue ranges of exp. confirmed functional LIRs
    non_func_lirs (dic of list of tuples of ints): residue ranges of exp. confirmed functional LIRs
    output (str): name of output file
    score (tuple (str, int or float, int or float)): each tuple contains the name of the score, the worst score, and the best score (in that order)
    min_moverlap (int): minimum length of a peak motif shared with exp. motif
    mod_types (list of str): modification types to consider
    use_peak_types (list of str): peak types to be used in the summary
    exclude_peak_types (list of str): peak types to not be used in the summary. Is only used if use_peak_types is an empty list
    limit_av_depth_to (float): upper cutoff (in nm) for residue depth
    """
    unconfirmed_lirs_list=[]
    confirmed_func_lirs_list=[]
    confirmed_non_func_lirs_list=[]
    confirmed_func_lirs_list_weak_cands=[]
    confirmed_non_func_lirs_list_weak_cands=[]    
    
    for system in systems:
        #Func LIRs
        for func_lir in func_lirs[system]:
            cand=None
            cand_score=None
            for peak in peak_list[system]:
                if peak["av. res. depth [nm]"]!=None and str(peak["av. res. depth [nm]"]).replace("'","").replace(" ","").replace('"','').replace("\n","").lower()!=str("none"):
                    if limit_av_depth_to!=None and float(peak["av. res. depth [nm]"])>=limit_av_depth_to:
                        continue
                if use_peak_types!=[] and peak["type"] not in use_peak_types:
                    continue
                elif exclude_peak_types!=[] and peak["type"] in exclude_peak_types: #Only use exclusion if inclusion is empty
                    continue
                else:
                    pass
                min_overlap_range=(max(int(peak["range"].replace("(","").replace(")","").split(",")[0]), int(func_lir[0])),min(int(peak["range"].replace("(","").replace(")","").split(",")[1]), int(func_lir[1])))
                if (min_overlap_range[0] >= int(peak["range"].replace("(","").replace(")","").split(",")[0]) and min_overlap_range[0] <= int(peak["range"].replace("(","").replace(")","").split(",")[1]) and min_overlap_range[1] >= int(peak["range"].replace("(","").replace(")","").split(",")[0]) and min_overlap_range[1] <= int(peak["range"].replace("(","").replace(")","").split(",")[1]) and
                    min_overlap_range[0] >= func_lir[0] and min_overlap_range[0] <= func_lir[1] and min_overlap_range[1] >= func_lir[0] and min_overlap_range[1] <= func_lir[1]) and min_overlap_range[1]-min_overlap_range[0] >= min_overlap-1:
                    #Calculate score:
                    mod_type_scores=[]
                    for mod_type in mod_types:
                        mod_type_scores.append(float(ast.literal_eval(peak[score[0]])[mod_type]))
                    current_peak_score=sum(mod_type_scores)/len(mod_type_scores)
                    if cand_score==None: #Catch None's in score input in case min / max can't be defined in advance.
                        cand=peak
                        cand_score=current_peak_score
                    elif (current_peak_score > cand_score and score[2]>score[1]) or (current_peak_score < cand_score and score[2]<score[1]):
                        if cand!=None:
                            confirmed_func_lirs_list_weak_cands.append(cand)
                        else:
                            pass
                        cand=peak
                        cand_score=current_peak_score
                    else:
                        confirmed_func_lirs_list_weak_cands.append(peak)
            if cand!=None:
                confirmed_func_lirs_list.append(cand)
            else:
                missed_signal={}
                missed_signal[score[0]]={}
                missed_signal["range"]=func_lir
                missed_signal["type"]="not-found"
                missed_signal["av. res. depth [nm]"]=None
                for mod_type in mod_types:
                    (missed_signal[score[0]])[mod_type]=score[1]
                print(f"Did not find a matching candidate for experimentally functional LIR {str(func_lir)} in {str(system)}.")
                confirmed_func_lirs_list.append(missed_signal)

        #Non func LIRs
        for non_func_lir in non_func_lirs[system]:
            cand=None
            cand_score=None
            for peak in peak_list[system]:
                if peak["av. res. depth [nm]"]!=None and str(peak["av. res. depth [nm]"]).replace("'","").replace(" ","").replace('"','').replace("\n","").lower()!=str("none"):
                    if limit_av_depth_to!=None and float(peak["av. res. depth [nm]"])>=limit_av_depth_to:
                        continue
                if use_peak_types!=[] and peak["type"] not in use_peak_types:
                    continue
                elif exclude_peak_types!=[] and peak["type"] in exclude_peak_types: #Only use exclusion if inclusion is empty
                    continue
                else:
                    pass
                min_overlap_range=(max(int(peak["range"].replace("(","").replace(")","").split(",")[0]), int(non_func_lir[0])),min(int(peak["range"].replace("(","").replace(")","").split(",")[1]), int(non_func_lir[1])))
                if (min_overlap_range[0] >= int(peak["range"].replace("(","").replace(")","").split(",")[0]) and min_overlap_range[0] <= int(peak["range"].replace("(","").replace(")","").split(",")[1]) and min_overlap_range[1] >= int(peak["range"].replace("(","").replace(")","").split(",")[0]) and min_overlap_range[1] <= int(peak["range"].replace("(","").replace(")","").split(",")[1]) and
                    min_overlap_range[0] >= non_func_lir[0] and min_overlap_range[0] <= non_func_lir[1] and min_overlap_range[1] >= non_func_lir[0] and min_overlap_range[1] <= non_func_lir[1]) and min_overlap_range[1]-min_overlap_range[0] >= min_overlap-1:
                    #Calculate score:
                    mod_type_scores=[]
                    for mod_type in mod_types:
                        mod_type_scores.append(float(ast.literal_eval(peak[score[0]])[mod_type]))
                    current_peak_score=sum(mod_type_scores)/len(mod_type_scores)
                    if cand_score==None: #Catch None's in score input in case min / max can't be defined in advance.
                        cand=peak
                        cand_score=current_peak_score
                    elif (current_peak_score > cand_score and score[2]>score[1]) or (current_peak_score < cand_score and score[2]<score[1]):
                        if cand!=None:
                            confirmed_non_func_lirs_list_weak_cands.append(cand)
                        else:
                            pass
                        cand=peak
                        cand_score=current_peak_score
                    else:
                        confirmed_non_func_lirs_list_weak_cands.append(peak)
            if cand!=None:
                confirmed_non_func_lirs_list.append(cand)
            else:
                missed_signal={}
                missed_signal[score[0]]={}
                missed_signal["range"]=func_lir
                missed_signal["type"]="not-found"
                missed_signal["av. res. depth [nm]"]=None
                for mod_type in mod_types:
                    (missed_signal[score[0]])[mod_type]=score[1]
                print(f"Did not find a matching candidate for experimentally non-functional LIR {str(non_func_lir)} in {str(system)}.")
                confirmed_non_func_lirs_list.append(missed_signal)
        #Unconfirmed LIRs
        for peak in peak_list[system]:
            if peak["av. res. depth [nm]"]!=None:
                if limit_av_depth_to!=None and float(peak["av. res. depth [nm]"])>=limit_av_depth_to:
                    continue #Here, one could collect all peaks that did not make it
            if use_peak_types!=[] and peak["type"] not in use_peak_types:
                continue
            elif exclude_peak_types!=[] and peak["type"] in exclude_peak_types: #Only use exclusion if inclusion is empty
                continue
            else:
                pass
            if peak not in confirmed_func_lirs_list+confirmed_func_lirs_list_weak_cands+confirmed_non_func_lirs_list+confirmed_non_func_lirs_list_weak_cands:
                unconfirmed_lirs_list.append(peak)
            else:
                pass
    #Output files
    #Construct name of output file
    if output[-4:] in [".csv", ".txt"]:
        name=output[:-4]
    else:
        name=output
    if use_peak_types!=[]:
        name+="_include"
        for peak_type in use_peak_types:
            name+="_"+str(peak_type)
    elif exclude_peak_types!=[]:
        name+="_exclude"
        for peak_type in exclude_peak_types:
            name+="_"+str(peak_type)
    else:
        pass
    if limit_av_depth_to!=None:
        name+=f"_av_depth_lim_{str(limit_av_depth_to)}"
    name+="_"+str(score[0])
    name=name.replace(" ","_").replace(' ','')
    write_unique_peaks_to_csv(unique_peaks=confirmed_func_lirs_list, csv_name=name+"_confirmed_func_lirs", overwrite=True)
    write_unique_peaks_to_csv(unique_peaks=confirmed_func_lirs_list_weak_cands, csv_name=name+"_confirmed_func_lirs_weaker_cands", overwrite=True)
    write_unique_peaks_to_csv(unique_peaks=confirmed_non_func_lirs_list, csv_name=name+"_confirmed_non_func_lirs", overwrite=True)
    write_unique_peaks_to_csv(unique_peaks=confirmed_non_func_lirs_list_weak_cands, csv_name=name+"_confirmed_non_func_lirs_weaker_cands", overwrite=True)
    write_unique_peaks_to_csv(unique_peaks=unconfirmed_lirs_list, csv_name=name+"_unconfirmed_lirs", overwrite=True)
    #Also returns output as dictionary
    return {"conf_func_lirs": confirmed_func_lirs_list,
            "conf_func_lirs_weak_cands": confirmed_func_lirs_list_weak_cands,
            "conf_non_func_lirs": confirmed_non_func_lirs_list,
            "conf_non_func_lirs_weak_cands": confirmed_non_func_lirs_list_weak_cands,
            "unconf_lirs": unconfirmed_lirs_list
           }
def plot_peak_score(unique_peaks, mod_types, output_name, colors=None, use_n_first_peaks=False, score=("length-weighted fraction", 0.0, 1.0), sort_by="all", plotwidth=6, use_peak_types=[], exclude_peak_types=[], limit_av_depth_to=None, hide_legend=False):
    """
    unique_peaks (list of dic): list of unique peaks updatedwith scores
    mod_types (list of str): modification types to consider
    output_name (str): name of output file
    colors (None or list of str): colors to be used for each entry in mod_types
    use_n_first_peaks (bool or int or list of int): If False, will show all peaks. If int will show the first int peaks after sorting. If list of ints, will show the peaks with the indices listed, e.g. [0,1,4,8] will show the first, the second, the fifth, and the ninth peak.
    score (tuple (str, int or float, int or float)): each tuple contains the name of the score, the worst score, and the best score (in that order)
    sort_by (str): always sort by a given mod_type (or use 'all' to sum scores over all mod types)
    plodwidth (int or float): in cm  
    use_peak_types (list of str): peak types to be used in the summary
    exclude_peak_types (list of str): peak types to not be used in the summary. Is only used if use_peak_types is an empty list
    limit_av_depth_to (float): upper cutoff (in nm) for residue depth
    hide_labels (bool): If True, will not show labels
    """
    #Apply filters
    filtered_peaks=[]
    for peak in unique_peaks:
        if peak["av. res. depth [nm]"]!=None and str(peak["av. res. depth [nm]"]).replace("'","").replace(" ","").replace('"','').replace("\n","").lower()!=str("none"):
            if limit_av_depth_to!=None and float(peak["av. res. depth [nm]"])>=limit_av_depth_to:
                continue
        if use_peak_types!=[] and peak["type"] not in use_peak_types:
            continue
        elif exclude_peak_types!=[] and peak["type"] in exclude_peak_types: #Only use exclusion if inclusion is empty
            continue
        else:
            filtered_peaks.append(peak)
    unique_peaks=filtered_peaks
    #Find the first n peaks, based on score
    if sort_by=="all":
        if float(score[1])<float(score[2]):
            peak_list_sorted=sorted(unique_peaks, key=lambda x: -float(sum([ast.literal_eval(x[str(score[0])])[key_] for key_ in mod_types])))
        elif float(score[1])>float(score[2]):
            peak_list_sorted=sorted(unique_peaks, key=lambda x: float(sum([ast.literal_eval(x[str(score[0])])[key_] for key_ in mod_types])))
        else:
            raise Exception("Score minimum and maximum value identical. It seems pointless to plot such a score.")
    else:
        if float(score[1])<float(score[2]):
            peak_list_sorted=sorted(unique_peaks, key=lambda x: -float(ast.literal_eval(x[str(score[0])])[sort_by]))
        elif float(score[1])>float(score[2]):
            peak_list_sorted=sorted(unique_peaks, key=lambda x: -float(ast.literal_eval(x[str(score[0])])[sort_by]))
        else:
            raise Exception("Score minimum and maximum value identical. It seems pointless to plot such a score.")
    if use_n_first_peaks==False:
        use_peaks=peak_list_sorted
        output_peak_info=""
    elif type(use_n_first_peaks) is list:
        indices=[int(i) for i in use_n_first_peaks]
        indices_str=[str(i) for i in use_n_first_peaks]
        use_peaks=np.array(peak_list_sorted)[np.array(indices)]
        output_peak_info="_peaks_shown_"+"_".join(indices_str)
    else:
        use_peaks=peak_list_sorted[:int(use_n_first_peaks)]
        output_peak_info=""

    labels=[]
    max_len=7 #Maximum number of letters before part of the label is ommitted
    for use_peak in use_peaks:
        use_peak=ast.literal_eval(str(use_peak))
        label=[]
        for index, aa in enumerate(use_peak["min_seq"]):
            try:
                if (use_peak["min_seq"])[index+1:index+6]=="(HP0)" or (use_peak["min_seq"])[index+1:index+6]=="(HP1)" or (use_peak["min_seq"])[index+1:index+6]=="(HP2)":
                    label.append(r"$\bf{"+str(aa)+"}$")
                else:
                    label.append(str(aa))
            except:
                label=append(str(aa))
        cleaned_label=[]
        i=0
        for j in range(0, len(label),1):
            if i >= len(label):
                break
            if label[i]=="(":
                i+=4 #Skip over (HP)), (HP1), and (HP2)
            else:
                cleaned_label.append(label[i])
            i+=1
        if len(cleaned_label) > max_len:
            cleaned_label=cleaned_label[:2]+["..."]+cleaned_label[-2:] 
        label_string="".join(cleaned_label)
        if use_peak["type"]=="DEWDE-LIR":
            full_label=r""+str((str(use_peak["range"]).replace("(","").replace(")","").split(","))[0])+"-"+label_string+"-"+str((str(use_peak["range"]).replace("(","").replace(")","").split(","))[1])+"\n"+str("[DE]W[DE]-LIR")
        elif use_peak["type"]=="low-conf-LIR":
            full_label=r""+str((str(use_peak["range"]).replace("(","").replace(")","").split(","))[0])+"-"+label_string+"-"+str((str(use_peak["range"]).replace("(","").replace(")","").split(","))[1])+"\n"+str("lcLIR")
        else:
            full_label=r""+str((str(use_peak["range"]).replace("(","").replace(")","").split(","))[0])+"-"+label_string+"-"+str((str(use_peak["range"]).replace("(","").replace(")","").split(","))[1])+"\n"+str(use_peak["type"])
        full_label=full_label.replace(" ","").replace(' ','')
        labels.append(full_label)
    x = np.arange(len(labels))  
    width = 0.72/len(mod_types)
    fig, ax = plt.subplots(figsize=(float(plotwidth)/2.54,6.0/2.54))
    #print([key for key in mpl.colormaps.keys()]) #Easy way to check available colors if you want to change them
    if colors==None:
        colors=mpl.colormaps["Paired"].colors
    all_data=[]
    for index, mod_type in enumerate(mod_types):
        data=[ast.literal_eval(str(use_peak[str(score[0])]))[mod_type] if ast.literal_eval(str(use_peak[str(score[0])]))[mod_type]!=None else 0 for use_peak in use_peaks]
        all_data=all_data+data
        ax.bar(x - width*((float(len(mod_types))/2-0.5)-float(index)), data, width, label=str(mod_type), edgecolor="black", color=colors[index])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    try:
        y_label_dic={"length-weighted fraction": "LO score"}
        ax.set_ylabel(y_label_dic[str(score[0])], fontsize=8)
    except:
        ax.set_ylabel(str(score[0]), fontsize=8)
    try:
        max_y=max(max(all_data),1.0) #Works for rel. occ. (1.0 is max), and abs. occ. (1.0 is min)
    except:
        max_y=1.0 #if all_data is empty
    ax.set_yticks(np.arange(0, max_y*1.01, max_y*0.25))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.tick_params(axis='both', which='both', labelsize=6)
    if hide_legend==False:
        ax.legend(fontsize=6, ncol=3, frameon=True, columnspacing=0.5, labelspacing=0.5, shadow=False, handlelength=0.7, loc="upper center")
    ax.set_ylim([0,max_y*1.2])
    fig.tight_layout()
    #Construct name of output file
    if output_name[-4:] in [".png", ".jpg", ".pdf", ".tga"]:
        name=output_name[:-4]
        file_ending=output_name[-4:]
    else:
        name=output_name
        file_ending=".pdf"
    if use_peak_types!=[]:
        name+="_include"
        for peak_type in use_peak_types:
            name+="_"+str(peak_type)
    elif exclude_peak_types!=[]:
        name+="_exclude"
        for peak_type in exclude_peak_types:
            name+="_"+str(peak_type)
    else:
        pass
    if limit_av_depth_to!=None:
        name+=f"_av_depth_lim_{str(limit_av_depth_to)}"
    name+="_"+str(score[0])
    name=name.replace(" ","_").replace(' ','')
    output=name+output_peak_info+file_ending
    print(f"Writing file {str(output)}.")
    plt.savefig(output, dpi=1200, bbox_inches="tight")
    plt.show(block=False)