#Python functions to summarize the output from the "seq_scan_analysis" class
#Version 2024-08-30, Jan Stuke

#Imports

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys

#Functions

def read_features_from_csv(file_name, frag_type):
    """
    file_name (str): name of csv file containing list of fragments
    frag_type (str): identifier of fragments in csv file, indicating e.g. fragment length and mutations / PTMs
    """
    #Read raw features
    raw_feature_indices={"Start": None, "End": None, "sequence": None, "type": None, "fragment": None, "av_pLDDT": None, "av_minPAE": None}
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
        refined_features_for_peak["type"]=raw_features_for_peak["type"]
        #min_seq (will be updated later when comparing peaks with overlapping range)
        refined_features_for_peak["min_seq"]=raw_features_for_peak["sequence"].split("-")[1]
        #Range
        refined_features_for_peak["range"]=(int(raw_features_for_peak["Start"]), int(raw_features_for_peak["End"]))
        #HP1 and HP2
        seq=raw_features_for_peak["sequence"].split("-")[1]
        for index, aa in enumerate(seq):
            try:
                if seq[index+1:index+6]=="(HP1)":
                    refined_features_for_peak["HP1"]=(str(aa), int(raw_features_for_peak["Start"])+index) #Amino acid name and residue number as tuple
                elif seq[index+1:index+6]=="(HP2)":
                    refined_features_for_peak["HP2"]=(str(aa), int(raw_features_for_peak["Start"])+index) #Amino acid name and residue number as tuple
                else:
                    continue
            except:
                break
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
        #Add to output
        refined_features_for_peaks.append(refined_features_for_peak)
    return refined_features_for_peaks

def summarize_peaks(input_peak_list, frag_types, min_motif_len=3):
    """
    input_peak_list (list of dic): list of all peaks to summarized
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
            if peak["HP1"]==unique_peak["HP1"] and peak["HP2"]==unique_peak["HP2"] and peak["type"]==unique_peak["type"]:
                #Not so straight forward comparisons (Redundant if HP1 or HP2 is not None. Important otherwise 
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
    return unique_peaks

def find_fragments(frag_name, fragment_main_dir, frag_name_add="", name_addition_optional=True, prefer_addition=True, msa_type="frag_msa", frag_len="any"): #If name_addition_optional, the function will still prefer the fragment with addition (self.name_add) if available
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
        for frag_type in frag_types:
            max_possible_occ=0
            for unique_range in unique_ranges_dic[frag_type]:
                if (unique_peak["range"])[0] >= unique_range[0] and (unique_peak["range"])[1] <= unique_range[1]:
                    max_possible_occ+=1
            if (unique_peak["occurences"])[frag_type]==0: #Avoid division by 0 for peaks that cannot occur for a certain fragment length
                (unique_peak["rel. occ."])[frag_type]=0
            else:
                (unique_peak["rel. occ."])[frag_type]=(unique_peak["occurences"])[frag_type]/max_possible_occ
        updated_unique_peaks.append(unique_peak)
    return updated_unique_peaks

def write_unique_peaks_to_csv(updated_unique_peaks, csv_name, overwrite=False):
        """
        updated_unique_peaks (list of dic): list of unique peaks updated with the occurence count
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
                    csv_name=csv_name[:-4]+f"_{str(add_to_file_name)}.csv"
                    add_to_file_name+=1
                print(f"NOTE: Chosen file name already exists and you have chosen not to overwrite it. Will write data to f'{csv_name}' instead.")

        #Write self.peaks to file
            #Ordered keys for output
        csv_keys=["occurences", "range", "min_seq", "type", "HP1", "HP2"]

        #Transform peak dictionary into a sorted list of lines for the csv file [sorted by i) Number of occurences. ii) min_seq start resnum.]
        peak_list=[]
        for peak in updated_unique_peaks:
            #Check peaks for completeness and setting missing values to None
            for csv_key in csv_keys:
                try:
                    peak[csv_key]
                except:
                    peak[csv_key]=None
            peak_list.append([peak[key] for key in csv_keys])
        peak_list_sorted=sorted(peak_list, key=lambda x: (-int(sum([(x[int(csv_keys.index("occurences"))])[key_] for key_ in x[int(csv_keys.index("occurences"))].keys()])), int((x[int(csv_keys.index("range"))])[0])))

        with open(csv_name, 'w') as f:
            if len(peak_list_sorted)==0:
                f.write(str(csv_keys).replace("[","").replace("]",""))
            else:
                f.write(str(csv_keys).replace("[","").replace("]","")+"\n")
            counter=1
            for peak_data in peak_list_sorted:
                if counter < len(peak_list_sorted):
                    f.write(str(peak_data).replace("[","").replace("]","")+"\n")
                else:
                    f.write(str(peak_data).replace("[","").replace("]",""))
                counter+=1

                
def plot_peak_summary(updated_unique_peaks, frag_types, output_name, use_n_first_peaks=False, sort_by="self", sorters=[(0,None)], plotwidth=6): #list, list, False or int, str ("self", "abs", "rel"), list of tuples, int or float (in cm)
    """
    updated_unique_peaks (list of dic): list of unique peaks updated with the occurence count
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
            peak_list_sorted=sorted(updated_unique_peaks, key=lambda x: [-float(sum([(x[counting_type])[key_] for key_ in frag_types[sorter[0]:sorter[1]]])) for sorter in sorters])
        elif sort_by=="abs":
            peak_list_sorted=sorted(updated_unique_peaks, key=lambda x: [-float(sum([(x["occurences"])[key_] for key_ in frag_types[sorter[0]:sorter[1]]])) for sorter in sorters])
        elif sort_by=="rel":
            peak_list_sorted=sorted(updated_unique_peaks, key=lambda x: [-float(sum([(x["rel. occ."])[key_] for key_ in frag_types[sorter[0]:sorter[1]]])) for sorter in sorters])
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
                    if (use_peak["min_seq"])[index+1:index+6]=="(HP1)" or (use_peak["min_seq"])[index+1:index+6]=="(HP2)":
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
                    i+=4 #Skip over (HP1) and (HP2)
                else:
                    cleaned_label.append(label[i])
                i+=1
            if len(cleaned_label) > max_len:
                cleaned_label=cleaned_label[:2]+["..."]+cleaned_label[-2:] 
            label_string="".join(cleaned_label)
            full_label=r""+str((use_peak["range"])[0])+"-"+label_string+"-"+str((use_peak["range"])[1])
            labels.append(full_label)
        x = np.arange(len(labels))  
        width = 0.12
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
        if counting_type=="rel. occ.":
            max_y=max(max(all_data),1.0)
        else:
            max_y=max(all_data)
        ax.set_yticks(np.arange(0, max_y*1.26, max_y*0.25))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.tick_params(axis='both', which='both', labelsize=6)
        ax.legend(fontsize=6, ncol=3, frameon=True, columnspacing=0.5, labelspacing=0.5, shadow=False, handlelength=0.7, loc="upper center")
        ax.set_ylim([0,max_y*1.5])
        fig.tight_layout()
        if output_name[-4:] in [".png", ".jpg", ".pdf", ".tga"]:
            output=output_name[:-4]+output_peak_info+output_add[counting_type]+output_name[-4:]
        else:
            output=output_name[:-4]+output_peak_info+output_add[counting_type]+".png"
        plt.savefig(output, dpi=600)
        plt.show(block=False)
    
