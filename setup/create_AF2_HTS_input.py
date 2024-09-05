#!/usr/bin/python3

"""
This script creates bait and candidate input files for the AlphaPulldown pipeline (https://doi.org/10.1093/bioinformatics/btac749). It takes fasta-files and fragmentation instructions as input and outputs fasta and txt files.
"""

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

#Version 2024-08-30

import math

#Input parameters

#Baits

baits_output = ("baits.fasta", "baits.txt") #Tuple of two strings, first for fasta and second for txt output
baits_input_dir = "./example_files/" #Directory containing INPUT files for the baits
baits_files = ["LC3B.fasta"] #List of fasta INPUT files
baits_start_resnums = [1] #List of resnums of first residue in the respective fasta file. Will use 1 if empty.
baits_scan_mode = ["full"] #List of modes: either str("full") or (int(FRAGMENT_LENGTH), int(FRAGMENT_OVERLAP)) for every sequence

#Candidates

candidates_output = ("candidates.fasta", "candidates.txt") #Tuple of two strings, first for fasta and second for txt output
candidates_input_dir = "./example_files/" #Directory containing INPUT files for the candidates
candidates_files = ["FUNDC1.fasta", "FUNDC1-STphosmim.fasta"] #List of fasta INPUT files
candidates_start_resnums = [1, 1] #List of resnums of first residue in the respective fasta file. Will use 1 if empty.
candidates_scan_mode = [(52, 39), (52, 39)] #List of modes: either str("full") or (int(FRAGMENT_LENGTH), int(FRAGMENT_OVERLAP)) for every sequence

#Functions

def load_seq(seq_file): 
    """
    seq_file (str): name of sequence file (in fasta format)
    """
    with open(seq_file, "r") as f:
        seq = ""
        for line in f:
            if line[0] == ">":
                header = line
                name = line[1:].split(" ")[0].replace("|","_").replace("\n", "")
            else:
                seq = seq + line.replace("\n","")
    return seq, name

def create_fragments(sequence, name, start_res_num, frag_length, frag_overlap):
    """
    sequence (str): single letter amino acid sequence
    name (str): name associated with sequence
    start_res_num (int): residue number of first amino acid listed in the sequence
    frag_length (int): length (in residues) of the desired fragments
    frag_overlap (int): overlap (in residues) between fragments
    """
    frag_seqs = []
    frag_names = []
    n_frags = int(math.ceil(float(len(sequence))/(frag_length-frag_overlap)))
    #All Fragments but the last one
    for i in range(0, n_frags, 1):
        seq_start = i*(frag_length-frag_overlap)
        seq_end = seq_start+frag_length
        frag_seq = sequence[seq_start:seq_end]
        frag_name = f"{name}-{str(seq_start+int(start_res_num))}-{str(seq_end-1+int(start_res_num))}".replace("|","_")
        if len(frag_seq) == frag_length: #Exclude the short fragment(s) at the end
            frag_seqs.append(frag_seq)
            frag_names.append(frag_name)
        else:
            pass
    #Last fragment
    frag_seq = sequence[-frag_length:]
    frag_name = f"{name}-{str(len(sequence)-frag_length+int(start_res_num))}-{str(len(sequence)-1+int(start_res_num))}".replace("|","_") # "|" in name strings is not compatible with alpha pulldown
    frag_seqs.append(frag_seq)
    frag_names.append(frag_name)
    return frag_seqs, frag_names

def attach_sequence(sequence, name, target_seq_list, target_name_list):
    """
    sequence (str): single letter amino acid sequence (potentially of a fragment)
    name (str): name associated with sequence (potentially of a fragment)
    target_seq_list (list of str): list of sequences the target sequence should be added to
    target_name_list (list of str): names of sequences in target_seq_list
    """
    if sequence in target_seq_list:
        print(f"Sequence {sequence} (with name: {name}) already in target list. Will not add it again.")
        pass
    else:
        target_seq_list.append(sequence)
        target_name_list.append(name)
    return target_seq_list, target_name_list

def write_output(target_seq_list, target_name_list, output_fasta, output_txt):
    """
    target_seq_list (list of str): list of output sequences
    target_name_list (list of str): names of sequences in target_seq_list
    output_fasta (str): name of output fasta file
    output_txt (str): name of output txt file
    """
    #FASTA
    if output_fasta[-6:] != ".fasta":
        output_fasta = output_fasta + ".fasta"
    with open(output_fasta, "w") as o:
        for index, seq in enumerate(target_seq_list):
            if index == 0:
                o.write(f">{target_name_list[index]}\n{seq}")
            else:
                o.write(f"\n>{target_name_list[index]}\n{seq}")
    #TXT
    if output_txt[-4:] != ".txt":
        output_txt = output_txt + ".txt"
    with open(output_txt, "w") as o:
        for index, name in enumerate(target_name_list):
            if index == 0:
                o.write(f"{name}")
            else:
                o.write(f"\n{name}")

#Main

if __name__ == "__main__":
    
    #Baits
    baits_seqs = []
    baits_names = []
    if baits_input_dir[-1] != "/":
        baits_input_dir = baits_input_dir + "/"
    for index, baits_file in enumerate(baits_files):
        raw_seq, raw_name = load_seq(baits_input_dir+baits_file)
        if baits_scan_mode[index] == "full":
            baits_seqs, baits_names = attach_sequence(raw_seq, raw_name, baits_seqs, baits_names)
        else:
            try:
                start_resnum = baits_start_resnums[index]
            except:
                start_resnum = 1
            proc_seqs, proc_names = create_fragments(raw_seq, raw_name, start_resnum, (baits_scan_mode[index])[0], (baits_scan_mode[index])[1])
            for also_index, proc_seq in enumerate(proc_seqs):
                baits_seqs, baits_names = attach_sequence(proc_seq, proc_names[also_index], baits_seqs, baits_names)
    write_output(baits_seqs, baits_names, baits_output[0], baits_output[1])

    #Candidates
    candidates_seqs = []
    candidates_names = []
    if candidates_input_dir[-1] != "/":
        candidates_input_dir = candidates_input_dir + "/"
    for index, candidates_file in enumerate(candidates_files):
        raw_seq, raw_name = load_seq(candidates_input_dir+candidates_file)
        if candidates_scan_mode[index] == "full":
            candidates_seqs, candidates_names = attach_sequence(raw_seq, raw_name, candidates_seqs, candidates_names)
        else:
            try:
                start_resnum = candidates_start_resnums[index]
            except:
                start_resnum = 1
            proc_seqs, proc_names = create_fragments(raw_seq, raw_name, start_resnum, (candidates_scan_mode[index])[0], (candidates_scan_mode[index])[1])
            for also_index, proc_seq in enumerate(proc_seqs):
                candidates_seqs, candidates_names = attach_sequence(proc_seq, proc_names[also_index], candidates_seqs, candidates_names)
    write_output(candidates_seqs, candidates_names, candidates_output[0], candidates_output[1])

    print(f"Done.")

