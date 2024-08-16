# the vast majority of the infrastructure in this program was inspired by: 
# https://github.com/Machine-Learning-for-Medical-Language/curate-mimic/blob/main/extract_mimic_temporal.py
# unlike `extract_temporal`, this script reads from a directory instead of a single file.
import argparse
import json
import os
import pathlib
import pdb
import pickle
import requests
import sys

from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.tokenize.util import align_tokens
from PyRuSH import RuSH
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", type=pathlib.Path, required=True, help="path to read data from. should be a directory (of json or txt files).")
parser.add_argument("-o", "--out_dir", type=pathlib.Path, required=True, help="directory in which to save output")
parser.add_argument("-s", "--sentence_dir", type=pathlib.Path, required=True, help="directory in which to save sentences")
parser.add_argument("-u", "--rest_url", type=str, default="http://0.0.0.0:8000/temporal/process",
                    help="Primary REST server. Use GPU REST server for high throughput.")
parser.add_argument("--backup_rest_url", type=str, default="http://0.0.0.0:8000/temporal/process",
                    help=("Backup REST server. This server will be used if the primary server fails to process "
                          "(often due to VRAM restrictions). Use a CPU REST server for stability, "
                          "especially with large or long documents."))
parser.add_argument("--input_format", choices=["json", "pkl", "txt"], default="json")
parser.add_argument("--text_name", type=str, default="text", help="key to access the text in a dictionary format")
parser.add_argument("--output_format", choices=["json", "pkl"], default="json")
args = parser.parse_args()

rush = RuSH("conf/rush_rules.tsv")
#rush = RuSH("conf/rush_rules_cr.tsv")  # Use this if you want to use <cr> as a paragraph splitter.


def read_file(filename):
    if args.input_format == "txt":
        with open(os.path.join(args.data_dir, filename), "r") as f:
            text = f.read()
    elif args.input_format == "json":
        with open(os.path.join(args.data_dir, filename), "r") as f:
            text = json.load(f)[args.text_name]
    elif args.input_format == "pkl":
        with open(os.path.join(args.data_dir, filename), "rb") as f:
            text = pickle.load(f)[args.text_name]
    return text


def write_file(data, out_filename):
    if args.output_format == "json":
        with open(out_filename, "w") as f:
            json.dump(data, f)
    elif args.output_format == "pkl":
        with open(out_filename, "wb") as f:
            pickle.dump(data, f)


def preprocess(sents):
    sent_tokens = []
    for sent in sents:
        sent_text = text[sent.begin:sent.end]
        tokens = tokenize(sent_text)
        # NOTE: `extract_mimic_temporal.py` has some commented-out code here that's supposed to fix alignment issues
        if text[sent.end-1] == "\n":
            tokens.append("<cr>")
        if len(tokens) > 0:
            sent_tokens.append(tokens)
    return sent_tokens


if __name__ == "__main__":
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.sentence_dir, exist_ok=True)
    
    in_files = [f for f in os.listdir(args.data_dir) if f.endswith("." + args.input_format)]
    
    retry_attempts_cnt = 0
     
    for filename in tqdm(in_files):
        bare_filename = filename.split(".")[0]
        out_filename = bare_filename + "." + args.output_format
        if os.path.exists(os.path.join(args.out_dir, out_filename)):
            continue
        text = read_file(filename)
        if len(text) == 0:
            sys.stderr.write(f"Empty file: {filename}")
            continue

        sents = rush.segToSentenceSpans(text)
        if len(sents) == 0:
            sys.stderr.write(f"No sentences found in {filename}; skipping.")
            continue
        sent_tokens = preprocess(sents)
        if len(sent_tokens) == 0:
            sys.stderr.write(f"No sentences in {filename} were tokenizable; skipping.")
            continue
        
        # send off to rest
        r = requests.post(args.rest_url, json={"sent_tokens": sent_tokens, "metadata": f"FNAME={filename}"})
        if r.status_code == 500:
            sys.stderr.write(f"Failed from primary server.\nRe-try with alternative server :{args.backup_rest_url}\n")
            r = requests.post(args.backup_rest_url, json={"sent_tokens": sent_tokens, "metadata": f"FNAME={filename}"})
            retry_attempts_cnt += 1
            sys.stderr.write(f"Current retry_attempts_cnt: {retry_attempts_cnt}\n")
        if r.status_code != 200:
            raise Exception(f"Problem processing {filename}: status code {r.status_code}")
            
        out_json = r.json()

        events_docs, timexes_docs, rels_docs = [], [], []
        sent_text_list = []
        for sent_idx, sent in enumerate(sents):
            events, timexes, rels = [], [], []
            sent_text = text[sent.begin:sent.end+1]
            sent_text_list.append(sent_text)
            sent_events = out_json["events"][sent_idx]
            sent_timexes = out_json["timexes"][sent_idx]
            sent_rels = out_json["relations"][sent_idx]
            token_spans = align_tokens(sent_tokens[sent_idx], sent_text)
            event_ids, timex_ids = [], []
            for timex in sent_timexes:
                timex_start_offset = token_spans[timex["begin"]][0] + sent.begin
                timex_end_offset = token_spans[timex["end"]][1] + sent.begin
                timex_text =  text[timex_start_offset:timex_end_offset]
                timex_id = f"Timex_{bare_filename}_Sent-{sent_idx}_Ind-{len(timex_ids)}"
                timex_ids.append(timex_id)
                timexes.append({
                    "note_id": bare_filename,  # NOTE: this is "row_id" in `extract_mimic_temporal`
                    "entity_id":timex_id,
                    "sent_index": sent_idx,
                    "begin":timex["begin"],
                    "end":timex["end"],
                    "sent_begin":sent.begin,
                    "begin_char": token_spans[timex["begin"]][0],
                    "endi_char": token_spans[timex["end"]][1],
                    "begin_origin": timex_start_offset,
                    "end_origin": timex_end_offset,
                    "text": timex_text,
                    "timeClas": timex["timeClass"]})
            for event in sent_events:
                event_start_offset = token_spans[event["begin"]][0] + sent.begin
                event_end_offset = token_spans[event["end"]][1] + sent.begin
                event_text =  text[event_start_offset:event_end_offset]
                event_id = f"Event_{bare_filename}_Sent-{sent_idx}_Ind-{len(event_ids)}"
                event_ids.append(event_id)
                events.append({"note_id": bare_filename,  # NOTE: this is "row_id" in `extract_mimic_temporal`
                               "entity_id": event_id,
                               "sent_index": sent_idx,
                               "begin":event["begin"],
                               "end":event["end"],
                               "sent_begin":sent.begin,
                               "begin_char": token_spans[event["begin"]][0] ,
                               "end_char": token_spans[event["end"]][1],
                               "begin_origin": event_start_offset,
                               "end_origin": event_end_offset,
                               "text": event_text,
                               "dtr": event["dtr"]})
            
            for rel in sent_rels:
                if rel["arg1"] is None or rel["arg2"] is None:
                    continue
                arg1_type, arg1_idx = rel["arg1"].split("-")
                arg2_type, arg2_idx = rel["arg2"].split("-")

                if arg1_type == "EVENT":
                    arg1 = event_ids[int(arg1_idx)]
                elif arg1_type == "TIMEX":
                    arg1 = timex_ids[int(arg1_idx)]
                if arg1 == -1:
                    continue

                if arg2_type == "EVENT":
                    arg2 = event_ids[int(arg2_idx)]
                elif arg2_type == "TIMEX":
                    arg2 = timex_ids[int(arg2_idx)]
                if arg2 == -1:
                    continue
                rels.append({"row_id": bare_filename, 
                            "sent_index": sent_idx, 
                            "arg1": arg1, 
                            "arg2": arg2, 
                            "category": rel["category"]})
            
            timexes_docs.append(timexes)
            events_docs.append(events)
            rels_docs.append(rels)
                
        temporal_info = {
                         "timexes": timexes_docs,  # TODO call these "timex" and "event"?
                         "events": events_docs, 
                         "relations": rels_docs,
                         }
        write_file(
            temporal_info, 
            os.path.join(args.out_dir, out_filename)
        )
        write_file(
            {"sentences": sent_text_list},
            os.path.join(args.sentence_dir, out_filename)
        )
    sys.stderr.write(f"Current retry_attempts_cnt: {retry_attempts_cnt}\n")
