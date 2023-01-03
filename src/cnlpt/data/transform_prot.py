import os
import csv
import sys
import pandas as pd
import spacy
from tqdm import tqdm
from pathlib import Path
from itertools import chain
from collections import defaultdict


TEST_DIR = "development"
TRAIN_DIR = "training"
nlp = spacy.load("en_core_sci_sm")


def to_stanza_style_dict(text):
    processed_doc = nlp(text)

    def sent_dict(spacy_sent):
        return [
            {
                "id": i + 1,
                "text": tok.text,
                "start_char": tok.idx,
                "end_char": tok.idx + len(tok) - 1,
            }
            for i, tok in enumerate(spacy_sent)
        ]

    return [sent_dict(sent) for sent in processed_doc.sents]


def file_type(filename):
    base_w_ext = os.path.basename(filename)
    base = base_w_ext.split(".")[0]
    return base.split("_")[-1].lower()


def order_files(file_dir):
    file_list = os.listdir(file_dir)
    # ChemProt spells it correctly and DrugProt doesn't
    abs_endings = {"abstracs", "abstracts"}
    abs_file = [*filter(lambda s: file_type(s) in abs_endings, file_list)][0]

    ents_file = [*filter(lambda s: file_type(s) == "entities", file_list)][0]

    rels_file = [*filter(lambda s: file_type(s) == "relations", file_list)][0]

    def full_path(fn):
        return os.path.join(file_dir, fn)

    return [*map(full_path, [abs_file, ents_file, rels_file])]


def build_abstract_dictionary(filename):
    identifier_to_processed_article = {}
    with open(filename) as fd:
        lines = sum(1 for line in fd)

    print("Tokenizing")
    with open(filename) as fd:
        rd = csv.reader(fd, delimiter="\t")  # quotechar='"')
        for row in tqdm(rd, total=lines):

            identifier, title, raw_article = row
            # character indices involve both title and abstract
            # processed_article = genia_pipeline("\t".join([title, raw_article]))
            identifier_to_processed_article[int(identifier)] = to_stanza_style_dict(
                "\t".join([title, raw_article])
            )  # processed_article.to_dict()

    return identifier_to_processed_article


def build_entity_dictionary(filename, mode="drugprot"):
    identifier_to_entity = defaultdict(lambda: {})
    with open(filename) as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            article_id, entity_id, raw_entity_type, begin, end, text = row
            # Since GENE-Y and GENE-N both become GENE in dev
            if mode == "drugprot":
                entity_type = raw_entity_type.split("-")[0]
            else:
                entity_type = raw_entity_type
            identifier_to_entity[int(article_id)][entity_id] = {
                "id": entity_id,
                "type": entity_type,
                "begin": int(begin),
                # according to DrugProt documentation
                # the 'ending character' is actually the first
                # character after the end of the token
                "end": int(end) - 1,
                "text": text,
            }
        print("Entities Loaded")
    return identifier_to_entity


def build_rel_dictionary(filename, mode="drugprot"):
    identifier_to_rel = defaultdict(lambda: {})
    with open(filename) as fd:
        rd = csv.reader(fd, delimiter="\t")
        for row in rd:
            if mode == "drugprot":
                article_id, rel_type, raw_arg1, raw_arg2 = row
                eval_q = "Y"
            else:
                article_id, rel_group, eval_q, rel_type, raw_arg1, raw_arg2 = row
            if eval_q.strip().upper() == "Y":
                entity_1 = raw_arg1.split(":")[-1]
                entity_2 = raw_arg2.split(":")[-1]
                # until we fix the issue
                # identifier_to_rel[int(article_id)][(entity_1, entity_2)] = rel_type
                out_rel = "_".join(rel_group.split(":"))
                identifier_to_rel[int(article_id)][(entity_1, entity_2)] = out_rel

    print("Relations Loaded")
    return identifier_to_rel


def build_entity_data(abstract_dict, entity_dict):
    def _coord(article_id, entity_to_info):
        stanza_sents = abstract_dict[article_id]
        return abs_ent_coord(entity_to_info, stanza_sents)

    return {
        article_id: _coord(article_id, ent2info)
        for article_id, ent2info in entity_dict.items()
    }


def get_sent_inds_table(stanza_sents):
    return {
        index: (sent[0]["start_char"], sent[-1]["end_char"])
        for index, sent in enumerate(stanza_sents)
    }


def get_tok_inds_table(stanza_sents):
    def tok_inds(sent):
        return sorted((stok["start_char"], stok["end_char"]) for stok in sent)

    return {index: tok_inds(sent) for index, sent in enumerate(stanza_sents)}


def abs_ent_coord(entity_to_info, stanza_sents):

    sent_inds_table = get_sent_inds_table(stanza_sents)
    tok_inds_table = get_tok_inds_table(stanza_sents)

    def sent_inside(info_dict, sent_inds):
        tok_begin = info_dict["begin"]
        tok_end = info_dict["end"]
        sent_begin, sent_end = sent_inds
        # We're not handling tokens which cross sentence boundaries
        return tok_begin in range(sent_begin, sent_end + 1) and tok_end in range(
            # want to include last index
            sent_begin,
            sent_end + 1,
        )

    def tok_inside(info_dict, stanza_tok_inds):
        tok_begin = info_dict["begin"]
        tok_end = info_dict["end"]
        stok_begin, stok_end = stanza_tok_inds
        # 'or' here in case the mention spans more than one discovered token
        # for spacy
        # if stok_begin == stok_end:
        #     return tok_begin == stok_begin
        return tok_begin in range(stok_begin, stok_end + 1) or tok_end in range(
            # here given the token index adjustment we risk
            # unwanted capture if we use the same end-inclusive policy as
            # with sentences
            stok_begin,
            stok_end  + 1,
        )

    def get_sent(info_dict):
        inds = [
            sent_index
            for sent_index, sent_inds in sent_inds_table.items()
            if sent_inside(info_dict, sent_inds)
        ]
        if len(inds) == 0:
            return -1
        return inds[0]

    def get_stanza_tokens(info_dict, sent_ind):
        stok_inds = tok_inds_table[sent_ind]

        def _inside(idx_stanza_tok_inds):
            _, stanza_tok_inds = idx_stanza_tok_inds
            return tok_inside(info_dict, stanza_tok_inds)

        raw_inds = [*filter(_inside, enumerate(stok_inds))]
        if len(raw_inds) == 0:
            print(f"Error! {info_dict}\n{stok_inds}\n{raw_inds}")
        return (
            (raw_inds[0][0], raw_inds[-1][0])
            if len(raw_inds) > 1
            else (raw_inds[0][0], raw_inds[0][0])
        )

    for ent_id, ent_info in entity_to_info.items():
        sent_ind = get_sent(ent_info)
        # Sometimes periods throw this off.
        # You really should use windowing and/or cTAKES tokenization
        # and/or efficient transformers if you're trying for the leaderboard
        first_token_idx, last_token_idx = (
            get_stanza_tokens(ent_info, sent_ind) if sent_ind > -1 else (-1, -1)
        )
        ent_info["stanza_location"] = sent_ind, first_token_idx, last_token_idx


def build_e2e_data_dict(entity_to_info, rel_ents_to_type):
    def clean(e2e_t):
        return str(e2e_t).replace("'", "")

    sent_idx_to_rels = defaultdict(lambda: [])
    for ent_pair, rel_type in rel_ents_to_type.items():
        entity_1, entity_2 = ent_pair
        sent_1_idx, ent_1_begin, ent_1_end = entity_to_info[entity_1]["stanza_location"]
        sent_2_idx, ent_2_begin, ent_2_end = entity_to_info[entity_2]["stanza_location"]
        # Not covering cross-sentence rels for now
        if sent_1_idx == sent_2_idx and sent_1_idx > -1:
            sent_idx_to_rels[sent_2_idx].append((ent_1_begin, ent_2_begin, rel_type))
    for sent_idx, labels in sent_idx_to_rels.items():
        sent_idx_to_rels[sent_idx] = " , ".join(
            map(clean, sorted(labels, key=lambda s: s[:2]))
        )
    return sent_idx_to_rels


def intervals_to_tags(intervals_dict, sent_len):
    final_tag_dict = {}
    for tag in ["CHEMICAL", "GENE"]:
        intervals = intervals_dict[tag]
        tags = ["O"] * sent_len
        for interval in intervals:
            begin, end, full_tag = interval
            for local_i, list_i in enumerate(range(begin, end + 1)):
                if local_i == 0:
                    tags[list_i] = "B-" + full_tag.upper()
                else:
                    tags[list_i] = "I-" + full_tag.upper()
        final_tag_dict[tag] = " ".join(tags)
    return final_tag_dict


def build_ner_data_dict(entity_to_info, stanza_sents, mode):
    sent_idx_to_tags = defaultdict(lambda: [])
    for entity, info_dict in entity_to_info.items():
        sent_idx, ent_begin, ent_end = info_dict["stanza_location"]
        entity_type = info_dict["type"]
        sent_idx_to_tags[sent_idx].append((ent_begin, ent_end, entity_type))
    print("Generating tags")
    for sent_idx, stanza_sent in enumerate(stanza_sents):
        tags = sent_idx_to_tags[sent_idx]
        sorted_tags = sorted(tags, key=lambda s: s[:2])
        final_tags = defaultdict(lambda: [])
        for i in range(0, len(sorted_tags)):

            curr_begin, curr_end, curr_type = sorted_tags[i]
            prev_ls = final_tags[curr_type]
            dict_type = curr_type.split("-")[0]
            (prev_begin, prev_end, prev_type) = (
                (-1, -1, curr_type) if len(prev_ls) == 0 else prev_ls[-1]
            )

            if (prev_begin >= curr_begin) and (curr_end >= prev_end):
                final_tags[dict_type] = [*prev_ls[:-1]]
                final_tags[dict_type].append((curr_begin, curr_end, curr_type))

            elif prev_end < curr_begin:

                final_tags[dict_type].append((curr_begin, curr_end, curr_type))

        sent_idx_to_tags[sent_idx] = intervals_to_tags(final_tags, len(stanza_sent))

    return sent_idx_to_tags


def coalesce(abs_dict, ent_dict, rel_dict, mode="drugprot"):

    columns = ["end_to_end", "chemical_ner", "gene_ner", "text"]

    build_entity_data(abs_dict, ent_dict)

    def to_samples(abs_id, stanza_sents):

        ent_info_dict = ent_dict.get(abs_id, {})

        rel_info_dict = rel_dict.get(abs_id, {})
        ner_data_dict = build_ner_data_dict(ent_info_dict, stanza_sents, mode)
        e2e_data_dict = build_e2e_data_dict(ent_info_dict, rel_info_dict)

        def to_list(stanza_ls, sent_index):
            tok_sent = " ".join(elem["text"] for elem in stanza_ls)
            raw_e2e_cell = e2e_data_dict[sent_index]
            e2e_cell = raw_e2e_cell if len(raw_e2e_cell) > 0 else "None"
            chemical_tags = ner_data_dict[sent_index]["CHEMICAL"]
            gene_tags = ner_data_dict[sent_index]["GENE"]
            return [e2e_cell, chemical_tags, gene_tags, tok_sent]

        return [
            to_list(stanza_sent, idx) for idx, stanza_sent in enumerate(stanza_sents)
        ]

    print("Converting Data Format")
    return pd.DataFrame(
        chain.from_iterable(
            to_samples(abs_id, stanza_sents)
            for abs_id, stanza_sents in abs_dict.items()
        ),
        columns=columns,
    )


def get_dataframe(tsv_dir, mode="drugprot"):
    abs_file, ents_file, rels_file = order_files(tsv_dir)
    abs_dict = build_abstract_dictionary(abs_file)
    entity_dict = build_entity_dictionary(ents_file, mode)
    rel_dict = build_rel_dictionary(rels_file, mode)
    return coalesce(abs_dict, entity_dict, rel_dict, mode)


def main():

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    mode = sys.argv[3]
    print("Generating Training Data")
    train_path = os.path.join(input_path, "training")
    print("Generating Development Data")
    dev_path = os.path.join(input_path, "development")

    train_df = get_dataframe(train_path, mode=mode.lower())
    dev_df = get_dataframe(dev_path, mode=mode.lower())

    # newline removal
    # train_df["text"] = train_df["text"].apply(remove_newline)
    # dev_df["text"] = dev_df["text"].apply(remove_newline)

    # quote removal
    train_df["text"] = train_df["text"].str.replace('"', "")
    dev_df["text"] = dev_df["text"].str.replace('"', "")

    # escape character removal
    train_df["text"] = train_df["text"].str.replace("//", "")
    dev_df["text"] = dev_df["text"].str.replace("//", "")
    
    output_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(
        output_path / "train.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        header=True,
        quoting=csv.QUOTE_NONE,
        escapechar=None,
    )
    dev_df.to_csv(
        output_path / "dev.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        header=True,
        quoting=csv.QUOTE_NONE,
        escapechar=None,
    )


if __name__ == "__main__":
    main()
