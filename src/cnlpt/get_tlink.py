import os
import sys
import pdb
from tqdm import tqdm

import anafora
from anafora import AnaforaData, AnaforaRelation
import requests

# sentence and token splitters:
from spacy.lang.en import English
nlp = English()
tokenizer = nlp.tokenizer

xml_name_regex = r'Temporal_(Entity|Relation)\.dave\.(completed|inprogress)\.xml'

reverse_label = {
    "AFTER": "BEFORE",
    "BEFORE": "AFTER",
    "BEGINS-ON": "BEGINS-ON-1",
    "BEGINS-ON-1": "BEGINS-ON",
    "CONTAINS": "CONTAINS-1",
    "CONTAINS-1": "CONTAINS",
    "CONTAINS-SUBEVENT": "CONTAINS-SUBEVENT-1",
    "CONTAINS-SUBEVENT-1": "CONTAINS-SUBEVENT",
    "ENDS-ON": "ENDS-ON-1",
    "ENDS-ON-1": "ENDS-ON",
    "NOTED-ON": "NOTED-ON-1",
    "NOTED-ON-1": "NOTED-ON",
    "OVERLAP": "OVERLAP"
}

relabel = {
    "BEGINS-ON-1": "BEFORE",
    "CONTAINS-SUBEVENT": "CONTAINS",
    "ENDS-ON": "BEFORE",
    "NOTED-ON": "OVERLAP",
    "NOTED-ON-1": "OVERLAP"
}

should_switch = {
    "AFTER": "BEFORE",
    "BEGINS-ON": "BEFORE",
    "CONTAINS-1": "CONTAINS",
    "CONTAINS-SUBEVENT-1": "CONTAINS",
    "ENDS-ON-1": "BEFORE"
}


def order_pair(arg1, label, arg2):
    if arg1.spans > arg2.spans:
        print("THIS SHOULD NOT HAPPEN")
        temp = arg1
        arg1 = arg2
        arg2 = temp
        label = reverse_label[label]
    if label in should_switch:
        temp = arg1
        arg1 = arg2
        arg2 = temp
        label = should_switch[label]
    elif label in relabel:
        label = relabel[label]
    return arg1, label, arg2


def main(args):
    if len(args) < 3:
        sys.stderr.write("Required arguments: <input directory> <tlink REST host> <ouput directory> [text_dir]")
        sys.exit(-1)
    
    text_dir = args[3] if len(args) > 3 else args[0]
    
    url = "http://%s/temporal/process" % args[1]
    for sub_dir, text_name, xml_names in tqdm(anafora.walk(args[0], xml_name_regex)):
        rel_idx = 0
        # print("Processing filename: %s" % (text_name))
        if len(xml_names) > 1:
            sys.stderr.write('There were multiple valid xml files for file %s\n' % (text_name))
            filtered_names = []
            for xml_name in xml_names:
                if 'dave' in xml_name:
                    filtered_names.append(xml_name)
            if len(filtered_names) == 1:
                sys.stderr.write('Picking the file with "dave" in the title: %s\n' % (filtered_names[0]) )
                xml_names = filtered_names
            else:
                sys.exit(-1)
        xml_name = xml_names[0]
        # if os.path.exists(os.path.join(args[2], sub_dir, xml_name)):
        #     continue

        entity_data = AnaforaData.from_file(os.path.join(args[0], sub_dir, xml_name))
        with open(os.path.join(text_dir, sub_dir, text_name)) as f:
            full_text = f.read()

        to_remove = []
        for annot in entity_data.annotations:
            if annot.type not in ["EVENT", "TIMEX3"]:
                to_remove.append(annot)
        for annot in to_remove:
            entity_data.annotations.remove(annot)

        tokenized_text = tokenizer(full_text)
        span_to_token = {(token.idx, token.idx+len(token)): token.i for token in tokenized_text}
        for entity in entity_data.annotations:
            try:
                entity.tokens = (span_to_token[entity.spans[0]], span_to_token[entity.spans[0]] + 1)
            except KeyError:
                start_token, end_token = -1, -1
                for key in span_to_token:
                    if key[0] <= entity.spans[0][0] and key[1] >= entity.spans[0][0]:
                        start_token = span_to_token[key]
                    if key[0] <= entity.spans[0][1] and key[1] >= entity.spans[0][1]:
                        end_token = span_to_token[key] + 1
                if start_token > -1 and end_token > -1:
                    entity.tokens = (start_token, end_token)
                else:
                    pdb.set_trace()
        
        # sorting entities for efficiency
        entity_annots = [a for a in entity_data.annotations] # if a.type in ["EVENT", "TIMEX3"]]
        sorted_entity_annots = sorted(entity_annots, key=lambda x: x.spans[0])
        
        for i, ent0 in enumerate(sorted_entity_annots):
            for j, ent1 in enumerate(sorted_entity_annots[i+1:]):
                # if ent1.type not in ["EVENT", "TIMEX3"] or ent0 == ent1 or ent0.tokens[0] > ent1.tokens[0] or ent1.tokens[1] - ent0.tokens[0] > 100:
                #     continue
                if ent1.tokens[1] - ent0.tokens[0] > 20:
                    break  # we've gone too far; skip to the next ent0
                text_start = max(0, ent0.tokens[0] - 12)
                text_end = ent1.tokens[1] + 12
                sent = [token.text for token in tokenized_text[text_start:ent0.tokens[0]]]
                sent += ["<e1>"] + [token.text for token in tokenized_text[ent0.tokens[0]:ent0.tokens[1]]] + ["</e1>"]
                sent += [token.text for token in tokenized_text[ent0.tokens[1]:ent1.tokens[0]]]
                sent += ["<e2>"] + [token.text for token in tokenized_text[ent1.tokens[0]:ent1.tokens[1]]] + ["</e2>"]
                sent += [token.text for token in tokenized_text[ent1.tokens[1]:text_end]]

                _r = requests.post(url, json={'sent_tokens': [sent], 'metadata':text_name})
                if _r.status_code != 200:
                    sys.stderr.write('Error: tlink rest call was not successful\n')
                    sys.exit(-1)

                if _r.json()["relations"][0][0]["category"] != "None":
                    new_rel = AnaforaRelation(_annotations=entity_data.annotations)
                    new_rel.id = f"{rel_idx}@r@{text_name}"
                    new_rel.type = "TLINK"
                    ent0, rel_type, ent1 = order_pair(ent0, _r.json()["relations"][0][0]["category"], ent1)
                    new_rel.properties["Source"] = ent0
                    new_rel.properties["Type"] = rel_type
                    new_rel.properties["Target"] = ent1
                    entity_data.annotations.append(new_rel)
                    # annots_to_add.append(new_rel)
                    rel_idx += 1
        
        # for annot in annots_to_add:
        #     entity_data.annotations.append(annot)
        os.makedirs(os.path.join(args[2], sub_dir), exist_ok=True)
        entity_data.to_file(os.path.join(args[2], sub_dir, xml_name))


if __name__ == "__main__":
    main(sys.argv[1:])
