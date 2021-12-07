import os
import sys

import anafora
from anafora import AnaforaData, AnaforaEntity, AnaforaRelation
import requests
from api.temporal_rest import TokenizedSentenceDocument

# sentence and token splitters:
from PyRuSH import RuSH
from nltk.tokenize import wordpunct_tokenize as tokenize
from nltk.tokenize.util import align_tokens

xml_name_regex = r'Temporal-(Entity|Relation)\.gold\.completed\.xml'

def main(args):
    if len(args) < 3:
        sys.stderr.write("Required arguments: <input directory> <rest host> <output directory>\n")
        sys.exit(-1)

    hostname = args[1]

    # initialize rest server
    init_url = 'http://%s:8001/temporal/initialize' % hostname
    process_url = 'http://%s:8001/temporal/process' % hostname

    # sentence segmenter
    rush = RuSH('conf/rush_rules.tsv')
    # tokenizer
    # tokenizer = TreebankWordTokenizer()

    r = requests.post(init_url)
    if r.status_code != 200:
        sys.stderr.write('Error: rest init call was not successful\n')
        sys.exit(-1)
    
    combine_sentences = True 
    token_threshold = 150

    for sub_dir, text_name, xml_names in anafora.walk(args[0], xml_name_regex):
        print("Processing filename: %s" % (text_name))
        if len(xml_names) > 1:
            sys.stderr.write('There were multiple valid xml files for file %s\n' % (text_name))
            filtered_names = []
            for xml_name in xml_names:
                if 'Relation' in xml_name:
                    filtered_names.append(xml_name)
            if len(filtered_names) == 1:
                sys.stderr.write('Picking the file with "Relation" in the title: %s\n' % (filtered_names[0]) )
                xml_names = filtered_names
            else:
                sys.exit(-1)
        xml_name = xml_names[0]

        with open(os.path.join(args[0], sub_dir, text_name)) as f:
            text = f.read()
        
        sentences = rush.segToSentenceSpans(text)
        sent_tokens = []
        merged_sentences = []

        if combine_sentences:
            for sentence_ind,sentence in enumerate(sentences):
                sent_txt = text[sentence.begin:sentence.end]
                tokens = tokenize(sent_txt)
                # print("Sentence number %d has %d tokens" % (sentence_ind, len(tokens)))

                if len(sent_tokens) > 0 and (len(sent_tokens[-1]) + len(tokens)) < token_threshold:
                    sent_tokens[-1].extend(tokens)
                    merged_sentences[-1].end = sentence.end
                else:
                    sent_tokens.append(tokens)
                    merged_sentences.append(sentence)
            sentences = merged_sentences
        else:
            for sentence in sentences:
                sent_txt = text[sentence.begin:sentence.end]
                sent_tokens.append(tokenize(sent_txt))
        
        r = requests.post(process_url, json={'sent_tokens': sent_tokens, 'metadata':text_name})
        if r.status_code != 200:
            sys.stderr.write('Error: rest call was not successful\n')
            sys.exit(-1)

        json = r.json()
        anafora_data = AnaforaData()
        cur_id = 0
        rel_id = 0

        for sent_ind,sentence in enumerate(sentences):
            sent_txt = text[sentence.begin:sentence.end]
            sent_events = json['events'][sent_ind]
            sent_timexes = json['timexes'][sent_ind]
            sent_rels = json['relations'][sent_ind]
            event_ids = []
            timex_ids = []

            try:
                token_spans = align_tokens(sent_tokens[sent_ind], sent_txt)
            except Exception as e:
                sys.stderr.write('In document %s, error \n%s\n processing sentence:\n*****\n%s\n******\n' % (text_name, str(e), sent_txt))
                sys.exit(-1)

            for event in sent_events:
                begin_token_ind = event['begin']
                end_token_ind = event['end']
                dtr = event['dtr']
                event_start_offset = token_spans[begin_token_ind][0] + sentence.begin
                event_end_offset = token_spans[end_token_ind][1] + sentence.begin
                event_text = text[event_start_offset:event_end_offset]                    

                annot = AnaforaEntity()
                annot.id = str(cur_id)+"@e@" + text_name

                if event_text.endswith('_date'):
                    annot.properties['datesectiontime'] = 'True'
                    event_ids.append(-1)
                else:                    
                    event_ids.append(annot.id)
                    annot.spans = ( (event_start_offset, event_end_offset), )
                    annot.type = "EVENT"
                    annot.properties['DocTimeRel'] = dtr                    
                    anafora_data.annotations.append(annot)

                cur_id += 1

                #print("Found event %s" % (event_text))

            for timex in sent_timexes:
                begin_token_ind = timex['begin']
                end_token_ind = timex['end']
                time_class = timex['timeClass']
                timex_start_offset = token_spans[begin_token_ind][0] + sentence.begin
                timex_end_offset = token_spans[end_token_ind][1] + sentence.begin
                timex_text = text[timex_start_offset:timex_end_offset]

                # create anafora entry
                annot = AnaforaEntity()
                annot.id = str(cur_id)+"@e@" + text_name
                timex_ids.append(annot.id)
                cur_id += 1
                annot.spans = ( (timex_start_offset, timex_end_offset), )
                annot.type = "TIMEX3"
                annot.properties['Class'] = time_class
                anafora_data.annotations.append(annot)

                #print("Found timex %s" % (timex_text))

            if not 'path' in text_name.lower():
                # no relations in pathology notes, so if we find any they are false positives.
                for rel in sent_rels:
                    arg1_type, arg1_ind = rel['arg1'].split('-')
                    arg2_type, arg2_ind = rel['arg2'].split('-')
                    if arg1_type == 'EVENT':
                        arg1 = event_ids[int(arg1_ind)]
                    elif arg1_type == 'TIMEX':
                        arg1 = timex_ids[int(arg1_ind)]

                    if arg1 == -1:
                        continue

                    if arg2_type == 'EVENT':
                        arg2 = event_ids[int(arg2_ind)]
                    elif arg2_type == 'TIMEX':
                        arg2 = timex_ids[int(arg2_ind)]

                    if arg2 == -1:
                        continue

                    reln = AnaforaRelation()
                    reln.id = str(rel_id)+'@r@'+text_name
                    rel_id += 1
                    reln.type = 'TLINK'
                    reln.properties['Type'] = rel['category']
                    reln.properties['Source'] = arg1
                    reln.properties['Target'] = arg2

                    anafora_data.annotations.append(reln)


        #break
        anafora_data.indent()
        os.makedirs(os.path.join(args[2], sub_dir), exist_ok=True)
        anafora_data.to_file(os.path.join(args[2], sub_dir, xml_name))
        

if __name__ == '__main__':
    main(sys.argv[1:])
