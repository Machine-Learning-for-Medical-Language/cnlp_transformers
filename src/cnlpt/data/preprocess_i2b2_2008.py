#!/usr/bin/env python

import sys
import xml.etree.ElementTree as ET
import json
from os.path import join
import random # for splitting train into train+dev

def add_notes_file(fn, ids, texts):
    tree = ET.parse(fn)
    root = tree.getroot()
    for doc in root.iter('doc'):
        id = doc.get('id')
        text = doc.find('text').text
        ids.add(id)
        texts[id] = text

def add_labels_file(fn, ids, labels):
    tree = ET.parse(fn)
    root = tree.getroot()
    for source in root.iter('diseases'):
        if source.get('source') == 'intuitive':            
            for disease in source.iter('disease'):
                for doc in disease.iter('doc'):
                    id = doc.get('id')
                    ids.add(id)
                    label = doc.get('judgment')
                    if not id in labels:
                        labels[id] = {}
                    labels[id][disease.get('name')] = label


def main(args):
    if len(args) < 2:
        sys.stderr.write('Required argument(s): <data directory> <json output directory>\n')
        sys.exit(-1)
    
    random_seed = 718
    random.seed(random_seed)

    training_ids = set()
    test_ids = set()
    training_texts = dict()
    test_texts = dict()
    training_labels = dict()
    test_labels = dict()

    add_notes_file(join(args[0], 'obesity_patient_records_training.xml'), training_ids, training_texts)   
    add_notes_file(join(args[0], 'obesity_patient_records_training2.xml'), training_ids, training_texts)
    print("After reading training text inputs, %d unique ids found" % (len(training_ids)))

    add_notes_file(join(args[0], 'obesity_patient_records_test.xml'), test_ids, test_texts)
    print("After reading test text inputs, %d unique ids found" % (len(test_ids)))
    
    add_labels_file(join(args[0], 'obesity_standoff_annotations_training.xml'), training_ids, training_labels)
    add_labels_file(join(args[0], 'obesity_standoff_annotations_training_addendum.xml'), training_ids, training_labels)
    add_labels_file(join(args[0], 'obesity_standoff_annotations_training_addendum2.xml'), training_ids, training_labels)
    add_labels_file(join(args[0], 'obesity_standoff_annotations_training_addendum3.xml'), training_ids, training_labels)
    print("After reading training labels, %d ids found, %d specifically we have labels for" % (len(training_ids), len(training_labels)))


    add_labels_file(join(args[0], 'obesity_standoff_annotations_test_intuitive.xml'), test_ids, test_labels)
    print("After reading test labels, %d total ids, %d specifically we have labels for" % (len(test_ids), len(test_labels)))

    missing_ids = set(training_texts.keys()) - set(training_labels.keys())
    print("Missing training labels for the following ids with text: ", missing_ids)
    missing_ids = set(training_labels.keys()) - set(training_texts.keys())
    print("Missing training texts for the following ids with labels: ", missing_ids)

    missing_ids = set(test_texts.keys()) - set(test_labels.keys())
    print("Missing test labels for the following ids: ", missing_ids)
    missing_ids = set(test_labels.keys()) - set(test_texts.keys())
    print("Missing test texts for the following ids with labels: ", missing_ids)

    training_data = {}
    dev_data = {}
    test_data = {}
    
    for training_id in set(training_texts.keys()).intersection(training_labels.keys()):
        data_point = {'text':training_texts[training_id], 'labels':training_labels[training_id]}
        if random.random() > 0.2:
            training_data[training_id] = data_point
        else:
            dev_data[training_id] = data_point
    
    for test_id in set(test_texts.keys()).intersection(test_labels.keys()):
        data_point = {'text':test_texts[test_id], 'labels':test_labels[test_id]}
        test_data[test_id] = data_point

    with open(join(args[1], 'training.json'), 'wt') as of:
        of.write(json.dumps(training_data))

    with open(join(args[1], 'dev.json'), 'wt') as of:
        of.write(json.dumps(dev_data))

    with open(join(args[1], 'test.json'), 'wt') as of:
        of.write(json.dumps(test_data))

    
if __name__ == '__main__':
    main(sys.argv[1:])
