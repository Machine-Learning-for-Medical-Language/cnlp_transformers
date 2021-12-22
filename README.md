# Clinical NLP Transformers (cnlp_transformers)
Transformers for Clinical NLP

This library was created to add abstractions on top of the Huggingface Transformers library for many clinical NLP research use cases.
Primary use cases include 1) simplifying multiple tasks related to fine-tuning of transformers for building models for clinical NLP, and 2) creating inference APIs that will allow downstream researchers easier access to clinical NLP outputs.

## Fine-tuning
The main entry point for fine-tuning is the ```train_system.py``` script. Run with no arguments to show an extensive list of options that are allowed, inheriting from and extending the Huggingface training options.

### Workflow
To use the library for fine-tuning, you'll need to take the following steps:
1. Write your dataset to a convenient format in a folder with train, dev, and test files.
2. Create a new entry for your dataset in ```cnlp_processors.py``` in the following places:
    1. Create a unique ```task_name``` for your task.
    2. ```cnlp_output_modes``` -- Add a mapping from a task name to a task type. Currently supported task types are sentence classification, tagging, relation extraction, and multi-task sentence classification.
    3. Processor class -- Create a subclass of DataProcessor for your data source. There are multiple examples to base off of, including intermediate abstractions like LabeledSentenceProcessor, RelationProcessor, SequenceProcessor, that simplify the implementation.
    4. ```cnlp_processors``` -- Add a mapping from your task name to the "processor" class you created in the last step.
3. Run train_system.py with the ```--task_name``` argument from Step 2.1 and the ```--data-dir``` argument from Step 1.

### End-to-end example
 
## Running existing APIs
There are existing APIs in the ```api``` folder for a few important clinical NLP tasks: 
1. Negation detection
2. Time expression tagging (spans + time classes)
3. Event detection (spans + document creation time relation)
4. End-to-end temporal relation extraction (event spans+DTR+timex spans+time classes+narrative container [CONTAINS] relation extraction)

### Negation API
To demo the negation API:
1. Go to the ```api``` sub-directory and run ```./start_negation_rest.sh```
2. Open a python console and run the following commands:
```
## Setup variables
>>> import requests
>>> init_url = 'http://localhost:8000/negation/initialize'
>>> process_url = 'http://nlp-gpu:8000/negation/process'

## Load the model
>>> r = requests.post(init_url)
>>> r.status_code
# should return 200

## Prepare the document
>>> sent = 'The patient has a sore knee and headache but denies nausea and has no anosmia.'
>>> ents = [[18, 27], [32, 40], [52, 58], [70, 77]]
>>> doc = {'doc_text':sent, 'entities':ents}

## Process the document
>>> r = requests.post(process_url, json=doc)
>>> r.json()
# Output: {'statuses': [-1, -1, -1, 1]}
# The model thinks only one of the entities is negated (anosmia). It missed "nausea" for some reason.
