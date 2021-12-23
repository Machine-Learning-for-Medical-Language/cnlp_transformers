# Clinical NLP Transformers (cnlp_transformers)
Transformers for Clinical NLP

This library was created to add abstractions on top of the Huggingface Transformers library for many clinical NLP research use cases.
Primary use cases include 1) simplifying multiple tasks related to fine-tuning of transformers for building models for clinical NLP, and 2) creating inference APIs that will allow downstream researchers easier access to clinical NLP outputs.
## Install
If you are installing just to run the REST APIs, you can just install without cloning with:
```pip install cnlp_transformers```

If you want to modify code (for fine-tuning), then install locally with:
1. ```git clone https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers.git```
2. ```cd cnlp_transformers```
3. ```pip install -e .```

## Fine-tuning
The main entry point for fine-tuning is the ```train_system.py``` script. Run with no arguments to show an extensive list of options that are allowed, inheriting from and extending the Huggingface training options.

### Workflow
To use the library for fine-tuning, you'll need to take the following steps:
1. Write your dataset to a convenient format in a folder with train, dev, and test files.
2. Create a new entry for your dataset in ```src/cnlpt/cnlp_processors.py``` in the following places:
    1. Create a unique ```task_name``` for your task.
    2. ```cnlp_output_modes``` -- Add a mapping from a task name to a task type. Currently supported task types are sentence classification, tagging, relation extraction, and multi-task sentence classification.
    3. Processor class -- Create a subclass of DataProcessor for your data source. There are multiple examples to base off of, including intermediate abstractions like LabeledSentenceProcessor, RelationProcessor, SequenceProcessor, that simplify the implementation.
    4. ```cnlp_processors``` -- Add a mapping from your task name to the "processor" class you created in the last step.
    5. (Optional) -- Modify cnlp_processors.cnlp_compute_metrics() to add you task. If your task is classification a reasonable default will be used so this step would be optional.
3. Run train_system.py with the ```--task_name``` argument from Step 2.1 and the ```--data-dir``` argument from Step 1.

### End-to-end example
1. Download data from [Drug Review Dataset (Drugs.com) Data Set](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29) and extract. Pay attention to their terms:
   1. only use the data for research purposes
   2. don't use the data for any commerical purposes
   3. don't distribute the data to anyone else
   4. cite us
2. Run ```python -m cnlpt.data.transform_uci_drug <input dir> <output dir>``` to preprocess the data from the extract directory into a new directory. This will create {train,dev,test}.tsv in the output directory specified, where the sentiment labels have been collapsed into 3 categories.
3. Fine-tune with something like: 
```python -m cnlpt.train_system --task_name ucidrug --data_dir ~/mnt/r/DeepLearning/mmtl/drug-sentiment/ --model_name_or_path roberta-base --do_train --cache cache/ --output_dir temp/ --overwrite_output_dir --evals_per_epoch 5 --do_eval --num_train_epochs 1 --learning_rate 1e-5```

On our hardware, that command results in the following eval performance:
```ucidrug = {'acc': 0.8127712337259765, 'f1': [0.8030439829743325, 0.49202644885258656, 0.9018332042344437], 'acc_and_f1': [0.8079076083501545, 0.6523988412892815, 0.8573022189802101], 'recall': [0.788500506585613, 0.524896265560166, 0.8935734752353663], 'precision': [0.8181340341655716, 0.4630307467057101, 0.9102470551443761]}```

For a demo of how to run the system in colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IVT53DBwFxLKftpIn5iKtF0g4xb9yuxm?usp=sharing)

### Fine-tuning options
Run ```python -m cnlpt.train_system -h``` to see all the available options. In addition to inherited Huggingface Transformers options, there are options to do the following:
* Run simple baselines (use ``--model_name_or_path cnn --tokenizer_name roberta-base`` -- since there is no HF model then you must specify the tokenizer explicitly)
* Use a different layer's CLS token for the classification (e.g., ```--layer 10```)
* Only update the weights of the classifier head and leave the encoder weights alone (```--freeze```)
* Classify based on a token embedding instead of the CLS embedding (```--token``` -- requires the input to have xml-style tags (<e>, </e>) around the tokens of interest)
* Use class-weighted loss function (```--class_weights```)

## Running REST APIs
There are existing REST APIs in the ```src/cnlpt/api``` folder for a few important clinical NLP tasks: 
1. Negation detection
2. Time expression tagging (spans + time classes)
3. Event detection (spans + document creation time relation)
4. End-to-end temporal relation extraction (event spans+DTR+timex spans+time classes+narrative container [CONTAINS] relation extraction)

### Negation API
To demo the negation API:
1. Install the `cnlp-transformers` package.
2. Run `cnlpt_negation_rest [-p PORT]`.
3. Open a python console and run the following commands:

#### Setup variables
```
>>> import requests
>>> init_url = 'http://hostname:8000/negation/initialize'  ## Replace hostname with your host name
>>> process_url = 'http://hostname:8000/negation/process'  ## Replace hostname with your host name
```

#### Load the model
```
>>> r = requests.post(init_url)
>>> r.status_code
```
should return ```200```

#### Prepare the document
```
>>> sent = 'The patient has a sore knee and headache but denies nausea and has no anosmia.'
>>> ents = [[18, 27], [32, 40], [52, 58], [70, 77]]
>>> doc = {'doc_text':sent, 'entities':ents}
```

#### Process the document
```
>>> r = requests.post(process_url, json=doc)
>>> r.json()
```
Output: {'statuses': [-1, -1, -1, 1]}

The model thinks only one of the entities is negated (anosmia). It missed "nausea" for some reason.

### Temporal API (End-to-end temporal information extraction)
To demo the temporal API:
1. Install the `cnlp-transformers` package.
2. Run `cnlpt_temporal_rest [-p PORT]`
3. Open a python console and run the following commands to test:
#### Setup variables
```
>>> import requests
>>> from pprint import pprint
>>> init_url = 'http://hostname:8000/temporal/initialize'  ## Replace hostname with your host name
>>> process_url = 'http://hostname:8000/temporal/process_sentence'  ## Replace hostname with your host name
```
#### Load the model
```
>>> r = requests.post(init_url)
>>> r.status_code
```
should return 200

#### Prepare and process the document
```
>>> sent = 'The patient was diagnosed with adenocarcinoma March 3, 2010 and will be returning for chemotherapy next week.'
>>> r = requests.post(process_url, json={'sentence':sent})
>>> pprint(r.json())
```
should return:
```
{'events': [[{'begin': 3, 'dtr': 'BEFORE', 'end': 3},
             {'begin': 5, 'dtr': 'BEFORE', 'end': 5},
             {'begin': 13, 'dtr': 'AFTER', 'end': 13},
             {'begin': 15, 'dtr': 'AFTER', 'end': 15}]],
 'relations': [[{'arg1': 'TIMEX-0', 'arg2': 'EVENT-0', 'category': 'CONTAINS'},
                {'arg1': 'EVENT-2', 'arg2': 'EVENT-3', 'category': 'CONTAINS'},
                {'arg1': 'TIMEX-1', 'arg2': 'EVENT-2', 'category': 'CONTAINS'},
                {'arg1': 'TIMEX-1',
                 'arg2': 'EVENT-3',
                 'category': 'CONTAINS'}]],
 'timexes': [[{'begin': 6, 'end': 9, 'timeClass': 'DATE'},
              {'begin': 16, 'end': 17, 'timeClass': 'DATE'}]]}
```
This output indicates the token spans of events and timexes, and relations between events and timexes, where the suffixes are indices into the respective arrays (e.g., TIMEX-0 in a relation refers to the 0th time expression found, which begins at token 6 and ends at token 9 -- ["March 3, 2010"])

To run only the time expression or event taggers, change the run command to:

```uvicorn api.timex_rest:app --host 0.0.0.0``` or

```uvicorn api.event_rest:app --host 0.0.0.0```

then run the same init and process commands as above. You will get similar json output, but only one of the dictionary elements (timexes or events) will be populated.

