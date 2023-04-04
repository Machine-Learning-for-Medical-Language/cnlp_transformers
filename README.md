# Clinical NLP Transformers (cnlp_transformers)
Transformers for Clinical NLP

This library was created to add abstractions on top of the Huggingface Transformers library for many clinical NLP research use cases.
Primary use cases include 
 1) simplifying multiple tasks related to fine-tuning of transformers for building models for clinical NLP research, and 
 2) creating inference APIs that will allow downstream researchers easier access to clinical NLP outputs. 

This library is _not_ intended to serve as a place for clinical NLP applications to live. If you build something cool that uses transformer models that take advantage of our model definitions, the best practice is probably to rely on it as a library rather than treating it as your workspace. This library is also not intended as a deployment-ready tool for _scalable_ clinical NLP. There is a lot of interest in developing methods and tools that are smaller and can process millions of records, and this library can potentially be used for research along those line. But it will probably never be extremely optimized or shrink-wrapped for applications. However, there should be plenty of examples and useful code for people who are interested in that type of deployment.

## Install

**Note: due to some dependency issues, this package does not officially
support macOS on Apple Silicon. If you want to install it on Apple Silicon,
you are on your own; we unofficially recommend trying it with Python 3.10, or using
the docker CPU image**

**Note:** When installing the library's dependencies, `pip` will probably install 
PyTorch with CUDA 10.2 support by default. If you would like to run the 
library in CPU-only mode or with a newer version of CUDA, [install PyTorch 
to your desired specifications](https://pytorch.org/get-started/locally/) 
in your virtual environment first before installing `cnlp-transformers`.

### Static installation

If you are installing just to fine-tune or run the REST APIs,
you can install without cloning:

```sh
$ # Note: if needed, install PyTorch first (see above)
$ pip install cnlp-transformers
```

If you prefer, [prebuilt Docker images](https://hub.docker.com/repository/docker/smartonfhir/cnlp-transformers) are also available to run the REST APIs in a network.
An example [Docker Compose configuration](./docker/compose.yaml) is also available for reference.


### Editable installation

If you want to modify code (e.g., for developing new models), then install locally:

1. Clone this repository:
   ```sh
   # Either the HTTPS method...
   $ git clone https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers.git
   # ...or the SSH method
   $ git clone git@github.com:Machine-Learning-for-Medical-Language/cnlp_transformers.git
   ```

2. Enter the repo: `cd cnlp_transformers`

3. Install the development dependencies: 
   ```sh
   $ pip install -r dev-requirements.txt
   ```

4. See above for the note about PyTorch; if needed, manually install it now.

5. Install `cnlp-transformers` in editable mode: 
   ```sh
   $ pip install -e .
   ```

## Fine-tuning
The main entry point for fine-tuning is the ```cnlp_transformers/src/cnlpt/train_system.py``` script. Run with no arguments to show an extensive list of options that are allowed, inheriting from and extending the Huggingface training options.

### Workflow
To use the library for fine-tuning, you'll need to take the following steps:
1. Write your dataset to one of the following formats in a folder with train, dev, and test files:
  1. csv or tsv: The first row should have column names separated by comma or tab. The name ```text``` has special meaning as the input string. Likewise if there are columns named ```text_a``` and ```text_b``` it will be interpreted as two parts of a transformer input string separated by a <sep>-token equivalent. All other columns are treated as potential targets -- their names can be passed to the ```train_system.py``` script as ```--task_name``` arguments. For tagging targets, the field must consist of space-delimited labels, one per space-delimited token in the ```text``` field. For relation extraction targets, the field must be a ``` , ``` delimited list of relation tuples, where each relation tuple is (<offset 1>, <offset 2>,label), where offset 1 and 2 are token indices into the space-delimited tokens in the ```text``` field.
  2. json: The file format must be the following:
  ```
    { 'data': [
        { 'text': <text of instance>,
          'id': <instance id>
          '<sub-task 1 name>': <instance label>,
          '<sub-task 2 name>': <instance label>,
          ... // other labels
          }
        { }, // instance 2
        ...  // instances 3...N
    ],
      'metadata': {
        'version': '<optional dataset versioning>',
        'task': <overall task/dataset name>,
        'subtasks': [
          {
            'task_name': '<sub-task 1 name>',
            'output_mode': <sub-task output mode (e.g. tagging, relex, classification)>,
          }
          ...
          {
            'task_name': '<sub-task n name>'
            'output_mode': <sub-task output mode (e.g. tagging, relex, classification)>,
          }
        ]
      }
    }
``` 
Instance labels should be formatted the same way as in the csv/tsv example above, see specifically the formats for tagging and relations. The 'metadata' field can either be included in the train/dev/test files or as a separate metadata.json file.



2. Run train_system.py with a ```--task_name``` from your data files and the ```--data-dir``` argument from Step 1.


### Step-by-step finetuning examples
We provided the following step-by-step examples how to finetune in clinical NLP tasks:

#### 1. [Classification task](examples/uci_drug/): using [Drug Review Dataset (Drugs.com) Data Set](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29)

#### 2. [Sequence tagging task](examples/chemprot/): using [ChemProt website](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/)


### Fine-tuning options
Run ```python -m cnlpt.train_system -h``` to see all the available options. In addition to inherited Huggingface Transformers options, there are options to do the following:
* Run simple baselines (use ``--model cnn --tokenizer_name roberta-base`` -- since there is no HF model then you must specify the tokenizer explicitly)
* Use a different layer's CLS token for the classification (e.g., ```--layer 10```)
* Only update the weights of the classifier head and leave the encoder weights alone (```--freeze```)
* Classify based on a token embedding instead of the CLS embedding (```--token``` -- applies to the event/entity classification setting only, and requires the input to have xml-style tags (<e>, </e>) around the tokens representing the event/entity)
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
>>> process_url = 'http://hostname:8000/negation/process'  ## Replace hostname with your host name
```

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
Output: {'statuses': [-1, -1, 1, 1]}

The model correctly classifies both nausea and anosmia as negated.

### Temporal API (End-to-end temporal information extraction)
To demo the temporal API:
1. Install the `cnlp-transformers` package.
2. Run `cnlpt_temporal_rest [-p PORT]`
3. Open a python console and run the following commands to test:
#### Setup variables
```
>>> import requests
>>> from pprint import pprint
>>> process_url = 'http://hostname:8000/temporal/process_sentence'  ## Replace hostname with your host name
```

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

```uvicorn cnlpt.api.timex_rest:app --host 0.0.0.0``` or

```uvicorn cnlpt.api.event_rest:app --host 0.0.0.0```

then run the same process commands as above (including the same URL). You will get similar json output, but only one of the dictionary elements (timexes or events) will be populated.

