# Clinical NLP Transformers (cnlp_transformers)

Transformers for Clinical NLP

This library was created to add abstractions on top of the Huggingface Transformers library for many clinical NLP research use cases.
Primary use cases include

 1) simplifying multiple tasks related to fine-tuning of transformers for building models for clinical NLP research, and
 2) creating inference APIs that will allow downstream researchers easier access to clinical NLP outputs.

This library is _not_ intended to serve as a place for clinical NLP applications to live. If you build something cool that uses transformer models that take advantage of our model definitions, the best practice is probably to rely on it as a library rather than treating it as your workspace. This library is also not intended as a deployment-ready tool for _scalable_ clinical NLP. There is a lot of interest in developing methods and tools that are smaller and can process millions of records, and this library can potentially be used for research along those line. But it will probably never be extremely optimized or shrink-wrapped for applications. However, there should be plenty of examples and useful code for people who are interested in that type of deployment.

## Install

> [!IMPORTANT]
> When installing the library's dependencies, PyTorch will probably be installed
> with CUDA 12.6 support by default on linux, and without CUDA support on other platforms.
> If you would like to run the library in CPU-only mode or with a specific version of CUDA,
> [install PyTorch to your desired specifications](https://pytorch.org/get-started/locally/)
> in your virtual environment first before installing `cnlp-transformers`.
> [See here](https://docs.astral.sh/uv/guides/integration/pytorch/#the-uv-pip-interface) if
> using uv.

### Static installation

If you are installing just to fine-tune or run the REST APIs,
you can install without cloning using [uv](https://docs.astral.sh/uv/):

```sh
uv pip install cnlp-transformers
```

Or with pip:

```sh
pip install cnlp-transformers
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

3. Follow the instructions [here](CONTRIBUTING.md#set-up-your-python-environment) to set up your Python environment.

## Fine-tuning

The main entry point for fine-tuning is the ```cnlp_transformers/src/cnlpt/train_system.py``` script. Run with no arguments to show an extensive list of options that are allowed, inheriting from and extending the Huggingface training options.

### Workflow

To use the library for fine-tuning, you'll need to take the following steps:

1. Write your dataset to one of the following formats in a folder with train, dev, and test files:
   1. csv or tsv: The first row should have column names separated by comma or tab. The name ```text``` has special meaning as the input string. Likewise if there are columns named ```text_a``` and ```text_b``` it will be interpreted as two parts of a transformer input string separated by a `<sep>`-token equivalent. All other columns are treated as potential targets -- their names can be passed to the ```train_system.py``` script as ```--task_name``` arguments. For tagging targets, the field must consist of space-delimited labels, one per space-delimited token in the ```text``` field. For relation extraction targets, the field must be a ``` , ``` delimited list of relation tuples, where each relation tuple is (<offset 1>, <offset 2>,label), where offset 1 and 2 are token indices into the space-delimited tokens in the ```text``` field.
   2. json: The file format must be the following:

      ```jsonc
      { 
        "data": [
          { 
            "text": "<text of instance>",
            "id": "<instance id>",
            "<sub-task 1 name>": "<instance label>",
            "<sub-task 2 name>": "<instance label>",
            // ... other labels
          },
          // ...
          {
            // instance N
          },
        ],
        "metadata": {
          "version": "<optional dataset versioning>",
          "task": "<overall task/dataset name>",
          "subtasks": [
            {
              "task_name": "<sub-task 1 name>",
              "output_mode": "<sub-task output mode (e.g. tagging, relex, classification)>",
            },
            // ...
            {
              "task_name": "<sub-task n name>",
              "output_mode": "<sub-task output mode (e.g. tagging, relex, classification)>",
            }
          ]
        }
      }
      ```

      Instance labels should be formatted the same way as in the csv/tsv example above, see specifically the formats for tagging and relations. The 'metadata' field can either be included in the train/dev/test files or as a separate metadata.json file.

2. Run train_system.py with a ```--task_name``` from your data files and the ```--data-dir``` argument from Step 1. If no ```--task_name``` is provided, all tasks will be trained.

### Step-by-step finetuning examples

We provided the following step-by-step examples how to finetune in clinical NLP tasks:

#### 1. [Classification task](examples/uci_drug/): using [Drug Reviews (Druglib.com) Data Set](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com)

#### 2. [Sequence tagging task](examples/chemprot/): using [ChemProt website](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/)

### Fine-tuning options

Run `cnlpt train -h` to see all the available options. In addition to inherited Huggingface Transformers options, there are options to do the following:

* Select different models: `--model hier` uses a hierarchical transformer layer on top of a specified encoder model. We recommend using a very small encoder: `--encoder microsoft/xtremedistil-l6-h256-uncased` so that the full model fits into memory.
* Run simple baselines (use ``--model cnn|lstm --tokenizer_name roberta-base`` -- since there is no HF model then you must specify the tokenizer explicitly)
* Use a different layer's CLS token for the classification (e.g., `--layer 10`)
* Probabilistically freeze weights of the encoder (leaving classifier weights all unfrozen) (`--freeze` alone freezes all encoder weights, `--freeze <float>` when given a parameter between 0 and 1, freezes that percentage of encoder weights)
* Classify based on a token embedding instead of the CLS embedding (`--token` -- applies to the event/entity classification setting only, and requires the input to have xml-style tags (`<e>`, `</e>`) around the tokens representing the event/entity)
* Use class-weighted loss function (`--class_weights`)

## Running REST APIs

There are existing REST APIs in the `src/cnlpt/api` folder for a few important clinical NLP tasks:

1. Negation detection
2. Time expression tagging (spans + time classes)
3. Event detection (spans + document creation time relation)
4. End-to-end temporal relation extraction (event spans+DTR+timex spans+time classes+narrative container [CONTAINS] relation extraction)

### Negation API

To demo the negation API:

1. Install the `cnlp-transformers` package.
2. Run `cnlpt rest --model-type negation [-p PORT]`.
3. Open a python console and run the following commands:

#### Setup variables for negation

```ipython
>>> import requests
>>> process_url = 'http://hostname:8000/negation/process'  ## Replace hostname with your host name
```

#### Prepare the document

```ipython
>>> sent = 'The patient has a sore knee and headache but denies nausea and has no anosmia.'
>>> ents = [[18, 27], [32, 40], [52, 58], [70, 77]]
>>> doc = {'doc_text':sent, 'entities':ents}
```

#### Process the document

```ipython
>>> r = requests.post(process_url, json=doc)
>>> r.json()
```

Output: `{'statuses': [-1, -1, 1, 1]}`

The model correctly classifies both nausea and anosmia as negated.

### Temporal API (End-to-end temporal information extraction)

To demo the temporal API:

1. Install the `cnlp-transformers` package.
2. Run `cnlpt rest --model-type temporal [-p PORT]`
3. Open a python console and run the following commands to test:

#### Setup variables for temporal

```ipython
>>> import requests
>>> from pprint import pprint
>>> process_url = 'http://hostname:8000/temporal/process_sentence'  ## Replace hostname with your host name
```

#### Prepare and process the document

```ipython
>>> sent = 'The patient was diagnosed with adenocarcinoma March 3, 2010 and will be returning for chemotherapy next week.'
>>> r = requests.post(process_url, json={'sentence':sent})
>>> pprint(r.json())
```

should return:

```json
{
  "events": [
    [
      {"begin": 3, "dtr": "BEFORE", "end": 3},
      {"begin": 5, "dtr": "BEFORE", "end": 5},
      {"begin": 13, "dtr": "AFTER", "end": 13},
      {"begin": 15, "dtr": "AFTER", "end": 15}
    ]
  ],
  "relations": [
    [
      {"arg1": "TIMEX-0", "arg2": "EVENT-0", "category": "CONTAINS"},
      {"arg1": "EVENT-2", "arg2": "EVENT-3", "category": "CONTAINS"},
      {"arg1": "TIMEX-1", "arg2": "EVENT-2", "category": "CONTAINS"},
      {"arg1": "TIMEX-1", "arg2": "EVENT-3", "category": "CONTAINS"}
    ]
  ],
  "timexes": [
    [
      {"begin": 6, "end": 9, "timeClass": "DATE"},
      {"begin": 16, "end": 17, "timeClass": "DATE"}
    ]
  ]
}
```

This output indicates the token spans of events and timexes, and relations between events and timexes, where the suffixes are indices into the respective arrays (e.g., TIMEX-0 in a relation refers to the 0th time expression found, which begins at token 6 and ends at token 9 -- ["March 3, 2010"])

## Citing cnlp_transformers

Please use the following bibtex to cite cnlp_transformers if you use it in a publication:

```latex
@misc{cnlp_transformers,
  author       = {CNLPT},
  title        = {Clinical {NLP} {Transformers} (cnlp\_transformers)},
  year         = {2021},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Machine-Learning-for-Medical-Language/cnlp_transformers}},
}
```

## Publications using cnlp_transformers

Please send us any citations that used this library!

1. Chen S, Guevara M, Ramirez N, Murray A, Warner JL, Aerts HJWL, et al. Natural Language Processing to Automatically Extract the Presence and Severity of Esophagitis in Notes of Patients Undergoing Radiotherapy. JCO Clin Cancer Inform. 2023 Jul;(7):e2300048.
2. Li Y, Miller T, Bethard S, Savova G. Identifying Task Groupings for Multi-Task Learning Using Pointwise V-Usable Information [Internet]. arXiv.org. 2024 [cited 2025 May 22]. Available from: <https://arxiv.org/abs/2410.12774v1>
3. Wang L, Li Y, Miller T, Bethard S, Savova G. Two-Stage Fine-Tuning for Improved Bias and Variance for Large Pretrained Language Models. In: Rogers A, Boyd-Graber J, Okazaki N, editors. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) [Internet]. Toronto, Canada: Association for Computational Linguistics; 2023 [cited 2025 May 22]. p. 15746–61. Available from: <https://aclanthology.org/2023.acl-long.877/>
4. Miller T, Bethard S, Dligach D, Savova G. End-to-end clinical temporal information extraction with multi-head attention. Proc Conf Assoc Comput Linguist Meet. 2023 Jul;2023:313–9.
5. Yoon W, Ren B, Thomas S, Kim C, Savova G, Hall MH, et al. Aspect-Oriented Summarization for Psychiatric Short-Term Readmission Prediction [Internet]. arXiv; 2025 [cited 2025 May 22]. Available from: <http://arxiv.org/abs/2502.10388>
6. Wang L, Zipursky AR, Geva A, McMurry AJ, Mandl KD, Miller TA. A computable case definition for patients with SARS-CoV2 testing that occurred outside the hospital. JAMIA Open. 2023 Oct 1;6(3):ooad047.
7. Bitterman DS, Goldner E, Finan S, Harris D, Durbin EB, Hochheiser H, et al. An End-to-End Natural Language Processing System for Automatically Extracting Radiation Therapy Events From Clinical Texts. Int J Radiat Oncol Biol Phys. 2023 Sep 1;117(1):262–73.
8. McMurry AJ, Gottlieb DI, Miller TA, Jones JR, Atreja A, Crago J, et al. Cumulus: A federated EHR-based learning system powered by FHIR and AI. medRxiv. 2024 Feb 6;2024.02.02.24301940.
9. LCD benchmark: long clinical document benchmark on mortality prediction for language models | Journal of the American Medical Informatics Association | Oxford Academic [Internet]. [cited 2025 Jan 23]. Available from: <https://academic.oup.com/jamia/article-abstract/32/2/285/7909835?redirectedFrom=fulltext>
