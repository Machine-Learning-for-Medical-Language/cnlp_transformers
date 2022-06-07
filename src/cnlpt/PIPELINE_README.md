## Instructions

Call via `python -m cnlpt.cnlp_pipeline`:
```
usage: cnlp_pipeline.py [-h] --models_dir MODELS_DIR --in_file IN_FILE
                        [--mode MODE] [--axis_task AXIS_TASK]

optional arguments:
  -h, --help            show this help message and exit
  --models_dir MODELS_DIR
                        Path where each entity model is stored in a folder
                        named after its corresponding cnlp_processor, models
                        with a 'tagging' output mode will be run first followed
                        by models with a 'classification' ouput mode over the
                        assembled data (default: None)
  --in_file IN_FILE     Path to file, with one raw sentenceper line in the
                        case of inference, and one <label> <annotated
                        sentence> per line in the case of evaluation (default:
                        None)
  --mode MODE           Use mode for full pipeline, inference, which outputs
                        annotated sentences and their relation, or eval, which
                        outputs metrics for a provided set of samples
                        (requires labels) (default: inf)
  --axis_task AXIS_TASK
                        key of the task in cnlp_processors which generates the
                        tag that will map to <a1> <mention> </a1> in pairwise
                        annotations (default: dphe_med)

```

The `cnlpt` models and their corresponding tokenizers used to generate the pipelines should be stored in the directory passed to the `models_dir` parameter,
where the folder each model is stored in has the name of the `cnlp_processors` task the model was trained on.
The `temp` folder used when training models with `cnlp_transformers` at the end of training will contain all the content needed for these task directories.

The code takes in data to run through the pipelines via the `in_file` parameter.  And the way it
will be processed will be determined by the `mode` parameter.
For inference mode, the file should contain raw unannotated sentences, any labels will be ignored.
For evaluation mode, the file should contain labels and sentences separated by tabs.
In evaluation the sentence may or may not be annotated, since all annotations are stripped from the sentence
before being passed to the pipeline.

Currently the code assumed that all models are either `cnlpt` entity taggers or sentential relation classifiers.  The taggers are run on a sentence and the results from all the taggers are converted into inputs which are appropriate for each relation classifier.  These relation labels are in between two types of entities,
and the vocabulary of relations is determined by the possible pairs of entities, and a central type of entity.
We call this type of entity the axis or anchor entity and tell the code to construct relations around this entity by providing the `axis_task` which is the `cnlp_processors` task which tags a sentence for the central type ofentity.

Added as well in `cnlp_processors.py` are some processors for DeepPhe/DeepPe-CR.  If you are at the BCH lab and would like help running some trained `cnlpt` models on the DeepPhe/DeepPhe-CR data please ask @Eli-Goldner.