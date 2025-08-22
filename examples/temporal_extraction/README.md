This code provides a wrapper which sends texts through our temporal REST API.

In one terminal, run the following to set up the REST entrypoint:

    cd cnlp_transformers/src/cnlpt/api
    python temporal_rest.py

In another, run the script in this folder, `extract_temporal.py`. The arguments are:
* Required:
    * `-d`, `--data_dir`: directory in which to find the texts. There should be one text per file.
    * `-o`, `--out_dir`: directory in which to save outputs
* Optional:
    * `-u`, `--rest_url`: default `"http://0.0.0.0:8000/temporal/process"`
    * `--input_format`: default `"json"`
    * `--text_name`: default `"text"`
    * `--output_format`: default `"json"`

Note: if this is being run on a cluster like E2, these programs must be run on the same node.

The post-processing code here was written for our temporal REST API, but can be modified to suit the outputs of our other REST APIs, e.g. negation detection.
