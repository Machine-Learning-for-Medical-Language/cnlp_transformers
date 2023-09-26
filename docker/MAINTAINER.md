# Creating cNLPT Docker Images

## Setup
- First authenticate with Docker with an account that has access to the
  [smartonfhir](https://hub.docker.com/u/smartonfhir/) organization.
- Make sure you have a local docker buildx setup that supports both amd64 and arm64.
  - Run `docker buildx ls` to see your current setup.
  - If you don't have a multi-platform instance already, you can create a new default one with:
    `docker buildx create --driver docker-container --name cross-builder --platform linux/amd64,linux/arm64 --use`

## Building
Use the `./build.py` script to build new images.
Pass `--help` to see all your options.

### Local Testing
Use the `./build.py` script to build the image you care about,
and then run something like one of the following, depending on your model and processor:

```
docker run --rm -p 8000:8000 smartonfhir/cnlp-transformers:termexists-latest-cpu
docker run --rm -p 8000:8000 --gpus all smartonfhir/cnlp-transformers:termexists-latest-gpu
```

With that specific example of the `termexists` model, you could smoke test it like so:
```shell
curl http://localhost:8000/termexists/process -H "Content-Type: application/json" -d '{"doc_text": "Patient has no cough", "entities": [[0, 6], [15, 19]]}'; echo
```
Which should print `{"statuses":[1,-1]}` (the word `cough` was negated, but `Patient` was not).

### Publishing to Docker Hub
Run the same `./build.py` command you tested with, but add the `--push` flag.
The built images will be pushed to Docker Hub.
