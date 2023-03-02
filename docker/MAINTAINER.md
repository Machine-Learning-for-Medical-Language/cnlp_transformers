To deploy images to dockerhub, first auth with docker with an account that
has access to the smartonfhir organization. Then, the following commands
should build and publish images (in the CPU case, for multiple architectures).


MODEL should be one of: [base, dtr, event, negation, temporal, timex]
PROCESSOR should be one of: [cpu, gpu]
PLATFORMS should be linux/amd64 for GPU, and linux/amd64,linux/arm64 for CPU
```
export MAJOR=0
export MINOR=4
export PATCH=0
export MODEL=negation
export PROCESSOR=cpu
export PLATFORMS=linux/amd64,linux/arm64

docker buildx build \
--push --platform $PLATFORMS \
--tag smartonfhir/cnlp-transformers:$MODEL-latest-$PROCESSOR \
--tag smartonfhir/cnlp-transformers:$MODEL-$MAJOR-$PROCESSOR \
--tag smartonfhir/cnlp-transformers:$MODEL-$MAJOR.$MINOR-$PROCESSOR \
--tag smartonfhir/cnlp-transformers:$MODEL-$MAJOR.$MINOR.$PATCH-$PROCESSOR \
-f Dockerfile.$PROCESSOR \
--target $MODEL . 
```
