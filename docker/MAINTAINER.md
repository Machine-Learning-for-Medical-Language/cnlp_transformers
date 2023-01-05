To deploy images to dockerhub, first auth with docker with an account that
has access to the smartonfhir organization. Then, the following commands
should build and publish images (in the CPU case, for multiple architectures).


GPU-enabled
```
export MAJOR=0
export MINOR=4
export PATCH=0
export MODEL=negation

docker buildx build \
--push --platform linux/amd64 \
--tag smartonfhir/cnlp-transformers:$MODEL-latest-gpu \
--tag smartonfhir/cnlp-transformers:$MODEL-$MAJOR-gpu \
--tag smartonfhir/cnlp-transformers:$MODEL-$MAJOR.$MINOR-gpu \
--tag smartonfhir/cnlp-transformers:$MODEL-$MAJOR.$MINOR.$PATCH-gpu \
-f Dockerfile.gpu \
--target $MODEL . 
```

CPU only
```
export MAJOR=0
export MINOR=4
export PATCH=0
export MODEL=negation

docker buildx build \
--push --platform linux/amd64,linux/arm64 \
--tag smartonfhir/cnlp-transformers:$MODEL-latest-cpu \
--tag smartonfhir/cnlp-transformers:$MODEL-$MAJOR-cpu \
--tag smartonfhir/cnlp-transformers:$MODEL-$MAJOR.$MINOR-cpu \
--tag smartonfhir/cnlp-transformers:$MODEL-$MAJOR.$MINOR.$PATCH-cpu \
-f Dockerfile.cpu \
--target $MODEL .
```