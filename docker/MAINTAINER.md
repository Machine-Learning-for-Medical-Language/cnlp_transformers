To deploy images to dockerhub, first auth with docker with an account that
has access to the smartonfhir organization. Then, the following commands
should build and publish images (in the CPU case, for multiple architectures).


GPU-enabled
```
set MAJOR=0
set MINOR=4
set PATCH=0

docker buildx build \
--platform linux/amd64 \
--tag smartonfhir/cnlp-transformers:latest-gpu \
--tag smartonfhir/cnlp-transformers:$MAJOR-gpu \
--tag smartonfhir/cnlp-transformers:$MAJOR.$MINOR-gpu \
--tag smartonfhir/cnlp-transformers:$MAJOR.$MINOR.$PATCH-gpu \
-f Dockerfile.gpu . --push

```

CPU only
```
set MAJOR=0
set MINOR=4
set PATCH=0

docker buildx build \
--platform linux/amd64,linux/arm64 \
--tag smartonfhir/cnlp-transformers:latest-cpu \
--tag smartonfhir/cnlp-transformers:$MAJOR-cpu \
--tag smartonfhir/cnlp-transformers:$MAJOR.$MINOR-cpu \
--tag smartonfhir/cnlp-transformers:$MAJOR.$MINOR.$PATCH-cpu \
-f Dockerfile.cpu . --push

```