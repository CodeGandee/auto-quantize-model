# TODO – trt-simple Docker image

- [ ] Build image and verify TensorRT import:
  - `docker build -t trt-simple -f dockers/trt-simple/Dockerfile .`
- [ ] Run container with GPU access and check `tensorrt`:
  - `docker run --gpus all --rm -it -v $PWD:/workspace trt-simple python3 -c "import tensorrt as trt; print(trt.__version__)"`
- [ ] Mount this repo and integrate with pixi/ModelOpt workflows.
- [ ] Add convenience scripts (e.g., `scripts/run-trt-simple.sh`) if useful.
- [ ] Use docker compose for build/run:
  - Build: `docker compose -f dockers/trt-simple/docker-compose.yml build`
  - Run: `docker compose -f dockers/trt-simple/docker-compose.yml run --rm trt-simple bash`
