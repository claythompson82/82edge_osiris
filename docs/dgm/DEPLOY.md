# Deploying the DGM Kernel

The Darwin Gödel Machine kernel can run directly from the source tree or via the provided Docker image.

## Building the Docker image

```bash
docker build -f docker/dgm-kernel.Dockerfile -t dgm-kernel:local .
```

## Running the DGM kernel

Invoke the kernel once from the command line:

```bash
python -m dgm_kernel --once
```

To run continuously just omit the flag:

```bash
python -m dgm_kernel
```

The Docker container exposes port 8000 for Prometheus metrics.
