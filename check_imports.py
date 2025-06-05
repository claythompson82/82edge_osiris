import importlib
import sys
import importlib.metadata

libraries = [
    "hypothesis",
    "fakeredis",
    "httpx",
    "redis",
    "requests",
    "lancedb",
    "sentry_sdk",
    "pandas",
    "opentelemetry.api", # Check API first
    "opentelemetry.trace",
    "opentelemetry.sdk",
    "opentelemetry.exporter.otlp",
    "torch",
    "torchaudio"
]
print(f"Python version: {sys.version}")

venv_path = getattr(sys, 'prefix', None)
if venv_path and venv_path != sys.base_prefix : # Check if prefix is different from base_prefix for venv
    print(f"Running in virtual environment: {venv_path}")
else:
    print("Not running in a detectable virtual environment, or venv is the same as base.")

for lib_name in libraries:
    try:
        module = importlib.import_module(lib_name)

        # Attempt to get version using standard attributes
        version = getattr(module, '__version__', None) or \
                  getattr(module, 'VERSION', None)

        # If version not found, try importlib.metadata for installed package name
        if not version:
            try:
                # Adjust package name for importlib.metadata if necessary
                # e.g. opentelemetry.api -> opentelemetry-api
                package_name_for_metadata = lib_name.replace('.', '-')
                if "opentelemetry-" in package_name_for_metadata and package_name_for_metadata.count('-') > 1:
                    # e.g. opentelemetry-exporter-otlp -> opentelemetry-exporter-otlp
                    pass # Already correct
                elif lib_name == "opentelemetry.api":
                     package_name_for_metadata = "opentelemetry-api"
                elif lib_name == "opentelemetry.sdk":
                     package_name_for_metadata = "opentelemetry-sdk"
                elif lib_name == "opentelemetry.trace": # trace is part of api
                    package_name_for_metadata = "opentelemetry-api"


                version = importlib.metadata.version(package_name_for_metadata)
            except importlib.metadata.PackageNotFoundError:
                version = "N/A (metadata)"
            except Exception: # Catch any other metadata error
                version = "Error getting version via metadata"

        if lib_name == "torch":
            cuda_available = module.cuda.is_available() if hasattr(module, 'cuda') else False
            print(f"Successfully imported {lib_name} - Version: {version} - CUDA available: {cuda_available}")
        elif lib_name == "opentelemetry.trace":
             # Version for opentelemetry.trace is effectively the opentelemetry-api version
             print(f"Successfully imported {lib_name} (opentelemetry.api version: {version})")
        elif version:
            print(f"Successfully imported {lib_name} - Version: {version}")
        else:
            print(f"Successfully imported {lib_name} (version not readily available)")

    except ModuleNotFoundError:
        print(f"Error: ModuleNotFoundError for {lib_name}")
    except Exception as e:
        print(f"Error importing {lib_name}: {e}")
