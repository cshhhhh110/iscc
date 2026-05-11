# Shared environment specs for ISCC-style tasks

Use the Conda environment `iscc-ml` with Python 3.10 as the common baseline for binary/security tasks.

Recommended setup:

```powershell
conda env create -f _env_specs/iscc-base.yml
conda activate iscc-ml
```

For lightweight projects, you can also install from `iscc-binsec-requirements.txt` inside the same environment.
