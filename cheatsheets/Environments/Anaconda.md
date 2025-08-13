# Developer Cheat Sheet

## ENVIRONMENT

### Anaconda Virtual Environment
```bash
# Create environment
conda create -n myenv python=3.11
conda create -n myenv python=3.11 numpy pandas

# Activate/deactivate
conda activate myenv
conda deactivate

# List environments
conda env list
conda info --envs

# Remove environment
conda env remove -n myenv

# Export/import environment
conda env export > environment.yml
conda env create -f environment.yml

# Install packages
conda install package_name
conda install -c conda-forge package_name
```
