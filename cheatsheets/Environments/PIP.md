# Developer Cheat Sheet

### Pip Installations
```bash
# Basic installation
pip install package_name
pip install package_name==1.2.3
pip install package_name>=1.2.3

# Install from requirements
pip install -r requirements.txt

# Generate requirements
pip freeze > requirements.txt

# Install in development mode
pip install -e .

# Install from Git
pip install git+https://github.com/user/repo.git

# Upgrade/uninstall
pip install --upgrade package_name
pip uninstall package_name

# List installed packages
pip list
pip show package_name
```