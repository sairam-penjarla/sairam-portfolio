# Python Virtual Environments Commands

```bash
# Create a virtual environment
python3 -m venv env_name

# Activate virtual environment

# On Windows (PowerShell)
.\env_name\Scripts\Activate.ps1

# On Windows (CMD)
.\env_name\Scripts\activate.bat

# On macOS/Linux
source env_name/bin/activate

# Deactivate virtual environment
deactivate

# Check active Python executable

# macOS/Linux
which python

# Windows
where python

# Delete a virtual environment (just remove the folder)

# macOS/Linux
rm -rf env_name

# Windows CMD
rmdir /s /q env_name
```