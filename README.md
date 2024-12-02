# Comet-Divers-AFSS
University of Maryland, College Park
ENAE484 Space Systems Design
Dr. David Akin
Comet Divers - AFSS Team
Sean McCurry, Nathan Kerns, Richard Huang, Owen Moran, Brendan Bullock
 
## Python Setup

Recommended python setup for consistent results.

Run in a command line the top level directory of the repo:

```bash
# Create a virtual environment to keep python packages isolated
python3 -m venv venv

# Activate the virtual environment
# Linux/MacOS:
source venv/bin/activate
# Windows command prompt:
venv\Scripts\activate.bat
# Windows powershell:
venv\Scripts\Activate.ps1

# Install packages (might take a while)
pip install -r requirements.txt
```

You need to activate the virtual environment before running a python script. Once activated, it will stay activated until you close that terminal. There should be a `(venv)` in your command prompt to indicated the virtual environment is activated. Inside vscode, you need to select the virtual environment so the autocomplete and intellisense know where to find the packages.

## Using Git

Command line:

```bash
# Clone repo (using either SSH or HTTP authentication)
git clone https://github.com/smccurry109/Comet-Divers-AFSS.git
git clone git@github.com:smccurry109/Comet-Divers-AFSS.git

# See current status - branch, modified files, staged files, etc
git status

# Stage modified files (README.md for example)
git add README.md

# Commit staged files - it will open a text editor for you to enter a short commit message
git commit

# Push local commits to github
git push

# Pull changes from github (do this before committing anything to make sure there's no conflicts)
git pull
```

In vscode, the source control panel (3rd icon down on the left side) has all the same functions in a nice GUI. This is equivalent to the commands, so it's up to personal preference which to use. You can click on any of the changes files to see what changed in them.
