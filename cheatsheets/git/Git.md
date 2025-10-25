# Developer Cheat Sheet

## GIT

### Git Commands
```bash
# Setup
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Repository operations
git init
git clone https://github.com/user/repo.git
git remote add origin https://github.com/user/repo.git

# Basic workflow
git status
git add .
git add file.txt
git commit -m "commit message"
git push origin main
git pull origin main

# Branching
git branch                    # list branches
git branch new-branch         # create branch
git checkout new-branch       # switch branch
git checkout -b new-branch    # create and switch
git merge branch-name
git branch -d branch-name     # delete branch

# History and changes
git log
git log --oneline
git diff
git diff --staged
git show commit-hash

# Undo changes
git reset HEAD~1              # undo last commit
git reset --hard HEAD~1       # undo and delete changes
git revert commit-hash        # create reverting commit
git checkout -- file.txt     # discard file changes
```
