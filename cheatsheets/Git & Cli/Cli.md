# Developer Cheat Sheet

### Bash Shortcuts
```bash
# Navigation
cd ~          # home directory
cd -          # previous directory
cd ..         # parent directory
pwd           # current directory

# File operations
ls -la        # list all files with details
mkdir dir     # create directory
mkdir -p a/b/c # create nested directories
rm file       # remove file
rm -rf dir    # remove directory recursively
cp src dest   # copy file
mv src dest   # move/rename file

# Text processing
cat file.txt      # display file content
head -n 5 file    # first 5 lines
tail -n 5 file    # last 5 lines
grep "pattern" file # search in file
find . -name "*.py" # find files

# Process management
ps aux            # list processes
kill PID          # kill process
kill -9 PID       # force kill
jobs              # list background jobs
ctrl+z            # suspend process
bg                # resume in background
fg                # bring to foreground

# Keyboard shortcuts
ctrl+c            # interrupt process
ctrl+d            # end of file/exit
ctrl+l            # clear screen
ctrl+r            # search command history
ctrl+a            # beginning of line
ctrl+e            # end of line
```