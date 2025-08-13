### SSH Setup

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"

# Start SSH agent
eval "$(ssh-agent -s)"

# Add key to agent
ssh-add ~/.ssh/id_ed25519

# Copy public key (macOS)
pbcopy < ~/.ssh/id_ed25519.pub

# Copy public key (Linux)
cat ~/.ssh/id_ed25519.pub | xclip -selection clipboard

# Test connection
ssh -T git@github.com

# SSH config file (~/.ssh/config)
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519

Host myserver
    HostName server.example.com
    User username
    Port 22
    IdentityFile ~/.ssh/id_rsa
```