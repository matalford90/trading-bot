#!/bin/bash

# Navigate to the Git repo
cd /Users/matalford/Documents/Mat\ Alfords\ VSCode\ files/Projects/trading\ bot/trading-bot || exit

# Add all changes
git add -A

# Commit changes with timestamp
git commit -m "Auto commit: $(date)"

# Push to GitHub
git push origin main
