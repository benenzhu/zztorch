#!/bin/bash
find . -size +1M
find . -size +1M | cat >> .gitignore
git status
git rm -r --cached .
read -r -p 'are you sure?' INPUT
git add .
git commit -m $INPUT 
git push 
