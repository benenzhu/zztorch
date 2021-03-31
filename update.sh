#!/usr/bin/bash
git status
git rm -r --cached .
read -r -p 'are you sure?' INPUT
git add .
git commit -m 'update'
git push 
