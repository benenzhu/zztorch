#!/bin/bash
mypath=`realpath $0`
cd `dirname $mypath`
pwd
pwd
1/0
find . -size +1M
find . -size +1M | cat >> .gitignore
git rm -r --cached .
git status
read -r -p 'are you sure?' INPUT
git add .
git commit -m $INPUT 
git push 
