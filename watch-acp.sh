#!/bin/bash
messages=("debugging" "editing" "commit & push" "development")

size=${#messages[@]}
index=$(($RANDOM % $size))
message=${messages[$index]}

git status
git add --all
git commit -m "${message}"

git push origin
