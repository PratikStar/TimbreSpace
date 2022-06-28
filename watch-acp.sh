#!/bin/bash
messages=("online debugging" "online editing" "online commit & push" "online development")

size=${#messages[@]}
index=$(($RANDOM % $size))
message=${messages[$index]}

git status
git add --all
git commit -m "${message}"
git push origin
