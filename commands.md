## Rsync

### rsync repo. local --> wisteria
```
cd /Users/pratik/repos/TimbreSpace
rsync -av --exclude-from=".rsyncignore_upload" "/Users/pratik/repos/TimbreSpace" w:/work/gk77/k77021/repos
```

### rsync data. local --> wisteria
```
cd /Users/pratik/data/timbre
rsync -av "/Users/pratik/data/timbre" w:/work/gk77/k77021/data
```

### rsync log files. wisteria --> local
```
rsync -av w:/work/gk77/k77021/repos/TimbreSpace/logs "/Users/pratik/repos/TimbreSpace"
```


## Watch rsync: Online editing

### repo. local --> wisteria
```
watch -d -n5 "rsync -av --exclude-from=\".rsyncignore_upload\" \"/Users/pratik/repos/TimbreSpace\" w:/work/gk77/k77021/repos"
```

### log files. wisteria --> local
```
watch -d -n5 "rsync -av w:/work/gk77/k77021/repos/TimbreSpace/logs \"/Users/pratik/repos/TimbreSpace\""
```

## Watch git push

```
watch -n60 ./watch-acp.sh
```

## Wisteria

```
# for debug jobs 
pjsub wisteria-scripts/wisteria-debug.sh

# for interactive jobs
pjsub --interact wisteria-scripts/wisteria-interactive.sh

```

## Tensorboard

## Install pyenv & python

```
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bashrc
source ~/.bash_profile

pyenv
pyenv install 3.10.4
pyenv global 3.10.4
```
## References
https://research.google.com/colaboratory/local-runtimes.html
https://www.concordia.ca/ginacody/aits/support/faq/ssh-tunnel.html
https://thedatafrog.com/en/articles/remote-jupyter-notebooks/
https://explainshell.com/explain?cmd=ssh+-L+-N+-f+-l
