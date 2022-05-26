## Rsync: Remote Development

1. rsync repo first

`cd /Users/pratik/repos/PyTorch-VAE
rsync -av --exclude-from=".rsyncignore_upload" "/Users/pratik/repos/PyTorch-VAE" w:/work/gk77/k77021/repos`

2. Rsync log files from wisteria
`rsync -av w:/work/gk77/k77021/repos/PyTorch-VAE/logs "/Users/pratik/repos/PyTorch-VAE"`

## Watch: Online editing

Pytorch-VAE repo
`
watch -d -n5 "rsync -av --exclude-from=\".rsyncignore_upload\" \"/Users/pratik/repos/PyTorch-VAE\" w:/work/gk77/k77021/repos"
`

log files from wisteria
`
watch -d -n5 "rsync -av w:/work/gk77/k77021/repos/PyTorch-VAE/logs \"/Users/pratik/repos/PyTorch-VAE\""
`

## Watch git push

`watch-2 acp-live`

## Wisteria

```
# for debug jobs 
pjsub wisteria-debug.sh

# for interactive jobs
pjsub --interact wisteria-interactive.sh

```
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
