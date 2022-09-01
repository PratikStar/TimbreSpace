import wandb
import random
# wandb sweep -p try analysis/wandb_sweeps.yaml


# Set up your default hyperparameters
hyperparameter_defaults = dict(
    lr= {"kld":1},
    hello="world"
)

# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults,
           entity="auditory-grounding",
           project="try")
# Access all hyperparameter values through wandb.config
config = wandb.config
print(config)
# Log metrics inside your training loop
loss = 100
for epoch in range(10):
    loss = random.uniform(0, loss)
    print({"loss": loss})
    wandb.log({"loss": loss})
