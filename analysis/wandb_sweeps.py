import wandb
# wandb sweep -p try analysis/wandb_sweeps.yaml


# Set up your default hyperparameters
hyperparameter_defaults = dict(
    channels=[16, 32],
    batch_size=100,
    learning_rate=0.001,
    optimizer="adam",
    epochs=2,
)

# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults)
# Access all hyperparameter values through wandb.config
config = wandb.config

# Set up your model
print(config)
print(config.learning_rate)
print(type(config.learning_rate))

# Log metrics inside your training loop
for i in range(5, 0, -1):
    wandb.log({"validation_loss": i})
