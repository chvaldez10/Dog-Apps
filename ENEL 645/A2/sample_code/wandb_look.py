# import wandb
# api = wandb.Api()
# run = api.run("enel-645-garbage-classifier/runs/xjrorumz")
# # Show history data:
# print("History from WandB:\n")
# print(run.history())


# ############ chat ##############
# import wandb

# # Initialize a wandb API object
# api = wandb.Api()

# # Specify the run path "entity/project/run_id"
# # run_path = "entity/project/run_id"
# run_path = "enel-645-garbage-classifier/runs/xjrorumz"


# # Access the run
# run = api.run(run_path)

# # Access run data
# print("Name:", run.name)
# print("Tags:", run.tags)
# print("Notes:", run.notes)
# print("Config:", run.config)

# # To get metrics logged during the run
# history = run.history()
# # print(history)  # Print available keys/metrics
# # print(history['accuracy'])  # Example to access the accuracy metric

############ wandb ##############
# https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs

import pandas as pd
import wandb



api = wandb.Api()
# entity, project = "enel-645", "enel-645-garbage-classifier" # Redge's
entity, project = "meng_team", "enel-645-garbage-classifier" # Carissa's 
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains output keys/values for
    # metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

    print("Run ID:", run.id)
    print("Name:", run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

# Write to .csv
# runs_df.to_csv("project.csv")

# access by index
print(runs_df.iloc[0]['summary'])
print(runs_df.iloc[0]['config'])
print(runs_df.iloc[0]['name'])