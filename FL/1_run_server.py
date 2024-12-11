#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%

import flwr as fl
import torch

from Custom_server import CustomServer
from Models import UNet_reduced_server
from Custom_Strategy import CustomStrategy
from Load import CustomDataset

import argparse
import json
from utils import is_interactive

# %%

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("--launch-id", type=int, default=0, help="Launch number")
parser.add_argument("--gpu-id", type=int, default=1, help="GPU number to use")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
parser.add_argument("--num-rounds", type=int, default=100, help="Number of rounds")

parser.add_argument(
    "--knowledge-distillation",
    type=int,
    default=1,
    help="Use knowledge distillation",
)

parser.add_argument(
    "--server-address",
    type=str,
    default="localhost:12389",
    help="Address of the server",
)

parser.add_argument(
    "--datasets-file",
    type=str,
    default="datasets_with_split_MRI_US.json",
    help="Data to load",
)
parser.add_argument(
    "--jobid",
    type=int,
    default=0,
    help="Job ID",
)
parser.add_argument(
    "--config-file",
    type=str,
    default="[]",
    help="Configuration",
)

parser.add_argument("--debug", type=int, default=1, help="Enable debug mode")
if is_interactive():
    args, _ = parser.parse_known_args()
else:
    args = parser.parse_args()

if not is_interactive() and not torch.cuda.is_available():
    raise ValueError("CUDA not available. Exiting...")

gpu_id = args.gpu_id

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

print("Device:", device)

config_file = args.config_file

if config_file != "[]":
    with open(config_file, "r") as f:
        config_file = json.load(f)

    launch_id = args.launch_id
    args.num_rounds = config_file[launch_id]["rounds"]
    args.modality = config_file[launch_id]["modality"]

if config_file == "[]":
    datasets_file = args.datasets_file

    with open(datasets_file, "r") as f:
        datasets = json.load(f)

    modalities = sorted(list(set(datasets.values())))
else:
    modalities = args.modality
    datasets_file = f"datasets_{modalities}.json"
    with open(datasets_file, "r") as f:
        datasets = json.load(f)

if modalities != ["MRI", "US"]:
    raise ValueError(
        f"Modalities {modalities} are invalid. Must contain ['CT', 'MRI', 'US']"
    )

print(args)

print("Datasets names:", list(datasets.keys()))
print("Datasets modalities:", list(datasets.values()))
print("Modalities:", modalities)

min_num_clients = len(datasets)
knowledge_distillation = args.knowledge_distillation != 0

modalities_clients = list(datasets.values())

# %%
batch_size = args.batch_size

strategies = {modality: fl.server.strategy.FedAvg() for modality in modalities}

custom_dataset = CustomDataset(
    [], [], batch_size=batch_size
)

# Load the data
(
    testloader_test_server,
    testloader_train_server_reduced,
    testloader_test_server_reduced,
) = custom_dataset.load_data_server(datasets, debug=args.debug)

print("Data loaded")
print(
    f"Testloader server: {len(testloader_test_server.dataset)} samples, Batch size: {batch_size}"
)
print(
    f"Trainloader server red: {len(testloader_train_server_reduced.dataset)} samples, Batch size: {batch_size}"
)
print(
    f"Testloader server red: {len(testloader_test_server_reduced.dataset)} samples, Batch size: {batch_size}"
)

# %%

# Start the server
knowledge_distillation = args.knowledge_distillation
print(f"Knowledge distillation: {knowledge_distillation}")
strategy = CustomStrategy(strategies=strategies, min_num_clients=min_num_clients)
model_reduced = UNet_reduced_server()

client_manager = fl.server.SimpleClientManager()
server = CustomServer(
    strategy=strategy,
    client_manager=client_manager,
    model_reduced_server=model_reduced,
    batch_size=batch_size,
    testloader_server=testloader_test_server,
    trainloader_server_red=testloader_train_server_reduced,
    testloader_server_red=testloader_test_server_reduced,
    DEVICE=device,
    know_distillation=knowledge_distillation,
    modalities_clients=modalities_clients,
    save_plots=True,
)

config = fl.server.ServerConfig(num_rounds=args.num_rounds)

# %%

if args.jobid > 0:
    status_file = f"{args.jobid}_status_server.txt"

    # touch the file
    open(status_file, "a").close()


fl.server.start_server(
    server_address=args.server_address,
    config=config,
    server=server,
    client_manager=client_manager,
    grpc_max_message_length=1024 * 1024 * 1024
)

# %%
