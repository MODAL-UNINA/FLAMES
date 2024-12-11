#!/usr/bin/env python
# -*- coding: utf-8 -*-
# %%
import grpc
import time
import flwr as fl
import torch
from Classic_model import UNet
from Custom_Classic_Client import FlowerClient
from Load import CustomDataset

import argparse
import json

from utils import is_interactive

# %%

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("--launch-id", type=int, default=0, help="Launch number")
parser.add_argument("--gpu-id", type=int, default=2, help="GPU number to use")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
parser.add_argument("--client-id", type=int, default=1, help="Client ID")
parser.add_argument("--round-epochs", type=int, default=5)
parser.add_argument(
    "--server-address",
    type=str,
    default="localhost:12389",
    help="Address of the server",
)

parser.add_argument(
    "--datasets-file",
    type=str,
    default='datasets_with_split.json',
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

print(args)

if not is_interactive() and not torch.cuda.is_available():
    raise ValueError("CUDA not available. Exiting...")

gpu_id = args.gpu_id

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

print("Device:", device)

server_address = args.server_address

datasets_file = args.datasets_file
config_file = args.config_file

round_epochs = args.round_epochs

with open(datasets_file, "r") as f:
    datasets = json.load(f)

modalities = sorted(list(set(datasets.values())))

if modalities != ["MRI", "US"]:
    raise ValueError(
        f"Modalities {modalities} are invalid. Must contain ['MRI', 'US']"
    )

print("Datasets names:", list(datasets.keys()))
print("Datasets modalities:", list(datasets.values()))
print("Modalities:", modalities)

client_id = args.client_id

dataset = list(datasets.keys())[client_id]
modality = datasets[dataset]
print(f"Selected dataset: {dataset}")
print(f"Modality: {modality}")

with open("config.json", "r") as f:
    config_data = json.load(f)

client_config = config_data[modality]

print("Client config:", client_config)

lr = client_config["lr"]

# %%
batch_size = args.batch_size

# Load data with batch size
custom_dataset = CustomDataset([], [], batch_size=batch_size)
trainloader, testloader = custom_dataset.load_data_client(
    dataset, data_augmentation=False and args.debug, debug=args.debug
)

print("Data loaded")
print(f"Trainloader: {len(trainloader.dataset)} samples, Batch size: {batch_size}")
print(f"Testloader: {len(testloader.dataset)} samples, Batch size: {batch_size}")

# Initialize the model for the selected modality
model = UNet()
model.to(device)

# %%
if args.jobid > 0:
    import os

    server_status_file = f"../0/{args.jobid}_status_server.txt"
    while not os.path.exists(server_status_file):
        print(f"Waiting for server status file {server_status_file}...")
        time.sleep(5)

# %%

def start_client(model, trainloader, testloader, modality, device):
    print(f"Starting client with model {model.__class__.__name__}...")
    client = FlowerClient(
        model,
        device,
        trainloader,
        testloader,
        modality,
        lr,
        num_examples=len(trainloader.dataset),
        modalities_list=modalities,
        save_plots=True,
        client_id=args.client_id,
        round_epochs=round_epochs,
    )

    delay_seconds = 10
    max_attempts = 10
    exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            print("Connection attempt number ", attempt)
            fl.client.start_client(
                server_address=args.server_address, client=client.to_client()
            )
            break
        except grpc.RpcError as e:
            print("Exception code: ", e.code())
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                print(
                    f"client {client_id} connection refused error on attempt {attempt}: {e}"
                )
                if attempt < max_attempts:
                    time.sleep(delay_seconds)
                continue
            else:
                print(f"client {client_id} error on attempt {attempt}: {e}")
                exception = e
                break
        except Exception as e:
            print(f"client {client_id} error on attempt {attempt}: {e}")
            exception = e
            break
    if exception is not None:
        print(f"client {client_id} failed to connect after {max_attempts} attempts")
        raise exception


start_client(model, trainloader, testloader, modality, device)

# %%
