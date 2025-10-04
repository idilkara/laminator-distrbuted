# Authors: Vasisht Duddu, Oskari Järvinen, Lachlan J Gunn, N Asokan
# Copyright 2025 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import os
import argparse
import hashlib
import json 
import time 
import io 

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import models
import utils
import data
import measured_file_read
import quote_generator

def get_distribution_z(z):
    df = pd.DataFrame()
    df['z'] = z.tolist()
    total_race = df['z'].value_counts()[0] + df['z'].value_counts()[1]
    print("ratio: {:.2f}".format(df['z'].value_counts()[0]/total_race), flush=True)
    actual_property = round(df['z'].value_counts()[0]/total_race,2)
    return actual_property

def get_distribution_z_y(z, y_train):
    df = pd.DataFrame()
    df['z'] = z.tolist()
    df['y'] = y_train.tolist()
    z0_y0 = df.value_counts(["z","y"]).iloc[[0,0]]
    z0_y1 = df.value_counts(["z","y"]).iloc[[0,1]]
    z1_y0 = df.value_counts(["z","y"]).iloc[[1,0]]
    z1_y1 = df.value_counts(["z","y"]).iloc[[1,1]]
    z0_y0 = round(z0_y0/len(z.tolist()),2)
    z0_y1 = round(z0_y1/len(z.tolist()),2)
    z1_y0 = round(z1_y0/len(z.tolist()),2)
    z1_y1 = round(z1_y1/len(z.tolist()),2)
    return z0_y0.tolist(), z0_y1.tolist(), z1_y0.tolist(), z1_y1.tolist()

def get_distribution_data(dataloader):
    labels_list, sattr_list = [], []
    for _, y, z in dataloader:
        labels_list.append(y)
        sattr_list.append(z)
    labels_list, sattr_list = np.array(labels_list), np.array(sattr_list)
    return labels_list, sattr_list

def distribution_attestation(args):
    # Load data
    data_load = {"UTKFACE": data.process_utkface, "CENSUS": data.process_census}

    if not (args.dataset == "UTKFACE" or args.dataset == "CENSUS"):
        print("Incorrect dataset", flush=True)
        exit()

    traindata, _, dataset_hash, input_measure_time, input_load_time, preprocess_time = data_load[args.dataset]()
    
    distribution_properties_results = {}

    # Get distribution
    compute_start = time.time()

    y_train, z_train = get_distribution_data(traindata)

    print("Checking distributional properties", flush=True)

    # Race_z
    compute_race_z_start = time.time()
    distribution_properties_results['race_z'] = get_distribution_z(z_train[:,0])
    compute_race_z_end = time.time()
    compute_race_z = compute_race_z_end - compute_race_z_start

    # Sex_z
    compute_sex_z_start = time.time()
    distribution_properties_results['sex_z'] = get_distribution_z(z_train[:,1])
    compute_sex_z_end = time.time()
    compute_sex_z = compute_sex_z_end - compute_sex_z_start

    # Race z given y
    compute_race_z_y_start = time.time()
    distribution_properties_results['race_z_y'] = get_distribution_z_y(z_train[:,0], y_train)
    compute_race_z_y_end = time.time()
    compute_race_z_y = compute_race_z_y_end - compute_race_z_y_start

    # Sex z given y
    compute_sex_z_y_start = time.time()
    distribution_properties_results['sex_z_y'] = get_distribution_z_y(z_train[:,1], y_train)
    compute_sex_z_y_end = time.time()
    compute_sex_z_y = compute_sex_z_y_end - compute_sex_z_y_start

    compute_end = time.time()
    compute_time = compute_end - compute_start
    print("Race z time to compute:", compute_race_z, flush=True)
    print("Sex z time to compute:", compute_sex_z, flush=True)
    print("Race z|y time to compute:", compute_race_z_y, flush=True)
    print("Sex z|y time to compute:", compute_sex_z_y, flush=True)
    print("Total time to compute:", compute_time, flush=True)
    
    attestation_time = "NA"
    output_measurement_time = "NA"
    output_storage_time = "NA"
    # Generate quote
    if args.with_sgx:
        payload = {
        'dataset-hash': base64.b64encode(dataset_hash).decode('utf-8'),
        'distribution_properties': distribution_properties_results
        }
   
        payload_bytes = json.dumps(payload).encode('utf-8')

        # Output measurement
        output_measurement_start = time.time()
        hasher = hashlib.sha512()
        hasher.update(payload_bytes)
        payload_hash = hasher.digest()
        output_measurement_end = time.time()
        output_measurement_time = output_measurement_end - output_measurement_start
        print("Output Measurement Time:", output_measurement_time, flush=True)

        # Quote generation/attestation time
        attestation_start = time.time()
        quote = quote_generator.generate_quote(user_data=payload_hash)
        json_data = {}
        json_data['sgx-quote'] = base64.b64encode(quote).decode('utf-8')
        json_data['payload'] = base64.b64encode(payload_bytes).decode('utf-8')
        attestation_end = time.time()
        attestation_time = attestation_end - attestation_start
        print("Time to form quote:", attestation_time, flush=True)

        output_storage_start = time.time()
        with open('output_distribution.json', 'w') as f:
            json.dump(json_data, f)
        output_storage_end = time.time()
        output_storage_time = output_storage_end - output_storage_start

    return preprocess_time, input_load_time, input_measure_time, compute_time, output_measurement_time, output_storage_time, attestation_time, compute_race_z, compute_sex_z, compute_race_z_y, compute_sex_z_y

def training_attestation(args):
    # Load Data
    print("Loading data", flush=True)
    data_load = {"CIFAR": data.process_cifar,"UTKFACE": data.process_utkface, "CENSUS": data.process_census, "IMDB": data.process_imdb}
    hidden_layer_sizes = {"VGG11": [128], "VGG13": [128, 256], "VGG16": [128, 256, 128], "VGG19": [128, 256, 512, 256]}
    hidden_layer_sizes_text = {"VGG11": 2, "VGG13": 4, "VGG16": 6, "VGG19": 8}
    train_data, _, dataset_hash, input_measure_time, input_load_time, preprocess_time = data_load[args.dataset]()

    if args.dataset != "IMDB":
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=256, shuffle=True)

    # Create model architecture
    compute_start = time.time()
    print("Creating model architecture", flush=True)
    if args.dataset == "UTKFACE":
        # args.epochs = 10
        model = models.VGGBinary(args.architecture).to(args.device)

    elif args.dataset == "CIFAR":
        # args.epochs = 25
        model = models.VGG(args.architecture).to(args.device)

    elif args.dataset == "CENSUS":
        # args.epochs = 5
        model = models.LinearNet(hidden_layer_sizes[args.architecture]).to(args.device)

    elif args.dataset == "IMDB":
        train_loader = DataLoader(train_data, shuffle=True, batch_size=50)
        model = models.SentimentRNN(args=args,no_layers=hidden_layer_sizes_text[args.architecture],vocab_size=1001,hidden_dim=256,embedding_dim=64,output_dim = 1,drop_prob=0.5).to(args.device)

    print(args.device)

    # Train model
    print("Training model", flush=True)
    if args.dataset != "IMDB":
        model = utils.train(args, model, train_loader)
    else:
        model = utils.train_text(args, model, train_loader)

    # Save model
    directory = './saved_models/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    path = f"{directory}{args.dataset.lower()}_model.pth"
    
    hasher = hashlib.sha256()
    bytes_object = io.BytesIO()
    torch.save(model.state_dict(), bytes_object)
    compute_end = time.time()
    compute_time = compute_end - compute_start
    print("Time to compute:", compute_time, flush=True)

    # Hash model
    model_measure_start = time.time()
    model_bytes = bytes_object.getvalue()
    hasher.update(model_bytes)
    model_hash = hasher.digest()
    model_measure_end = time.time()
    model_measure_time = model_measure_end - model_measure_start

    with open(path, 'wb') as f:
        f.write(model_bytes)

    print(f"Hash of model: {model_hash}", flush=True)


    attestation_time = "NA"
    output_measurement_time = "NA"
    output_storage_time = "NA"
    # Generate quote
    if args.with_sgx:
        payload = {
        'name': base64.b64encode(model_hash).decode('utf-8'),
        'dataset-hash': base64.b64encode(dataset_hash).decode('utf-8'),
        'training-parameters': {
            'epochs': args.epochs,
            'model_architecture': args.architecture,
        }}

        payload_bytes = json.dumps(payload).encode('utf-8')

        # Output measurement
        output_measurement_start = time.time()
        hasher = hashlib.sha512()
        hasher.update(payload_bytes)
        payload_hash = hasher.digest()
        output_measurement_end = time.time()
        output_measurement_time = output_measurement_end - output_measurement_start + model_measure_time

        # Quote generation/attestation time
        attestion_start = time.time()
        quote = quote_generator.generate_quote(user_data=payload_hash)
        json_data = {}
        json_data['sgx-quote'] = base64.b64encode(quote).decode('utf-8')
        json_data['payload'] = base64.b64encode(payload_bytes).decode('utf-8')
        attestation_end = time.time()
        attestation_time = attestation_end - attestion_start
        print("Time to form quote:", attestation_time, flush=True)

        output_storage_start = time.time()
        with open('output_train.json', 'w') as f:
            json.dump(json_data, f)
        output_storage_end = time.time()
        output_storage_time = output_storage_end - output_storage_start

    return preprocess_time, input_load_time, input_measure_time, compute_time, output_measurement_time, output_storage_time, attestation_time 

def accuracy_attestation(args, hidden_layer_sizes, hidden_layer_sizes_text):
    # Load data
    data_load = {"CIFAR": data.process_cifar,"UTKFACE": data.process_utkface, "CENSUS": data.process_census, "IMDB": data.process_imdb}
    _, testdata, dataset_hash, data_measure_time, data_load_time, preprocess_time = data_load[args.dataset]()

    if args.dataset == "IMDB":
        test_loader = DataLoader(testdata, shuffle=True, batch_size=50)
    else:
        test_loader = DataLoader(dataset=testdata, batch_size=256, shuffle=True)

    # Create model architecture
    if args.dataset == "UTKFACE":
        model = models.VGGBinary(args.architecture).to(args.device)

    elif args.dataset == "CIFAR":
        model = models.VGG(args.architecture).to(args.device)

    elif args.dataset == "CENSUS":
        model = models.LinearNet(hidden_layer_sizes[args.architecture]).to(args.device)

    elif args.dataset == "IMDB":
        model = models.SentimentRNN(args=args,no_layers=hidden_layer_sizes_text[args.architecture],vocab_size=1001,hidden_dim=256,embedding_dim=64,output_dim = 1,drop_prob=0.5).to(args.device)

    # Load model
    directory = './saved_models/'
    path = f"{directory}{args.dataset.lower()}_model.pth"

    model_measure_start = time.time()
    measured_bytes_model, model_load_time = measured_file_read.open_measured(path, "rb")
    model_measure_end = time.time()
    model_measure_time = model_measure_end - model_measure_start - model_load_time

    model.load_state_dict(torch.load(measured_bytes_model))

    input_measure_time = data_measure_time + model_measure_time
    input_load_time = data_load_time + model_load_time

    # Get accuracy
    compute_start = time.time()
    if args.dataset != "IMDB":
        accuracy = utils.test(args, model, test_loader)
    else:
        accuracy = utils.test_text(args, model, test_loader)
    compute_end = time.time()
    compute_time = compute_end - compute_start
    print("Time to compute:", compute_time, flush=True)

    print("Test Accuracy: {:.2f}".format(accuracy), flush=True)

    attestation_time = "NA"
    output_measurement_time = "NA"
    output_storage_time = "NA"
    # Generate quote
    if args.with_sgx:
        payload= {
            'name': base64.b64encode(measured_bytes_model.hasher.digest()).decode('utf-8'),
            'results': {
                'task': "classification",
                'dataset': f"{args.dataset}",
                'metrics': {
                    'type': 'accuracy',
                    'value': accuracy,
                    'dataset-hash': base64.b64encode(dataset_hash).decode('utf-8'),
                }
            }
        }

        payload_bytes = json.dumps(payload).encode('utf-8')
        
        # Output measurement
        output_measurement_start = time.time()
        hasher = hashlib.sha512()
        hasher.update(payload_bytes)
        payload_hash = hasher.digest()
        output_measurement_end = time.time()
        output_measurement_time = output_measurement_end - output_measurement_start 

        # Quote generation/attestation time
        attestation_start = time.time()
        quote = quote_generator.generate_quote(user_data=payload_hash)
        json_data = {}
        json_data['sgx-quote'] = base64.b64encode(quote).decode('utf-8')
        json_data['payload'] = base64.b64encode(payload_bytes).decode('utf-8')
        attestation_end = time.time()
        attestation_time = attestation_end - attestation_start
        print("Time to form quote:", attestation_time, flush=True)

        output_storage_start = time.time()
        with open('output_accuracy.json', 'w') as f:
            json.dump(json_data, f)
        output_storage_end = time.time()
        output_storage_time = output_storage_end - output_storage_start

    return preprocess_time, input_load_time, input_measure_time, compute_time, output_measurement_time, output_storage_time, attestation_time 

def io_attestation(args, hidden_layer_sizes, hidden_layer_sizes_text):
    # Load input
    input_path = f"./input/{args.dataset.lower()}_input.pt"
    data_measure_start = time.time()
    measured_input, data_load_time = measured_file_read.open_measured(input_path, "rb")
    data_measure_end = time.time()
    data_measure_time = data_measure_end - data_measure_start - data_load_time
    input_tensor = torch.load(measured_input).unsqueeze(0).to(args.device)

    # Create model architecture
    if args.dataset == "UTKFACE":
        model = models.VGGBinary(args.architecture).to(args.device)

    elif args.dataset == "CIFAR":
        model = models.VGG(args.architecture).to(args.device)

    elif args.dataset == "CENSUS":
        model = models.LinearNet(hidden_layer_sizes[args.architecture]).to(args.device)

    elif args.dataset == "IMDB":
        model = models.SentimentRNN(args=args,no_layers=hidden_layer_sizes_text[args.architecture],vocab_size=1001,hidden_dim=256,embedding_dim=64,output_dim = 1,drop_prob=0.5).to(args.device)

    # Load model
    directory = './saved_models/'
    path = f"{directory}{args.dataset.lower()}_model.pth"
    model_measure_start = time.time()
    measured_bytes_model, model_load_time = measured_file_read.open_measured(path, "rb")
    model_measure_end = time.time()
    model_measure_time = model_measure_end - model_measure_start - model_load_time

    model.load_state_dict(torch.load(measured_bytes_model))

    input_load_time = data_load_time + model_load_time
    input_measure_time = data_measure_time + model_measure_time

    # Get output
    compute_start = time.time()
    model = model.eval()
    for i in range(100):
        if args.dataset == "IMDB":
            val_h = model.init_hidden(1)
            val_h = tuple([each.data for each in val_h])
            with torch.no_grad():
                output, val_h = model(input_tensor, val_h)
                prediction = output.item()
        else:
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_classes = torch.max(output, 1)
                prediction = predicted_classes.sum().item()

    print("Prediction:", prediction, flush=True)
    compute_end = time.time()
    compute_time = (compute_end - compute_start) / 100
    print("Time to compute:", compute_time, flush=True)
    
    attestation_time = "NA"
    output_measurement_time = "NA"
    output_storage_time = "NA"
    # Generate quote
    if args.with_sgx:
        payload={
            "name": base64.b64encode(measured_bytes_model.hasher.digest()).decode('utf-8'),
            "results": {
                "task": "classification",
                "metrics": {
                    'type': 'inference',
                    'input': base64.b64encode(measured_input.hasher.digest()).decode('utf-8'),
                    'value': prediction,
                }
            }
        }

        payload_bytes = json.dumps(payload).encode('utf-8')

        # Output measurement
        output_measurement_start = time.time()
        hasher = hashlib.sha512() 
        hasher.update(payload_bytes)
        payload_hash = hasher.digest()
        output_measurement_end = time.time()
        output_measurement_time = output_measurement_end - output_measurement_start 

        # Quote generation/attestation time
        attestation_start = time.time()
        quote = quote_generator.generate_quote(user_data=payload_hash)
        json_data = {}
        json_data['sgx-quote'] = base64.b64encode(quote).decode('utf-8')
        json_data['payload'] = base64.b64encode(payload_bytes).decode('utf-8')
        attestation_end = time.time()
        attestation_time = attestation_end - attestation_start
        print("Time to form quote:", attestation_time, flush=True)

        output_storage_start = time.time()
        with open('output_io.json', 'w') as f:
            json.dump(json_data, f)
        output_storage_end = time.time()
        output_storage_time = output_storage_end - output_storage_start

    return 0, input_load_time, input_measure_time, compute_time, output_measurement_time, output_storage_time, attestation_time

def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",type=str,default="UTKFACE",help="One of: [UTKFACE, CIFAR, CENSUS, IMDB]")
    parser.add_argument("--device",type=str,default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),help="GPU ID for this process")
    parser.add_argument("--epochs",type=int,default=5,help="number of epochs to train")
    parser.add_argument("--architecture",type=str,default="VGG11",help="[One of: [VGG11, VGG13, VGG16, VGG19]")
    parser.add_argument("--model_size",type=str,default="S",help="[One of: [S, L]")
    parser.add_argument("--attestation_type",type=str, default="train", help="One of: [train, distribution, accuracy, io]")
    parser.add_argument("--with_sgx",type=bool, default=False, help="Whether the script is being run inside Gramine or not")
    parser.add_argument("--exp_id",type=int, default=0, help="For reporting purposes.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = handle_args()

    hidden_layer_sizes = {"VGG11": [128], "VGG13": [128, 256], "VGG16": [128, 256, 128], "VGG19": [128, 256, 512, 256]}
    hidden_layer_sizes_text = {"VGG11": 2, "VGG13": 4, "VGG16": 6, "VGG19": 8}

    # Change the model size S/L to the architecture.
    if args.model_size == "S":
        args.architecture = "VGG11"
    elif args.model_size == "L":
        if args.dataset == "CENSUS":
            args.architecture = "VGG19"
        elif args.dataset == "UTKFACE":
            args.architecture = "VGG16"
        elif args.dataset == "IMDB":
            args.architecture = "VGG13"
    

    print("Starting", flush=True)

    if args.attestation_type == "distribution":
        preprocess_time, input_load_time, input_measure_time, compute_time, output_measurement_time, output_storage_time, attestation_time, compute_race_z, compute_sex_z, compute_race_z_y, compute_sex_z_y = distribution_attestation(args)
    elif args.attestation_type == "train":
        preprocess_time, input_load_time, input_measure_time, compute_time, output_measurement_time, output_storage_time, attestation_time  = training_attestation(args)
    elif args.attestation_type == "accuracy":
        preprocess_time, input_load_time, input_measure_time, compute_time, output_measurement_time, output_storage_time, attestation_time  = accuracy_attestation(args, hidden_layer_sizes, hidden_layer_sizes_text)
    elif args.attestation_type == "io":
        preprocess_time, input_load_time, input_measure_time, compute_time, output_measurement_time, output_storage_time, attestation_time  = io_attestation(args, hidden_layer_sizes, hidden_layer_sizes_text)
    else:
        print("Incorrect attestation type", flush=True)
        exit()

    if args.attestation_type == "distribution":
        with open("./distribution_results.csv", "a") as f:
            f.write(f"{args.dataset},{args.epochs},{args.architecture},{args.attestation_type},{args.with_sgx}, {preprocess_time}, {input_load_time}, {input_measure_time}, {compute_time}, {output_measurement_time}, {output_storage_time}, {attestation_time},{compute_race_z},{compute_sex_z},{compute_race_z_y},{compute_sex_z_y}\n")
    else:
        with open("./results.csv", "a") as f:
            f.write(f"{args.dataset},{args.epochs},{args.architecture},{args.attestation_type},{args.with_sgx}, {preprocess_time}, {input_load_time}, {input_measure_time}, {compute_time}, {output_measurement_time}, {output_storage_time}, {attestation_time}\n")