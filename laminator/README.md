# Laminator: Verifiable ML Property Cards using Hardware-assisted Attestations

Code for "Laminator: Verifiable ML Property Cards using Hardware-assisted Attestations" published in ACM Conference on Data and Application Security and Privacy (CODASPY), 2025.

Link to paper: https://arxiv.org/abs/2406.17548

Please cite our work as:
```
@inproceedings{duddu2024laminator,
  title={Laminator: Verifiable ML Property Cards using Hardware-assisted Attestations},
  author={Duddu, Vasisht and J{\"a}rvinen, Oskari and Gunn, Lachlan J and Asokan, N},
  booktitle={ACM Conference on Data and Application Security and Privacy (CODASPY)},
  year={2025}
}

```

## Setup
First you need to setup your own DCAP setup for a computer:
https://www.intel.com/content/www/us/en/developer/articles/guide/intel-software-guard-extensions-data-center-attestation-primitives-quick-install-guide.html

and

https://github.com/intel/SGXDataCenterAttestationPrimitives/blob/main/QuoteGeneration/pccs/README.md

Npm package "n" might help you to get a proper nodejs version to work: https://www.npmjs.com/package/n

## Datasets

Link to CELEBA: https://drive.google.com/file/d/1KTaJraB9Koa4h5EVTJQ3y2Dig_vgE5MZ/view?usp=sharing

Link to UTKFACE: https://drive.google.com/file/d/1iLCJEu2bwVdd0SiZFzNMkzvhH3TjTWN4/view?usp=sharing


## Install Python Libraries
When running the code in the docker container, all the required libraries will be installed automatically. To run the code outside of Docker using your Python environment, first install PyTorch, and then install the rest of the dependencies using `requirements.txt`

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Run Code
The script `main.py` has the following command-line arguments:
- `dataset`: One of `[UTKFACE, CIFAR, CENSUS, IMDB]`. Default `UTKFACE`.
- `epochs`: Number of epochs to train for. Default `5`.
- `architecture`: One of: `[VGG11, VGG13, VGG16, VGG19]`. Default `VGG11`.
- `model_size`: Map model size to architecture. One of: `[S, L]`. Default `S`.
- `attestation_type`: One of: `[train, distribution, accuracy, io]`. Default `train`.
- `with_sgx`: Boolean argument set to 1 if training from within gramine using SGX. Default `False`

Example in your own environment:
```
python main.py --dataset CENSUS --epochs 5 --model_size S --attestation_type train
```

Example when running in Docker:
```
python3 main.py --dataset CENSUS --epochs 5 --model_size S --attestation_type train
```

To run the code in an SGX enclave using Gramine, use the following code to setup a Docker image and run a docker container
```
cd generate_attestations
docker build -t {container_name} .
docker run --device /dev/sgx_enclave --device /dev/sgx_provision -it {container_name}
```

To run the attestation once inside the docker container:
```
make -f Makefile.main clean
make -f Makefile.main SGX=1
gramine-sgx ./main main.py --dataset CENSUS --epochs 5 --model_size S --attestation_type train --with_sgx 1
```
Note: Run training first to get a model.

## Run full experiment
Use the following bash script to run an experiment 10 times using the default architecture and epochs without SGX. 
```
bash run_test_base.sh <dataset> <attestation_type>
```

Use the following bash script to run an experiment 10 times using the default architecture and epochs with SGX. 
```
bash run_test_SGX.sh <dataset> <attestation_type>
```

Furthermore, the following scripts can be used to run all the datasets, attestations, architectures, 10 times. 
```
bash run_everything_base.sh
bash run_everything_sgx.sh
```

Additionally, we use [PyNaCl](https://pynacl.readthedocs.io/en/latest/) to compare the time taken for signing using a key pair versus hardware-based attestation for input/output attestation. The software signing approach can be implemented using the following code:

```python
from nacl.signing import SigningKey

key_signing_start = time.time()

# Generate a new random signing 
signing_key = SigningKey.generate()

# Sign a payload_hash, which is the user data used to generate the SGX quote.
keysigned = signing_key.sign(payload_hash)

key_signing_end = time.time()
key_signing_total = key_signing_end - key_signing_start
```

Useful links:
- https://gramine.readthedocs.io/en/stable/
- https://gramine.readthedocs.io/en/stable/manifest-syntax.html
- https://gramine.readthedocs.io/en/stable/attestation.html
- https://gramine.readthedocs.io/en/stable/run-sample-application.html
- https://github.com/intel/SGXDataCenterAttestationPrimitives
- https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/Intel_SGX_ECDSA_QuoteLibReference_DCAP_API.pdf
- https://learn.microsoft.com/en-us/azure/security/fundamentals/trusted-hardware-identity-management