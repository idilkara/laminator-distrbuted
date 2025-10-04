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

import sgx_dcap_quote_verify
import os
import sys
import measured_read
import hashlib
import base64

from pathlib import Path
from datetime import datetime
from azure.security.attestation import AttestationClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import requests

# You can download the trusted root CA in PEM format directly from Intel at :
# <https://certificates.trustedservices.intel.com/Intel_SGX_Provisioning_Certification_RootCA.pem>
load_dotenv()
host=input("choose hosting type (self | azure): ")
if host == "self":
    trusted_root_ca_certificate = Path("./test_data/trustedRootCaCert.pem").read_text()

    # Get the quote and the collateral from the service you want to attest
    pck_certificate = Path("./test_data/pckCert.pem").read_text()
    pck_signing_chain = Path("./test_data/pckSignChain.pem").read_text()
    root_ca_crl = Path("./test_data/trustedCaCrl.pem").read_text()
    intermediate_ca_crl = Path("./test_data/intermediateCaCrl.pem").read_text()
    tcb_info = Path("./test_data/tcbInfo.json").read_text()
    tcb_signing_chain = Path("./test_data/tcbSignChain.pem").read_text()
    quote_test = Path("./test_data/quote.dat").read_bytes()
    qe_identity = Path("./test_data/qe_identity.json").read_text()

    # Set the date used to check if the collateral (certificates,CRLs...) is still valid
    expiration_date = datetime.now()

    # Use the package to check the validity of the quote
    attestation_result = sgx_dcap_quote_verify.verify(
        trusted_root_ca_certificate,
        pck_certificate,
        pck_signing_chain,
        root_ca_crl,
        intermediate_ca_crl,
        tcb_info,
        tcb_signing_chain,
        quote_test,
        qe_identity,
        expiration_date,
    )

    assert attestation_result.ok
    assert (
        attestation_result.pck_certificate_status
        == sgx_dcap_quote_verify.VerificationStatus.STATUS_OK
    )
    assert (
        attestation_result.tcb_info_status
        == sgx_dcap_quote_verify.VerificationStatus.STATUS_OK
    )
    assert (
        attestation_result.qe_identity_status
        == sgx_dcap_quote_verify.VerificationStatus.STATUS_OK
    )
    assert (
        attestation_result.quote_status
        == sgx_dcap_quote_verify.VerificationStatus.STATUS_OK
    )

    # The attestation result contains the report data, which includes the MR_ENCLAVE
    print("Quote is valid")
elif host == "azure":

    attest_client = AttestationClient(
        endpoint=os.environ.get("AZURE_ATTESTER_URL"),
        credential=DefaultAzureCredential()
    )

    quote = b'0'
    with open("../pytorch_setup/server/quote.dat", "rb") as f:
        quote = base64.b64encode(f.read())
    runtime_data= base64.b64encode(quote[368:432])

    quote_str = quote.decode('utf-8')
    runtime_data_str = runtime_data.decode('utf-8')
    url=os.environ.get("AZURE_ATTESTER_URL") + "/attest/SgxEnclave?api-version=2022-08-01"
    data = {
        "quote": quote_str,
        "runtimeData": {
            "data": runtime_data_str,
            "dataType": "Binary"
        }
    }

    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
else:
    print("choose host or azure")
    exit()
input_data_path= "../pytorch_setup/client/input.jpg"
print(input_data_path)
input_data = measured_read.open_measured(input_data_path, "rb")
print(input_data.hasher.hexdigest())

model_path = "../pytorch_setup/server/alexnet-pretrained.pt"
print(model_path)
ml_model = measured_read.open_measured(model_path, "rb")
print(ml_model.hasher.hexdigest())
output_data= measured_read.open_measured("../pytorch_setup/server/result.txt", 'rb')
hasher = hashlib.sha512()
hasher.update(input_data.hasher.digest() + ml_model.hasher.digest() + output_data.hasher.digest())
with open("../pytorch_setup/server/quote.dat", "rb") as f:
    quote = f.read()
print("Checking if report_data matches input, model and output")
assert(
    quote[368:432].hex() == hasher.hexdigest()
)
print("OK")
print(f" Generated hash from input, model and output: {hasher.hexdigest()}")
print(f" REPORT_DATA_FIELD: {quote[368:432].hex()}")
print(f" MRENCLAVE: {quote[112:144].hex()}")
