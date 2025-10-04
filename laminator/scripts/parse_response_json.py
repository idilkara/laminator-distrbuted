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

import re
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from datetime import datetime, timezone
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')



# Binary data
binary_data = b'0'

# with open('')

# Decode binary data to a string to use regular expressions
try:
    decoded_data = binary_data.decode('utf-8')
except UnicodeDecodeError:
    # If the data cannot be decoded to UTF-8, you might try 'latin-1' which maps bytes directly to characters
    decoded_data = binary_data.decode('latin-1')

# Regex to find PEM formatted certificates
certificates = re.findall(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", decoded_data, re.DOTALL)


# Print found certificates

for i, cert in enumerate(certificates, start=1):
    print(f"Certificate {i}:\n{cert}\n")



def load_certificates(cert_pems):
    certificates = []
    for pem in cert_pems:
        certificates.append(x509.load_pem_x509_certificate(pem.encode(), default_backend()))
    return certificates

def verify_chain(certificates):
    try:
        for i in range(len(certificates) - 1):
            issuer_cert = certificates[i + 1]
            issuer_cert.public_key().verify(
                certificates[i].signature,
                certificates[i].tbs_certificate_bytes,
                ec.ECDSA(hashes.SHA256())
            )
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False

def check_validity(certificates):
    now = datetime.now(timezone.utc)
    for cert in certificates:
        if not (cert.not_valid_before.replace(tzinfo=timezone.utc) <= now <= cert.not_valid_after.replace(tzinfo=timezone.utc)):
            print(f"Certificate with subject {cert.subject.rfc4514_string()} is invalid.")
            print(f"Valid from {cert.not_valid_before} to {cert.not_valid_after}. Current time: {now}.")
            return False
    return True

certificates = load_certificates(certificates)
if verify_chain(certificates) and check_validity(certificates):
    print("The certificate chain is valid and all certificates are currently valid.")
else:
    print("The certificate chain is invalid or a certificate is out of date.")
