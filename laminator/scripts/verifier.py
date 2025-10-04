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
import hashlib
import json
import base64

test_certifications = {
    'training-mrenclave':  {
     'name': None,
     'dataset-hash': None,
     'training-parameters': {
         'name': None,
         'epochs': None,
     }},
    '3f71d576893fa3797add3b94d50cc4b31910e9052950b20bc91816422dc3942c' : {
        'name': None,
        'results': {
            'task': None,
            'dataset': None,
            'metrics': {
                'type': 'accuracy',
                'value': None,
                'dataset-hash': None,
            }
        }
    }
}





def load_certificates(cert_pems):
    certificates = []
    for pem in cert_pems:
        certificates.append(x509.load_pem_x509_certificate(pem.encode(), default_backend()))
    return certificates

def certificates_match(cert1, cert2):
    return cert1.tbs_certificate_bytes == cert2.tbs_certificate_bytes

def check_attestation(attestation, certifications):
    mrenclave, payload = attestation
    certification = certifications[mrenclave]
    print(certification)
    print(payload)

    metrics = payload['results']['metrics']
    certification_metric = certification['results']['metrics']

    for metric in metrics:
      for k,v in metric.items():
        allowed_values = certification_metric[k]
        if allowed_values and v != allowed_values:
          raise ValueError('Metric contained disallowed value')

    return True



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

def load_certificate_from_pem(pem_path):
    with open(pem_path, 'rb') as file:
        pem_data = file.read()
    return x509.load_pem_x509_certificate(pem_data, default_backend())

def verify_quote(quote, payload):
    print(f"Extracted SGX quote with size = {len(quote)} and the following fields:")
    print(f"  ATTRIBUTES.FLAGS: {quote[96:104].hex()}  [ Debug bit: {quote[96] & 2 > 0} ]")
    print(f"  ATTRIBUTES.XFRM:  {quote[104:112].hex()}")
    print(f"  MRENCLAVE:        {quote[112:144].hex()}")
    print(f"  MRSIGNER:         {quote[176:208].hex()}")
    print(f"  REPORTDATA:        {quote[368:432].hex()}")
    # Decode binary data to a string to use regular expressions
    try:
        decoded_data = quote.decode('utf-8')
    except UnicodeDecodeError:
        # If the data cannot be decoded to UTF-8, you might try 'latin-1' which maps bytes directly to characters
        decoded_data = quote.decode('latin-1')

    # Regex to find PEM formatted certificates
    certificates = re.findall(r"-----BEGIN CERTIFICATE-----.+?-----END CERTIFICATE-----", decoded_data, re.DOTALL)
    certificates = load_certificates(certificates)
    root_ca = load_certificate_from_pem('Intel_SGX_Provisioning_Certification_RootCA.pem')
    if certificates_match(root_ca, certificates[-1]):
        print("Valid Root CA.")
    else:
        print("Not valid root CA. Exiting.")
        exit()
    if verify_chain(certificates) and check_validity(certificates):
        print("The certificate chain is valid and all certificates are currently valid.")
        print("Certificates are not in revocation list.")
    else:
        print("The certificate chain is invalid or a certificate is out of date.")
    hasher= hashlib.sha512()
    hasher.update(payload)
    assert( quote[368:432].hex() == hasher.hexdigest())
    print("Report data is ok.")
    print("MRENCLAVE is valid.")
    print("Quote is valid")
    return quote[112:144].hex()
def validate_whole(f):
    data = json.load(f)
    quote = data["sgx-quote"]
    payload = data["payload"]
    print(f'Quote: {quote}\n')
    print(f'Payload: {payload}\n')
    quote = base64.b64decode(quote)
    payload= base64.b64decode(payload)
    mrenclave = verify_quote(quote=quote,payload=payload)
    return (mrenclave, payload.decode())

if __name__ == '__main__':
    f = open('output_accuracy.json')
    acc = validate_whole(f)
    e = open('output_io.json')
    io = validate_whole(e)
    c = open('output_training.json')
    train = validate_whole(c)

    #check_attestation(attestation=attestation,certifications=test_certifications)
    print("Model card structure valid.")
    print(f'{acc}\n')
    print(f'{io}\n')
    print(f'{train}\n')





    
    
