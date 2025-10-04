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

import json


with open("tcbSignChain.json") as f:
    json_input = f.read()

data = json.loads(json_input)
issuer_chain = data['SGX-TCB-Info-Issuer-Chain'].replace("%20", " ").replace("%0A", "\n").replace("%2B", "+").replace("%2F", "/").replace("%3D", "=")


with open('../quote_verifier/tcbSignChain.pem', 'w') as f:
    f.write(issuer_chain)

print("Certificates saved as 'pck_cert.pem' and 'issuer_chain.pem'.")
