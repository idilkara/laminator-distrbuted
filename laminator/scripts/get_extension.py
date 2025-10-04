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

from cryptography import x509
from cryptography.hazmat.backends import default_backend
import base64

pck_certificate_der_base64 = """
MIIE8zCCBJmgAwIBAgIVAMEPCs5sJ751LFwvkwkZ38rNcs0EMAoGCCqGSM49BAMC
MHAxIjAgBgNVBAMMGUludGVsIFNHWCBQQ0sgUGxhdGZvcm0gQ0ExGjAYBgNVBAoM
EUludGVsIENvcnBvcmF0aW9uMRQwEgYDVQQHDAtTYW50YSBDbGFyYTELMAkGA1UE
CAwCQ0ExCzAJBgNVBAYTAlVTMB4XDTIzMDgyNDIxMzMyMVoXDTMwMDgyNDIxMzMy
MVowcDEiMCAGA1UEAwwZSW50ZWwgU0dYIFBDSyBDZXJ0aWZpY2F0ZTEaMBgGA1UE
CgwRSW50ZWwgQ29ycG9yYXRpb24xFDASBgNVBAcMC1NhbnRhIENsYXJhMQswCQYD
VQQIDAJDQTELMAkGA1UEBhMCVVMwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAATU
INpoisk51yvTDue+AyKBEyH59V3XQPwvrej+LoQs/sHxERREUK+YrnS+XIoQ2TvQ
naz74sITRVvvuTotKepNo4IDDjCCAwowHwYDVR0jBBgwFoAUlW9dzb0b4elAScnU
9DPOAVcL3lQwawYDVR0fBGQwYjBgoF6gXIZaaHR0cHM6Ly9hcGkudHJ1c3RlZHNl
cnZpY2VzLmludGVsLmNvbS9zZ3gvY2VydGlmaWNhdGlvbi92My9wY2tjcmw/Y2E9
cGxhdGZvcm0mZW5jb2Rpbmc9ZGVyMB0GA1UdDgQWBBQJrHmZu+rGZ/2dFwQ8V0yi
znjXCTAOBgNVHQ8BAf8EBAMCBsAwDAYDVR0TAQH/BAIwADCCAjsGCSqGSIb4TQEN
AQSCAiwwggIoMB4GCiqGSIb4TQENAQEEENmbMGyW+S6Wd3gC2SET0TQwggFlBgoq
hkiG+E0BDQECMIIBVTAQBgsqhkiG+E0BDQECAQIBDDAQBgsqhkiG+E0BDQECAgIB
DDAQBgsqhkiG+E0BDQECAwIBAzAQBgsqhkiG+E0BDQECBAIBAzARBgsqhkiG+E0B
DQECBQICAP8wEQYLKoZIhvhNAQ0BAgYCAgD/MBAGCyqGSIb4TQENAQIHAgEBMBAG
CyqGSIb4TQENAQIIAgEAMBAGCyqGSIb4TQENAQIJAgEAMBAGCyqGSIb4TQENAQIK
AgEAMBAGCyqGSIb4TQENAQILAgEAMBAGCyqGSIb4TQENAQIMAgEAMBAGCyqGSIb4
TQENAQINAgEAMBAGCyqGSIb4TQENAQIOAgEAMBAGCyqGSIb4TQENAQIPAgEAMBAG
CyqGSIb4TQENAQIQAgEAMBAGCyqGSIb4TQENAQIRAgENMB8GCyqGSIb4TQENAQIS
BBAMDAMD//8BAAAAAAAAAAAAMBAGCiqGSIb4TQENAQMEAgAAMBQGCiqGSIb4TQEN
AQQEBgBgagAAADAPBgoqhkiG+E0BDQEFCgEBMB4GCiqGSIb4TQENAQYEEAAa+Zmv
F0af/6zM4ELyY1UwRAYKKoZIhvhNAQ0BBzA2MBAGCyqGSIb4TQENAQcBAQH/MBAG
CyqGSIb4TQENAQcCAQEAMBAGCyqGSIb4TQENAQcDAQEAMAoGCCqGSM49BAMCA0gA
MEUCIQDRpTDsl/+glhSPbyTjOVbqryHXGcfRNKNXPH9BLh4PuAIgJ6moPBH7LNcJ
LTrZwxL+OdJuu6PTQ3cvHDroA3Al170=
"""
pck_certificate_der = base64.b64decode(pck_certificate_der_base64.strip())
pck_certificate = x509.load_der_x509_certificate(pck_certificate_der, default_backend())

fm_spc_oid = x509.ObjectIdentifier('1.2.840.113741.1.13.1')
fm_spc = None
try:
    extension = pck_certificate.extensions.get_extension_for_oid(fm_spc_oid)
    fm_spc = extension.value.value
except x509.ExtensionNotFound:
    print("FMSPC extension not found in the certificate")

# If FMSPC was found and extracted, display it
if fm_spc:
    # Assuming FMSPC is a direct string or needs decoding from bytes
    print("FMSPC:", fm_spc.hex())
else:
    print("Failed to extract FMSPC.")
