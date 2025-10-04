#!/usr/bin/env python3

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

test_payloads = [
    ('training-mrenclave', {
     'name': 'abc1234',
     'dataset-hash': 'def5678',
     'training-parameters': {
         'foo': 1,
         'bar': 2
     }}),
    ('accuracy-mrenclave', {
        'name': 'abc1234',
        'results': [{
            'metrics': [
                {'type': 'accuracy', 'value': '0.9', 'dataset-hash': 'def5678'}
            ],
        }]}),
     ('accuracy-mrenclave', {
     'name': 'abc1234',
         'results': [{
             'metrics': [
                 {'type': 'fairness', 'value': '0.9', 'dataset-hash': 'def5678'}
             ]
         }]
     }),
    ]

test_certifications = {
    'training-mrenclave': {
        'name': None,
        'results': {
            'task': None,
            'dataset': None,
            'metrics': {
                'type': 'training',
                'value': None,
            }
        }
    },
    'accuracy-mrenclave': {
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


def check_spec(tree, spec):
    if spec is None:
        return True
    else:
        if isinstance(tree, list):
            for obj in tree:
                if not check_spec(obj, spec):
                    return False
            return True

        elif isinstance(spec, str) or isinstance(spec, bytes):
            if tree == spec:
                return True
            else:
                print("%s did not match %s" % (tree, spec))
                return False

        elif isinstance(spec, dict):
            if not isinstance(tree, dict):
                print("Expected dict, got %s" % str(tree))
                return False
            else:
                for k,v in tree.items():
                    if k in spec:
                        if not check_spec(v, spec[k]):
                            print("No match for %s" % k)
                            return False
                    else:
                        print("Unexpected key %s" % k)
                        return False


                return True

        else:
            raise ValueError('Invalid specification "%s"' % str(spec))


def check_attestation(attestation, certifications):
    mrenclave, payload = attestation
    certification = certifications[mrenclave]


    metrics = payload['metrics']
    certification_metric = certification['metrics']

    for metric in metrics:
      for k,v in metric.items():
        allowed_values = certification_metric[k]
        if allowed_values and v != allowed_values:
          raise ValueError('Metric contained disallowed value')

    return True

if __name__ == '__main__':
    for payload in test_payloads:
        try:
            check_attestation(payload, test_certifications)
            print("Success")
        except Exception as e:
            print("Rejected: %s" % str(e))
