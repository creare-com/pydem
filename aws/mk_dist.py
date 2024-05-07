#!python

"""
   Copyright 2015-2024 Creare

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import boto3
import subprocess

with open('zip_package_sizes.txt', 'r') as fid:
    zps = fid.read()

with open('package_sizes.txt', 'r') as fid:
    ps = fid.read()

def parse_ps(ps):
    lns = ps.split('\n')
    pkgs = {}
    for ln in lns:
        try:
            parts = ln.split('\t')
            pkgs[parts[1]] = int(parts[0])
        except:
            pass
    return pkgs

pgz = parse_ps(zps)
pg = parse_ps(ps)

data = {}
for p, s in list(pgz.items()):
    os = pg.get(p[:-4], 0)
    data[p] = {"zip_size": s, "size": os, 'ratio':os*1.0/s}

sdata = sorted(list(data.items()), key=lambda t: t[1]['ratio'])

zipsize = data['pydem_dist.zip']['zip_size']
totsize = sum([pg[k] for k in pg if (k + '.zip') not in list(pgz.keys())])
pkgs = []
for val in sdata[::-1]:
    if val[0] == 'pydem_dist.zip':
        continue
    key = val[0]
    pkgs.append(key)

    zipsize += data[key]['zip_size']
    totsize += data[key]['size']

    if (zipsize > 100000 or totsize > 250000):
        k = pkgs.pop()
        zipsize -= data[k]['zip_size']
        totsize -= data[k]['size']

core = [k[:-4] for k in pkgs if k != 'pydem_dist.zip']
deps = [k[:-4] for k in data if k[:-4] not in core and k != 'pydem_dist.zip']
dep_size = sum([data[k+'.zip']['size'] for k in deps])
dep_size_zip = sum([data[k+'.zip']['zip_size'] for k in deps])

# add core to pydem_dist.zip
cmd = ['zip', '-9', '-rq', 'pydem_dist.zip'] + core
subprocess.call(cmd)
cmd = ['zip', '-9', '-rqy', 'pydem_deps.zip'] + deps
subprocess.call(cmd)

# upload zip files to s3
bucket = 'twi-processing'
s3 = boto3.client('s3')
s3.upload_file('pydem_dist.zip', bucket, 'pydem/pydem_dist.zip')
s3.upload_file('pydem_deps.zip', bucket, 'pydem/pydem_deps.zip')
