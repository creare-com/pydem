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

import os

with open('so_files.txt', 'r') as fid:
    sf = fid.read().split('\n')

links ={}
for s in sf:
    key = s.split('/')[-1]
    links[key] = links.get(key, []) + [s]

for l,v in list(links.items()):
    if len(v) > 1:
        for f in v[1:]:
            os.remove(f)
            os.symlink(v[0], f)
