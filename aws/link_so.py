import os

with open('so_files.txt', 'r') as fid:
    sf = fid.read().split('\n')

links ={}
for s in sf:
    key = s.split('/')[-1]
    links[key] = links.get(key, []) + [s]

for l,v in links.items():
    if len(v) > 1:
        for f in v[1:]:
            os.remove(f)
            os.symlink(v[0], f)
