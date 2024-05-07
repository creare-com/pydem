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

import json
import boto3
import botocore
import subprocess
import sys, os
if sys.version_info.major == 2:
    import urllib.request, urllib.parse, urllib.error
else:
    import urllib.parse as urllib
if not os.path.isdir('/tmp/'):
    os.mkdir('/tmp/')
sys.path.append('/tmp/')

s3 = boto3.resource('s3')

api_root = 'https://.'
s3_bucket = 'twi-processing'
deps = 'pydem_deps.zip'

def return_exception(e, event, context):
    print(e)
    exc_type, exc_obj, tb = sys.exc_info()
    line_no = tb.tb_lineno
    events = str(event)
    contexts = str(context)
    try:
        events = json.dumps(event, sort_keys=True, indent=2, separators=(',', ': '))
        contexts = json.dumps(contexts, sort_keys=True, indent=2, separators=(',', ': '))
    except:
        pass

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html',
            'Access-Control-Allow-Origin': '*',
        },
        'body': '<h1>Event</h1><br><br><br>' + str(event)\
                + '<h1>Context</h1><br><br><br>' + str(context)
                + '<h1>Exception</h1><br><br><br>' + str(e)
                + '<h1>Line number</h1><br><br><br>' + str(line_no),
        'isBase64Encoded': False,
    }

def getDeps():
    # Download additional dependencies
    # if get_deps: # Not sure what this flag is supposed to be.
    srcKey = 'pydem/' + deps
    try:
        s3.Bucket(s3_bucket).download_file(srcKey, '/tmp/' + deps)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

    print("got deps zip, time to unzip it...")
    subprocess.call(['unzip', '/tmp/' + deps, '-d', '/tmp/'])
    sys.path.append('/tmp/GDAL-1.11.3-py2.7-linux-x86_64.egg')
    subprocess.call(['rm', '/tmp/' + deps])

def lambda_handler(event=None, context=None, callback=None, get_deps=True):
    print("Printing sys.path at start of handler.")
    print(sys.path)
    if event is None:
        try:
            with open('event.json') as f:
                event = json.load(f)
        except Exception as e:
            return return_exception(e, event, context)
    else:
        print("Event is not None - see it below:")
    try:
        print(json.dumps(event))
        srcKey = urllib.parse.unquote(event['Records'][0]['s3']['object']['key'].replace(r'/\+/g', " "))
        print("srcKey = " + srcKey + " at line 78")
        # Infer the file type.
        filename, fileType = os.path.splitext(srcKey)
        if not fileType:
            if callback:
                callback("Could not determine the file type.")
                return
            else:
                raise Exception("Uh oh, could not determine file type. srcKey = " + str(srcKey))
        fileType = fileType.upper()
        if ("TIFF" not in fileType and "TIF" not in fileType and "NPZ" not in fileType):
            if callback:
                callback('Unsupported file type: ' + fileType)
                return
            else:
                raise Exception("Uh oh, unsupported file type. fileType = " + str(fileType))

        if 'elev' in srcKey:
            # Do nothing
            return
        elif 'mag' in srcKey:
            s3.download_file(s3_bucket, 'pydem/' + srcKey, '/tmp/' + srcKey) # mag npz file
            elev = srcKey.replace('mag', 'elev')
            ang = srcKey.replace('mag', 'ang')
            s3.download_file(s3_bucket, 'data/' + elev, '/tmp/' + elev) # elev npz file
            s3.download_file(s3_bucket, 'data/' + ang, '/tmp/' + ang) # ang npz file
            getDeps()
            from pydem.dem_processing import DEMProcessor

            # TODO create dem_proc, set slope/ang/elev to files downloaded from s3
            # then calculate TWI. Something like:
            #
            # demproc = DEMProcessor('/tmp/' + srcKey.replace('mag', ''))
            #
            # demproc.elev = '/tmp/' + elev # ?
            # demproc.ang = '/tmp/' + ang # ?
            # demproc.mag = '/tmp/' + mag # ?
            #
            # uca, twi = demproc.calc_twi()
        elif 'ang' in srcKey:
            # Ignore this - handled by mag case
            return
        elif 'uca' in srcKey:
            # Do nothing
            return
        elif 'twi' in srcKey:
            # Do nothing
            return
        elif 'tif' in srcKey or 'tiff' in srcKey:
            # This is the case of a raw tiff file being uploaded.
            try:
                print("Trying to get tif file from s3")
                s3.Bucket(s3_bucket).download_file(srcKey, '/tmp/' + filename.split('/')[1])
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("The object does not exist.")
                else:
                    raise
            print("Got tif file from s3, now to getDeps()")
            getDeps()
            print("Got past getDeps() call")
            print("Printing new sys.path")
            print(sys.path)
            from pydem.dem_processing import DEMProcessor
            print("Got past import pydem!")

            # demproc = DEMProcessor('/tmp/' + srcKey)
            # demproc.calc_slopes_directions()

            # TODO Save slope/direction to file and upload to s3, see
            # processing_manager.tile_edge.update_edges(esfile, dem_proc) or
            # save/load array in DEMProcessor

            # demproc.save_elevation('/tmp', raw=True)
            # demproc.save_direction('/tmp', raw=True)
            # demproc.save_slope('/tmp', raw=True)

            s3.upload_file('/tmp/' + filename.split('/')[1], s3_bucket, 'DSullivan/mag.tif')
            print("Got past upload mag tif file!!!")
            return


    except Exception as e:
        return return_exception(e, event, context)

# print(lambda_handler())
