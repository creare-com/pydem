from __future__ import division, unicode_literals, print_function, absolute_import

import json
import boto3
import subprocess
import sys, os
if sys.version_info.major == 2:
    import urllib
else:
    import urllib.parse as urllib
sys.path.append('/tmp')

api_root = 'https://.'
s3_bucket = 'twi-processing'
s3 = boto3.client('s3')
deps = 'pydem_deps.zip'

def return_exception(e, event, context):
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
                + '<h1>Exception</h1><br><br><br>' + str(e),
        'isBase64Encoded': False,
    }

def getDeps():
    # Download additional dependencies
    if get_deps:
        s3.download_file(s3_bucket, 'pydem/' + deps, '/tmp/' + deps)

        subprocess.call(['unzip', '/tmp/' + deps, '-d', '/tmp'])
        subprocess.call(['rm', '/tmp/' + deps])

def handler(event, context, callback, get_deps=True):

    try:
        srcKey = urllib.unquote(event.Records[0].s3.object.key.replace(r'/\+/g', " "))
        # Infer the file type.
        typeMatch = srcKey.match(r'/\.([^.]*)$/')
        if !typeMatch:
            callback("Could not determine the file type.")
            return
        fileType = typeMatch[1]
        if (fileType != "tiff" && fileType != "tif" && fileType != "npz"):
            callback('Unsupported file type: ' + fileType)
            return

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
            s3.download_file(s3_bucket, 'pydem/' + srcKey, '/tmp/' + srcKey) # raw tiff file
            getDeps()
            from pydem.dem_processing import DEMProcessor
            demproc = DEMProcessor('/tmp/' + srcKey)
            demproc.calc_slopes_directions()
            # TODO Save slope/direction to file and upload to s3, see
            # processing_manager.tile_edge.update_edges(esfile, dem_proc) or
            # save/load array in DEMProcessor
            demproc.save_elevation('/tmp', raw=True)
            demproc.save_direction('/tmp', raw=True)
            demproc.save_slope('/tmp', raw=True)
            s3.upload_file(s3_bucket, 'pydem/' + )


    except Exception as e:
        return return_exception(e, event, context, pipeline)

if __name__ == "__main__":
    # test
