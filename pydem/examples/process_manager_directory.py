# -*- coding: utf-8 -*-
"""
   Copyright 2015 Creare

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
if __name__ == "__main__":
    import os
    from pydem.processing_manager import ProcessManager
    path = r'testtiff'

    # Make at least one test file
    if len(os.listdir(path)) == 0:
        from pydem.test_pydem import get_test_data
        get_test_data(13, 64)

        from pydem.utils import rename_files
        elev_source_files = [os.path.join(path, fn)
                                      for fn in os.listdir(path)
                                      if os.path.splitext(fn)[-1]
                                      in ['.tif', '.tiff'] and '_elev' in fn]
        other_source_files = [os.path.join(path, fn)
                                      for fn in os.listdir(path)
                                      if os.path.splitext(fn)[-1]
                                      in ['.tif', '.tiff'] and '_elev' not in fn]
        rename_files(elev_source_files)
        for fil in other_source_files:
            os.remove(fil)

    savepath = os.path.join('testtiff', 'processed_data')
    pm = ProcessManager(path, savepath)
    #pm._DEBUG = False
    pm.process()