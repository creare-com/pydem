# Installation Instructions
NoteL: pyDEM currently only supports Python 2.7.

## Windows

### Install gdal:

* Using a python distribution like Anaconda
   
   ```conda install gdal```
   
* Or download from here: http://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
   
   ```pip install *.whl```

In both cases, if trying to import gdal fails, try installing this windows
update: http://www.microsoft.com/en-us/download/details.aspx?id=26368

### Install pyDEM:

With a python distribution like Anaconda use:
```pip install pyDEM```

### Install Taudem:

For pitremoval we use taudem in some of the examples. Also, for some of the 
comparisons we use taudem. For windows, Taudem can be found [here](http://svn.mapwindow.org/svnroot/MapWindow4Dev/Bin/Taudem5Exe/).
Download everything to the ```<ANACONDA_ROOT>\<lib><site-packages>\pydem\taudem\taudem_Windows``` directory.

## Linux

```
conda install gdal

pip install pyDEM
```

# Developers Instructions

Clone the git repository to a directory. Run:
```
python setup.py develop
```

to install the developer version to the system. 

If using pyDEM just from the local directory use:

```
python setup.py build_ext --inplace
```
