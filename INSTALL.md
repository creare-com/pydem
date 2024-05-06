# Installation Instructions


## Windows

### Install rasterio:

* Using a python distribution like Anaconda

   ```conda install rasterio```

### Install pyDEM:

With a python distribution like Anaconda use:
```pip install pydem```


## Linux

```
conda install rasterio

pip install pydem
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