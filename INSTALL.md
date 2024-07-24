# Installation Instructions

## Windows:
We recommend using Windows Subsystem for Linux (WSL). See below.

## Linux / WSL

You can use various flavors of a Python virtual environment. Here's an example using `pyenv`:
```
pyenv virtualenv 3.11.2 pydem
pyenv activate pydem
pip install numpy cython
pip install pydem
```

PyDEM published a "many linux wheel", which is a binary file. In case you need to actually compile the Cython functions, you'll likely need to install a compiler. For example, on Ubuntu use:

```bash
apt update
apt install build-essential
```

## Docker
See `docker/pydem_user_guide.md` for using `PyDEM` with Docker.

# Developers Instructions

Clone the git repository to a directory. You can set up a virtual environment if you like (see Linux / WSL install instructions). After that, run:
```
pip install numpy cython
pip install -e .
```

to install the developer version to the system.

If using pyDEM and changing the Cython functions, use:

```
python setup.py build_ext --inplace
```

to rebuild the Cython extensions.
