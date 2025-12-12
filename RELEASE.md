# Release

How to release `pydem`

## Updating main

1. Create a release branch: `release/x.y.z`.

2. Ensure unit tests are passing and new ones have been added.

3. Update `pyproject.toml`, `__init__.py`, and `CHANGELOG.md` with the new version. Make sure to also update the changelog with the new changes.

4. Merge changes into main and then tag the release (from main).

5. Build source and binary wheels for pypi. There are also instructions available here: https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#packaging-your-project. To build, run the following:

```bash
$ git clean -xdf  # this deletes all uncommited changes!
$ python setup.py bdist_wheel sdist
```

6. Repair the wheels to be compatible with manylinux format. It will look something like this:

```bash
$ cd dist
$ auditwheel pydem-1.2.0-cp311-cp311-linux_x86_64.whl
```

7. Upload package to [TestPypi](https://packaging.python.org/guides/using-testpypi/). You will need to be listed as a package owner at
https://pypi.python.org/pypi/pydem for this to work. You now need to use a pypi generated token, can no longer use your password. 

```bash
$ twine upload --repository-url https://test.pypi.org/legacy/ dist/pydem*
```

Note you may need to move files to make sure the manylinux .whl and .tar.gz file are in the same location. You only want to upload those two files and not hte originally produced .whl file.

8. Use twine to register and upload the release on pypi. Be careful, you can't
take this back! You will need to be listed as a package owner at
https://pypi.python.org/pypi/pydem for this to work. Again, make sure you just upload the .tar.gz and the manylinux version of the .whl file.

```bash
$ twine upload dist/pydem*
```
