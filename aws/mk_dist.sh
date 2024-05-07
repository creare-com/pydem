#!bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

cd ~

sudo yum groupinstall "Development Tools" -q -y

# checkout PyDEM
git clone https://github.com/creare-com/pydem.git

# Create the virtualenv now so gdal can find it
virtualenv env_pip2

export PYTHONPATH="/home/ec2-user/env_pip2/lib64/python2.7/site-packages/:/home/ec2-user/env_pip2/lib64/python2.7/dist-packages/"

wget https://github.com/OSGeo/proj.4/archive/4.9.2.tar.gz
tar -zvxf 4.9.2.tar.gz
cd proj.4-4.9.2/
./configure --prefix=/home/ec2-user/env_pip2/lib64/python2.7/site-packages
make
make install

wget http://download.osgeo.org/gdal/1.11.3/gdal-1.11.3.tar.gz
tar -xzvf gdal-1.11.3.tar.gz
cd gdal-1.11.3
./configure --prefix=/home/ec2-user/env_pip2/lib64/python2.7/site-packages \
            --with-geos=/home/ec2-user/env_pip2/local/bin/geos-config \
            --with-static-proj4=/home/ec2-user/env_pip2/local \
            --with-python=/home/ec2-user/env_pip2/bin/python
make
make install

export LD_LIBRARY_PATH="/home/ec2-user/env_pip2/lib64/python2.7/site-packages/lib/:$LD_LIBRARY_PATH"
sudo ldconfig

cd ~

source env_pip2/bin/activate
pip install --upgrade pip
pip install boto3
pip install cython
sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=1024
sudo /sbin/mkswap /var/swap.1
sudo /sbin/swapon /var/swap.1
pip install scipy
sudo swapoff /var/swap.1
sudo rm /var/swap.1
export GDAL_CONFIG=/home/ec2-user/env_pip2/local/bin/gdal-config
pip install pyDEM
deactivate

cd ~

# Copy python packages from virtual environment for manipulation
mkdir dist
cd dist
cp -r ~/env_pip2/lib/python2.7/dist-packages/* .
cp -r ~/env_pip2/lib64/python2.7/dist-packages/* .
cp -r ~/env_pip2/lib/python2.7/site-packages/* .
cp -r ~/env_pip2/lib64/python2.7/site-packages/* .

# Remove the egg-info directories
rm -r *-info*

# Remove any .pyc files
find . -name "*.pyc" | xargs rm

# Link any repeated .so files together, and delete repeats
cp ~/aws/link_so.py .
find . -name "*.so" > so_files.txt
python link_so.py
rm so_files.txt
rm link_so.py

# Add the podpac library
cp -r ~/pydem/pydem .
cp -r ~/aws/handler.py .

# Zip all the directories, and the files in the current directory
find * -maxdepth 0 -type f | grep ".zip" -v | grep -v ".pyc" | xargs zip -9 -rqy pydem_dist.zip
find * -maxdepth 0 -type d -exec zip -9 -rqy {}.zip {} \;

# Figure out the package sizes (for python script)
du -s *.zip > zip_package_sizes.txt
du -s * | grep .zip -v > package_sizes.txt

# Run python script to assemble zip files, and upload to s3
cp ~/aws/mk_dist.py .
python mk_dist.py
