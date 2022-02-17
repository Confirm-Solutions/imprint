# kevlar

The Kevlar Project.

## Installation instructions:

git clone git@gitlab.com:libeigen/eigen.git
cd eigen/
git checkout 3.4
cd ..
conda update conda
conda create -n kevlar-revamp python=3.9.7 anaconda
conda activate kevlar-revamp
conda install pybind11
git clone git@github.com:mikesklar/kevlar.git
cd kevlar/
git checkout james.yang/kevlar_revamp
cd python/
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:../../eigen
python setup.py install # review compiler opts
pip install -vvv -e .

## Reinstall

cd python/
rm -rf ./build ./dist ./*egg-info
find . -type f -name "*.so" -exec rm {} \;
find . -type f -name "*.o" -exec rm {} \;
python setup.py install # review compiler opts
pip install -vvv -e .

# Smoke test

export PYTHONPATH=.
time python ./examples/fit_driver_example.py