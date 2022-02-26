# kevlar

The Kevlar Project.

## Installation:
```
conda update conda
conda create -n kevlar-revamp python=3.9.7 anaconda
conda activate kevlar-revamp
conda install pybind11
git clone git@github.com:mikesklar/kevlar.git
cd kevlar/
git checkout james.yang/kevlar_revamp
```
Refer to ./src/kevlar/README.md to install C++ driver deps
```
cd python/
pip install -vvv -e .
```

## Reinstall
```
cd python/
rm -rf ./build ./dist ./*egg-info
find . -type f -name "*.so" -exec rm {} \;
find . -type f -name "*.o" -exec rm {} \;
pip install -vvv -e .
```

# Smoke test
```
PYTHONPATH=. time python ./examples/fit_driver_example.py
```