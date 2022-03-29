# Kevlar

The Kevlar Project.

## Dependencies

- [pre-commit](https://pre-commit.com/)

## Install

Run the following commands:
```
git clone git@github.com:mikesklar/kevlar.git
cd kevlar/
pre-commit install
./generate_bazelrc
```

From here, we refer to the installation instructions
for each of the sub-components:

- [PyKevlar](./python/README.md): Kevlar Python package.
- [kevlar](./kevlar/README.md): Kevlar C++ core engine.
