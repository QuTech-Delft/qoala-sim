# Qoala Simulator
A comprehensive and configurable software library for simulating quantum network applications and nodes. It allows users to mimic both the software and hardware of real quantum network nodes, execute quantum programs, and gather statistics. This flexibility enables users to test and optimize various hardware parameters, investigate their effects on application performance and overall network performance, and even prototype new quantum network node architectures.


## Qoala
Qoala is an execution environment specification which describes:
- the compiled format of programs that run on processing nodes (or end nodes) in a quantum network
- the software/hardware architecture of processing nodes and how they execute programs

This simulator provides tools for simulating a network in which nodes have implemented the Qoala specification.
Processing nodes and their internal components and behavior are presented as configurable classes, which can be
connected and used to mimick both the software and hardware of real quantum network nodes. This mimicked software
and hardware can then execute quantum programs and statistics can be obtained.

With these tools, simulations may be run in order to (among other things):
- run and obtain results of quantum network applications
- investigate the effect of changing hardware parameters on the performance of specific applications
- investigate the effect of changing hardware parameters on the overall performance of the quantum network
- test or create new prototypes of quantum network node architectures


## Installation

### Prerequisites
The Qoala Simulator uses the [NetSquid](https://netsquid.org/) Python package.
To install and use NetSquid, you need to first create an account for it.
The username and password for this account are also needed to install `qoala-sim`.

### From PyPI
The Qoala Simulator is available as [a package on PyPI](https://pypi.org/project/qoala/) and can be installed with
```
pip install qoala --extra-index-url=https://pypi.netsquid.org
```

This will prompt for your NetSquid account name and password.

### From source
Clone this repository and make an editable install with

```
pip install -e . --extra-index-url=https://pypi.netsquid.org
```
which will prompts for your NetSquid account name and password.

Additionally, you may want to install the extra `dev` packackes, so you can run the tests and linter:

```
pip install -e .[dev] --extra-index-url=https://pypi.netsquid.org
```

You can also use the `make install` and `make install-dev` Makefile commands.
These commands require you to have the `NETSQUIDPYPI_USER` and
`NETSQUIDPYPI_PWD` environment variables set to your NetSquid username and password respectively.

To verify the installation and run all tests and examples:
```sh
make verify
```

If this command completes without any errors, and instead ends with a message saying `Everything works!`, then everything is set up correctly.

## Usage
The Qoala Simulator provides building blocks for many (sub)components of quantum network nodes, which may be used in any way desired.
For example, one may simply create a single processing node object and only run local quantum programs on it.
Or, one might create multiple processing nodes and connect them in a certain way, and also provides many quantum network applications as input
for these, in order to test different scheduling techniques.

A typical way this package is used is in the form of a Python script that contains code for setting up a simulation and then running it.
Such a script hence `import`s from the `qoala` package the objects and functions that are required for whatever simulation is desired.

See the `examples` directory for examples.

Additionally, one may look into the `tests` directory for code that uses only subcomponents of quantum networks.
The `tests/integration` directory does include more comprehensive simulations, e.g. simulating execution of full quantum network applications.
