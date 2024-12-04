# ML Algorithm

Library with some simple implementations of many known machine learning algorithms used over many different areas.

## Structure

Inside the `algorithms` folder you will find these sub-folders:

* ensemble: combination of multiple models in an attempt to improve the performance when compared to a single instance.
* reinforcement: used to mimic action-consequence scenarios where the algorithms will work by making decisions that will return awards or penalizations.
* supervised: provides algorithms that are expected to be trained on labeled data (input with a known output). These models will learn to map the provided inputs to its appropriate output.
* unsupervised: is expected to execute without a known output, these algorithms unfold patterns in the input data.

## Running

This package uses [uv](https://docs.astral.sh/uv) as its environment manager. Follow the steps [here](https://docs.astral.sh/uv/getting-started/installation/) to configure `uv` in your machine.

Once you have `uv`, it is a breeze to use the package, run the following command to install all necessary dependencies

```sh
uv sync
```

Now you can run any of the currently available algorithms using the command below:

```sh
python -m algorithms run <algorithm>
```

If you are not sure of which algorithms are available you can use the `--help` flag to display them to you:

```sh
python -m algorithms list
```