# nanograd

![The most beautiful creature God ever made.](https://www.zoo-berlin.de/fileadmin/user_upload/header_flachlandtapir.jpg)

An unapologetic mock-off of the awesome [micrograd](https://github.com/karpathy/micrograd). This repo is mostly a learning exercise for me to get a deeper understanding of the inner works of PyTorch. All credits go to the original author.

## Some small changes compared to micrograd

* stricter typing
* alternative algorithm to build up topological order of DAG
* slightly different API
* extensive test suite

## Setup

Install dependencies with poetry:

```
poetry install
```

Enter the virtual environment:

```
poetry shell
```

Run tests:

```
poetry run pytest
```

