.RECIPEPREFIX := >

install:
>python -m pip install --upgrade pip
>pip install -r requirements.txt

lint:
>black --check src

format:
>black src

test:
>python src/test_pipeline.py

train:
>python src/train.py

all: install lint test train