# Deep Comedy

## Introduction
This project was developed for the Deep Learning course held by professor Andrea Asperti at University of Bologna. 

Our goal was to build a neural model capable of producing the correct syllabification of Italian poetry, and using this same model to produce poetry in hendecasyllables in the style of Dante.

We only used the Divine Comedy as training set. For the syllabified version we rely on the outputs of [this project](https://github.com/asperti/Dante) from professor Asperti.

## Results
We provide three notebooks which explain our model and the results we obtained:
- **Initial syllabification experiments** --> contains our first experiments with syllabification using the transformer architecture
- **Char2Char generation and syllabification** --> uses a transformer architecture for both syllabification and text generation; as the name suggests, both encoder and decoder work at character-level.
- **Word2Char generation** --> we tried to improve the semantics of generated text using a word-level encoder, however the results are only slightly better

The `deepcomedy` folder contains the custom libraries we use in the notebooks.

The `nlgpoetry` folder contains an alternative syllabification algorithm from [Neural Poetry](https://gitlab.com/zugo91/nlgpoetry). We used this for comparison.

In `outputs` we provide a syllabification of the "Orlando Furioso" by Ludovico Ariosto, obtained using the Char2Char model.

For a deep dive check the `docs` folder, which contains a report of our experiments and discoveries.

TODOs:
- Using a syllable-based decoder for generation
- Longer training
- Bigger models :)

## Running the code

### With poetry

We manage dependencies using [poetry](https://python-poetry.org/).

You can create a virtual environment and install all dependencies using the following poetry command:
```
poetry install
```

Then activate the environment with:
```
poetry shell
```

Then you should be able to run the notebooks.

### Without poetry
We also provide the freezed dependencies in the `requirements.txt` file. We highly recommend installing them in a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
``` 

## Credits
Some of the code is adapted from these sources:
- [Transformer model for language understanding](https://www.tensorflow.org/text/tutorials/transformer) from Tensorflow
- https://github.com/AlessandroLiscio/DeepComedy for some metrics
- [Neural Poetry](https://gitlab.com/zugo91/nlgpoetry) for some metrics and the alternative syllabification
- [Overleaf](https://www.overleaf.com/) as LaTeX editor for the report
