# MarkLogic CNTK sample functions
Sample functions for NLP using CNTK of MarkLogic10.

## Overview
The following functions are implemented by XQuery.
These functions uses word2vec ONNX model learned using the Japanese Wikipedia.
This sample model has learned about 350,000 Japanese words.

### cosine-distance.xqy

This sample calculate the cosine distance between two words learned by word2vec.

The following MarkLogic CNTK functions are mainly used.

  - cntk:one-hot-op
  - cntk:embedding-layer
  - cntk:cosine-distance
  - cntk:batch-of-sequences
  - cntk:evaluate

### top-k.xqy

This sample gets the top-k values from the argument array. 

The following MarkLogic CNTK functions are mainly used.

  - cntk:top-k
  - cntk:evaluate

### most-similar.xqy

This sample searches for similar words learned by word2vec.

The following MarkLogic CNTK functions are mainly used.

  - cntk:one-hot-op
  - cntk:embedding-layer
  - cntk:cosine-distance
  - cntk:batch-of-sequences
  - cntk:evaluate

# word2vec model and vocabulary list
The sample codes above uses the following word2vec ONNX model and vocabulary list.

* wikipedia_w2v_model.onnx

This is a sample ONNX model generated using Keras with CNTK.
This model was learned over 350,000 Japanse words on wikipedia.

* wikipedia_vocab.csv
This is a list of word indices. Load this file into the MarkLogic in JSON format by MLCP.

# How to use
1. Load wikipedia_vocab.csv to MarkLogic

     Convert this file to JSON format, load it under /vocab/.
     See load_vocab.sh for the MLCP command.
  
2. Load wikipedia_w2v_model.onnx

     Load this file into the MarkLogic under /model/.
     See load_model.sh for the MLCP command.

3. Execute XQuery samples
