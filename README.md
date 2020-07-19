# MaLSTM

This project aims to largely replicate the architecture of the model outlined in "Siamese Recurrent Architectures for Learning Sentence Similarity" by Jonas 
Mueller and Aditya Thyagarajan. However, instead of using pre-trained word embeddings, we train an embedding layer for the vocabulary as part of the training routine.

The arguments of the model can be modified by editing "MaLSTM.conf". Specifically, the supported arguments are:

  1. OUTPUT_UNITS: dimension of the LSTM's final output state 
  2. MAXLEN: maximum sequence length to be considered by the LSTM
  3. NO_OF_UNIQUE_WORDS: number of unique words in the vocabulary
  4. EMBED_DIM: dimension of each word embedding vector
  5. HOLDOUT_RATIO: proportion of the dataset to be used for hyperparameter tuning 
  6. BATCH_SIZE: number of examples per minibatch
  7. LEARNING_RATE: learning rate
  8. NO_OF_ITERATIONS: total number of iterations for model training. Note that 1 epoch = (dataset_size//batch_size) iterations.
  9. SEED: random seed to ensure reproducibility
  10. ROOT_LOGDIR: folder in which model checkpoints are to be saved 

