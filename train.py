from MaLSTM import MaLSTM
from config import OUTPUT_UNITS, MAXLEN, NO_OF_UNIQUE_WORDS, EMBED_DIM, HOLDOUT_RATIO, BATCH_SIZE, LEARNING_RATE, NO_OF_ITERATIONS, SEED, ROOT_LOGDIR
from process_data import process_df

if __name__ == "__main__":
    #  import data in correct form
    X_sentence_1, X_sentence_2, X_sentence_1_length, X_sentence_2_length, y_data = process_df(MAXLEN)
    #  train the model
    model = MaLSTM(OUTPUT_UNITS, MAXLEN, NO_OF_UNIQUE_WORDS, EMBED_DIM)
    model.train(X_sentence_1, X_sentence_2, X_sentence_1_length, X_sentence_2_length, y_data,\
        HOLDOUT_RATIO, BATCH_SIZE, LEARNING_RATE, NO_OF_ITERATIONS, SEED, ROOT_LOGDIR)
