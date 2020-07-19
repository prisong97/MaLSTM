import pandas as pd
import tensorflow as tf
from os.path import join
import itertools
from utils import batch_generator

class MaLSTM:
    """
    This largely replicates the architecture of the MaLSTM (without calibration) outlined
    in the paper 'Siamese Recurrent Architectures for Learning Sentence Similarity' by Jonas 
    Mueller and Aditya Thyagarajan.
    
    However, we train an embedding layer from scratch in the training routine instead of 
    using word2vec. Thus, the embedding layer's dimensions are determined by the vocabulary
    of the dataset used for training. 
    """
    
    def __init__(self, output_units, maxlen, no_of_unique_words, embed_dim):
        
        """
        :param output_units: int
            no of output units of the LSTM
        :param maxlen: int
            maximum length of the sequence that we will consider
        :param no_of_unique_words: int
            number of unique words in the vocabulary
        :param embed dim: int
            dimension of each word vector
        """
        
        tf.reset_default_graph()
        
        self.output_units = output_units 
        self.maxlen = maxlen 
        self.activation_fn = tf.nn.relu
        self.initializer =  tf.compat.v1.initializers.glorot_normal(seed=42)
        self.W_embed = tf.get_variable(name = 'Embedding_weights', shape = [no_of_unique_words, embed_dim], initializer=tf.contrib.layers.xavier_initializer(seed = 42), trainable=True) # acts as code weight

        self.LSTM = tf.nn.rnn_cell.LSTMCell(num_units=self.output_units, initializer=self.initializer,
                                       forget_bias=2.5,state_is_tuple=True)
        

    def build_network(self, X_sentence_series_flattened, X_sentence_length):
        
        """
        :param X_sentence_series_flattened: pd.Series[int]
            Series containing the integer mappings of the words in the entire minibatch. 
        :param X_sentence_length: pd.Series[int]
            Series with shape=[minibatch_size,] containing the length of each sentence in the minibatch.
        """
        
        X_sentence_embedding = tf.nn.embedding_lookup(self.W_embed, X_sentence_series_flattened)     
        X_sentence_list = tf.RaggedTensor.from_row_lengths(X_sentence_embedding, row_lengths = X_sentence_length)        
        X_sentence_padded = X_sentence_list.to_tensor()
        
        #  pad the ragged tensor up till specified max length
        tensor_shape = tf.shape(X_sentence_padded)
        num_examples = tensor_shape[0]
        num_time_steps = tensor_shape[1]
        vector_size = tensor_shape[2]       
        padding = tf.fill(dims = [num_examples, self.maxlen - num_time_steps, vector_size], value = 0.0)    
        X_sentence_padded_final = tf.concat([padding, X_sentence_padded], axis=1)        
        lstm_output, _ = tf.compat.v1.nn.dynamic_rnn(self.LSTM, inputs = X_sentence_padded_final, 
                                   dtype=tf.float32)
        
        return lstm_output
    
    
    def compute_contrastive_loss(self, lstm_output_1, lstm_output_2, y_series):
        
        """
        :param lstm_output_1: tf.tensor
            Matrix with shape=[batch_size, output_units] containing the output state of the LSTM for the first set of sentences.
        :param lstm_output_2: tf.tensor
            Matrix with shape=[batch_size, output_units] containing the output state of the LSTM for the second set of sentences.
        :param y_series: pd.Series
            Series with shape=[batch_size,] containing values \in {0,1}, indicating whether each sentence pair is semantically similar.            
        """
        
        y_prediction_exp = tf.compat.v1.norm(lstm_output_1 - lstm_output_2, ord = 1, axis = 1) #l1 norm         
        y_prediction = tf.exp(-tf.reduce_sum(y_prediction_exp, axis = 1))
        total_loss = tf.reduce_sum(tf.math.square(y_series - y_prediction))
        
        return total_loss
    
    
    def train(self, X_sentence_1, X_sentence_2, X_sentence_1_length, X_sentence_2_length, y_data, batch_size, learning_rate,\
              no_of_iterations, seed, root_logdir):
        
        """
        :param X_sentence_1: pd.Series[int]
            Series with dim=[batch_size,] containing the indices of words present in 1st set of sentences.
        :param X_sentence_2: pd.Series[int]
            Series with dim=[batch_size,] containing the indices of words present in 2nd set of sentences.
        :param X_sentence_1_length: pd.Series[int]
            Series with shape=[batch_size,] containing the length of each sentence in the 1st set of sentences.
        :param X_sentence_2_length: pd.Series[int]
            Series with shape=[batch_size,] containing the length of each sentence in the 2nd set of sentences.
        :param y_data: pd.Series[int]
            Series with shape=[batch_size,] containing values \in {0,1}, indicating whether each sentence pair is semantically similar.            
        :param batch_size: int
            Integer specifying the minibatch size to use during training.
        :learning_rate: float
            Value specifying the learning rate to use during training.
        :no_of_iterations: int
            Integer specifying the total number of iterations for training.    
        :seed: int
            Random seed that ensures reproducibility in experiments.
        :root_logdir: str(directory_path)
            valid directory path in the local machine to save model checkpoints.        
        """
        
        self.X_sentence_1 = tf.placeholder(tf.int32, shape = [None])
        self.X_sentence_2 = tf.placeholder(tf.int32, shape = [None])
        self.X_sentence_1_length = tf.placeholder(tf.int32, shape = [None])
        self.X_sentence_2_length = tf.placeholder(tf.int32, shape = [None])
        self.y_series = tf.placeholder(tf.float32, shape = [None]) 

        sentence_1_output = self.build_network(self.X_sentence_1, self.X_sentence_1_length)
        sentence_2_output = self.build_network(self.X_sentence_2, self.X_sentence_2_length)
        
        self.total_loss = self.compute_contrastive_loss(sentence_1_output, sentence_2_output, self.y_series)
        
        all_vars = tf.trainable_variables()
  
        self.train_rnn = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(self.total_loss, var_list=all_vars)        
        
        ## TRAINING ##
        val_loss = tf.placeholder(dtype=tf.float32, shape=[], name='val_loss')
        val_loss_summary = tf.summary.scalar('val_loss', val_loss)
        
        scalar_summaries = [tf.summary.scalar(var_.name, var_) for var_ in all_vars if len(var_.shape) == 0]
        array_summaries = [tf.summary.histogram(var_.name, var_) for var_ in all_vars if len(var_.shape) > 0]
       
        writer = tf.summary.FileWriter(root_logdir)

        saver = tf.train.Saver(max_to_keep=1)
        init = tf.global_variables_initializer()
        
        batch_generator_ = batch_generator(len(y_data), batch_size = batch_size, seed = seed)  
        
        with tf.Session() as sess:
            
            tf.set_random_seed(seed)
            
            init.run()
                                 
            for iteration in range(no_of_iterations):
                
                mini_batch_indices = batch_generator_.generate_minibatch_indices()
                X_sentence_1_mini_batch = X_sentence_1.iloc[mini_batch_indices]
                X_sentence_2_mini_batch = X_sentence_2.iloc[mini_batch_indices]
                X_sentence_1_mini_batch_length = X_sentence_1_length.iloc[mini_batch_indices]
                X_sentence_2_mini_batch_length = X_sentence_2_length.iloc[mini_batch_indices]
                y_mini_batch = y_data.iloc[mini_batch_indices]
               
                X_sentence_1_mini_batch_final = pd.Series(itertools.chain.from_iterable(X_sentence_1_mini_batch))
                X_sentence_2_mini_batch_final = pd.Series(itertools.chain.from_iterable(X_sentence_2_mini_batch))
               
                batch_dict = {
                    self.X_sentence_1: X_sentence_1_mini_batch_final,
                    self.X_sentence_2: X_sentence_2_mini_batch_final,
                    self.X_sentence_1_length: X_sentence_1_mini_batch_length,
                    self.X_sentence_2_length: X_sentence_2_mini_batch_length,
                    self.y_series: y_mini_batch               
                }
                                
                batch_loss_, _ = sess.run([self.total_loss, self.train_rnn], feed_dict=batch_dict)               
                
                if iteration%20==0:
                    
                    print(iteration, end=" ")
                    
                    val_indices = batch_generator_.retrieve_holdout_indices()
                                   
                    X_sentence_1_mini_batch = X_sentence_1.iloc[val_indices]
                    X_sentence_2_mini_batch = X_sentence_2.iloc[val_indices]
                    X_sentence_1_mini_batch_length = X_sentence_1_length.iloc[val_indices]
                    X_sentence_2_mini_batch_length = X_sentence_2_length.iloc[val_indices]
                    y_mini_batch = y_data.iloc[val_indices]

                    X_sentence_1_mini_batch_final = pd.Series(itertools.chain.from_iterable(X_sentence_1_mini_batch))
                    X_sentence_2_mini_batch_final = pd.Series(itertools.chain.from_iterable(X_sentence_2_mini_batch))

                    batch_dict = {
                        self.X_sentence_1: X_sentence_1_mini_batch_final,
                        self.X_sentence_2: X_sentence_2_mini_batch_final,
                        self.X_sentence_1_length: X_sentence_1_mini_batch_length,
                        self.X_sentence_2_length: X_sentence_2_mini_batch_length,
                        self.y_series: y_mini_batch               
                    }

                    val_loss_ = sess.run(self.total_loss, feed_dict=batch_dict)
                    
                    print("\tval loss: %.4f" % val_loss_)
                    
                    val_loss_summary_str = sess.run(val_loss_summary, feed_dict={val_loss: val_loss_})
                    writer.add_summary(val_loss_summary_str, iteration)
                    
                scalar_summaries_str = sess.run(scalar_summaries)
                array_summaries_str = sess.run(array_summaries)
                for summary_ in scalar_summaries_str + array_summaries_str:
                    writer.add_summary(summary_, iteration)
                    
            #  save the model
            saver.save(sess, join(root_logdir, "model.ckpt"))

        #  close the file writer
        writer.close()
        
        