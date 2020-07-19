import numpy as np

class batch_generator:
    """Minibatch Generator."""
    
    def __init__(self, dataset_size, batch_size=None, holdout_ratio=0.05, seed=None):
        """
        Splits the dataset into the training and validation sets. Subsequently, feeds
        the training indices in minibatches. 
        
        Use it as follows:
        
            batch_generator_ = batch_generator(dataset_size, batch_size, holdout_ratio, seed)
            minibatch_training_indices = batch_generator_.generate_minibatch_indices()
            validation_indices = batch_generator_.retrieve_holdout_indices()
            
        :param dataset_size: int
            Size of the entire dataset that is to be used for model training and validation.
        :param batch_size: int
            Size of the minibatch for model training.
        :param holdout_ratio: float
            Ratio that determines the proportion of the dataset that should be used for model validation.
        :param seed: int
            Random seed that ensures reproducibility in dataset split. 
        """
            
        self.dataset_indices = list(range(dataset_size))
        np.random.seed(seed)
        np.random.shuffle(self.dataset_indices)
        self.start = 0
        
        if holdout_ratio is not None:
            holdout_set_size = int(holdout_ratio * dataset_size)
            self.training_set_indices = self.dataset_indices[holdout_set_size:]
            self.training_set_size = len(self.training_set_indices)
            self.holdout_set_indices = self.dataset_indices[:holdout_set_size]            
        else:
            self.training_set_size = len(self.dataset_indices)
            self.training_set_indices = self.dataset_indices
            self.holdout_set_indices = None
        
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = self.training_set_size
            print("No minibatch size specified. Using batch learning.")
                   
        self.no_of_minibatches = self.training_set_size//self.batch_size + 1*(self.training_set_size%self.batch_size != 0)
        print(f"Total number of minibatches: {self.no_of_minibatches}.")


    def generate_minibatch_indices(self):
        """
        Retrieve the minibatched dataset indices for model training.    
        """
        if self.start + self.batch_size >= len(self.training_set_indices):
            print("This is the last mini-batch for this epoch.")
            indices = self.training_set_indices[self.start:]
            self.start = 0 # reset index
        else:          
            indices = self.training_set_indices[self.start: self.start + self.batch_size]
            self.start += self.batch_size
        return indices
    
    def retrieve_holdout_indices(self):
        """
        Retrieve the holdout indices for model validation.
        """
        if self.holdout_set_indices in [None, 0]:
            raise ValueError("No holdout set generated. Ensure that the specified holdout ratio > 0.")
        return self.holdout_set_indices
