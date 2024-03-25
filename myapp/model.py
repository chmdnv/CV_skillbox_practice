import os
import pickle
import numpy as np
import logging


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Model:
    def __init__(self):
        model_path = os.path.join('myapp', 'model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} does not exist!")
        
        with open(model_path,'rb') as file:
            self.meta = pickle.load(file)
            self.model = self.meta['model']
            self.char_map = self.meta['char map dict']
        
    def __repr__(self):
        return repr(self.model)

    def __str__(self):
        return self.model.__class__.__name__
    
    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred_char : str
            Символ-предсказание 
        '''
        logger.info(f"Model: {self}, accuracy: {self.meta['accuracy'] :.2f}")
        
        x = np.array((np.array(x.reshape((28*28,))),))
        logger.info(f"Starting prediction on X with shape {x.shape}")

        prediction = self.model.predict(x)[0]
        logger.info(f"Predicted class: {prediction}")

        if prediction not in self.char_map:
            logger.warninig(f"{repr(prediction)} not found in char map dict!")

        pred_char = self.char_map.get(prediction, None)
        logger.info(f"Predicted character: {pred_char}")
            
        return pred_char
