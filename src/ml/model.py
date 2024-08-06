# 
# type: ignore

from keras.models import Sequential 
from keras.optimizers import Adam 
from keras.layers import LSTM,Activation,Dense,Dropout,Input,Embedding,SpatialDropout1D
from src.entity.config_entity import ModelTraninerConfig
from src.constants import *





class ModelArchitecture:
    def __init__(self) -> None:
        pass

    def get_model(self):
        model=Sequential([
        Embedding(MAX_WORDS,100,input_length=MAX_LEN),
        SpatialDropout1D(.3),
        LSTM(128,dropout=.2,return_sequences=True),
        LSTM(64,dropout=.3),
        Dense(1,activation=ACTIVATION)])

        model.summary()
        model.compile(loss=LOSS,optimizer=Adam(),metrics=METRICS)

        return model        
        


