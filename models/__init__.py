import json

# from models.simple_rnn_model import SimpleRNNModel
# from models.lstm_model import LSTMModel
# from models.gru_model import GRUModel
from models.leaky_rnn_model import LeakyRNNCell, RandomLeakyRNNCell, ATCLRNNCell
from models.leaky_rnn_model import LeakyRNNModel, RandomLeakyRNNModel, ATCLRNNModel

ALL_MODEL = [ LeakyRNNModel, RandomLeakyRNNModel, ATCLRNNModel,]
ALL_MODEL_DICT = {m.__name__: m for m in ALL_MODEL}


def from_json(filepath):
  with open(filepath, 'r') as f:
    model_cnf = json.load(f)
  cls = ALL_MODEL_DICT[ model_cnf["class_name"] ]
  return cls.from_config( model_cnf["config"] )