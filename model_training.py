from tensorflow.python.keras.backend import dropout
from preprocess import *
import transformers
from transformers import ElectraTokenizer, TFElectraModel
from os import path
import tensorflow as tf
import math
from sklearn.model_selection import train_test_split
import numpy as np

seed = 232
model_name = 'google/electra-base-discriminator'
cache_dir = path.join(path.dirname(path.dirname(__file__)),'cache')
tokenizer = ElectraTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
use_token_type_ids = "token_type_ids" in tokenizer.model_input_names

use_iob2_format = True
model_meta = ModelMeta()
model_meta.model_type = 'bert'
batch_size = 5

def read_data(filepath):
    examples, annotations_list, class_list = read_annotation_file(filepath)
    converted_examples = convert_platform_data_to_ner(examples, annotations_list, class_list, use_iob2_format = use_iob2_format)
    class_map = {i:label for i, label in enumerate(class_list)}
    features = convert_examples_to_features(model_meta, converted_examples,class_list,tokenizer,use_iob2_format = use_iob2_format)
    return features, class_map

def create_tensorflow_dataset(features):
    def gen():
        for ex in features:
            yield (
                {
                    "input_ids": ex.input_ids,
                    "attention_mask": ex.attention_mask,
                    "token_type_ids": ex.token_type_ids,
                },
                ex.label_ids,
            )
    return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([None]),
            ),
        )
    
def build_model(num_labels, use_dropout=True, dropout_rate=0.15):
    model = TFElectraModel.from_pretrained(model_name, cache_dir = cache_dir)
    input_ids = tf.keras.layers.Input(shape=(model_meta.max_seq_length,), name='input_ids', dtype='int32')
    attention_mask = tf.keras.layers.Input(shape=(model_meta.max_seq_length,), name='attention_mask', dtype='int32')
    token_type_ids = tf.keras.layers.Input(shape=(model_meta.max_seq_length,), name='token_type_ids', dtype='int32')
    model_inputs = [input_ids, attention_mask, token_type_ids]
    outputs = model(model_inputs)
    logits = outputs[0]
    if use_dropout and dropout_rate>0:
        logits = tf.keras.layers.Dropout(dropout_rate)(logits)
    model_op = tf.keras.layers.Dense(num_labels, activation = 'softmax', kernel_initializer='glorot_uniform')(logits)
    keras_model = tf.keras.Model(inputs= model_inputs, outputs = model_op)
    

def run_examples(texts, model, tokenizer, class_map:Dict, saved_model_format:bool = True):
    features_2d = preprocess_for_inferencing(texts, tokenizer)
    attention_mask, input_ids, token_type_ids = [],[],[]
    for feature_index, features in enumerate(features_2d):
        attention_mask.append(features.attention_mask)
        input_ids.append(features.attention_mask)
        token_type_ids.append(features.attention_mask)
    attention_mask = np.array(attention_mask).astype('int32')
    input_ids = np.array(input_ids).astype('int32')
    token_type_ids = np.array(token_type_ids).astype('int32')
    model_inputs = []
    if "token_type_ids" in tokenizer.model_input_names:
        model_inputs = [input_ids, attention_mask, token_type_ids]
    else:
        model_inputs = [input_ids, attention_mask]
    if saved_model_format:
        predictions = model(model_inputs, training = False)
    else:
        predictions = model.predict(model_inputs)
    predictions = np.argmax(predictions, axis=-1)
    result = []
    for i,text in enumerate(texts):
        tokens, labels = [], []
        for j in range(1,predictions.shape[1]-1):
            if attention_mask[i,j]==1:
                lbl = class_map[predictions[i,j]]
                tk = tokenizer.convert_ids_to_tokens(input_ids[i,j].item())
                tokens.append(tk)
                labels.append(lbl)
        result.append(TFNERResult(tokens, labels, text))
    return result