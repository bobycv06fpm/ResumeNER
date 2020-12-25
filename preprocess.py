import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from spacy.lang.en import English # updated
import string
import re
import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer

reg_whitespace = '[{0}]'.format(string.whitespace)
reg_weird = '[^0-9a-zA-Z!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\r ]'
reg_CONTINUOUS_newlines = '\\n\s*\\n'
html_tags = '<(?!tag).*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'

#load sentencizer at global level
sentencizer = English()
sentencizer.add_pipe(sentencizer.create_pipe('sentencizer')) # updated

@dataclass
class ModelMeta:
    max_seq_length:int = 512
    sep_token_extra:bool = False
    model_type:str = None
    pad_token_label_id:int =0
    sequence_a_segment_id:int =0
    mask_padding_with_zero:bool =True

@dataclass
class Annotation:

    start_index: int
    end_index: int
    value: str
    name: str
    
@dataclass
class SentenceMeta:
    words: List[str]
    labels: Optional[List[str]]
    
@dataclass
class InputExample:
    num_sentences: int
    num_tokens: int
    sentence_meta_list : List[SentenceMeta]

@dataclass
class InputFeatures:

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None

def recurive_replacement(value, replacement, text):
    while value in text:
        text= text.replace(value, replacement)
    return text

def parse_text(text):
    text = ''.join( [ch if ord(ch)<128 else ' ' for ch in text])
    #special case
    text = text.replace("\\r\\n","\n")
    text = text.replace("\\n","\n")
    text = text.replace("\\t","\t")
    text = text.replace("\\r","\r")
    text = text.replace("\\x0b"," ")
    text = text.replace("\\x0c"," ")
    text = text.replace("\r\n","\n")
    text = text.replace("\r","\n")
    text = re.sub(reg_CONTINUOUS_newlines,'\n', text)
    text = re.sub(reg_weird,' ', text)
    text = re.sub(html_tags,'', text)
    text = recurive_replacement('  ',' ', text)
    return text

def sent_tokenize(text):
    doc = sentencizer(text)
    return [sent.string for sent in doc.sents]

def tokenize(text:str)->List[str]:
    tokens = []
    word = ''
    for ch in text:
        if re.match('^[a-zA-Z0-9]',ch):
            word += ch
        else:
            if len(word)>0:
                tokens.append(word)
                word = ''
            tokens.append(ch)
    if len(word)>0:#check for leftover word if any
        tokens.append(word)
    return tokens

def correct_annotation_indice(label, label_si, label_ei, text):
    if label not in text:
        raise Exception('Cannot recover label indice from text')
    new_si = text.index(label)
    while new_si>=0:
        new_ei = new_si + len(label)
        # check for intersection
        recovered_label = text[new_si:new_ei]
        if new_si <= label_ei and label_si <= new_ei and recovered_label == label:
            return (new_si, new_ei)
        new_si = text.index(label, new_si+1)
    raise Exception('Could not recover correct label indice!')

def convert_platform_data_to_ner(examples:List[str], annotations_list:List[Annotation], class_list:List[str], use_iob2_format:bool=False)->List[InputExample]:
    assert(len(examples)== len(annotations_list))
    converted_examples = []
    for example,annotations in tqdm(zip(examples, annotations_list), total=len(examples),desc='converting samples'):
        sentences = sent_tokenize(example)
        tokens = []
        chr_to_tok_map = {}
        tok_to_sent_map = []
        num_tokens = 0
        token_idx = 0
        ch_index = 0
        for sent_index, sentence in enumerate(sentences):
            sent_tokens = tokenize(sentence)
            for token  in sent_tokens:
                for ch in token:
                    chr_to_tok_map[ch_index] = token_idx
                    ch_index += 1
                token_idx += 1
            tok_to_sent_map.append(( num_tokens, num_tokens+len(sent_tokens)))
            num_tokens += len(sent_tokens)
            tokens.extend(sent_tokens)
        labels = [class_list[0],]* len(tokens)# by default all the tokens would be 'O'
        for annotation in annotations:
            token_si = chr_to_tok_map[annotation.start_index]
            # if annotation.end_index not in chr_to_tok_map and annotation.end_index==len(example):
            #     token_ei = chr_to_tok_map[annotation.end_index-1]
            # else:
            token_ei = chr_to_tok_map[annotation.end_index-1]
            #start: sanity check
            original_label = ''.join(tokens[token_si: token_ei+1])
            assert(original_label == annotation.value or annotation.value in original_label)
            # if original_label != annotation.value:
            #     print(annotation.value)
            #     print('here')
            #end: sanity check
            # assert(len(set(labels[token_si:token_ei+1]))==1)#Entity over Entity not allowed
            if len(set(labels[token_si:token_ei+1]))>1:
                continue
            if use_iob2_format:
                iob_start = 'B-'+annotation.name.upper()
                iob_end = 'I-'+annotation.name.upper()
                assert(iob_start in class_list and iob_end in class_list)
                labels[token_si:token_ei+1] = [iob_end]*(token_ei+1-token_si)
                labels[token_si]=iob_start
            else:
                assert(annotation.name in class_list)
                labels[token_si:token_ei+1] = [annotation.name]*(token_ei+1-token_si)
        #create sentence level tokens
        sentence_meta_list = []
        for sent_si, sent_ei in tok_to_sent_map:
            sent_tokens = tokens[sent_si:sent_ei]
            sent_labels = labels[sent_si:sent_ei]
            sentence_meta_list.append( SentenceMeta(sent_tokens, sent_labels))
        converted_examples.append(InputExample(len(sentences), len(tokens), sentence_meta_list))
    return converted_examples

def convert_tokens_to_features(
    tokenizer:object,
    model_meta:ModelMeta,
    feature_tokens:List[str],
    feature_labels: List[str]=[],
    )->InputFeatures:

    inference_mode = len(feature_labels)==0
    cls_token_at_end = bool(model_meta.model_type in ["xlnet"])
    cls_token_segment_id= 2 if model_meta.model_type in ["xlnet"] else 0
    pad_token = tokenizer.pad_token_id
    pad_token_segment_id =tokenizer.pad_token_type_id
    pad_on_left =bool(tokenizer.padding_side == "left")
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    special_tokens_count = tokenizer.num_special_tokens_to_add()
    max_allowed_tokens = model_meta.max_seq_length - special_tokens_count - int(model_meta.sep_token_extra)

    if len(feature_tokens) > max_allowed_tokens:
        feature_tokens = feature_tokens[: max_allowed_tokens]
        feature_labels = feature_labels[: max_allowed_tokens]
    feature_tokens += [sep_token]
    feature_labels += [model_meta.pad_token_label_id]
    if model_meta.sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        feature_tokens += [sep_token]
        feature_labels += [model_meta.pad_token_label_id]
    segment_ids = [model_meta.sequence_a_segment_id] * len(feature_tokens)

    if cls_token_at_end:
        feature_tokens += [cls_token]
        feature_labels += [model_meta.pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        feature_tokens = [cls_token] + feature_tokens
        feature_labels = [model_meta.pad_token_label_id] + feature_labels
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(feature_tokens)
    input_mask = [1 if model_meta.mask_padding_with_zero else 0] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = model_meta.max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if model_meta.mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        feature_labels = ([model_meta.pad_token_label_id] * padding_length) + feature_labels
    else:
        input_ids += [pad_token] * padding_length
        input_mask += [0 if model_meta.mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        feature_labels += [model_meta.pad_token_label_id] * padding_length

    assert len(input_ids) == model_meta.max_seq_length
    assert len(input_mask) == model_meta.max_seq_length
    assert len(segment_ids) == model_meta.max_seq_length
    if not inference_mode:
        assert len(feature_labels) == model_meta.max_seq_length
    return InputFeatures(
            input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=feature_labels
        )

def convert_examples_to_features(
    model_meta: ModelMeta,
    input_examples:List[InputExample],
    class_list: List[str],
    tokenizer: PreTrainedTokenizerFast,
    use_iob2_format:bool=False) -> List[InputFeatures]:
    max_allowed_tokens = model_meta.max_seq_length - tokenizer.num_special_tokens_to_add() + int(model_meta.sep_token_extra)
    class_map = {label: i for i, label in enumerate(class_list)}
    features = []
    for ex_index, example in tqdm(enumerate(input_examples), total=len(input_examples), desc='generating features'):
        #loop over sentences in an example
        context_subsets = []
        current_subset = []
        last_count = 0
        for sentence_meta in example.sentence_meta_list:
            tokens = []
            label_ids = []
            for word, label in zip(sentence_meta.words, sentence_meta.labels):
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    temp_pad_id = model_meta.pad_token_label_id
                    label_id = class_map[label]
                    if label!='O' :
                        if use_iob2_format:
                            temp_pad_id = class_map['I-'+ label[2:]]
                        else:
                            temp_pad_id = class_map[label]
                    label_ids.extend([label_id] + [temp_pad_id] * (len(word_tokens) - 1))
            assert(len(tokens)== len(label_ids))
            num_tokens = len(tokens)
            sent_feature = SentenceMeta(tokens, label_ids)
            if last_count + num_tokens >= max_allowed_tokens:
                if len(current_subset)==0:
                    #meaning a single sentence is exceeding the max_seq_len of the model
                    context_subsets.append([sent_feature])
                else:
                    context_subsets.append(current_subset)
                    current_subset = [sent_feature]
                    last_count = num_tokens
            else:
                last_count += num_tokens
                current_subset.append(sent_feature)
        if len(current_subset)>0:
            context_subsets.append(current_subset)
        #finally create concatenated features based on sent features
        for context_subset in context_subsets:
            feature_tokens = []
            feature_labels = []
            for sent_feature in context_subset:
                feature_tokens.extend( sent_feature.words)
                feature_labels.extend( sent_feature.labels)
            input_features = convert_tokens_to_features(
                tokenizer = tokenizer,
                model_meta = model_meta,
                feature_tokens = feature_tokens,
                feature_labels = feature_labels
            )
            features.append(input_features)
    return features

def read_annotation_file(filepath, use_iob2_format=True):
    examples = []
    all_labels = []
    tag_master = []
    labels_skipped = 0
    with open(filepath, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            text = data['content']
            annotations = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]
                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    # entities.append((point['start'], point['end'] + 1 ,label))
                    if label not in tag_master:
                        tag_master.append(label)
                    label_text = point['text']
                    label_si = point['start']
                    label_ei = point['end']+1
                    use_annotation = True
                    if text[label_si:label_ei]!=label_text or label_ei>=len(text)+1:
                        use_annotation = False
                        if label_text in text:
                            label_si = text.index(label_text)
                            label_ei = label_si + len(label_text)
                            use_annotation = True
                    if use_annotation:
                        annotations.append(Annotation(label_si, label_ei, label_text, label))
                    else:
                        labels_skipped += 1
                        print('annotation skipped due to inconsitent indices..')
            if len(annotations) >0 and len(text.strip())>0:
                examples.append(text)
                all_labels.append(annotations)
        class_list = ['O']
    for tag_name in tag_master:
        if use_iob2_format:
            class_list.append('B-'+tag_name.upper())
            class_list.append('I-'+tag_name.upper())
        else:
            class_list.append(tag_name)
    print('Total Samples Processed:', len(examples))
    print('labels skipped:', labels_skipped)
    return examples, all_labels, class_list
            
if __name__ == "__main__":
    
    use_iob2_format = True
    model_meta = ModelMeta()
    model_meta.model_type = 'bert'
    examples, annotations_list, class_list = read_annotation_file(r'traindata.json')
    converted_examples = convert_platform_data_to_ner(examples, annotations_list, class_list, use_iob2_format = use_iob2_format)
    class_map = {i:label for i, label in enumerate(class_list)}
    features = convert_examples_to_features(converted_examples,class_list,tokenizer,use_iob2_format = use_iob2_format)
    use_token_type_ids = "token_type_ids" in tokenizer.model_input_names
    print('here')
            