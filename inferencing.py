import numpy as np

is_begining = lambda tag: len(tag)>1 and tag[:2]=='B-'
is_inside = lambda tag: len(tag)>1 and tag[:2]=='I-'

def apply_postprocessing(tokens, labels):
    new_tokens, new_labels = [], []
    assert(len(tokens)== len(labels))
    for i,(tk,lbl) in enumerate(zip(tokens[:-1], labels[:-1])):
        if '##' in tk and i>0:
            new_tokens[-1] += tk.replace('##','')
            last_lbl = new_labels[-1]
            new_lbl = 'O'
            if last_lbl!='O' and lbl=='O':
                new_lbl = last_lbl
            elif last_lbl=='O' and lbl!='O':
                new_lbl = lbl
            else:
                new_lbl = last_lbl
            new_labels[-1] = new_lbl
        else:
            new_tokens.append(tk)
            new_labels.append(lbl)
    return new_tokens, new_labels

def prepare_field(self, name, value, probs, use_iob = True):
    probability = round(probs[0] if len(probs)==1 else np.average(probs).item(), 3)
    return {'name': name[2:] if use_iob else name, 'value': self.remove_whitespaces(value), 'confidence': probability}

def extract_fields_from_IOB2_sequence(tokens, labels, probabilities):
    fields = []
    last_field_name = 'O'
    last_field_val = ''
    probs_set = []
    for i,(tk,lbl) in enumerate(zip(tokens, labels)):
        if is_begining(lbl):
            if last_field_name!='O':
                fields.append(prepare_field(last_field_name, last_field_val, probs_set))
            last_field_val = tk
            probs_set = [ probabilities[i]]
        elif is_inside(lbl):
            if last_field_name=='O':
                last_field_val = tk
                probs_set = [ probabilities[i]]
            elif last_field_name[2:]== lbl[2:]:
                last_field_val += ' '+ tk
                probs_set.append(probabilities[i])
            else:
                fields.append(prepare_field(last_field_name, last_field_val, probs_set))
                last_field_val = tk
                probs_set = [ probabilities[i]]
        elif last_field_name!='O':
            fields.append(prepare_field(last_field_name, last_field_val, probs_set))
        last_field_name = lbl
    if last_field_name!='O':
        fields.append(prepare_field(last_field_name, last_field_val, probs_set))
    return fields