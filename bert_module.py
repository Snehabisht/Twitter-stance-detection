import nltk
import numpy as np 
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

def get_tokeniser():
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    #model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
def get_vector(vec_type,max_squence_len,label_map=None):
    if vec_type == 'train':
        with open("/home/sneha/Desktop/major-20/Capsule_Network/TagMyNews/trainCorpora.txt") as f:
            lines = f.readlines()
    else:
        with open("/home/sneha/Desktop/major-20/Capsule_Network/TagMyNews/validCorpora.txt") as f:
            lines = f.readlines()
    labels = []
    text = []
    for i in range(len(lines)):
        temp = lines[i].split(' ',1)
        if len(temp)<2: continue
        labels.append(temp[0])
        text.append(temp[1].strip())
    map = get_glove_map("glove.twitter.27B.25d.txt")
    text_vec = []
    


    for item in text:
        temp = []
        words = nltk.word_tokenize(item)
        for sub in words:
            if sub in map:
                temp.append(map[sub])
            else:
                temp.append([0]*25)
        if len(temp) > max_squence_len: temp = temp[:max_squence_len]
        else: 
            for _ in range(max_squence_len-len(temp)):
                temp.append([0]*25)
        text_vec.append(temp)
    text_vec = np.array(text_vec)
    s_label = set(labels)
    map_label = {}
    for i,item in enumerate(s_label):
        map_label[item] = i
    if label_map != None:
        map_label = label_map
    dig_label = np.array([map_label[item] for item in labels])
    one_hot_label = to_categorical(dig_label)
    #print(text_vec)
    #print(one_hot_label)
    #print(map_label)
    return text_vec,one_hot_label,map_label


def get_glove_map(glove_path):
    map = {}
    with open(glove_path,encoding='utf-8') as f:
        lines = f.readlines()
        for item in lines:
            line = item.strip().split(' ')
            map[line[0]]=[item for item in line[1:]]
    return map

get_vector('train',12,None)