import os
import json
import jieba
import sys

def get_sys_argvs(help_callback=None):
    try:

        argvs = sys.argv
        l = len(argvs)
        if(l == 2):
            if(argvs[1] == '-h' or argvs[1] == 'help'):
                help_callback()
                return None
        config = {}
        for i in range(l-1):
            text = argvs[i+1]
            s = text.split('=')
            config[s[0]] = s[1]
    except Exception as e:
        if(help_callback is not None):
            help_callback()
        raise e

    return config

def get_option(data,key,default=None):
    res = default
    if(key in data):
        res = data[key]

    return res

def set_jieba_name_entities():
    import pymongo
    client = pymongo.MongoClient('192.168.0.254',27017)
    db = client['betalpha_corpus']
    col = db['name_entities']
    datas = col.find()
    for data in datas:
        text = data['name']
        jieba.suggest_freq(text,True)

    client.close()

def read_json(path):
    with open(path) as f:
        return json.load(f)

def write_json(path,data):
    with open(path,'w') as f:
        json.dump(data,f,ensure_ascii=False)

def parse_config_from_str(content):
    l = content.split(',')
    results = {}
    for text in l:
        t = text.split('=')

        key = t[0]
        value = t[1]
        results[key] = value
    
    return results