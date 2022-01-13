from flask import Flask
from TFNet_model_stft import Trainer
from TFNet_model import Trainer

import os
app = Flask(__name__)

@app.route('/')
def getLabel():
    pm.inference()
    return "ok"

if __name__ == '__main__':
    para_dict={}
    f = open("./config/config.ini", encoding="utf-8")
    contents=f.read().splitlines()
    for line in contents:
        tmp=line.split('=')
        para_dict[tmp[0]]=tmp[1]
    f.close()
    if not os.path.exists(para_dict['data_path']):  # 如果路径不存在
        os.makedirs(para_dict['data_path'])
    pm = Trainer(para_dict['data_path'])
    app.run(host='127.0.0.1', port=para_dict['server_port'], debug=False)

