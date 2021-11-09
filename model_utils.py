import yaml
import os
from keras.models import model_from_yaml

# 检查文件夹是否存在，不存在则创建
def makedir(path):
    if(not os.path.exists(path)):
        os.makedirs(path)
    return path

def check_path(path,eflag=False):

    if(not os.path.exists(path)):
        if(eflag):
            raise Exception('no file:%s'%path)
        return False
    return True

# 保存模型 
def save_model(model,uid='default',out_dir='/mnt/mydata/deep_siren_models'):
    '''
        将模型导出成yaml文件，并将权重保存为HDF5格式文件

        param:
            model: [keras model] 模型实例
            uid : [string] 模型编号
            out_dir: 保存的文件夹
        
        return:
            yml_path: [string] yaml文件路径
            model_path [string] 权重文件路径
    '''
    makedir(out_dir)
    
    model_dir = os.path.join(out_dir,uid)
    
    makedir(model_dir)

    yml_path = os.path.join(model_dir,'model.yml')
    model_path = os.path.join(model_dir,'model.h5')
    if(os.path.exists(yml_path)):
        os.remove(yml_path)
    if(os.path.exists(model_path)):
        os.remove(model_path)
        
    yaml_string = model.to_yaml()

    with open(yml_path,'w') as wf:
        wf.write(yaml.dump(yaml_string,default_flow_style=True))

    model.save_weights(model_path)

    return yml_path,model_path

def load_model(uid='default',indir='/mnt/mydata/deep_siren_models',custom_objects=None):

    yaml_path = os.path.join(indir,uid,'model.yml')
    check_path(yaml_path,True)

    model_path = os.path.join(indir,uid,'model.h5')
    check_path(model_path,True)
    with open(yaml_path,'r') as yml:
        yaml_string = yaml.load(yml)
        model = model_from_yaml(yaml_string,custom_objects)

    model.load_weights(model_path)

    return model



if __name__ == "__main__":
    
    model = load_model('ca5023a0-aeb9-11e9-9175-1c1b0d0c830c')