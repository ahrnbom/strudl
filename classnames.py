""" A module for getting the class names for a dataset, in a single canonical order.
"""
import json

from folder import datasets_path

def get_class_data(dataset):
    path = '{}{}/classes.json'.format(datasets_path, dataset)
    
    with open(path, 'r') as f:
        class_data = json.load(f)
    class_data.sort(key=lambda d: d['name'])
    return class_data

def get_classnames(dataset):
    class_data = get_class_data(dataset)
    names = [d['name'] for d in class_data]
    return names

def set_class_data(dataset, class_data):
    class_data.sort(key=lambda d: d['name'])
    path = '{}{}/classes.json'.format(datasets_path, dataset)
    with open(path, 'w') as f:
        json.dump(class_data, f)

    
if __name__ == '__main__':
    print(get_classnames('sweden2'))


