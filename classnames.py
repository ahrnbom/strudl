from folder import datasets_path

def get_classnames(dataset):
    path = '{}{}/classes.txt'.format(datasets_path, dataset)
    
    with open(path, 'r') as f:
        lines = [x.strip('\n') for x in f.readlines()]
    
    lines = [line for line in lines if (len(line) > 0) and (not line.isspace())]
    lines.sort()
    
    return lines
    
def set_classnames(dataset, classnames):
    classnames.sort()

    path = '{}{}/classes.txt'.format(datasets_path, dataset)
    with open(path, 'w') as f:
        for cn in classnames:
            f.write("{}\n".format(cn))
    
    
if __name__ == '__main__':
    print(get_classnames('sweden2'))


