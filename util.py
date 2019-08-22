""" A module for very simple utility functions """

import sys
from math import sqrt

def left_remove(text, to_remove):
    """ Removes a part of a string, if it starts with it.
        Similar to str.lstrip, see note below on right_remove
    """
    if text.startswith(to_remove):
        return text.replace(to_remove, '', 1)
    else:
        return text

def right_remove(text, to_remove):
    """ Removes a part of a string, if it ends with it.
        str.rstrip is similar, but can remove too much. 
        For example, '4.mp4'.rstrip('.mp4') will remove the leading four!
        This function does not do that.
    """
    if text.endswith(to_remove):
        return text[:len(text)-len(to_remove)]
    else:
        return text 

def parse_resolution(s, expected_length=None):
    """ 
        Takes a string on the format '(WIDTH,HEIGHT,CHANNELS)' and 
        evaluates to a tuple with ints. Unlike literal_eval in ast, 
        this should be a bit more safe against unsanitized input.
    """
    tup = tuple([int(x.strip()) for x in (s.strip('(').strip(')').split(','))])
    if expected_length is None:
        return tup
    
    assert(len(tup) == expected_length)
    return tup
    
def print_flush(text):
    """ Prints text and immidiately flushes stdout. When running python code
        as subprocesses, with stdout mapped to a log file, print does not flush
        until the subprocess ends. This is inconvenient, as we want to observe progress
        while it is running. Therefore, this function is used in all long-running jobs.
    """
    print(text)
    sys.stdout.flush()
    
def clamp(x, mi, ma):
    """ Makes sure x is within the interval [mi, ma]. I've always liked this implementation.
        '.sort' is ever so slightly faster than 'sorted', because no copy of the list is made. 
    """
    tmp = [mi, x, ma]
    tmp.sort()
    return tmp[1]
    
def split_lambda(some_list, some_function, as_list=False):
    """ Splits a list into a dict such that all elements 'x' in 'some_list' that get one value from 'some_function(x)' 
        end up in the same list. 
        
        Example:
        split_lambda([1,2,3,4,5,6,7], lambda x: x%2) -> {0: [2, 4, 6], 1: [1, 3, 5, 7]}
        
        if as_list is True, then you would instead get [[2,4,6], [1,3,5,7]] in the above example (although order is not guaranteed)

    """
    
    lists = dict()
    for x in some_list:
        y = some_function(x)
        if y in lists:
            lists[y].append(x)
        else:
            lists[y] = [x]
    
    if as_list:
        return list(lists.values())
    else:    
        return lists
    
    
def to_hex(col):
    """ Takes an OpenCV/ImageIO-style color (as a tuple/list of three values in BGR) 
        and converts to RGB hex to be shown in the Web UI.
    """
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(col[0], 0, 255), clamp(col[1], 0, 255), clamp(col[2], 0, 255))
    
def pandas_loop(dataframe, stop=0):
    """ Essentially iterrows on a pandas DataFrame, except much faster. 
        The only real difference in use is that you need to use
        indexing (row['some_column_name']) instead of direct referencing (row.some_column_name)
        If stop is a positive integer, only that many rows are looped over.
        
        The speed difference between this and any built-in functionality in pandas
        is really large, as looping through pandas dataframe was a huge bottleneck
        in some scripts before implementing this.
    """
    
    columns = ['_']
    columns.extend(dataframe.columns.values.tolist())

    should_stop = False
    if stop > 0:
        should_stop = True
        i = 0

    for row in dataframe.to_records():
        obj = {}
        for val, col in zip(row, columns):

            obj[col] = val
        
        yield obj
        
        if should_stop:
            i += 1
            if i >= stop:
                return 
    return
    
def normalize(dx, dy):
    l = sqrt(dx**2 + dy**2)
    if l == 0:
        return 0,0
    else:
        return dx/l, dy/l
        
def rep_last(some_list, n):
    """ Makes a list have a number of elements divisible by n, by repeating the last entry """
    while not (len(some_list)%n == 0):
        some_list.append(some_list[-1])
    
    return some_list
        
if __name__ == '__main__':
    print(parse_resolution('(640, 480,3)', 3))
    
    print(split_lambda([1,2,3,4,5,6,7], lambda x: x%2))
    print(split_lambda([1,2,3,4,5,6,7], lambda x: x%2, as_list=True))

    print(left_remove('abcabcdefg', 'abc'))
    print(right_remove('defabcabc', 'abc'))
    
