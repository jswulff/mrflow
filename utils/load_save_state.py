#! /usr/bin/env python2

import shelve

def save_state(fname, variables=None, scope=None):
    """ Save all current variables to file

    If variables is not None, save all variables in that list.
    For example, call as

        save_state('some_file.dat', scope=locals())

    to save all variables in the current local scope.

    If variables is None, save all variables in global scope (can be a lot!)

    Contains code taken from
    http://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session
    """

    shelf = shelve.open(fname, 'n')

    if scope is None:
        scope = globals()

    if variables is None:
        variable_list = scope.keys()
    else:
        variable_list = variables

    for key in variable_list:
        try:
            print('  Saving state: Adding {}'.format(key))
            shelf[key] = scope[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('  Saving state: ERROR shelving: {0}'.format(key))
        except KeyError:
            # Variable not found in global state
            print('  Saving state: Variable {} not found in global scope'.format(key))
    shelf.close()


def load_state(fname, scope):
    """ Load current variables from file.

    Load variables from file and put into scope (which is usually a dict)

    In the code, this can be used as such:

        load_state('some_file.dat', scope=globals())

    Contains code taken from
    http://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session

    """
    shelf = shelve.open(fname)
    for key in shelf:
        print('  Adding {} to global variables.'.format(key))
        scope[key] = shelf[key]
    shelf.close()


def test_save():
    import numpy as np
    A = np.random.rand(10,100)
    b = np.random.rand(100)
    c = np.random.rand(1000*1000)

    save_state('./statefile.dat', scope=locals())

def test_load():
    load_state('./statefile.dat', scope=globals())
    print(A)
    print(A.shape)
    print(b)
    print(b.shape)

