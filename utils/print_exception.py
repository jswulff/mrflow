import linecache
import sys
import traceback


def print_exception():
    exc_type, exc_obj, tb = sys.exc_info()
    traceback.print_exception(exc_type, exc_obj, tb, limit=10, file=sys.stdout)
