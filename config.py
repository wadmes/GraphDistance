# Used for the configuration of the model
import torch
import warnings
import argparse
import sys
import os

class ArgParser(argparse.ArgumentParser):
    """ ArgumentParser with better error message
    """
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


parser = argparse.ArgumentParser("Graph Distance Project")
parser.add_argument("--graph_data", type=str, default="random",
                    help="graph_data for analysis, it can be \
                    iscas, cep, arith,mcnc,random, itc99,reg_ran,")   
parser.add_argument("--distance_metric", type=str, default="hamming",
                    help="distance metric, it can be ,")
# How the str argument is set

opt = parser.parse_args()

# import ast
addable_types = [
    "buf",
    "and",
    "or",
    "xor",
    "not",
    "nand",
    "nor",
    "xnor",
    "0",
    "1",
    "x",
    "output",
    "input",
]

supported_types = addable_types + ["bb_input", "bb_output","key","virtual_key"]

print(opt)