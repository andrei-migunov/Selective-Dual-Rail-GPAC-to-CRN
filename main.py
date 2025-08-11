from basic_ops import *
import pickle 
from plotting import *
from sympy import *
import argparse
import os
import sys




class CompileHistory:

    def __init__(self):
        self.input_system = None
        self.input_iv = None
        self.input_mainvar = None
        self.crn = None
        self.crn_iv = None

    def print(self):
        if self.input_system and self.input_iv:
            print(f'Input system with initial value {self.input_iv}: \n')
            print(format_dict(self.input_system))

        if self.crn and self.crn_iv:
            print(f'CRN system with initial value {self.crn_iv}: \n')
            print(format_dict(self.crn))
            
    def write(self, filename):
        """
        Writes all attribute values of the object to a text file.
        Handles dictionaries and other types appropriately.
        """
        with open(filename, 'w') as f:
            for attr, value in self.__dict__.items():
                f.write(f"{attr}:\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"  {value}\n")
                f.write("\n")
            

    



        
'''
Input: A bounded general-purpose analog computer G, in the form of a Python dictionary G = {x:x',y:y',...,z:z'} mapping variable names to variable ODEs, 
where lim (t -> infty) x(t) = r, i.e. G computes r via x.

This function 
1. Zeroes the system (modifies the system to start with all-0 initial values).
2. Adjusts the real by a constant to account for change resulting from the zero-ing.

In the resulting system, the variable x_1 computes the number r.
'''
def pre_processing(sys, in_iv, *args, **kwargs):
    
    log = bool(kwargs.get("log",False)) # Value of kwargs["log"] otherwise False
    zsys = zeroed_system(sys,in_iv)

    #before dual-railing the zeroed system, we need to reintroduce a variable that 
    #goes to the actual desired, original value from the input system
    csys = add_const_to_x1(zsys,[0]*len(zsys),.5) #TODO: why is this hardcoded
    #clean up the names so they are x_i
    csys = clean_names(csys,"x_1")

    return csys

'''Serializes obj. filename must end in .pkl .'''
def cache_obj(obj,filename):
    if os.path.exists(filename):
        print(f"Warning: {filename} already exists and will be overwritten.")
        
    try:
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
        print(f"Object {filename} of type {type(obj)} cached successfully")
    except Exception as e:
        raise Exception(f"Failed to cache object {filename} of type {type(obj)}. Error: {e}")

def fetch_cached(filename):
    obj = None
    try: 
        obj = unpickle(filename) 
        return obj
    except:
        raise Exception(f"Tried to deserialize an object that probably had not been cached in a previous execution, or simply {filename} does not exist locally.")


''' The main function.

system: a general-purpose analog computer represented as a PIVP represented as a dictionary mapping variable names to those variables' ODEs' expressions.
mainvar: a variable in system.keys() which represents the species converging to the desired real number. '''
def compile(system, mainvar, iv, cache_filename=None, filename=None, checks = False, *args, **kwargs):

    debug = False
    verbose = kwargs.get("verbose",False) # Set to True to produce intermediate system console output.
    checks = kwargs.get("conversionchecks",True) # Set to True to conduct form checks on interemdiate systems (i.e. - is the CRN actually a CRN? More for debugging.)
    sim = kwargs.get("sim",["INPUT","CRN"]) # ["INPUT","CRN"]
    simtime = kwargs.get("simtime",20) 

    #INITIAL SYSTEM 
    ch = CompileHistory()
    ch.input_iv = iv
    ch.input_mainvar = mainvar
    ch.input_system = system

    # First verify that system is actually a GPAC
    if not is_valid_gpac_system(system):
        raise TypeError(f'Input system is not a GPAC.')

    system, iv = clean_names(system, mainvar, iv)
    # Expand all sympy expressions in the system dictionary
    system = {k: expand(v) for k, v in system.items()}

    #CHEMICAL REACTION NETWORK OF ARBITRARY DEGREE
    crn, crn_iv, leader = selective_dual_rail(system, iv, Symbol("x_1"))
    ch.crn = crn
    ch.crn_iv = crn_iv
    ch.crn_mainvar = leader

    if checks:
        if not crn_implementable(crn): raise ValueError('Internal issue: CRN form is not CRN implementable (conversion failed)- please report!')

    if verbose:
        print(f'CRN translation complete, dual railed system below:')
        print(format_dict(crn))

    print("Saving files...")
    if cache_filename:
        cache_obj(ch, cache_filename)

    if filename:
        ch.write(filename)

    # SIMULATIONS, IF SELECTED
    if sim != []:
        print("Running requested simulations. This may or may not take a some time. ...")
        run_simulations(ch, sim, simtime, debug, verbose)
    print (f'Complete. Returning an object containing the full conversion history and output as a sympy dictionary.')
    return ch

'''Simulate the intermediate systems as requested by user input.'''
def run_simulations(ch, sim,simtime,debug,verbose):
    
    if "INPUT" in sim:
        print(f"Simulating input system for {simtime} time units...")
        soln, lim = fsp(ch.input_system,list(ch.input_iv.values()),mainvar=ch.input_mainvar,time_span=(0,simtime))
        if debug or verbose:
            print(f'(Input) Latest simulation value of main variable {ch.input_mainvar} is {lim}.')

    if "CRN" in sim:
        print(f"Simulating CRN-implementable (dual-railed) system for {simtime} time units...")
        soln, lim = fsp(ch.crn,list(ch.crn_iv.values()),time_span=(0,simtime),mainvar=ch.crn_mainvar)
        if debug or verbose:
            print(f'(CRN) Latest simulation value of main variable {ch.crn_mainvar} is {lim}.')

 
"""
Reads a .txt file describing a system, initial values, and optional flags. 
See example_input.txt
"""
def compile_from_file(input_filename):
    def read_next_dict(lines, start_idx):
        """Read a dictionary block enclosed by matching { ... } braces."""
        block_lines = []
        brace_count = 0
        started = False
        i = start_idx

        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            if "{" in line:
                brace_count += line.count("{")
                started = True
            if "}" in line:
                brace_count -= line.count("}")

            if started:
                block_lines.append(line)
            if started and brace_count == 0:
                break
            i += 1

        if brace_count != 0:
            raise ValueError("Unbalanced braces in dictionary block.")

        return "\n".join(block_lines), i + 1

    # Read all lines first
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    # Parse system dictionary
    system_str, next_idx = read_next_dict(lines, 0)
    try:
        system = sympify(ast.literal_eval(system_str))
    except Exception as e:
        raise ValueError(f"Could not parse system dictionary: {e}")

    # Parse IV dictionary
    iv_str, next_idx = read_next_dict(lines, next_idx)
    try:
        iv = sympify(ast.literal_eval(iv_str))
    except Exception as e:
        raise ValueError(f"Could not parse initial values: {e}")

    # Remaining lines: flags
    kwargs = {}
    for line in lines[next_idx:]:
        if "=" in line and not line.strip().startswith("#"):
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            try:
                val = ast.literal_eval(val)
            except:
                pass
            kwargs[key] = val

    # Identify primary variable
    primary_var = Symbol(kwargs.pop("primary_var")) if "primary_var" in kwargs else Symbol(list(system.keys())[0])

    return compile(system, primary_var, iv, **kwargs)



if __name__ == '__main__':
    compile_from_file(sys.argv[1])