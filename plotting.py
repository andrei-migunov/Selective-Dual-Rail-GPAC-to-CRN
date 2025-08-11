import plotly.express as px
import numpy as np
import scipy.integrate as intg
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from sympy.utilities.lambdify import *
from sympy import *
from sympy import symbols, sympify
import pickle
import basic_ops
import datetime
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, ImplicitEuler, PIDController, Kvaerno5
import jax.numpy as jnp
import sympy2jax as s2j

'''You only have to functionize a dictionary to solve and plot it. This is not a part of the actual transformation process.'''
def functionize_dict(system):
    spfy = sympify(system)
    def sys_rhs(t, y):
        # Create a dictionary mapping variable names to their current values
        var_values = {var: value for var, value in zip(spfy.keys(), y)}

        # Evaluate each expression in the system
        dxdts = [expression.evalf(subs=var_values) for expression in spfy.values()]
        return dxdts

    # Return the system of equations and the variable names
    return sys_rhs, list(spfy.keys())



def sympy_to_diffrax_term(sympy_system):
    """
    Convert a dict of SymPy SYSTEM (var -> expression) into a diffrax.ODETerm.
    """
    symbols = list(sympy_system.keys())
    expressions = [sympy_system[var] for var in symbols]
    
    # Build the Sympy2JAX module
    mod = s2j.SymbolicModule(expressions)
    
    def vector_field(t, y, args):
        # Convert all values to jnp-compatible floats
        param_dict = {str(sym): jnp.asarray(y_i) for sym, y_i in zip(symbols, y)}
        return jnp.asarray(mod(**param_dict))  # Ensure result is JAX array too
    
    return ODETerm(vector_field)

def format_dict(dict):
    """
    Formats a dictionary to be more readable.
    """
    formatted = ""
    for key, value in dict.items():
        formatted += f"{key}: {value}\n"
    return formatted

''' Solves the ODE system and plots it. 
Option to show the sum of all variable values default false.
Option to show the running maximum of the sum of variable values default false.'''
def solve_and_plot_ode(ode_func, initial_values, time_span=(0,20), num_points=101,showSum=False, showMax = False, var_names =None,plot=True, sympy_system = None):
    # Time grid and initial value as JAX array
    times = np.linspace(time_span[0], time_span[1], num_points)
    y0 = jnp.array(initial_values, dtype=jnp.float32)
    
    # Get term from sympy system or fall back to ode_func
    if sympy_system:
        term = sympy_to_diffrax_term(sympy_system)
    elif ode_func:
        term = ODETerm(ode_func)
    else:
        raise ValueError("Must provide either a sympy_system or an ode_func.")

    solver = Kvaerno5()#ImplicitEuler()#Dopri5()
    # For ImplicitEuler, a controller is required for adaptive step sizing.
    # Here we use a PIDController with default tolerances.
    controller = PIDController(rtol=1e-4, atol=1e-6)

    soln = diffeqsolve(
        term,
        solver,
        t0=time_span[0],
        t1=time_span[1],
        dt0=1e-4,
        y0=y0,
        saveat=SaveAt(ts=times),
        stepsize_controller = controller,
        max_steps = None
    )
    if plot:
        fig, ax = plt.subplots()
        plt.subplots_adjust(right=0.7)  # Space for checkbox widget
        lines = []

        # Plot each variable
        for i in range(soln.ys.shape[1]):
            label = var_names[i] if var_names else f'Variable {i+1}'
            line, = ax.plot(soln.ts, soln.ys[:, i], '-', label=label)
            lines.append(line)
        
        # Plot sum of variables if requested
        if showSum:
            total = jnp.sum(soln.ys, axis=1)
            line, = ax.plot(soln.ts, total, '-', label="Sum")
            lines.append(line)

        # Plot running max of total
        if showMax:
            total = jnp.sum(soln.ys, axis=1)
            running_max = jnp.maximum.accumulate(total)
            line, = ax.plot(soln.ts, running_max, '-', label="Max")
            lines.append(line)

        # Interactive checkboxes
        labels = [line.get_label() for line in lines]
        visibility = [line.get_visible() for line in lines]
        check = CheckButtons(plt.axes([0.75, 0.4, 0.2, 0.5]), labels, visibility)

        def func(label):
            index = labels.index(label)
            lines[index].set_visible(not lines[index].get_visible())
            plt.draw()

        check.on_clicked(func)

        # Labels and legend
        ax.set_xlabel("Time")
        ax.set_ylabel("Values")
        ax.set_title("Solution of the ODE System")
        ax.legend()
        plt.show()

    return soln


'''Functionize sys, solve it using the provided initial values (iv), write all of that to a logfile.'''

def fsp(sys, iv, debug=True, log=True, time_span=(0, 20), num_points=101, showSum=False, showMax=False, logfilename='ode_log.txt', plot=True, mainvar = Symbol('x_1')):
    fsys, names = functionize_dict(sys)
    
    if log:
        write_log_before_solving(sys, iv, names, logfilename)
    
    soln = solve_and_plot_ode(fsys, iv, var_names=names, time_span=time_span, plot=plot, sympy_system=sys)
    leader_limit = soln.ys[-1, names.index(mainvar)] if mainvar in names else None
    
    if log:
        write_log_after_solving(names, soln, logfilename)
        
    return soln, leader_limit

def write_simple(sys,iv,names,filename):
    """
    Writes the given ODE system and initial values to a log file in a human-readable form before solving the ODEs.

    :param sys: The ODE system to be logged (dictionary with sympy expressions).
    :param iv: Initial values of the variables (list).
    :param names: List of variable names corresponding to the system's variables.
    :param filename: The name of the log file (default is 'ode_log.txt').
    """
    with open(filename, 'a') as log_file:
        log_file.write(f'ODE System (Before Solving), Timestamp ({datetime.datetime.now()}):\n')
        for v, eq in sys.items():
            log_file.write(f"{v}' = {eq}\n")
        log_file.write('\n\n')  # Extra space before the next section

        log_file.write("Initial Values:\n")
        for name, val in zip(names, iv):
            log_file.write(f"{name} = {val}\n")
        log_file.write('\n\n')  # Extra space before the next section

def write_log_before_solving(sys, iv, names, filename='ode_log.txt'):
    write_simple(sys,iv,names,filename)

def write_log_after_solving(names, soln, filename='ode_log.txt'):
    """
    Writes the solutions of the ODE system to a log file in a human-readable form after solving the ODEs.

    :param names: List of variable names corresponding to the system's variables.
    :param soln: Solutions of the ODE system (dictionary).
    :param filename: The name of the log file (default is 'ode_log.txt').
    """
    # Unpack the tuple returned by sample_solution into sampled_soln and selected_times
    sampled_soln, selected_times = sample_solution(soln, names)

    with open(filename, 'a') as log_file:
        log_file.write(f'Sampled Solutions (After Solving) Timestamp ({datetime.datetime.now()}):\n')
        # Log selected times first
        log_file.write(f"Selected Times: {selected_times}\n\n")

        # Now log the sampled solutions
        for var, values in sampled_soln.items():
            log_file.write(f"{var}: {values}\n")

        log_file.write('\n\n')  # Extra space before the end

# def sample_solution(soln, names):
#     """
#     Samples the solution at eight specific times over the span of the execution and returns the sampled solutions along with the selected times.
#     Two times are early, and six times are fairly late to observe stabilization.

#     :param soln: Solution object returned by solve_ivp, expected to have 't' for times and 'y' for solution values.
#     :param names: List of variable names corresponding to the solution's variables.
#     :return: A tuple containing a dictionary with sampled values for each variable at the specified times, and the list of selected times.
#     """
#     sampled_soln = {}
#     total_times = len(soln.ts)
    
#     # Determine sample times: 2 early, 6 late
#     early_times = [0, total_times // 8]
#     late_times = [total_times * 5 // 8, total_times * 6 // 8, total_times * 7 // 8, total_times * 15 // 16, total_times * 31 // 32, total_times - 1]
#     sampled_times = early_times + late_times
    
#     # Convert sampled times from indices to actual times
#     selected_times = [soln.t[t] for t in sampled_times]
    
#     # Sample for each variable using the provided names
#     for i, var_series in enumerate(soln.y):
#         sampled_values = [var_series[t] for t in sampled_times]
#         sampled_soln[names[i]] = sampled_values

#     return sampled_soln, selected_times

'''Above fn but diffrax'''
def sample_solution(soln, names):
    """
    Samples the solution at eight specific times over the span of the execution and returns the sampled solutions along with the selected times.
    Two times are early, and six times are fairly late to observe stabilization.

    :param soln: Diffrax solution object with 'ts' (times) and 'ys' (values with shape [n_times, n_vars]).
    :param names: List of variable names corresponding to the solution's variables.
    :return: A tuple containing a dictionary with sampled values for each variable at the specified times, and the list of selected times.
    """
    sampled_soln = {}
    total_times = len(soln.ts)
    
    # Indices: 2 early, 6 late
    early_indices = [0, total_times // 8]
    late_indices = [
        total_times * 5 // 8,
        total_times * 6 // 8,
        total_times * 7 // 8,
        total_times * 15 // 16,
        total_times * 31 // 32,
        total_times - 1
    ]
    sampled_indices = early_indices + late_indices
    
    # Convert indices to actual times
    selected_times = [soln.ts[i] for i in sampled_indices]
    
    # For each variable, sample values at those times
    for var_index, var_name in enumerate(names):
        sampled_values = [float(soln.ys[i][var_index]) for i in sampled_indices]
        sampled_soln[var_name] = sampled_values

    return sampled_soln, selected_times

def unpickle(filename):
    sys = {}
    try: 
        with open(filename, 'rb') as file:
            sys = pickle.load(file)
        return sys
    except:
        raise Exception(f'Error during de-serialization. Are you sure file {filename} exists?')