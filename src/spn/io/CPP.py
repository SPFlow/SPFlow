'''
Created on March 22, 2018

@author: Alejandro Molina
'''
import subprocess

from spn.algorithms.Inference import log_likelihood
from spn.io.Text import spn_to_str_equation
from spn.structure.Base import get_nodes_by_type, Leaf, eval_spn, Sum, Product

from spn.structure.leaves.parametric.Parametric import Gaussian
import math

def to_cpp(node, c_data_type="double"):
    eval_functions = {}

    def logsumexp_sum_to_cpp(n, children, input_vals, c_data_type="double"):
        val = "\n".join(children)

        operations = []
        for i, c in enumerate(n.children):
            operations.append("log({log_weight})+result_node_{child_id}".format(log_weight=n.weights[i],
                                                                                child_id=c.id))

        return val + "\n{vartype} result_node_{node_id} = logsumexp({operation}); //sum node".format(vartype=c_data_type,
                                                                                          node_id=n.id,
                                                                                          operation=",".join(operations))

    def log_prod_to_cpp(n, children, input_vals, c_data_type="double"):
        val = "\n".join(children)
        operation = "+".join(["result_node_" + str(c.id) for c in n.children])

        return val + "\n{vartype} result_node_{node_id} = {operation}; //prod node".format(vartype=c_data_type,
                                                                                           node_id=n.id,
                                                                                           operation=operation)

    def gaussian_to_cpp(n, input_vals, c_data_type="double"):
        return "{vartype} result_node_{node_id} = 0; //leaf node gaussian".format(vartype=c_data_type, node_id=n.id)

    eval_functions[Sum] = logsumexp_sum_to_cpp
    eval_functions[Product] = log_prod_to_cpp
    eval_functions[Gaussian] = gaussian_to_cpp

    return eval_spn(node, eval_functions, c_data_type=c_data_type)


_leaf_to_cpp = {}


def register_spn_to_cpp(leaf_type, func):
    _leaf_to_cpp[leaf_type] = func


def histogram_to_cpp(node, leaf_name, vartype):
    import numpy as np
    inps = np.arange(int(max(node.breaks))).reshape((-1, 1))

    leave_function = """
    {vartype} {leaf_name}_data[{max_buckets}];
    inline {vartype} {leaf_name}(uint8_t v_{scope}){{
        return {leaf_name}_data[v_{scope}];
    }}
    """.format(vartype=vartype, leaf_name=leaf_name, max_buckets=len(inps), scope=node.scope[0])

    leave_init = ""

    for bucket, value in enumerate(np.exp(log_likelihood(node, inps, log_space=False))):
        leave_init += "\t{leaf_name}_data[{bucket}] = {value};\n".format(leaf_name=leaf_name, bucket=bucket,
                                                                         value=value)
    leave_init += "\n"

    return leave_function, leave_init


# register_spn_to_cpp(Histogram, histogram_to_cpp)


def to_cpp2(node):
    vartype = "double"

    spn_eqq = spn_to_str_equation(node,
                                  node_to_str={Histogram: lambda node, x, y: "leaf_node_%s(data[i][%s])" % (
                                      node.name, node.scope[0])})

    spn_function = """
    {vartype} likelihood(int i, {vartype} data[][{scope_size}]){{
        return {spn_eqq};
    }}
    """.format(vartype=vartype, scope_size=len(node.scope), spn_eqq=spn_eqq)

    init_code = ""
    leaves_functions = ""
    for l in get_nodes_by_type(node, Leaf):
        leaf_name = "leaf_node_%s" % (l.name)
        leave_function, leave_init = _leaf_to_cpp[type(l)](l, leaf_name, vartype)

        leaves_functions += leave_function
        init_code += leave_init

    return """
#include <iostream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <iomanip>
#include <chrono>


using namespace std;

{leaves_functions}

{spn_function}

int main() 
{{

    {init_code}
 
    vector<string> lines;
    for (string line; getline(std::cin, line);) {{
        lines.push_back( line );
    }}
    
    int n = lines.size()-1;
    int f = {scope_size};
    auto data = new {vartype}[n][{scope_size}]();
    
    for(int i=0; i < n; i++){{
        std::vector<std::string> strs;
        boost::split(strs, lines[i+1], boost::is_any_of(";"));
        
        for(int j=0; j < f; j++){{
            data[i][j] = boost::lexical_cast<{vartype}>(strs[j]);
        }}
    }}
    
    auto result = new {vartype}[n];
    
    chrono::high_resolution_clock::time_point begin = chrono::high_resolution_clock::now();
    for(int j=0; j < 1000; j++){{
        for(int i=0; i < n; i++){{
            result[i] = likelihood(i, data);
        }}
    }}
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();

    delete[] data;
    
    long double avglikelihood = 0;
    for(int i=0; i < n; i++){{
        avglikelihood += log(result[i]);
        cout << setprecision(60) << log(result[i]) << endl;
    }}
    
    delete[] result;

    cout << setprecision(15) << "avg ll " << avglikelihood/n << endl;
    
    cout << "size of variables " << sizeof({vartype}) * 8 << endl;

    cout << setprecision(15)<< "time per instance " << (chrono::duration_cast<chrono::nanoseconds>(end-begin).count()  / 1000.0) /n << " ns" << endl;
    cout << setprecision(15) << "time per task " << (chrono::duration_cast<chrono::nanoseconds>(end-begin).count()  / 1000.0)  << " ns" << endl;


    return 0;
}}
    """.format(spn_function=spn_function, vartype=vartype, leaves_functions=leaves_functions,
               scope_size=len(node.scope), init_code=init_code)


def generate_native_executable(spn, cppfile="/tmp/spn.cpp", nativefile="/tmp/spnexe"):
    code = to_cpp(spn)

    text_file = open(cppfile, "w")
    text_file.write(code)
    text_file.close()

    nativefile_fast = nativefile + '_fastmath'

    return subprocess.check_output(['g++', '-O3', '--std=c++11', '-o', nativefile, cppfile],
                                   stderr=subprocess.STDOUT).decode("utf-8"), \
           subprocess.check_output(['g++', '-O3', '-ffast-math', '--std=c++11', '-o', nativefile_fast, cppfile],
                                   stderr=subprocess.STDOUT).decode("utf-8"), \
           code
