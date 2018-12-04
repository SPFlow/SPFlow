"""
Created on March 22, 2018

@author: Alejandro Molina
"""
import subprocess

from spn.algorithms.Inference import log_likelihood
from spn.io.Text import spn_to_str_equation
from spn.structure.Base import get_nodes_by_type, Leaf, eval_spn_bottom_up, Sum, Product
from spn.structure.leaves.parametric.Parametric import Gaussian
import math


def to_cpp(node, c_data_type="double"):
    eval_functions = {}

    def logsumexp_sum_to_cpp(n, c_data_type="double"):
        operations = []
        for i, c in enumerate(n.children):
            operations.append(
                "result_node_{child_id}+{log_weight:.20}".format(log_weight=math.log(n.weights[i]), child_id=c.id)
            )

        return "\n{vartype} result_node_{node_id} = logsumexp({num_children},{operation}); //sum node".format(
            vartype=c_data_type, node_id=n.id, num_children=len(n.children), operation=",".join(operations)
        )

    def log_prod_to_cpp(n, c_data_type="double"):
        operation = "+".join(["result_node_" + str(c.id) for c in n.children])

        return "\n{vartype} result_node_{node_id} = {operation}; //prod node".format(
            vartype=c_data_type, node_id=n.id, operation=operation
        )

    def gaussian_to_cpp(n, c_data_type="double"):
        operation = " - log({stdev}) - (pow(x_{scope} - {mean}, 2.0) / (2.0 * pow({stdev}, 2.0))) - K".format(
            mean=n.mean, stdev=n.stdev, scope=n.scope[0]
        )
        return "{vartype} result_node_{node_id} = {operation}; //leaf node gaussian".format(
            vartype=c_data_type, node_id=n.id, operation=operation
        )

    eval_functions[Sum] = logsumexp_sum_to_cpp
    eval_functions[Product] = log_prod_to_cpp
    eval_functions[Gaussian] = gaussian_to_cpp

    params = ",".join([c_data_type + " x_" + str(s) for s in range(len(node.scope))])

    spn_code = ""
    for n in reversed(get_nodes_by_type(node)):
        spn_code += eval_functions[type(n)](n, c_data_type=c_data_type)
        spn_code += "\n"

    header = """
    #include <stdarg.h>
    
    using namespace std;
    
    const {vartype} K = 0.91893853320467274178032973640561763986139747363778341281;

    {vartype} logsumexp(size_t count, ...){{
        va_list args;
        va_start(args, count);
        double max_val = va_arg(args, double);
        for (int i = 1; i < count; ++i) {{
            double num = va_arg(args, double);
            if(num > max_val){{
                max_val = num;
            }}
        }}
        va_end(args);

        double result = 0.0;

        va_start(args, count);
        for (int i = 0; i < count; ++i) {{
            double num = va_arg(args, double);
            result += exp(num - max_val);
        }}
        va_end(args);
        return ({vartype})(max_val + log(result));
    }}
    """.format(
        vartype=c_data_type
    )

    spn_execution_params = ",".join(["data_in[r+%s]" % s for s in range(len(node.scope))])

    function_code = """
    {vartype} spn({parameters}){{
        {spn_code}
        return result_node_0;
    }}
    
    void spn_many({vartype}* data_in, {vartype}* data_out, size_t rows){{
        #pragma omp parallel for
        for (int i=0; i < rows; ++i){{
            unsigned int r = i * {scope_len};
            data_out[i] = spn({spn_execution_params});
        }}
    }}
        
    """.format(
        vartype=c_data_type,
        parameters=params,
        spn_code=spn_code,
        spn_execution_params=spn_execution_params,
        scope_len=len(node.scope),
    )
    return header + function_code


def get_cpp_function(node):
    c_code = to_cpp(node, c_data_type="double")
    import cppyy

    cppyy.cppdef(c_code)
    # print(c_code)
    from cppyy.gbl import spn_many

    import numpy as np

    def python_eval_func(data):
        results = np.zeros((data.shape[0], 1))
        spn_many(data, results, data.shape[0])
        return results

    return python_eval_func


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
    """.format(
        vartype=vartype, leaf_name=leaf_name, max_buckets=len(inps), scope=node.scope[0]
    )

    leave_init = ""

    for bucket, value in enumerate(np.exp(log_likelihood(node, inps, log_space=False))):
        leave_init += "\t{leaf_name}_data[{bucket}] = {value};\n".format(
            leaf_name=leaf_name, bucket=bucket, value=value
        )
    leave_init += "\n"

    return leave_function, leave_init


# register_spn_to_cpp(Histogram, histogram_to_cpp)


def to_cpp2(node):
    vartype = "double"

    spn_eqq = spn_to_str_equation(
        node, node_to_str={Histogram: lambda node, x, y: "leaf_node_%s(data[i][%s])" % (node.name, node.scope[0])}
    )

    spn_function = """
    {vartype} likelihood(int i, {vartype} data[][{scope_size}]){{
        return {spn_eqq};
    }}
    """.format(
        vartype=vartype, scope_size=len(node.scope), spn_eqq=spn_eqq
    )

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
    """.format(
        spn_function=spn_function,
        vartype=vartype,
        leaves_functions=leaves_functions,
        scope_size=len(node.scope),
        init_code=init_code,
    )


def generate_native_executable(spn, cppfile="/tmp/spn.cpp", nativefile="/tmp/spnexe"):
    code = to_cpp(spn)

    text_file = open(cppfile, "w")
    text_file.write(code)
    text_file.close()

    nativefile_fast = nativefile + "_fastmath"

    return (
        subprocess.check_output(
            ["g++", "-O3", "--std=c++11", "-o", nativefile, cppfile], stderr=subprocess.STDOUT
        ).decode("utf-8"),
        subprocess.check_output(
            ["g++", "-O3", "-ffast-math", "--std=c++11", "-o", nativefile_fast, cppfile], stderr=subprocess.STDOUT
        ).decode("utf-8"),
        code,
    )
