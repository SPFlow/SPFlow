"""
Created on March 22, 2018

@author: Alejandro Molina
"""
import subprocess

from spn.algorithms.Inference import log_likelihood
from spn.io.Text import spn_to_str_equation
from spn.structure.Base import get_nodes_by_type, Leaf, eval_spn_bottom_up, Sum, Product
from spn.structure.leaves.parametric.Parametric import Gaussian, Bernoulli
import math
import logging

logger = logging.getLogger(__name__)


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


def get_header(num_inputs, num_nodes, c_data_type="double", header_guard=False):
    header = """
    #include <stdlib.h> 
    #include <stdarg.h>
    #include <cmath> 
    #include <vector> 
    // # include <fenv.h>
    #include <cstdio> 

    using namespace std;

    #define SPN_NUM_INPUTS {num_inputs}
    #define SPN_NUM_NODES {num_nodes}

    void spn_mpe(const vector<{c_data_type}>& evidence, 
                 vector<{c_data_type}>& completion);

    void spn_mpe({c_data_type}* evidence, 
                 {c_data_type}* completion, 
                 size_t data_size);

    void spn_mpe_many({c_data_type}* evidence, {c_data_type}* completion, 
                    size_t data_size, size_t rows);

    {c_data_type} spn(const vector<{c_data_type}>& x, 
                vector<{c_data_type}>& result_node);
    
    {c_data_type} spn({c_data_type}* x, {c_data_type}* result_node);

    void spn_many({c_data_type}* data_in, {c_data_type}* data_out, size_t rows);

    """.format(
        c_data_type=c_data_type, num_inputs=num_inputs, num_nodes=num_nodes
    )

    if header_guard:
        header = (
            """
    #ifndef __SPN_H
    #define __SPN_H
        """
            + header
            + """
    #endif
        """
        )
    return header


def mpe_to_cpp(root, c_data_type="double"):
    eval_functions = {}

    def mpe_prod_to_cpp(node, c_data_type="dobule"):
        ## If I have been selected
        operation = "if (selected[{my_id}]) {{".format(my_id=n.id)
        ## Select all my children.
        for c in node.children:
            operation += """
            selected[{child_id}] = true; 
            max_llh[ {my_id} ] = ll_result[ {my_id} ]; 
            """.format(
                my_id=node.id, child_id=c.id
            )
        # No double when no format?
        operation += "}\n"
        return operation

    def mpe_sum_to_cpp(node, c_data_type="double"):
        ## If I have been selected before (root is always selected)
        operation = "if (selected[{my_id}]) {{".format(my_id=n.id)
        for i, c in enumerate(node.children):
            operation += """
            if ( ll_result[ {child_id} ] + {node_weight:.20}  > max_llh[ {my_id} ] ) {{
                winning_nodes[ {my_id} ] = {child_id}; 
                max_llh[ {my_id} ] = ll_result[{child_id}] + {node_weight:.20} ; 
            }} 
            """.format(
                my_id=node.id, child_id=c.id, node_weight=math.log(node.weights[i])
            )
        operation += """
            selected[winning_nodes[{my_id}]] = true; // now select the node that won. 
        """.format(
            my_id=node.id
        )
        # No double when no format
        operation += "}\n"  # Close if selected.
        return operation

    def mpe_gaussian_to_cpp(node, c_data_type="double"):
        return """if (selected[{my_id}]) {{
                completion[{input_map}] = {mean};
            }}
        """.format(
            my_id=node.id, input_map=node.scope[0], mean=node.mean
        )

    def mpe_bernoulli_to_cpp(node, c_data_type="double"):
        completion_val = float(int(node.p > 0.5))
        return """if (selected[{my_id}] && isnan(completion[{input_map}]) ) {{
                completion[{input_map}] = {completion_val}; 
            }}
        """.format(
            my_id=node.id, input_map=node.scope[0], completion_val=completion_val
        )

    eval_functions[Product] = mpe_prod_to_cpp
    eval_functions[Sum] = mpe_sum_to_cpp
    eval_functions[Gaussian] = mpe_gaussian_to_cpp
    eval_functions[Bernoulli] = mpe_bernoulli_to_cpp

    all_nodes = get_nodes_by_type(root)

    top_down_code = ""
    for n in all_nodes:
        top_down_code += eval_functions[type(n)](n, c_data_type)
        # top_down_code += """
        #     for (size_t n_idx = 0; n_idx < {num_nodes}; n_idx++)
        #     {{
        #         printf(\"%d\", selected[n_idx] ? 1 : 0);
        #     }}
        #     printf(\"\\n\");
        # """.format(num_nodes = len(all_nodes))
        top_down_code += "\n\t\t"

    function_code = """
        void spn_mpe(const vector<{c_data_type}>& evidence, 
                        vector<{c_data_type}>& completion) {{
            // Copy the evidence to completion. 
            completion = evidence; 
            vector<bool> selected( (size_t) {num_nodes}, false);
            selected[0] = true; // Root is always selected.  

            // To hold max_llh values for each node 
            // For sum nodes, we take max over children. 
            // For prod nodes, -INFTY if not selected, llh of itself if selected. 
            vector<{c_data_type}> max_llh((size_t) {num_nodes}, -INFINITY);

            // For each node_id (of sum nodes), keep track of winning nodes. 
            vector<int> winning_nodes((size_t) {num_nodes}, -1);

            // Log likelihood at each node (bottom-up pass)
            vector<{c_data_type}> ll_result; 
            // Do a bottom up pass. 
            {c_data_type} ll = spn(evidence, ll_result); 
            // Top down code
            {top_down_code}
        }}

        void spn_mpe({c_data_type}* evidence, {c_data_type}* completion, size_t data_size)
        {{
            vector<double> _evidence(data_size); 
            vector<double> _completion(data_size); 
            for (size_t i = 0; i < data_size; i++)
            {{
                _evidence[i] = evidence[i]; 
            }}
            spn_mpe(_evidence, _completion);
            for (size_t i = 0; i < data_size; i++)
            {{
                completion[i] = _completion[i]; 
            }}
        }}

        void spn_mpe_many({c_data_type}* evidence, {c_data_type}* completion, 
                        size_t data_size, size_t rows){{
            #pragma omp parallel for
            for (int i=0; i < rows; ++i){{
                vector<double> _evidence(data_size); 
                vector<double> _completion(data_size); 
                unsigned int r = i * data_size;

                for (size_t j = 0; j < data_size; j++)
                {{
                    _evidence[j] = evidence[r + j]; 
                }}
                spn_mpe(_evidence, _completion);
                for (size_t j = 0; j < data_size; j++)
                {{
                    completion[r + j] = _completion[j]; 
                }}
            }}
        }}        
    """.format(
        top_down_code=top_down_code, num_nodes=len(all_nodes), c_data_type=c_data_type
    )
    return function_code


def eval_to_cpp(node, c_data_type="double"):
    eval_functions = {}

    def logsumexp_sum_eval_to_cpp(n, c_data_type="double"):
        operations = []
        for i, c in enumerate(n.children):
            operations.append(
                "result_node[{child_id}]+{log_weight:.40}".format(log_weight=math.log(n.weights[i]), child_id=c.id)
            )
        return "result_node[{node_id}] = logsumexp({num_children},{operation}); // sum node".format(
            vartype=c_data_type, node_id=n.id, num_children=len(n.children), operation=",".join(operations)
        )

    def log_prod_eval_to_cpp(n, c_data_type="double"):
        operation = "+".join(["result_node[" + str(c.id) + "]" for c in n.children])

        return "result_node[{node_id}] = {operation}; //prod node".format(
            vartype=c_data_type, node_id=n.id, operation=operation
        )

    def gaussian_eval_to_cpp(n, c_data_type="double"):
        operation = " - log({stdev}) - (pow(x[{scope}] - {mean}, 2.0) / (2.0 * pow({stdev}, 2.0))) - K".format(
            mean=n.mean, stdev=n.stdev, scope=n.scope[0]
        )
        return """result_node[{node_id}] = {operation};""".format(
            vartype=c_data_type, node_id=n.id, operation=operation
        )

    def bernoulli_eval_to_cpp(n, c_data_type="double"):
        # If isnan, return 1, if not, return proper probability.
        return "result_node[{node_id}] = isnan(x[{scope}]) ? 20.0 : ( x[{scope}] > 0.5 ? log({p_true}) : log(1 - {p_true}) ); //leaf node bernoulli".format(
            vartype=c_data_type, node_id=n.id, scope=n.scope[0], p_true=n.p
        )

    eval_functions[Sum] = logsumexp_sum_eval_to_cpp
    eval_functions[Product] = log_prod_eval_to_cpp
    eval_functions[Gaussian] = gaussian_eval_to_cpp
    eval_functions[Bernoulli] = bernoulli_eval_to_cpp

    num_nodes = len(get_nodes_by_type(node))
    spn_code = ""
    for n in reversed(get_nodes_by_type(node)):
        # spn_code += "\t\t"
        spn_code += eval_functions[type(n)](n, c_data_type=c_data_type)
        spn_code += "\n\t\t"

    # header = get_header(c_data_type=c_data_type)

    function_code = """
    const {c_data_type} K = 0.91893853320467274178032973640561763986139747363778341281;

    {c_data_type} logsumexp(size_t count, ...){{
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
        return ({c_data_type})(max_val + log(result));
    }}

    {vartype} spn(const vector<{vartype}>& x, 
                vector<{vartype}>& result_node){{
        // feenableexcept(FE_INVALID | FE_OVERFLOW);
        // 3.0 is just a temporary number. 
        result_node.resize({num_nodes}, 3.0);
        {spn_code}
        return result_node[0];
    }}
    
    void spn_many({vartype}* data_in, {vartype}* data_out, size_t rows){{
        #pragma omp parallel for
        for (int i=0; i < rows; ++i){{
            vector<double> result_node; 

            // data_in is rows by num_input
            unsigned int r = i * {num_input};
            
            vector<double> input((size_t) {num_input}); 
            // Explicit copy is required for correct operation. 
            for ( size_t i = 0; i < input.size(); i++)
            {{
                input[i] = data_in[i + r];
            }}
            spn(input, result_node);

            // data_out is rows by num_nodes
            unsigned int r2 = i * {num_nodes}; 
            for ( size_t i = 0; i < result_node.size(); i++)
            {{
                data_out[i + r2] = result_node[i]; 
            }}
        }}
    }}
    """.format(
        c_data_type=c_data_type, num_nodes=num_nodes, vartype=c_data_type, spn_code=spn_code, num_input=len(node.scope)
    )
    return function_code


def generate_cpp_code(node, c_data_type="double", outfile=None):

    num_input = len(node.scope)
    num_nodes = len(get_nodes_by_type(node))

    code = (
        get_header(num_input, num_nodes, c_data_type) + eval_to_cpp(node, c_data_type) + mpe_to_cpp(node, c_data_type)
    )
    if outfile:
        f = open(outfile, "w")
        f.write(code)
        f.close()
    return code


def generate_cpp_code_with_header(node, c_data_type="double", filename="spn"):

    num_input = len(node.scope)
    num_nodes = len(get_nodes_by_type(node))

    header = get_header(num_input, num_nodes, c_data_type=c_data_type, header_guard=True)

    code = """
    #include \"{header_file_name}\"
    """.format(
        header_file_name=filename + ".h"
    )
    code += eval_to_cpp(node, c_data_type=c_data_type)
    code += mpe_to_cpp(node, c_data_type=c_data_type)

    with open(filename + ".h", "w") as f_header, open(filename + ".cpp", "w") as f_code:
        f_header.write(header)
        f_code.write(code)


def setup_cpp_bridge(node):
    c_code = generate_cpp_code(node, c_data_type="double")
    import cppyy

    cppyy.cppdef(c_code)
    # logger.info(c_code)


def get_cpp_function(node):
    from cppyy.gbl import spn_many

    import numpy as np

    def python_eval_func(data):
        num_nodes = len(get_nodes_by_type(node))
        # It has to be this way - otherwise the data doesn't appear contiguous in CPP.
        # np.ascontiguousarray doesn't seem to work either.
        results = []
        for _ in range(num_nodes):
            results += np.zeros(shape=(data.shape[0]), dtype="float32").tolist()
        results = np.array(results).reshape((data.shape[0], num_nodes))
        spn_many(data, results, results.shape[0])
        return results

    return python_eval_func


def get_cpp_mpe_function(node):
    from cppyy.gbl import spn_mpe_many

    import numpy as np

    def python_mpe_func(data):
        results = np.zeros((data.shape[0], data.shape[1]))
        spn_mpe_many(data, results, data.shape[1], data.shape[0])
        return results

    return python_mpe_func


def generate_native_executable(spn, cppfile="/tmp/spn.cpp", nativefile="/tmp/spn.o"):
    code = generate_cpp_code(spn, cppfile)

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


_leaf_to_cpp = {}


def register_spn_to_cpp(leaf_type, func):
    _leaf_to_cpp[leaf_type] = func


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
