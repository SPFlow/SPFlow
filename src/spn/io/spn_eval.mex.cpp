#include "mex.h"
#include "spn.h"
#include <cstring> 
// compile together with spn.h

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if ( (nrhs != 1) || (nlhs != 1) ) {
        mexErrMsgIdAndTxt("spn:eval:io", "Expected one input and one output");
    }

    mwSize input_size = mxGetNumberOfElements(prhs[0]); 
    mwSize num_data = mxGetM(prhs[0]); // number of rows

    if (input_size != SPN_NUM_INPUTS){
        mexErrMsgIdAndTxt("spn:eval:size", 
            sprintf("Expected input size is %d, got %d", SPN_NUM_INPUTS, input_size)); 
    }

    double *input_data = mxGetDoubles(prhs[0]);
    plhs[0] = mxCreateDoubleMatrix(num_data, SPN_NUM_NODES, mxREAL); 

    double *result = mxGetDoubles(plhs[0]); 

    spn_many(input_data, result, num_data); 
}