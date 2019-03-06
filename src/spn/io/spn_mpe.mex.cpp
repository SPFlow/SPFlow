#include "mex.hpp"
#include "mexAdapter.hpp"
// #include "matrix.h"
#include "spn.h"
#include <algorithm> 
// #include <string> 
// compile together with spn.h

using namespace std; 
using namespace matlab; 

class MexFunction : public mex::Function {
public:
    void operator()(mex::ArgumentList outputs, 
                    mex::ArgumentList inputs) {
        
        shared_ptr<engine::MATLABEngine> matlabPtr = getEngine();
        data::ArrayFactory factory; 
        
        if ( (inputs.size() != 1) || (outputs.size() != 1) ) {
            matlabPtr->feval(engine::convertUTF8StringToUTF16String("error"),
            0, vector<data::Array>({ factory.createScalar("Expected one input/output")} ));
        }



        data::TypedArray<double> mw_input = move(inputs[0]); 

        if ( mw_input.getNumberOfElements() != SPN_NUM_INPUTS ) {
            matlabPtr->feval(engine::convertUTF8StringToUTF16String("error"),
            0, vector<data::Array>({ factory.createScalar("Number of input mismatch")} ));
        }


        vector<double> input( mw_input.getNumberOfElements() ); 
        // for (size_t i = 0; i < inputs[0].size(); i++)
        // {
        //     input[i] = inputs[0][i]; 
        // }
        copy(mw_input.begin(), mw_input.end(), input.begin()); 
        vector<double> output; 

        spn_mpe(input, output); 

        data::ArrayDimensions dims({1, output.size() }); 
        data::TypedArray<double> mw_output = factory.createArray(
            dims, output.begin(), output.end()
        ); 
        // mw_output.resize(output.size()); 
        // copy(output.begin(), output.end(), mw_output.begin()); 

        outputs[0] = move(mw_output); 
        // outputs[0].resize(output.size()); 
        // for (size_t i = 0; i < output.size(); i++)
        // {
        //     outputs[0][i] = output[i]; 
        // }

    }
};



// void mexFunction(int nlhs, mxArray *plhs[],
//                  int nrhs, const mxArray *prhs[])
// {

//     mwSize input_size = mxGetNumberOfElements(prhs[0]); 
//     mwSize num_data = mxGetM(prhs[0]); // number of rows

//     if (input_size != SPN_NUM_INPUTS){
//         mexErrMsgIdAndTxt("spn:eval:size", "Input size mismatch"); 
//     }

//     double *input_data = mxGetDoubles(prhs[0]);
//     plhs[0] = mxCreateDoubleMatrix(num_data, SPN_NUM_NODES, mxREAL); 

//     double *result = mxGetDoubles(plhs[0]); 

//     spn_mpe_many(input_data, result, input_size, num_data); 
// }