#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void printMatrix(const unsigned char* result, const int rows, const int cols, std::string name){
    std::cout<<"Matrix "<<name<<" = "<<std::endl;
    for(int i=0; i<rows; ++i){
        std::cout<<"line"<<i<<" = ";
        for(int j=0; j<cols; ++j){
            std::cout<<static_cast<int>(result[i*cols+j])<<" ";
            if(j == cols-1){
                std::cout<<std::endl;
            }
        }
    }
}

void printMatrix(const double* result, const int rows, const int cols, std::string name){
    std::cout<<"Matrix "<<name<<" = "<<std::endl;
    for(int i=0; i<rows; ++i){
        std::cout<<"line"<<i<<" = ";
        for(int j=0; j<cols; ++j){
            std::cout<<result[i*cols+j]<<" ";
            if(j == cols-1){
                std::cout<<std::endl;
            }
        }
    }
}

void reduceSumRow(const double *A, double *&result, const int rows_a, const int cols_a){
    for(int i=0; i<rows_a; ++i){
        double sum = 0;
        for(int j=0; j<cols_a; ++j){
            sum += A[i*cols_a+j];
        }
        result[i] = sum;
    }
}

void matrix_formUnitVector(const unsigned char *vector_label, double *&result, const int rows_a, const int cols_a){
    //Need to set result to all zero first.
    for(int i=0; i<rows_a; ++i){
        result[i*cols_a + (int)vector_label[i]] = 1.0;
    }
}

void matrix_normVector(const double *A, const double *vector_sum, double *&result, const int rows_a, const int cols_a){
    for(int i=0; i<rows_a; ++i){
        for(int j=0; j<cols_a; ++j){
            result[i*cols_a+j] = A[i*cols_a+j]/vector_sum[i];
        }
    }
}

void matrix_exp(const double *A, double *&result, const int rows_a, const int cols_a){
    for(int i=0; i<rows_a; ++i){
        for(int j=0; j<cols_a; ++j){
            result[i*cols_a+j] = std::exp(A[i*cols_a+j]);
        }
    }
}

void matrix_transpose(const double *A, double *&result, const int rows_a, const int cols_a){
    int new_rows = cols_a;
    int new_cols = rows_a;

    for(int i=0; i<new_rows; ++i){
        for(int j=0; j<new_cols; ++j){
            result[i*new_cols+j] = A[j*cols_a+i];
        }
    }
}

void matrix_converToFloat(const unsigned char *A,  double *&result, const int rows_a, const int cols_a){
    for(int i=0; i<rows_a; ++i){
        for(int j=0; j<cols_a; ++j){
            result[i*cols_a+j] = static_cast<double>(A[i*cols_a+j]);
        }
    }
}

void matrix_constMultiply(const double *A, double *&result, const double mult_const, const int rows_a, const int cols_a){
    for(int i=0; i<rows_a; ++i){
        for(int j=0; j<cols_a; ++j){
            result[i*cols_a+j] = A[i*cols_a+j]*mult_const;
        }
    }
}

void matrix_addition(const double *A, const double*B, double *&result, const int rows_a, const int cols_a){
    for(int i=0; i<rows_a; ++i){
        for(int j=0; j<cols_a; ++j){
            result[i*cols_a+j] = A[i*cols_a+j]+B[i*cols_a+j];
        }
    }
}
void matrix_addition(const float *A, const double*B, float *&result, const int rows_a, const int cols_a){
    for(int i=0; i<rows_a; ++i){
        for(int j=0; j<cols_a; ++j){
            result[i*cols_a+j] = A[i*cols_a+j]+B[i*cols_a+j];
        }
    }
}
void matrix_convertDoubleType(const float *A, double *&result, const int rows_a, const int cols_a){
    for(int i=0; i<rows_a; ++i){
        for(int j=0; j<cols_a; ++j){
            result[i*cols_a+j] = static_cast<double>(A[i*cols_a+j]);
        }
    }
}
void matrix_convertFloatType(const double *A, float *&result, const int rows_a, const int cols_a){
    for(int i=0; i<rows_a; ++i){
        for(int j=0; j<cols_a; ++j){
            result[i*cols_a+j] = static_cast<float>(A[i*cols_a+j]);
        }
    }
}


void matrix_multiplication(const double *A, const double*B, double *&result, const int rows_a, const int cols_a, const int rows_b, const int cols_b){
    for(int i=0; i<rows_a; ++i){
        for(int k=0; k<cols_b; ++k){
            for(int j=0; j<cols_a; ++j){
                result[i*cols_b+k] += A[i*cols_a+j]*B[j*cols_b+k];
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int num_iter = static_cast<int>(m)/batch;
    for (int iteration=0; iteration<num_iter; ++iteration){
        int rows_x = batch;
        int cols_x = static_cast<int>(n);
        int rows_theta = static_cast<int>(n);
        int cols_theta = static_cast<int>(k);
        const float *subset_X = &X[iteration*batch*cols_x];
        const unsigned char *subset_y = &y[iteration*batch];
        double *subset_doubleX = new double [rows_x*cols_x];
        double *I_y = new double [rows_x*cols_theta]();
        double* Z = new double [rows_x*cols_theta]();
        double* sum_rowZ = new double [rows_x]();
        double* subset_XT = new double [cols_x*rows_x]();
        double *dTheta = new double [rows_theta*cols_theta]();
        double *theta_double = new double [rows_theta*cols_theta]();
        matrix_convertDoubleType(subset_X, subset_doubleX, rows_x, cols_x);
        matrix_convertDoubleType(theta, theta_double, rows_theta, cols_theta);
        matrix_multiplication(subset_doubleX, theta_double, Z, rows_x, cols_x, rows_theta, cols_theta);
        matrix_exp(Z, Z, rows_x, cols_theta);
        reduceSumRow(Z, sum_rowZ, rows_x, cols_theta);
        matrix_normVector(Z, sum_rowZ, Z, rows_x, cols_theta);
        matrix_formUnitVector(subset_y, I_y, rows_x, cols_theta);
        matrix_constMultiply(I_y, I_y, -1, rows_x, cols_theta);
        matrix_addition(Z, I_y, Z, rows_x, cols_theta);
        matrix_transpose(subset_doubleX, subset_XT, rows_x, cols_x);
        matrix_multiplication(subset_XT, Z, dTheta, cols_x, rows_x, rows_x, cols_theta);
        matrix_constMultiply(dTheta, dTheta, -1.f*lr/(float)batch, rows_theta, cols_theta);
        matrix_addition(theta_double, dTheta, theta_double, rows_theta, cols_theta);
        matrix_convertFloatType(theta_double, theta, rows_theta, cols_theta);

        delete [] I_y; I_y = nullptr;
        delete [] Z; Z = nullptr;
        delete [] sum_rowZ; sum_rowZ = nullptr;
        delete [] subset_XT; subset_XT = nullptr;
        delete [] dTheta; dTheta = nullptr;
        delete [] theta_double; theta_double = nullptr;
        delete [] subset_doubleX; subset_doubleX = nullptr;

        /*
        printMatrix(X, m, cols_x, "X");
        printMatrix(y, m, 1, "y");
        printMatrix(subset_X, rows_x, cols_x, "subset_X");
        printMatrix(theta, rows_theta, cols_theta, "theta");
        printMatrix(subset_y, rows_x, 1, "subset_y");
        printMatrix(I_y, rows_x, cols_theta, "I_y");
        printMatrix(Z, rows_x, cols_theta, "Z");
        printMatrix(subset_XT, cols_x, rows_x, "subset_XT");
        printMatrix(dTheta, rows_theta, cols_theta, "dTheta");
        printMatrix(theta, rows_theta, cols_theta, "theta");
        */
    }
    
    /*Test Matrix Operations*/
    /*
    int rows_a = 6;
    int cols_a = 6;
    int rows_b = 6;
    int cols_b = 4;
    float* matrix_a = new float [rows_a*cols_a]();
    float* matrix_apron = new float [rows_a*cols_a]();
    float* matrix_b = new float [rows_b*cols_b]();
    unsigned char* matrix_y_label = new unsigned char [rows_a/2]();
    float* result = new float [rows_a/2*cols_b]();
    float* result_pron = new float [rows_a/2*cols_a]();
    float* result_reduced = new float [rows_a/2*1]();
    float* result_transpose = new float [cols_a*rows_a/2]();
    float* result_normVector = new float [rows_a/2*cols_a]();
    unsigned char* result_formUnitVector = new unsigned char [rows_a/2*cols_a]();
    for(int i=0; i<rows_a*cols_a; ++i){
        matrix_a[i] = i;
        matrix_apron[i] = i+1;
    }
    for(int i=0; i<rows_b*cols_b; ++i){
        matrix_b[i] = 2*i;
    }
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for(int i=0; i<rows_a/2; ++i){
        matrix_y_label[i] = static_cast<unsigned char>(std::rand()%cols_a);
    }
    float *subset_A = &matrix_a[3*cols_a];
    float *subset_Apron = &matrix_apron[2*cols_a];
    std::cout<<"> Test matrix_multiplication()..."<<std::endl;
    matrix_multiplication(matrix_a, matrix_b, result, rows_a/2, cols_a, rows_b, cols_b);
    printMatrix(matrix_a, rows_a/2, cols_a, "matrix_a");
    printMatrix(matrix_b, rows_b, cols_b, "matrix_b");
    printMatrix(result, rows_a/2, cols_b, "result");
    std::cout<<"========"<<std::endl;
    std::fill(result, result+rows_a/2*cols_b, 0);
    matrix_multiplication(subset_A, matrix_b, result, rows_a/2, cols_a, rows_b, cols_b);
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(matrix_b, rows_b, cols_b, "matrix_b");
    printMatrix(result, rows_a/2, cols_b, "result");
    std::cout<<"========"<<std::endl;
    std::cout<<"> Test matrix_addition()..."<<std::endl;
    matrix_addition(subset_A, subset_Apron, result_pron, rows_a/2, cols_a);
    printMatrix(matrix_apron, rows_a, cols_a, "matrix_apron");
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(subset_Apron, rows_a/2, cols_a, "subset_Apron");
    printMatrix(result_pron, rows_a/2, cols_a, "result_pron");
    std::cout<<"========"<<std::endl;
    std::cout<<"> Test matrix_constMultiply()..."<<std::endl;
    std::fill(result_pron, result_pron+rows_a/2*cols_a, 0);
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(result_pron, rows_a/2, cols_a, "result_pron");
    std::cout<<"----after operation----"<<std::endl;
    matrix_constMultiply(subset_A, result_pron, 3.5, rows_a/2, cols_a);
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(result_pron, rows_a/2, cols_a, "result_pron");
    std::cout<<"========"<<std::endl;
    std::cout<<"> Test reduceSumRow()..."<<std::endl;
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(result_reduced, rows_a/2, 1, "result_reduced");
    std::cout<<"----after operation----"<<std::endl;
    reduceSumRow(subset_A, result_reduced, rows_a/2, cols_a);
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(result_reduced, rows_a/2, 1, "result_reduced");
    std::cout<<"========"<<std::endl;
    std::cout<<"> Test matrix_transpose()..."<<std::endl;
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(result_transpose, cols_a, rows_a/2, "result_transpose");
    std::cout<<"----after operation----"<<std::endl;
    matrix_transpose(subset_A, result_transpose, rows_a/2, cols_a);
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(result_transpose, cols_a, rows_a/2, "result_transpose");
    std::cout<<"========"<<std::endl;
    std::cout<<"> Test matrix_exp()..."<<std::endl;
    std::fill(result_pron, result_pron+rows_a/2*cols_a, 0);
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(result_pron, rows_a/2, cols_a, "result_pron");
    std::cout<<"----after operation----"<<std::endl;
    matrix_exp(subset_A, result_pron, rows_a/2, cols_a);
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(result_pron, rows_a/2, cols_a, "result_pron");
    std::cout<<"========"<<std::endl;
    std::cout<<"> Test matrix_normVector()..."<<std::endl;
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(result_reduced, rows_a/2, 1, "result_reduced");
    std::cout<<"----after operation----"<<std::endl;
    matrix_normVector(subset_A, result_reduced, result_normVector, rows_a/2, cols_a);
    printMatrix(subset_A, rows_a/2, cols_a, "subset_A");
    printMatrix(result_reduced, rows_a/2, 1, "result_reduced");
    printMatrix(result_normVector, rows_a/2, cols_a, "result_normVector");
    std::cout<<"========"<<std::endl;
    std::cout<<"> Test matrix_formUnitVector()..."<<std::endl;
    printMatrix(matrix_y_label, rows_a/2, 1, "matrix_y_label");
    printMatrix(result_formUnitVector, rows_a/2, cols_a, "result_formUnitVector");
    std::cout<<"----after operation----"<<std::endl;
    matrix_formUnitVector(matrix_y_label, result_formUnitVector, rows_a/2, cols_a);
    printMatrix(matrix_y_label, rows_a/2, 1, "matrix_y_label");
    printMatrix(result_formUnitVector, rows_a/2, cols_a, "result_formUnitVector");
    */
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
