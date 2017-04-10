#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

    VectorXd rmse = VectorXd(estimations[0].array().size());
     rmse.fill(0.0);

    if(estimations.size() != ground_truth.size()  || estimations.size() == 0){ 
        std::cout << "Invalid estimation or ground_truth data" << std::endl; 
        return rmse; 
    }  

    //accumulate squared residuals 
    for(unsigned int i = 0; i < estimations.size(); ++i){  
        VectorXd residual = estimations[i] - ground_truth[i];  

        //coefficient-wise multiplication 
        residual = residual.array()*residual.array(); 
        rmse += residual;
    }

     rmse = rmse/estimations.size();
     rmse = rmse.array().sqrt();
    return rmse;
}
