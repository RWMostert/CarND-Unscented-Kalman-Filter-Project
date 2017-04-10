#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.6;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    // state dimension
    n_x_ = 5;

    time_us_ = 0;

    // augmented dimension
    n_aug_ = 7;

    NIS_radar_ = 0.0;
    NIS_laser_ = 0.0;

    x_ = VectorXd(n_x_);
    x_ << 0.1, 0.1, 0.1, 0.1, 0.01;

    //create example covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    P_ <<   0.2,    0,      0,      0,      0,
            0,      0.2,    0,      0,      0,
            0,      0,      0.2,    0,      0,
            0,      0,      0,      0.3,    0,
            0,      0,      0,      0,      0.3;


    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

    // set weights
    weights_ = VectorXd(2 * n_aug_ + 1);

    lambda_ = 3 - n_aug_;

    double weight_0 = lambda_/(lambda_+n_aug_);
    weights_(0) = weight_0;
    for(int i = 1; i < 2 * n_aug_ + 1; i++) {
        double weight = 0.5/(n_aug_ + lambda_);
        weights_(i) = weight;
    }

    H_ = MatrixXd(2, 5);
    H_ <<   1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;

    R_lidar_ = MatrixXd(2, 2);
    R_lidar_ << pow(std_laspx_, 2.0),   0,
            0,                      pow(std_laspy_, 2.0);

    R_radar_ = MatrixXd(3, 3);
    R_radar_ << pow(std_radr_, 2.0),    0,                      0,
            0,                      pow(std_radphi_, 2.0),  0,
            0,                      0,                      pow(std_radrd_, 2.0);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /*****************************************************************************
    *  Initialization
    ****************************************************************************/
    if (!is_initialized_) {
        // first measurement
        if (meas_package.raw_measurements_(0) != 0 && meas_package.raw_measurements_(1) != 0 ) {
            if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
                /**
                Convert radar from polar to cartesian coordinates and initialize state.
                */
                double x =  meas_package.raw_measurements_(0) * cos(meas_package.raw_measurements_(1));
                double y =  meas_package.raw_measurements_(0) * sin(meas_package.raw_measurements_(1));
                double vx = meas_package.raw_measurements_(2) * cos(meas_package.raw_measurements_(1));
                double vy = meas_package.raw_measurements_(2) * sin(meas_package.raw_measurements_(1));

                if(fabs(vx) < 0.0001) x_(3) = vy/vx;
                else x_(3) = 0.1;

                x_(0) = x;  x_(1) = y;
                x_(2) = sqrt(pow(vx, 2.0) + pow(vy, 2.0));
                x_(4) = 0.01;
            }
            else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
                x_(0) = meas_package.raw_measurements_(0);
                x_(1) = meas_package.raw_measurements_(1);
            }

            is_initialized_ = true;
            time_us_ = meas_package.timestamp_;
        }
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        UpdateRadar(meas_package);

    } else {
        // Laser updates
        UpdateLidar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

    VectorXd x_aug = VectorXd(7);
    x_aug(5) = 0;   x_aug(6) = 0;
    x_aug.head(5) = x_;

    MatrixXd P_aug = MatrixXd(7, 7);
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(5,5) = pow(std_a_, 2.0);
    P_aug(6,6) = pow(std_yawdd_, 2.0);

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    for (int i = 0; i < n_aug_; i++)
    {
        Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_)= x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //extract values for better readability
        double p_x  = Xsig_aug(0, i);
        double p_y  = Xsig_aug(1, i);
        double v    = Xsig_aug(2, i);
        double yaw  = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p = 0.0;  double py_p = 0.0;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p  = v_p + nu_a * delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }

    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        x_ = x_ + weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        //angle normalization
        while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
        while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

    VectorXd z_pred = H_ * x_;
    VectorXd y = meas_package.raw_measurements_ - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_lidar_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
    P_ = (I - K * H_) * P_;

    NIS_laser_ = y.transpose() * Si * y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v   = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1  = cos(yaw) * v;
        double v2  = sin(yaw) * v;

        // measurement model
        Zsig(0,i) = sqrt( pow(p_x, 2.0) + pow(p_y, 2.0) );
        Zsig(1,i) = atan2(p_y, p_x);
        Zsig(2,i) = (p_x * v1  +  p_y * v2) / Zsig(0,i);
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(3);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(3,3);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
        while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    S = S + R_radar_;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, 3);
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        while (x_diff(3) >  M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    //angle normalization
    while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();

    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
