#include "initial_alignment.h"

namespace
{
    void solveGyroscopeBias(const std::map<double, ImageFrame> &frames, Eigen::Vector3d *Bgs)
    {
        Eigen::Matrix3d A;
        Eigen::Vector3d b;
        Eigen::Vector3d delta_bg;
        A.setZero();
        b.setZero();

        for (auto curFrame = frames.begin(); std::next(curFrame) != frames.end(); curFrame++)
        {
            auto nextFrame = next(curFrame);
            Eigen::Matrix3d tmp_A;
            Eigen::Vector3d tmp_b;
            tmp_A.setZero();
            tmp_b.setZero();

            Eigen::Quaterniond q_ij(curFrame->second.R.transpose() * nextFrame->second.R);
            tmp_A = nextFrame->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
            tmp_b = 2 * (nextFrame->second.pre_integration->delta_q.inverse() * q_ij).vec();
            A += tmp_A.transpose() * tmp_A;
            b += tmp_A.transpose() * tmp_b;
        }
        delta_bg = A.ldlt().solve(b);
        ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

        for (int i = 0; i <= WINDOW_SIZE; i++)
            Bgs[i] += delta_bg;

        for (auto curFrame = frames.begin(); next(curFrame) != frames.end(); curFrame++)
        {
            auto nextFrame = next(curFrame);
            nextFrame->second.pre_integration->repropagate(Eigen::Vector3d::Zero(), Bgs[0]);
        }
    }

    Eigen::MatrixXd TangentBasis(const Eigen::Vector3d &g0)
    {
        Vector3d b, c;
        Vector3d a = g0.normalized();
        Vector3d tmp(0, 0, 1);
        if (a == tmp)
            tmp << 1, 0, 0;
        b = (tmp - a * (a.transpose() * tmp)).normalized();
        c = a.cross(b);
        MatrixXd bc(3, 2);
        bc.block<3, 1>(0, 0) = b;
        bc.block<3, 1>(0, 1) = c;
        return bc;
    }

    void RefineGravity(const std::map<double, ImageFrame> &frames, Eigen::Vector3d &g, Eigen::VectorXd &x)
    {
        Vector3d g0 = g.normalized() * G.norm();
        Vector3d lx, ly;
        // VectorXd x;
        int all_frame_count = frames.size();
        int n_state = all_frame_count * 3 + 2 + 1;

        MatrixXd A{n_state, n_state};
        A.setZero();
        VectorXd b{n_state};
        b.setZero();

        for (int k = 0; k < 4; k++)
        {
            MatrixXd lxly(3, 2);
            lxly = TangentBasis(g0);
            int i = 0;
            for (auto frame_i = frames.begin(); next(frame_i) != frames.end(); frame_i++, i++)
            {
                auto frame_j = next(frame_i);

                MatrixXd tmp_A(6, 9);
                tmp_A.setZero();
                VectorXd tmp_b(6);
                tmp_b.setZero();

                double dt = frame_j->second.pre_integration->sum_dt;

                tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
                tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
                tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
                tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

                tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
                tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
                tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
                tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;

                Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
                // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
                // MatrixXd cov_inv = cov.inverse();
                cov_inv.setIdentity();

                MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
                VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

                A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
                b.segment<6>(i * 3) += r_b.head<6>();

                A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
                b.tail<3>() += r_b.tail<3>();

                A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
                A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
            }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            // double s = x(n_state - 1);
        }
        g = g0;
    }

    bool LinearAlignment(map<double, ImageFrame> &frames, Vector3d &g, VectorXd &x)
    {
        int all_frame_count = frames.size();
        int n_state = all_frame_count * 3 + 3 + 1;

        MatrixXd A{n_state, n_state};
        VectorXd b{n_state};
        A.setZero();
        b.setZero();

        int i = 0;
        for (auto frame_i = frames.begin(); next(frame_i) != frames.end(); frame_i++, i++)
        {
            auto frame_j = next(frame_i);

            MatrixXd tmp_A(6, 10);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
            tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            // MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
            b.tail<4>() += r_b.tail<4>();

            A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
            A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        double s = x(n_state - 1) / 100.0;
        ROS_DEBUG("estimated scale: %f", s);
        g = x.segment<3>(n_state - 4);
        ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
        if (fabs(g.norm() - G.norm()) > 1.0 || s < 0)
        {
            return false;
        }

        RefineGravity(frames, g, x);
        s = (x.tail<1>())(0) / 100.0;
        (x.tail<1>())(0) = s;
        ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
        return !(s < 0.0);
    }
}

bool VisualIMUAlignment(const std::map<double, ImageFrame> &frames, Eigen::Vector3d *Bgs, Eigen::Vector3d &g, Eigen::VectorXd &x)
{
    solveGyroscopeBias(frames, Bgs);
    return LinearAlignment(frames, g, x);
}
