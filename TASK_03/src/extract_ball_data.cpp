#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>

// ============================================================================
// 残差函数类 - 用于Ceres优化
// ============================================================================
class TrajectoryResidual {
public:
    TrajectoryResidual(double time, double observed_x, double observed_y, 
                      double initial_x, double initial_y) 
        : time_(time), observed_x_(observed_x), observed_y_(observed_y),
          initial_x_(initial_x), initial_y_(initial_y) {}
    
    template <typename T>
    bool operator()(const T* const parameters, T* residual) const {
        // 参数解析
        const T& initial_velocity_x = parameters[0];  // x方向初始速度 (px/s)
        const T& initial_velocity_y = parameters[1];  // y方向初始速度 (px/s)
        const T& gravity = parameters[2];             // 重力加速度 (px/s²)
        const T& drag_coefficient = parameters[3];    // 阻力系数 (1/s)
        
        const T delta_time = T(time_);  // 时间增量
        
        // x方向运动方程: x(t) = x0 + (vx0/k) * (1 - exp(-k*t))
        const T predicted_x = T(initial_x_) + 
                            (initial_velocity_x / drag_coefficient) * 
                            (T(1.0) - ceres::exp(-drag_coefficient * delta_time));
        
        // y方向运动方程: y(t) = y0 + (vy0 + g/k)/k * (1 - exp(-k*t)) - (g/k) * t
        const T predicted_y = T(initial_y_) + 
                            ((initial_velocity_y + gravity / drag_coefficient) / drag_coefficient) * 
                            (T(1.0) - ceres::exp(-drag_coefficient * delta_time)) - 
                            (gravity / drag_coefficient) * delta_time;
        
        // 计算残差
        residual[0] = T(observed_x_) - predicted_x;
        residual[1] = T(observed_y_) - predicted_y;
        
        return true;
    }

private:
    const double time_;
    const double observed_x_;
    const double observed_y_;
    const double initial_x_;
    const double initial_y_;
};

// ============================================================================
// 轨迹拟合器类
// ============================================================================
class TrajectoryFitter {
public:
    // 参数边界常量
    static constexpr double MIN_GRAVITY = 100.0;
    static constexpr double MAX_GRAVITY = 1000.0;
    static constexpr double MIN_DRAG_COEFFICIENT = 0.01;
    static constexpr double MAX_DRAG_COEFFICIENT = 1.0;
    static constexpr double FPS = 60.0;

    explicit TrajectoryFitter(const std::string& video_path) 
        : video_path_(video_path),
          output_video_path_("/home/danny/TASK_03/video/output.mp4") {}

    bool Run() {
        std::cout << "=== Projectile Trajectory Fitting ===" << std::endl;
        
        if (!ExtractTrajectory()) {
            std::cerr << "Trajectory extraction failed" << std::endl;
            return false;
        }
        
        if (!FitParameters()) {
            std::cerr << "Parameter fitting failed" << std::endl;
            return false;
        }
        
        VisualizeResults();
        return true;
    }

    void PrintResults() const {
        std::cout << "\n=== FITTING RESULTS ===" << std::endl;
        std::cout << "Initial velocity X: " << initial_velocity_x_ << " px/s" << std::endl;
        std::cout << "Initial velocity Y: " << initial_velocity_y_ << " px/s" << std::endl;
        std::cout << "Gravity: " << gravity_ << " px/s²" << std::endl;
        std::cout << "Drag coefficient: " << drag_coefficient_ << " 1/s" << std::endl;
        std::cout << "Average fitting error: " << CalculateAverageError() << " pixels" << std::endl;
    }

private:
    // 坐标系转换函数
    cv::Point2f ConvertToPhysicsCoords(const cv::Point2f& image_point, int image_height) const {
        return {image_point.x, static_cast<float>(image_height) - image_point.y};
    }

    cv::Point2f ConvertToImageCoords(const cv::Point2f& physics_point, int image_height) const {
        return {physics_point.x, static_cast<float>(image_height) - physics_point.y};
    }

    bool CheckVideoFile() const {
        if (!std::filesystem::exists(video_path_)) {
            std::cerr << "Error: Video file not found: " << video_path_ << std::endl;
            return false;
        }

        cv::VideoCapture test_cap(video_path_);
        if (!test_cap.isOpened()) {
            std::cerr << "Error: Cannot open video file: " << video_path_ << std::endl;
            return false;
        }

        const int total_frames = static_cast<int>(test_cap.get(cv::CAP_PROP_FRAME_COUNT));
        const double fps = test_cap.get(cv::CAP_PROP_FPS);
        
        std::cout << "Video file: " << video_path_ << std::endl;
        std::cout << "Total frames: " << total_frames << std::endl;
        std::cout << "FPS: " << fps << std::endl;

        test_cap.release();
        return true;
    }

    bool ExtractTrajectory() {
        if (!CheckVideoFile()) {
            return false;
        }

        cv::VideoCapture capture(video_path_);
        if (!capture.isOpened()) {
            return false;
        }

        // 获取视频信息
        cv::Mat first_frame;
        capture.read(first_frame);
        const int image_height = first_frame.rows;
        const int image_width = first_frame.cols;
        capture.set(cv::CAP_PROP_POS_FRAMES, 0);

        // 初始化输出视频
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = FPS;
        output_video_writer_.open(output_video_path_, fourcc, fps, cv::Size(image_width, image_height));
        if (!output_video_writer_.isOpened()) {
            std::cerr << "Error: Cannot open output video file: " << output_video_path_ << std::endl;
            return false;
        }

        std::cout << "Image size: " << image_width << " x " << image_height << std::endl;
        std::cout << "Extracting trajectory..." << std::endl;

        // 蓝色HSV阈值
        const cv::Scalar blue_lower(90, 90, 0);
        const cv::Scalar blue_upper(140, 255, 255);

        std::vector<cv::Point2f> image_trajectory;
        std::vector<cv::Point2f> physics_trajectory;

        cv::Mat frame;
        int frame_count = 0;

        while (capture.read(frame) && !frame.empty()) {
            if (ProcessFrame(frame, blue_lower, blue_upper, image_trajectory, physics_trajectory)) {
                VisualizeFrame(frame, image_trajectory, physics_trajectory, frame_count, image_height);
            }

            // 保存每帧到输出视频
            output_video_writer_.write(frame);

            if (cv::waitKey(1) == 27) { // ESC key
                break;
            }
            frame_count++;
        }

        capture.release();
        output_video_writer_.release(); // 释放输出视频
        cv::destroyAllWindows();

        return FinalizeTrajectoryData(physics_trajectory, image_height);
    }

    bool ProcessFrame(const cv::Mat& frame, 
                     const cv::Scalar& lower_bound, 
                     const cv::Scalar& upper_bound,
                     std::vector<cv::Point2f>& image_trajectory,
                     std::vector<cv::Point2f>& physics_trajectory) const {
        cv::Mat hsv_frame;
        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);

        cv::Mat mask;
        cv::inRange(hsv_frame, lower_bound, upper_bound, mask);

        // 形态学处理
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        const cv::Point2f ball_center = FindBallCenter(contours, image_trajectory);
        
        if (ball_center.x >= 0 && ball_center.y >= 0) {
            image_trajectory.push_back(ball_center);
            physics_trajectory.push_back(ConvertToPhysicsCoords(ball_center, frame.rows));
            return true;
        }

        return false;
    }

    cv::Point2f FindBallCenter(const std::vector<std::vector<cv::Point>>& contours,
                              const std::vector<cv::Point2f>& previous_trajectory) const {
        cv::Point2f best_center(-1, -1);
        double best_area = 0;

        for (const auto& contour : contours) {
            const double area = cv::contourArea(contour);
            if (area < 10 || area <= best_area) {
                continue;
            }

            const cv::Rect bounding_box = cv::boundingRect(contour);
            const cv::Point2f center(bounding_box.x + bounding_box.width / 2.0f,
                                   bounding_box.y + bounding_box.height / 2.0f);

            // 连续性检查
            if (!previous_trajectory.empty()) {
                const cv::Point2f& previous_center = previous_trajectory.back();
                const double distance = cv::norm(center - previous_center);
                if (distance > 50.0) {
                    continue;
                }
            }

            best_center = center;
            best_area = area;
        }

        return best_center;
    }

    void VisualizeFrame(cv::Mat& frame,
                       const std::vector<cv::Point2f>& image_trajectory,
                       const std::vector<cv::Point2f>& physics_trajectory,
                       int frame_count,
                       int image_height) const {
        if (image_trajectory.empty()) return;

        // 绘制当前检测到的小球
        const cv::Point2f& current_center = image_trajectory.back();
        const cv::Rect visualization_box(current_center.x - 10, current_center.y - 10, 20, 20);
        
        cv::rectangle(frame, visualization_box, cv::Scalar(0, 255, 0), 2);
        cv::circle(frame, current_center, 5, cv::Scalar(0, 0, 255), -1);

        // 显示物理坐标
        const cv::Point2f physics_pos = physics_trajectory.back();
        std::string coordinate_text = "(" + std::to_string(static_cast<int>(physics_pos.x)) + 
                                    ", " + std::to_string(static_cast<int>(physics_pos.y)) + ")";
        
        cv::putText(frame, coordinate_text, current_center + cv::Point2f(15, -15),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

        // 绘制轨迹线
        for (size_t i = 1; i < image_trajectory.size(); ++i) {
            cv::line(frame, image_trajectory[i-1], image_trajectory[i],
                    cv::Scalar(255, 0, 0), 2);
        }

        // 显示信息
        cv::putText(frame, "Frame: " + std::to_string(frame_count),
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(frame, "Points: " + std::to_string(image_trajectory.size()),
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Trajectory Detection", frame);
    }

    bool FinalizeTrajectoryData(const std::vector<cv::Point2f>& physics_trajectory, int image_height) {
        if (physics_trajectory.size() < 10) {
            std::cerr << "Error: Insufficient trajectory points: " 
                      << physics_trajectory.size() << std::endl;
            return false;
        }

        // 设置初始位置
        initial_position_x_ = physics_trajectory[0].x;
        initial_position_y_ = physics_trajectory[0].y;

        // 准备优化数据
        for (size_t i = 0; i < physics_trajectory.size(); ++i) {
            const double time = i / FPS;
            time_data_.push_back(time);
            observed_x_data_.push_back(physics_trajectory[i].x);
            observed_y_data_.push_back(physics_trajectory[i].y);
        }

        std::cout << "Extracted " << physics_trajectory.size() << " trajectory points" << std::endl;
        std::cout << "Initial position: (" << initial_position_x_ 
                  << ", " << initial_position_y_ << ")" << std::endl;

        return true;
    }

    bool FitParameters() {
        if (time_data_.empty()) {
            return false;
        }

        std::cout << "Fitting trajectory parameters..." << std::endl;

        // 初始参数估计
        double parameters[4] = {500.0, 500.0, 500.0, 0.1};

        ceres::Problem problem;
        SetupResidualBlocks(problem, parameters);
        SetParameterBounds(problem, parameters);

        ceres::Solver::Summary summary;
        SolveOptimizationProblem(problem, parameters, summary);

        std::cout << summary.BriefReport() << std::endl;

        // 保存结果
        initial_velocity_x_ = parameters[0];
        initial_velocity_y_ = parameters[1];
        gravity_ = parameters[2];
        drag_coefficient_ = parameters[3];

        return true;
    }

    void SetupResidualBlocks(ceres::Problem& problem, double* parameters) const {
        for (size_t i = 0; i < time_data_.size(); ++i) {
            auto* cost_function = 
                new ceres::AutoDiffCostFunction<TrajectoryResidual, 2, 4>(
                    new TrajectoryResidual(time_data_[i], observed_x_data_[i], 
                                         observed_y_data_[i], initial_position_x_, 
                                         initial_position_y_));
            
            problem.AddResidualBlock(cost_function, nullptr, parameters);
        }
    }

    void SetParameterBounds(ceres::Problem& problem, double* parameters) const {
        problem.SetParameterLowerBound(parameters, 2, MIN_GRAVITY);
        problem.SetParameterUpperBound(parameters, 2, MAX_GRAVITY);
        problem.SetParameterLowerBound(parameters, 3, MIN_DRAG_COEFFICIENT);
        problem.SetParameterUpperBound(parameters, 3, MAX_DRAG_COEFFICIENT);
    }

    void SolveOptimizationProblem(ceres::Problem& problem, 
                                 double* parameters, 
                                 ceres::Solver::Summary& summary) const {
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;

        ceres::Solve(options, &problem, &summary);
    }

    void VisualizeResults() const {
        std::cout << "Generating fitted trajectory..." << std::endl;
        
        std::vector<cv::Point2f> fitted_trajectory;
        for (double t = 0; t <= time_data_.back(); t += 0.01) {
            const double x = initial_position_x_ + (initial_velocity_x_ / drag_coefficient_) * 
                           (1 - exp(-drag_coefficient_ * t));
            const double y = initial_position_y_ + 
                           ((initial_velocity_y_ + gravity_ / drag_coefficient_) / drag_coefficient_) * 
                           (1 - exp(-drag_coefficient_ * t)) - 
                           (gravity_ / drag_coefficient_) * t;
            fitted_trajectory.emplace_back(x, y);
        }

        std::cout << "Generated " << fitted_trajectory.size() << " fitted points" << std::endl;
    }

    double CalculateAverageError() const {
        double total_error = 0.0;
        
        for (size_t i = 0; i < time_data_.size(); ++i) {
            const double t = time_data_[i];
            const double predicted_x = initial_position_x_ + 
                                     (initial_velocity_x_ / drag_coefficient_) * 
                                     (1 - exp(-drag_coefficient_ * t));
            const double predicted_y = initial_position_y_ + 
                                     ((initial_velocity_y_ + gravity_ / drag_coefficient_) / drag_coefficient_) * 
                                     (1 - exp(-drag_coefficient_ * t)) - 
                                     (gravity_ / drag_coefficient_) * t;
            
            const double error = std::hypot(observed_x_data_[i] - predicted_x,
                                          observed_y_data_[i] - predicted_y);
            total_error += error;
        }

        return total_error / time_data_.size();
    }

    // 成员变量
    std::string video_path_;
    
    // 拟合参数
    double initial_velocity_x_ = 0.0;
    double initial_velocity_y_ = 0.0;
    double gravity_ = 0.0;
    double drag_coefficient_ = 0.0;
    
    // 初始位置
    double initial_position_x_ = 0.0;
    double initial_position_y_ = 0.0;
    
    // 轨迹数据
    std::vector<double> time_data_;
    std::vector<double> observed_x_data_;
    std::vector<double> observed_y_data_;
    std::string output_video_path_;
    cv::VideoWriter output_video_writer_;
};

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    const std::string video_path = "/home/danny/TASK_03/video/video.mp4";
    
    TrajectoryFitter fitter(video_path);
    
    if (fitter.Run()) {
        fitter.PrintResults();
        std::cout << "\n=== PROGRAM COMPLETED ===" << std::endl;
        return 0;
    } else {
        std::cerr << "\n=== PROGRAM FAILED ===" << std::endl;
        return -1;
    }
}