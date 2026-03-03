#include <gtest/gtest.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

using namespace ::executorch::runtime;
using namespace ::executorch::extension;

constexpr int kNumIterations = 300;

static int getDelayMs() {
    if (const char* env = std::getenv("DELAY_MS")) {
        return std::atoi(env);
    }
    return 0;
}

static int getExcludeN() {
    if (const char* env = std::getenv("EXCLUDE_N")) {
        return std::atoi(env);
    }
    return 0;
}

static std::string getModelPath() {
    const char* env = std::getenv("MODEL_PATH");
    if (!env || std::string(env).empty()) {
        throw std::runtime_error("MODEL_PATH environment variable is required");
    }
    return std::string(env);
}

TEST(Benchmark, XLModelDummy) {
    const std::string model_path = getModelPath();
    const int delayMs = getDelayMs();
    const int excludeN = getExcludeN();

    std::cout << "\n=== Benchmark: XL Model Dummy ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Iterations: " << kNumIterations << std::endl;
    std::cout << "Delay between iterations: " << delayMs << " ms" << std::endl;
    std::cout << "Exclude first N from stats: " << excludeN << std::endl;

    Module module(model_path);

    auto input = rand({1, 1, 432, 640}, exec_aten::ScalarType::Float);

    std::cout << "Running warmup..." << std::endl;
    auto warmup_result = module.forward(input);
    ASSERT_TRUE(warmup_result.ok()) << "Warmup forward() failed";

    std::cout << "Running " << kNumIterations << " iterations..." << std::endl;

    std::vector<double> iteration_times;
    iteration_times.reserve(kNumIterations);

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumIterations; ++i) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        auto result = module.forward(input);
        ASSERT_TRUE(result.ok()) << "forward() failed on iteration " << i;

        auto iter_end = std::chrono::high_resolution_clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(iter_end - iter_start).count();
        iteration_times.push_back(iter_ms);

        if (delayMs > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    int statsCount = kNumIterations - excludeN;
    ASSERT_GT(statsCount, 0) << "EXCLUDE_N must be less than kNumIterations";

    double sum = 0.0;
    double min_time = iteration_times[excludeN];
    double max_time = iteration_times[excludeN];

    for (int i = excludeN; i < kNumIterations; ++i) {
        double t = iteration_times[i];
        sum += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }

    double avg_time = sum / statsCount;
    double execution_time_only = sum;

    std::cout << "\n=== Results (excluding first " << excludeN << ") ===" << std::endl;
    std::cout << "Iterations used for stats: " << statsCount << std::endl;
    std::cout << "Total wall time: " << total_ms << " ms" << std::endl;
    std::cout << "Total execution time (forward only): " << execution_time_only << " ms" << std::endl;
    std::cout << "Average per iteration: " << avg_time << " ms" << std::endl;
    std::cout << "Min iteration time: " << min_time << " ms" << std::endl;
    std::cout << "Max iteration time: " << max_time << " ms" << std::endl;

    if (delayMs > 0) {
        double delay_overhead = total_ms - execution_time_only;
        std::cout << "Delay overhead: " << delay_overhead << " ms" << std::endl;
    }

    std::cout << "==============================\n" << std::endl;
}
