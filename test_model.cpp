#include <gtest/gtest.h>


#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <cstdlib>

using namespace ::executorch::runtime;
using namespace ::executorch::extension;


const std::string kTestPTEPath = [] {
    if (const char* env_p = std::getenv("ET_TESTING_MODEL_PATH")) {
        return std::string(env_p);
    }
    // fallback if env var not set
    return std::string("model.pte");
}();

const int kNumThreads = [] {
    if (const char* env_p = std::getenv("ET_TESTING_NUM_THREADS")) {
        try {
            return std::stoi(env_p);
        } catch (...) {
            // if conversion fails, fall back
        }
    }
    // fallback default
    return 7;
}();

std::vector<TensorPtr> get_inputs(Module& module) {
  const auto method_meta = module.method_meta("forward");
  const auto num_inputs = method_meta->num_inputs();
  std::cout << "num_inputs: " << num_inputs << std::endl;

  // Num outputs
  const auto num_outputs = method_meta->num_outputs();
  std::cout << "num_outputs: " << num_outputs << std::endl;

  std::vector<TensorPtr> tensors;
  tensors.reserve(num_inputs);

  for (auto index = 0; index < num_inputs; ++index) {
    const auto input_tag = method_meta->input_tag(index);

    switch (*input_tag) {
      case Tag::Tensor: {
        const auto tensor_meta = method_meta->input_tensor_meta(index);
        const auto sizes = tensor_meta->sizes();
        tensors.emplace_back(
            rand({sizes.begin(), sizes.end()}, tensor_meta->scalar_type()));
      } break;
      default:
        throw std::runtime_error("Unsupported tag");
    }
  }
  return tensors;
}

std::vector<TensorPtr> get_outputs(Module& module) {
  const auto method_meta = module.method_meta("forward");
  const auto num_outputs = method_meta->num_outputs();

  std::vector<TensorPtr> tensors;
  tensors.reserve(num_outputs);

  for (auto index = 0; index < num_outputs; ++index) {
    const auto output_tag = method_meta->output_tag(index);

    switch (*output_tag) {
      case Tag::Tensor: {
        const auto tensor_meta = method_meta->output_tensor_meta(index);
        const auto sizes = tensor_meta->sizes();
        tensors.emplace_back(
            zeros({sizes.begin(), sizes.end()}, tensor_meta->scalar_type()));
      } break;
      default:
        throw std::runtime_error("Unsupported tag");
    }
  }
  return tensors;
}



void run_predict(int i, const std::string& model_path, std::atomic<size_t>& success_count) {
  Module module(model_path);

  auto inputs = get_inputs(module);
  for (int i = 0; i < inputs.size(); i++) {
    module.set_input(inputs[i], i);
  }

  auto outputs = get_outputs(module);
  for (int i = 0; i < outputs.size(); i++) {
    module.set_output(outputs[i], i);
  }

  // Perform an inference.
  const auto result = module.forward();

  if (result.ok()) {
    // Retrieve the output data.
    success_count++;
  }
}

TEST(ModelTest, MultipleThreads) {
  const int num_threads = 1;

  ASSERT_NE(kTestPTEPath.size(), 0);
  ASSERT_NE(num_threads, 0);

  std::vector<std::thread> threads(num_threads);
  std::atomic<size_t> success_count{0};
  int i = 0;

  for (int i = 0; i < num_threads; i++) {
    threads[i] = std::thread([&, i]() {
      run_predict(i, kTestPTEPath, success_count);
    });
  }
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }
  ASSERT_EQ(success_count, num_threads);
}
