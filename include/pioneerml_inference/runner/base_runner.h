#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace pioneerml::inference::runner {

struct RunOptions {
  std::string model_path;
  nlohmann::json input_spec;
  std::string output_path;
  std::string config_json;
  std::string device{"cpu"};
  bool check_accuracy{false};
  std::string metrics_output_path;
  double threshold{0.5};
};

class BaseRunner {
 public:
  virtual ~BaseRunner() = default;
  virtual void Run(const RunOptions& options) = 0;
};

}  // namespace pioneerml::inference::runner
