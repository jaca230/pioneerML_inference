#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "pioneerml_inference/runner/group_classifier_runner.h"
#include "pioneerml_inference/runner/group_classifier_event_runner.h"

namespace {

void PrintUsage() {
  std::cerr << "Usage: pioneerml_inference --mode group_classifier|group_classifier_event --model <path> --input <file> [--input <file> ...] --output <path> [--config <json>] [--device cpu|cuda] [--check-accuracy] [--metrics-out <path>] [--threshold <float>]\n";
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open config file: " + path);
  }
  std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return contents;
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode;
  std::string model_path;
  std::vector<std::string> inputs;
  std::string output_path;
  std::string config_json;
  std::string config_path;
  std::string device = "cpu";
  bool check_accuracy = false;
  std::string metrics_output_path;
  double threshold = 0.5;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg == "--model" && i + 1 < argc) {
      model_path = argv[++i];
    } else if (arg == "--input" && i + 1 < argc) {
      inputs.emplace_back(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      device = argv[++i];
    } else if (arg == "--check-accuracy") {
      check_accuracy = true;
    } else if (arg == "--metrics-out" && i + 1 < argc) {
      metrics_output_path = argv[++i];
    } else if (arg == "--threshold" && i + 1 < argc) {
      threshold = std::stod(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      PrintUsage();
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n";
      PrintUsage();
      return 1;
    }
  }

  if (!config_path.empty()) {
    config_json = ReadFile(config_path);
  }

  if (mode.empty() || model_path.empty() || inputs.empty() || output_path.empty()) {
    PrintUsage();
    return 1;
  }

  try {
    if (mode == "group_classifier") {
      pioneerml::inference::runner::GroupClassifierRunner runner;
      pioneerml::inference::runner::RunOptions options;
      options.model_path = model_path;
      options.parquet_paths = inputs;
      options.output_path = output_path;
      options.config_json = config_json;
      options.device = device;
      options.check_accuracy = check_accuracy;
      options.metrics_output_path = metrics_output_path;
      options.threshold = threshold;
      runner.Run(options);
      return 0;
    }
    if (mode == "group_classifier_event") {
      pioneerml::inference::runner::GroupClassifierEventRunner runner;
      pioneerml::inference::runner::RunOptions options;
      options.model_path = model_path;
      options.parquet_paths = inputs;
      options.output_path = output_path;
      options.config_json = config_json;
      options.device = device;
      options.check_accuracy = check_accuracy;
      options.metrics_output_path = metrics_output_path;
      options.threshold = threshold;
      runner.Run(options);
      return 0;
    }
    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "Inference failed: " << ex.what() << "\n";
    return 1;
  }
}
