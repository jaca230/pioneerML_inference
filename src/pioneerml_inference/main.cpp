#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "pioneerml_inference/runner/group_classifier_runner.h"
#include "pioneerml_inference/runner/group_classifier_event_runner.h"
#include "pioneerml_inference/runner/group_splitter_runner.h"
#include "pioneerml_inference/runner/group_splitter_event_runner.h"

namespace {

void PrintUsage() {
  std::cerr << "Usage: pioneerml_inference --mode group_classifier|group_classifier_event|group_splitter|group_splitter_event --model <path> --input <file> [--input <file> ...] [--input-group <file0,file1,...>] --output <path> [--config <json>] [--device cpu|cuda] [--check-accuracy] [--metrics-out <path>] [--threshold <float>]\n";
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
  std::vector<std::vector<std::string>> input_groups;
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
    } else if (arg == "--input-group" && i + 1 < argc) {
      std::string raw = argv[++i];
      std::vector<std::string> group;
      std::stringstream ss(raw);
      std::string item;
      while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
          group.push_back(item);
        }
      }
      if (group.empty()) {
        std::cerr << "Invalid --input-group value: " << raw << "\n";
        return 1;
      }
      input_groups.push_back(std::move(group));
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

  if (input_groups.empty()) {
    for (const auto& input : inputs) {
      input_groups.push_back({input});
    }
  }

  if (mode.empty() || model_path.empty() || input_groups.empty() || output_path.empty()) {
    PrintUsage();
    return 1;
  }

  auto build_input_spec = [&](const std::string& current_mode) -> nlohmann::json {
    nlohmann::json files = nlohmann::json::array();
    for (const auto& group : input_groups) {
      if (group.empty()) {
        continue;
      }
      nlohmann::json item;
      item["mainFile"] = group[0];
      if ((current_mode == "group_splitter" || current_mode == "group_splitter_event") && group.size() >= 2) {
        item["group_probs"] = group[1];
      }
      files.push_back(item);
    }
    nlohmann::json spec;
    spec["files"] = files;
    return spec;
  };

  try {
    if (mode == "group_classifier") {
      pioneerml::inference::runner::GroupClassifierRunner runner;
      pioneerml::inference::runner::RunOptions options;
      options.model_path = model_path;
      options.input_spec = build_input_spec(mode);
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
      options.input_spec = build_input_spec(mode);
      options.output_path = output_path;
      options.config_json = config_json;
      options.device = device;
      options.check_accuracy = check_accuracy;
      options.metrics_output_path = metrics_output_path;
      options.threshold = threshold;
      runner.Run(options);
      return 0;
    }
    if (mode == "group_splitter") {
      pioneerml::inference::runner::GroupSplitterRunner runner;
      pioneerml::inference::runner::RunOptions options;
      options.model_path = model_path;
      options.input_spec = build_input_spec(mode);
      options.output_path = output_path;
      options.config_json = config_json;
      options.device = device;
      options.check_accuracy = check_accuracy;
      options.metrics_output_path = metrics_output_path;
      options.threshold = threshold;
      runner.Run(options);
      return 0;
    }
    if (mode == "group_splitter_event") {
      pioneerml::inference::runner::GroupSplitterEventRunner runner;
      pioneerml::inference::runner::RunOptions options;
      options.model_path = model_path;
      options.input_spec = build_input_spec(mode);
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
