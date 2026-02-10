#include "pioneerml_inference/runner/endpoint_regressor_event_runner.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <arrow/api.h>
#include <nlohmann/json.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include "pioneerml_dataloaders/batch/endpoint_regressor_batch.h"
#include "pioneerml_dataloaders/configurable/input_adapters/graph/endpoint_regressor_event_input_adapter.h"
#include "pioneerml_dataloaders/configurable/output_adapters/graph/endpoint_regressor_event_output_adapter.h"

namespace pioneerml::inference::runner {
namespace {

torch::Tensor ArrowToTensor(const std::shared_ptr<arrow::Array>& array,
                            const std::vector<int64_t>& shape,
                            torch::ScalarType dtype) {
  if (!array) {
    throw std::runtime_error("Arrow array is null.");
  }
  auto data = array->data();
  if (!data || data->buffers.size() < 2 || !data->buffers[1]) {
    throw std::runtime_error("Arrow array missing values buffer.");
  }
  auto buffer = data->buffers[1];
  void* ptr = const_cast<uint8_t*>(buffer->data());
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
  return torch::from_blob(ptr, shape, [buf = buffer](void*) mutable { buf.reset(); }, options);
}

int64_t LastValue(const std::shared_ptr<arrow::Array>& array) {
  auto numeric = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(array);
  if (!numeric) {
    throw std::runtime_error("Expected int64 array.");
  }
  if (numeric->length() == 0) {
    return 0;
  }
  return numeric->Value(numeric->length() - 1);
}

torch::Tensor BuildBatchFromNodePtr(const std::shared_ptr<arrow::Array>& node_ptr,
                                    int64_t total_nodes) {
  auto numeric = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(node_ptr);
  if (!numeric) {
    throw std::runtime_error("Expected int64 node_ptr array.");
  }
  auto batch = torch::empty({total_nodes}, torch::TensorOptions().dtype(torch::kInt64));
  auto* batch_ptr = batch.data_ptr<int64_t>();
  const int64_t* ptr = numeric->raw_values();
  const int64_t num_graphs = numeric->length() - 1;
  for (int64_t g = 0; g < num_graphs; ++g) {
    int64_t start = ptr[g];
    int64_t end = ptr[g + 1];
    for (int64_t i = start; i < end; ++i) {
      batch_ptr[i] = g;
    }
  }
  return batch;
}

std::shared_ptr<arrow::Array> TensorToArrowFloat(const torch::Tensor& tensor) {
  auto t = tensor.contiguous().to(torch::kCPU);
  if (t.scalar_type() != torch::kFloat32) {
    throw std::runtime_error("Expected float32 tensor for output.");
  }
  auto* ptr = reinterpret_cast<const uint8_t*>(t.data_ptr<float>());
  auto buffer = arrow::Buffer::Wrap(ptr, t.numel() * sizeof(float));
  auto data = arrow::ArrayData::Make(arrow::float32(), t.numel(), {nullptr, buffer});
  return arrow::MakeArray(data);
}

torch::Tensor ArrowToFloatMatrix(const std::shared_ptr<arrow::Array>& array,
                                 int64_t rows,
                                 int64_t cols) {
  return ArrowToTensor(array, {rows, cols}, torch::kFloat32);
}

void WriteMetrics(const nlohmann::json& metrics, const std::string& path) {
  if (path.empty()) {
    return;
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open metrics output path: " + path);
  }
  out << metrics.dump(2) << std::endl;
}

bool ConfigBool(const nlohmann::json& cfg, const std::string& key, bool default_value) {
  if (cfg.contains(key)) {
    return cfg.at(key).get<bool>();
  }
  if (cfg.contains("inference") && cfg.at("inference").is_object() &&
      cfg.at("inference").contains(key)) {
    return cfg.at("inference").at(key).get<bool>();
  }
  if (cfg.contains("loader") && cfg.at("loader").is_object() && cfg.at("loader").contains(key)) {
    return cfg.at("loader").at(key).get<bool>();
  }
  return default_value;
}

double ConfigDouble(const nlohmann::json& cfg, const std::string& key, double default_value) {
  if (cfg.contains(key)) {
    return cfg.at(key).get<double>();
  }
  if (cfg.contains("inference") && cfg.at("inference").is_object() &&
      cfg.at("inference").contains(key)) {
    return cfg.at("inference").at(key).get<double>();
  }
  if (cfg.contains("loader") && cfg.at("loader").is_object() && cfg.at("loader").contains(key)) {
    return cfg.at("loader").at(key).get<double>();
  }
  return default_value;
}

int64_t ConfigInt64(const nlohmann::json& cfg, const std::string& key, int64_t default_value) {
  if (cfg.contains(key)) {
    return cfg.at(key).get<int64_t>();
  }
  if (cfg.contains("inference") && cfg.at("inference").is_object() &&
      cfg.at("inference").contains(key)) {
    return cfg.at("inference").at(key).get<int64_t>();
  }
  if (cfg.contains("loader") && cfg.at("loader").is_object() && cfg.at("loader").contains(key)) {
    return cfg.at("loader").at(key).get<int64_t>();
  }
  return default_value;
}

std::filesystem::path ResolveMetaPathCandidates(const std::filesystem::path& model_path) {
  const auto parent = model_path.parent_path();

  std::filesystem::path legacy = model_path;
  legacy.replace_extension();
  legacy += "_meta.json";
  if (std::filesystem::exists(legacy)) {
    return legacy;
  }

  std::string stem = model_path.stem().string();
  constexpr const char* suffix = "_torchscript";
  constexpr size_t suffix_len = 12;
  if (stem.size() > suffix_len &&
      stem.compare(stem.size() - suffix_len, suffix_len, suffix) == 0) {
    stem = stem.substr(0, stem.size() - suffix_len);
  }
  std::filesystem::path canonical = parent / (stem + "_meta.json");
  if (std::filesystem::exists(canonical)) {
    return canonical;
  }

  return {};
}

void NormalizeFeatures(torch::Tensor& x, torch::Tensor& edge_attr, double eps) {
  if (x.numel() > 0) {
    auto x_feat = x.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    auto mean = x_feat.mean(0, false);
    auto std = x_feat.std(0, false).clamp_min(eps);
    x_feat.sub_(mean).div_(std);
  }
  if (edge_attr.numel() > 0) {
    auto e_feat = edge_attr.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    auto mean = e_feat.mean(0, false);
    auto std = e_feat.std(0, false).clamp_min(eps);
    e_feat.sub_(mean).div_(std);
  }
}

const int64_t* RawInt64Values(const std::shared_ptr<arrow::Array>& array, const char* name) {
  auto numeric = std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(array);
  if (!numeric) {
    throw std::runtime_error(std::string("Expected int64 array for ") + name + ".");
  }
  return numeric->raw_values();
}

}  // namespace

void EndpointRegressorEventRunner::Run(const RunOptions& options) {
  if (options.model_path.empty()) {
    throw std::runtime_error("model_path is required");
  }
  if (!options.input_spec.is_object() || !options.input_spec.contains("files") ||
      !options.input_spec.at("files").is_array() || options.input_spec.at("files").empty()) {
    throw std::runtime_error("input_spec.files is required");
  }
  if (options.output_path.empty()) {
    throw std::runtime_error("output_path is required");
  }

  torch::Device device(torch::kCPU);
  if (options.device == "cuda") {
    if (torch::cuda::is_available()) {
      device = torch::Device(torch::kCUDA);
    } else {
      throw std::runtime_error("CUDA requested but not available.");
    }
  }

  torch::jit::script::Module model = torch::jit::load(options.model_path, device);
  model.eval();

  pioneerml::input_adapters::graph::EndpointRegressorEventInputAdapter input_adapter;
  nlohmann::json config;
  if (!options.config_json.empty()) {
    config = nlohmann::json::parse(options.config_json);
  } else {
    const std::filesystem::path meta_path = ResolveMetaPathCandidates(options.model_path);
    if (!meta_path.empty()) {
      std::ifstream in(meta_path);
      if (in) {
        nlohmann::json meta;
        in >> meta;
        if (meta.contains("pipeline_config") && !meta.at("pipeline_config").is_null()) {
          config = meta.at("pipeline_config");
        }
      }
    }
  }
  if (!config.is_null()) {
    input_adapter.LoadConfig(config);
  }

  auto bundle = input_adapter.LoadInference(options.input_spec);
  auto* inputs = dynamic_cast<pioneerml::EndpointRegressorInputs*>(bundle.inputs.get());
  if (!inputs) {
    throw std::runtime_error("Failed to cast inputs to EndpointRegressorInputs");
  }

  const int64_t num_graphs = static_cast<int64_t>(inputs->num_graphs);
  const int64_t num_groups = static_cast<int64_t>(inputs->num_groups);
  const int64_t total_nodes = LastValue(inputs->node_ptr);
  const int64_t total_edges = LastValue(inputs->edge_ptr);

  auto x = ArrowToTensor(inputs->node_features, {total_nodes, 4}, torch::kFloat32);
  auto edge_index_pairs = ArrowToTensor(inputs->edge_index, {total_edges, 2}, torch::kInt64);
  auto edge_index = edge_index_pairs.t().contiguous();
  auto edge_attr = ArrowToTensor(inputs->edge_attr, {total_edges, 4}, torch::kFloat32);
  auto time_group_ids = ArrowToTensor(inputs->time_group_ids, {total_nodes}, torch::kInt64);
  auto group_probs = ArrowToTensor(inputs->group_probs, {num_groups, 3}, torch::kFloat32);
  auto splitter_probs = ArrowToTensor(inputs->splitter_probs, {total_nodes, 3}, torch::kFloat32);
  auto group_ptr = ArrowToTensor(inputs->group_ptr, {num_graphs + 1}, torch::kInt64);
  auto batch = BuildBatchFromNodePtr(inputs->node_ptr, total_nodes);
  const int64_t* node_ptr_raw = RawInt64Values(inputs->node_ptr, "node_ptr");
  const int64_t* edge_ptr_raw = RawInt64Values(inputs->edge_ptr, "edge_ptr");
  const int64_t* group_ptr_raw = RawInt64Values(inputs->group_ptr, "group_ptr");

  int64_t graph_batch_size = 128;
  if (!config.is_null()) {
    graph_batch_size = ConfigInt64(config, "inference_graph_batch_size", 128);
  }
  if (graph_batch_size <= 0) {
    graph_batch_size = 128;
  }

  auto preds_all =
      torch::zeros({num_groups, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  torch::Tensor targets;
  if (options.check_accuracy) {
    if (!inputs->y) {
      throw std::runtime_error("check_accuracy requested but y targets are missing.");
    }
    targets = ArrowToFloatMatrix(inputs->y, num_groups, 6);
  }

  if (!config.is_null()) {
    bool normalize = ConfigBool(config, "normalize", false);
    if (normalize) {
      double eps = ConfigDouble(config, "normalize_eps", 1e-6);
      NormalizeFeatures(x, edge_attr, eps);
    }
  }

  double squared_error_sum = 0.0;
  double abs_error_sum = 0.0;
  int64_t error_element_count = 0;

  for (int64_t graph_start = 0; graph_start < num_graphs; graph_start += graph_batch_size) {
    const int64_t graph_end = std::min<int64_t>(num_graphs, graph_start + graph_batch_size);
    const int64_t node_start = node_ptr_raw[graph_start];
    const int64_t node_end = node_ptr_raw[graph_end];
    const int64_t edge_start = edge_ptr_raw[graph_start];
    const int64_t edge_end = edge_ptr_raw[graph_end];
    const int64_t group_start = group_ptr_raw[graph_start];
    const int64_t group_end = group_ptr_raw[graph_end];

    auto x_chunk = x.index({torch::indexing::Slice(node_start, node_end)});
    auto edge_index_chunk =
        edge_index.index({torch::indexing::Slice(), torch::indexing::Slice(edge_start, edge_end)}).contiguous();
    edge_index_chunk = edge_index_chunk - node_start;
    auto edge_attr_chunk = edge_attr.index({torch::indexing::Slice(edge_start, edge_end)});
    auto batch_chunk = batch.index({torch::indexing::Slice(node_start, node_end)}).to(torch::kInt64);
    batch_chunk = batch_chunk - graph_start;
    auto group_ptr_chunk = group_ptr.index({torch::indexing::Slice(graph_start, graph_end + 1)}).to(torch::kInt64);
    group_ptr_chunk = group_ptr_chunk - group_start;
    auto time_group_ids_chunk = time_group_ids.index({torch::indexing::Slice(node_start, node_end)}).to(torch::kInt64);
    auto group_probs_chunk = group_probs.index({torch::indexing::Slice(group_start, group_end)});
    auto splitter_probs_chunk = splitter_probs.index({torch::indexing::Slice(node_start, node_end)});

    if (device.is_cuda()) {
      x_chunk = x_chunk.to(device);
      edge_index_chunk = edge_index_chunk.to(device);
      edge_attr_chunk = edge_attr_chunk.to(device);
      batch_chunk = batch_chunk.to(device);
      group_ptr_chunk = group_ptr_chunk.to(device);
      time_group_ids_chunk = time_group_ids_chunk.to(device);
      group_probs_chunk = group_probs_chunk.to(device);
      splitter_probs_chunk = splitter_probs_chunk.to(device);
    }

    std::vector<torch::jit::IValue> ivals = {
        x_chunk,
        edge_index_chunk,
        edge_attr_chunk,
        batch_chunk,
        group_ptr_chunk,
        time_group_ids_chunk,
        group_probs_chunk,
        splitter_probs_chunk,
    };

    auto output = model.forward(ivals);
    if (!output.isTensor()) {
      throw std::runtime_error("Endpoint regressor TorchScript forward must return a tensor.");
    }
    auto preds_chunk = output.toTensor();
    if (device.is_cuda()) {
      preds_chunk = preds_chunk.to(torch::kCPU);
    }
    preds_chunk = preds_chunk.contiguous();
    preds_all.index_put_({torch::indexing::Slice(group_start, group_end), torch::indexing::Slice()}, preds_chunk);

    if (options.check_accuracy) {
      auto target_chunk = targets.index({torch::indexing::Slice(group_start, group_end), torch::indexing::Slice()});
      auto diff = preds_chunk - target_chunk;
      squared_error_sum += diff.pow(2).sum().item<double>();
      abs_error_sum += diff.abs().sum().item<double>();
      error_element_count += diff.numel();
    }
  }

  auto pred_arr = TensorToArrowFloat(preds_all);

  if (options.check_accuracy) {
    if (error_element_count <= 0) {
      throw std::runtime_error("check_accuracy requested but no prediction elements were evaluated.");
    }
    double mse = squared_error_sum / static_cast<double>(error_element_count);
    double mae = abs_error_sum / static_cast<double>(error_element_count);

    nlohmann::json metrics;
    metrics["loss"] = mse;
    metrics["mae"] = mae;
    std::cout << metrics.dump() << std::endl;
    WriteMetrics(metrics, options.metrics_output_path);
  }

  pioneerml::output_adapters::graph::EndpointRegressorEventOutputAdapter output_adapter;
  output_adapter.WriteParquet(
      options.output_path,
      pred_arr,
      inputs->group_ptr,
      inputs->graph_event_ids);
}

}  // namespace pioneerml::inference::runner
