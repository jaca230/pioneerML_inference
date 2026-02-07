#include "pioneerml_inference/runner/group_splitter_event_runner.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <arrow/api.h>
#include <nlohmann/json.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include "pioneerml_dataloaders/batch/group_splitter_event_batch.h"
#include "pioneerml_dataloaders/configurable/input_adapters/graph/group_splitter_event_input_adapter.h"
#include "pioneerml_dataloaders/configurable/output_adapters/graph/group_splitter_event_output_adapter.h"

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
  return torch::from_blob(
      ptr,
      shape,
      [buf = buffer](void*) mutable { buf.reset(); },
      options);
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
  if (cfg.contains("loader") && cfg.at("loader").is_object() && cfg.at("loader").contains(key)) {
    return cfg.at("loader").at(key).get<bool>();
  }
  return default_value;
}

double ConfigDouble(const nlohmann::json& cfg, const std::string& key, double default_value) {
  if (cfg.contains(key)) {
    return cfg.at(key).get<double>();
  }
  if (cfg.contains("loader") && cfg.at("loader").is_object() && cfg.at("loader").contains(key)) {
    return cfg.at("loader").at(key).get<double>();
  }
  return default_value;
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

}  // namespace

void GroupSplitterEventRunner::Run(const RunOptions& options) {
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

  pioneerml::input_adapters::graph::GroupSplitterEventInputAdapter input_adapter;
  nlohmann::json config;
  if (!options.config_json.empty()) {
    config = nlohmann::json::parse(options.config_json);
  } else {
    std::filesystem::path model_path(options.model_path);
    std::filesystem::path meta_path = model_path;
    meta_path.replace_extension();
    meta_path += "_meta.json";
    if (std::filesystem::exists(meta_path)) {
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
  auto* inputs = dynamic_cast<pioneerml::GroupSplitterEventInputs*>(bundle.inputs.get());
  if (!inputs) {
    throw std::runtime_error("Failed to cast inputs to GroupSplitterEventInputs");
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
  auto u = ArrowToTensor(inputs->u, {num_graphs, 1}, torch::kFloat32);
  auto group_probs = ArrowToTensor(inputs->group_probs, {num_groups, 3}, torch::kFloat32);
  auto group_ptr = ArrowToTensor(inputs->group_ptr, {num_graphs + 1}, torch::kInt64);

  auto batch = BuildBatchFromNodePtr(inputs->node_ptr, total_nodes);

  if (!config.is_null()) {
    bool normalize = ConfigBool(config, "normalize", false);
    if (normalize) {
      double eps = ConfigDouble(config, "normalize_eps", 1e-6);
      NormalizeFeatures(x, edge_attr, eps);
    }
  }

  if (device.is_cuda()) {
    x = x.to(device);
    edge_index = edge_index.to(device);
    edge_attr = edge_attr.to(device);
    batch = batch.to(device);
    u = u.to(device);
    group_ptr = group_ptr.to(device);
    time_group_ids = time_group_ids.to(device);
    group_probs = group_probs.to(device);
  }

  std::vector<torch::jit::IValue> ivals = {
      x,
      edge_index,
      edge_attr,
      batch,
      u,
      group_ptr,
      time_group_ids,
      group_probs,
  };

  auto output = model.forward(ivals);
  if (!output.isTensor()) {
    throw std::runtime_error("Group splitter event TorchScript forward must return a tensor.");
  }
  auto node_logits = output.toTensor();
  auto node_probs = torch::sigmoid(node_logits);

  if (device.is_cuda()) {
    node_probs = node_probs.to(torch::kCPU);
  }

  auto node_pred_arr = TensorToArrowFloat(node_probs);

  if (options.check_accuracy) {
    if (!inputs->y_node) {
      throw std::runtime_error("check_accuracy requested but y_node targets are missing.");
    }
    auto targets = ArrowToFloatMatrix(inputs->y_node, total_nodes, 3);
    auto preds_cpu = node_probs.to(torch::kCPU);
    double threshold = options.threshold;
    auto preds_binary = (preds_cpu >= threshold).to(torch::kFloat32);
    auto targets_f = targets.to(torch::kFloat32);
    auto accuracy = (preds_binary == targets_f).to(torch::kFloat32).mean().item<double>();
    auto exact_match = (preds_binary == targets_f).all(1).to(torch::kFloat32).mean().item<double>();

    int64_t num_classes = targets_f.size(1);
    nlohmann::json confusion = nlohmann::json::array();
    for (int64_t cls = 0; cls < num_classes; ++cls) {
      auto truth = targets_f.select(1, cls);
      auto pred = preds_binary.select(1, cls);
      auto tn = ((truth == 0) & (pred == 0)).sum().item<int64_t>();
      auto fp = ((truth == 0) & (pred == 1)).sum().item<int64_t>();
      auto fn = ((truth == 1) & (pred == 0)).sum().item<int64_t>();
      auto tp = ((truth == 1) & (pred == 1)).sum().item<int64_t>();
      double total = static_cast<double>(tp + fp + fn);
      double tp_rate = total > 0.0 ? static_cast<double>(tp) / total : 0.0;
      double fp_rate = total > 0.0 ? static_cast<double>(fp) / total : 0.0;
      double fn_rate = total > 0.0 ? static_cast<double>(fn) / total : 0.0;
      confusion.push_back({{"tp", tp_rate}, {"fp", fp_rate}, {"fn", fn_rate}});
    }

    nlohmann::json metrics;
    metrics["loss"] = nullptr;
    metrics["accuracy"] = accuracy;
    metrics["exact_match"] = exact_match;
    metrics["confusion"] = confusion;
    metrics["threshold"] = threshold;
    std::cout << metrics.dump() << std::endl;
    WriteMetrics(metrics, options.metrics_output_path);
  }

  pioneerml::output_adapters::graph::GroupSplitterEventOutputAdapter output_adapter;
  output_adapter.WriteParquet(options.output_path,
                              node_pred_arr,
                              inputs->node_ptr,
                              inputs->time_group_ids,
                              inputs->graph_event_ids);
}

}  // namespace pioneerml::inference::runner
