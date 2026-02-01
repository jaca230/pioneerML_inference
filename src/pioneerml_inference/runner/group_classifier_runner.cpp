#include "pioneerml_inference/runner/group_classifier_runner.h"

#include <stdexcept>

#include <arrow/api.h>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "pioneerml_dataloaders/batch/group_classifier_batch.h"
#include "pioneerml_dataloaders/configurable/input_adapters/graph/group_classifier_input_adapter.h"
#include "pioneerml_dataloaders/configurable/output_adapters/graph/group_classifier_output_adapter.h"

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
  return torch::from_blob(ptr, shape,
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

}  // namespace

void GroupClassifierRunner::Run(const RunOptions& options) {
  if (options.model_path.empty()) {
    throw std::runtime_error("model_path is required");
  }
  if (options.parquet_paths.empty()) {
    throw std::runtime_error("parquet_paths is required");
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

  pioneerml::input_adapters::graph::GroupClassifierInputAdapter input_adapter;
  if (!options.config_json.empty()) {
    input_adapter.LoadConfig(nlohmann::json::parse(options.config_json));
  }

  auto bundle = input_adapter.LoadInference(options.parquet_paths);
  auto* inputs = dynamic_cast<pioneerml::GroupClassifierInputs*>(bundle.inputs.get());
  if (!inputs) {
    throw std::runtime_error("Failed to cast inputs to GroupClassifierInputs");
  }

  int64_t total_nodes = LastValue(inputs->node_ptr);
  int64_t total_edges = LastValue(inputs->edge_ptr);
  int64_t num_graphs = static_cast<int64_t>(inputs->num_graphs);

  auto node_ptr = ArrowToTensor(inputs->node_ptr, {num_graphs + 1}, torch::kInt64);
  auto edge_ptr = ArrowToTensor(inputs->edge_ptr, {num_graphs + 1}, torch::kInt64);
  (void)edge_ptr;
  auto group_ptr = ArrowToTensor(inputs->group_ptr, {num_graphs + 1}, torch::kInt64);

  auto x = ArrowToTensor(inputs->node_features, {total_nodes, 4}, torch::kFloat32);
  auto edge_index = ArrowToTensor(inputs->edge_index, {2, total_edges}, torch::kInt64);
  auto edge_attr = ArrowToTensor(inputs->edge_attr, {total_edges, 4}, torch::kFloat32);
  auto time_group_ids = ArrowToTensor(inputs->time_group_ids, {total_nodes}, torch::kInt64);

  auto batch = BuildBatchFromNodePtr(inputs->node_ptr, total_nodes);

  if (device.is_cuda()) {
    x = x.to(device);
    edge_index = edge_index.to(device);
    edge_attr = edge_attr.to(device);
    batch = batch.to(device);
    time_group_ids = time_group_ids.to(device);
    group_ptr = group_ptr.to(device);
  }

  std::vector<torch::jit::IValue> ivals = {x, edge_index, edge_attr, batch, group_ptr, time_group_ids};
  auto logits = model.forward(ivals).toTensor();
  auto preds = torch::sigmoid(logits);

  if (device.is_cuda()) {
    preds = preds.to(torch::kCPU);
  }

  auto pred_arr = TensorToArrowFloat(preds);

  pioneerml::output_adapters::graph::GroupClassifierOutputAdapter output_adapter;
  output_adapter.WriteParquet(options.output_path,
                              pred_arr,
                              nullptr,
                              inputs->node_ptr,
                              inputs->time_group_ids);
}

}  // namespace pioneerml::inference::runner
