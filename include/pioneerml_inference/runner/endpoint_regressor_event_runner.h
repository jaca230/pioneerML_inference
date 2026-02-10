#pragma once

#include "pioneerml_inference/runner/base_runner.h"

namespace pioneerml::inference::runner {

class EndpointRegressorEventRunner final : public BaseRunner {
 public:
  void Run(const RunOptions& options) override;
};

}  // namespace pioneerml::inference::runner
