#pragma once

#include "visp/ml.h"
#include "visp/vision.h"

#include <vector>

namespace visp::da3 {

tensor residual_unit(model_ref m, tensor x);
tensor fusion(model_ref m, tensor x0, tensor x1, int64_t const* size = nullptr);
std::vector<tensor> encode(model_ref m, tensor image, depthany3_params const& p);
depthany3_prediction head(model_ref m, span<tensor> features, depthany3_params const& p);

} // namespace visp::da3
