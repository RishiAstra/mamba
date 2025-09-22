#pragma once

#include <tuple>
#include <cstdint>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include "common.h"

// TODO: change return type to 2-tuple or merge into fully fused kernel
std::tuple<c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>>
mamba2_bmm_chunk_fwd_cuda(const Mamba2SSDArgs& args);
