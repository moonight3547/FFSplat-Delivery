#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector_types.h>

namespace NEWTON
{
	void assemble(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* background,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* pixel_weights,
		float* g_mean2D,
		float* H_mean2D,
		float* g_conic,
		float* H_conic,
		float* g_opacity,
		float* H_opacity,
		float* g_color,
		float* H_color,
		float* vis);
}
