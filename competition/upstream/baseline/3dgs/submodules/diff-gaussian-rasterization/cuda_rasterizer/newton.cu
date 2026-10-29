#include "newton.h"
#include "auxiliary.h"
#include "config.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Assemble per-Gaussian local gradient / Hessian of L w.r.t. 2D mean, conic,
// opacity and RGB, following 3DGS2 Eq. (5):
//   g = J^T L_c
//   H = J^T (d2L/dc2) J + L_c · d2c/dy2
// Color and opacity have vanished renderer second derivatives. Position /
// conic keep the 2D Gaussian second derivatives of G.

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
assembleNewtonCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ pixel_weights,
	float* __restrict__ g_mean2D,
	float* __restrict__ H_mean2D,
	float* __restrict__ g_conic,
	float* __restrict__ H_conic,
	float* __restrict__ g_opacity,
	float* __restrict__ H_opacity,
	float* __restrict__ g_color,
	float* __restrict__ H_color,
	float* __restrict__ vis)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float wpixel[C];
	if (inside)
	{
		for (int i = 0; i < C; i++)
		{
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
			wpixel[i] = pixel_weights[i * H * W + pix_id];
		}
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int ch = 0; ch < C; ch++)
				collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + ch];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor--;
			if (contributor >= last_contributor)
				continue;

			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float T_before = T;
			const float w = alpha * T_before;
			const int global_id = collected_id[j];

			float dC_dalpha[C];
			float dL_dalpha = 0.0f;
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;
				dC_dalpha[ch] = T_before * (c - accum_rec[ch]);
				dL_dalpha += dC_dalpha[ch] * dL_dpixel[ch];
				atomicAdd(&(g_color[global_id * C + ch]), w * dL_dpixel[ch]);
				atomicAdd(&(H_color[global_id * C + ch]), w * wpixel[ch] * w);
			}

			const float bg_scale = -T_final / (1.f - alpha);
			for (int ch = 0; ch < C; ch++)
			{
				dC_dalpha[ch] += bg_scale * bg_color[ch];
				dL_dalpha += bg_scale * bg_color[ch] * dL_dpixel[ch];
			}

			last_alpha = alpha;
			atomicAdd(&(vis[global_id]), w);

			// Opacity: dc/dσ = G * dc/dα, d2c/dσ2 = 0 (Eq. 27).
			float J_op[C];
			float g_op = 0.f, H_op = 0.f;
			for (int ch = 0; ch < C; ch++)
			{
				J_op[ch] = G * dC_dalpha[ch];
				g_op += J_op[ch] * dL_dpixel[ch];
				H_op += J_op[ch] * wpixel[ch] * J_op[ch];
			}
			atomicAdd(&(g_opacity[global_id]), g_op);
			atomicAdd(&(H_opacity[global_id]), H_op);

			const float dC_dG[C] = { con_o.w * dC_dalpha[0], con_o.w * dC_dalpha[1], con_o.w * dC_dalpha[2] };
			float dL_dG = 0.f;
			for (int ch = 0; ch < C; ch++)
				dL_dG += dC_dG[ch] * dL_dpixel[ch];

			const float a = con_o.x, b = con_o.y, ccoef = con_o.z;
			const float gx = a * d.x + b * d.y;
			const float gy = ccoef * d.y + b * d.x;
			const float dG_dx = -G * gx;
			const float dG_dy = -G * gy;
			const float d2G_dxx = G * (gx * gx - a);
			const float d2G_dxy = G * (gx * gy - b);
			const float d2G_dyy = G * (gy * gy - ccoef);

			float Jx[C], Jy[C];
			float gmx = 0.f, gmy = 0.f;
			float Hxx = 0.f, Hxy = 0.f, Hyy = 0.f;
			for (int ch = 0; ch < C; ch++)
			{
				Jx[ch] = dC_dG[ch] * dG_dx;
				Jy[ch] = dC_dG[ch] * dG_dy;
				gmx += Jx[ch] * dL_dpixel[ch];
				gmy += Jy[ch] * dL_dpixel[ch];
				Hxx += Jx[ch] * wpixel[ch] * Jx[ch];
				Hxy += Jx[ch] * wpixel[ch] * Jy[ch];
				Hyy += Jy[ch] * wpixel[ch] * Jy[ch];
			}
			Hxx += dL_dG * d2G_dxx;
			Hxy += dL_dG * d2G_dxy;
			Hyy += dL_dG * d2G_dyy;
			atomicAdd(&(g_mean2D[global_id * 2 + 0]), gmx);
			atomicAdd(&(g_mean2D[global_id * 2 + 1]), gmy);
			atomicAdd(&(H_mean2D[global_id * 3 + 0]), Hxx);
			atomicAdd(&(H_mean2D[global_id * 3 + 1]), Hxy);
			atomicAdd(&(H_mean2D[global_id * 3 + 2]), Hyy);

			const float dG_da = -0.5f * G * d.x * d.x;
			const float dG_db = -G * d.x * d.y;
			const float dG_dc = -0.5f * G * d.y * d.y;
			float Ja[C], Jb[C], Jc[C];
			float ga = 0.f, gb = 0.f, gc = 0.f;
			float Haa = 0.f, Hab = 0.f, Hac = 0.f, Hbb = 0.f, Hbc = 0.f, Hcc = 0.f;
			for (int ch = 0; ch < C; ch++)
			{
				Ja[ch] = dC_dG[ch] * dG_da;
				Jb[ch] = dC_dG[ch] * dG_db;
				Jc[ch] = dC_dG[ch] * dG_dc;
				ga += Ja[ch] * dL_dpixel[ch];
				gb += Jb[ch] * dL_dpixel[ch];
				gc += Jc[ch] * dL_dpixel[ch];
				Haa += Ja[ch] * wpixel[ch] * Ja[ch];
				Hab += Ja[ch] * wpixel[ch] * Jb[ch];
				Hac += Ja[ch] * wpixel[ch] * Jc[ch];
				Hbb += Jb[ch] * wpixel[ch] * Jb[ch];
				Hbc += Jb[ch] * wpixel[ch] * Jc[ch];
				Hcc += Jc[ch] * wpixel[ch] * Jc[ch];
			}
			const float dx2 = d.x * d.x, dy2 = d.y * d.y, dxdy = d.x * d.y;
			Haa += dL_dG * (0.25f * G * dx2 * dx2);
			Hab += dL_dG * (0.5f * G * dx2 * dxdy);
			Hac += dL_dG * (0.25f * G * dx2 * dy2);
			Hbb += dL_dG * (G * dx2 * dy2);
			Hbc += dL_dG * (0.5f * G * dxdy * dy2);
			Hcc += dL_dG * (0.25f * G * dy2 * dy2);
			atomicAdd(&(g_conic[global_id * 3 + 0]), ga);
			atomicAdd(&(g_conic[global_id * 3 + 1]), gb);
			atomicAdd(&(g_conic[global_id * 3 + 2]), gc);
			atomicAdd(&(H_conic[global_id * 6 + 0]), Haa);
			atomicAdd(&(H_conic[global_id * 6 + 1]), Hab);
			atomicAdd(&(H_conic[global_id * 6 + 2]), Hac);
			atomicAdd(&(H_conic[global_id * 6 + 3]), Hbb);
			atomicAdd(&(H_conic[global_id * 6 + 4]), Hbc);
			atomicAdd(&(H_conic[global_id * 6 + 5]), Hcc);
		}
	}
}

void NEWTON::assemble(
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
	float* vis)
{
	assembleNewtonCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges, point_list, W, H, background, means2D, conic_opacity, colors,
		final_Ts, n_contrib, dL_dpixels, pixel_weights,
		g_mean2D, H_mean2D, g_conic, H_conic, g_opacity, H_opacity, g_color, H_color, vis);
}
