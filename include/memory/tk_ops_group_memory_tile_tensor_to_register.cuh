/**
 * @file
 * @brief Minimal FP32 TMEM to BF16 register-tile load for Blackwell.
 *
 * Included inside kittens::group.  The warpgroup path partitions a 128-row
 * TMEM tile into one 32-row slice per warp and reuses the warp implementation.
 */

template<ducks::rt::row_layout RT, ducks::tt::all TM>
__device__ static inline void load_async(RT &dst, const TM &src) {
    static_assert(std::is_same_v<typename TM::dtype, float>);
    static_assert(std::is_same_v<typename RT::T, bf16>);

    if constexpr (GROUP_WARPS == 1) {
        static_assert(RT::rows == TM::rows);
        static_assert(RT::cols == TM::cols);
        static_assert(RT::width % 2 == 0,
                      "minimal TMEM load expects an even number of 16-column tiles");

        using dst_packed = typename RT::dtype;
        using src_packed = float2;

        #pragma unroll
        for (int i = 0; i < RT::height; ++i) {
            #pragma unroll
            for (int j = 0; j < RT::width; j += 2) {
                src_packed data[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32 "
                    "{%0, %1, %2, %3, %4, %5, %6, %7, "
                    "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];\n"
                    : "=f"(data[0].x), "=f"(data[0].y),
                      "=f"(data[1].x), "=f"(data[1].y),
                      "=f"(data[2].x), "=f"(data[2].y),
                      "=f"(data[3].x), "=f"(data[3].y),
                      "=f"(data[4].x), "=f"(data[4].y),
                      "=f"(data[5].x), "=f"(data[5].y),
                      "=f"(data[6].x), "=f"(data[6].y),
                      "=f"(data[7].x), "=f"(data[7].y)
                    : "r"(src.addr +
                          ((i * RT::tile_size_row) << 16) +
                          j * RT::tile_size_col)
                    : "memory");

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    dst.tiles[i][j].data[k] =
                        base_types::convertor<dst_packed, src_packed>::convert(
                            data[k]);
                    dst.tiles[i][j + 1].data[k] =
                        base_types::convertor<dst_packed, src_packed>::convert(
                            data[k + 4]);
                }
            }
        }
    } else {
        static_assert(GROUP_WARPS == 4,
                      "minimal TMEM load supports warp or warpgroup scope");
        constexpr int warp_rows = TM::rows / GROUP_WARPS;
        static_assert(warp_rows == RT::rows);
        static_assert(TM::cols == RT::cols);
        auto warp_src = src.template subtile<
            tt<typename TM::dtype, warp_rows, TM::cols>>(32 * warpid(), 0);
        ::kittens::group<1>::load_async(dst, warp_src);
    }
}
