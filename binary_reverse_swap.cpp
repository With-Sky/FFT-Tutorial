#include <complex>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <cstdint>
#include <cassert>

namespace fft
{
    // 求整数的对数
    template <typename T>
    constexpr int hint_log2(T n)
    {
        constexpr int bits = sizeof(n) * 8;
        int l = -1, r = bits;
        while ((l + 1) != r)
        {
            int mid = (l + r) / 2;
            if ((T(1) << mid) > n)
            {
                r = mid;
            }
            else
            {
                l = mid;
            }
        }
        return l;
    }
    // 计算32位整数的二进制逆序
    // 类似于大小端转换
    constexpr uint32_t bitrev32(uint32_t n)
    {
        constexpr uint32_t mask55 = 0x55555555;
        constexpr uint32_t mask33 = 0x33333333;
        constexpr uint32_t mask0f = 0x0f0f0f0f;
        constexpr uint32_t maskff = 0x00ff00ff;
        n = ((n & mask55) << 1) | ((n >> 1) & mask55);
        n = ((n & mask33) << 2) | ((n >> 2) & mask33);
        n = ((n & mask0f) << 4) | ((n >> 4) & mask0f);
        n = ((n & maskff) << 8) | ((n >> 8) & maskff);
        return (n << 16) | (n >> 16);
    }

    class BitRev8
    {
    public:
        constexpr BitRev8()
        {
            for (size_t i = 0; i < 256; i++)
            {
                table[i] = bitrev8(i);
            }
        }
        constexpr uint8_t operator()(uint8_t n) const
        {
            return table[n];
        }
        static constexpr uint8_t bitrev8(uint8_t n)
        {
            constexpr uint8_t mask55 = 0x55;
            constexpr uint8_t mask33 = 0x33;
            n = ((n & mask55) << 1) | ((n >> 1) & mask55);
            n = ((n & mask33) << 2) | ((n >> 2) & mask33);
            n = (n << 4) | (n >> 4);
            return n;
        }

    private:
        uint8_t table[256]{};
    };

    constexpr BitRev8 bitrev8;
    constexpr uint32_t bitrev32_fast(uint32_t n)
    {
        uint32_t n0 = uint8_t(n);
        uint32_t n1 = uint8_t(n >> 8);
        uint32_t n2 = uint8_t(n >> 16);
        uint32_t n3 = n >> 24;
        n0 = bitrev8(n0);
        n1 = bitrev8(n1);
        n2 = bitrev8(n2);
        n3 = bitrev8(n3);
        n0 = (n0 << 24) | (n2 << 8);
        n1 = (n1 << 16) | n3;
        return n0 | n1;
    }

    // 计算len位整数的二进制逆序,长度len小于等于32
    constexpr size_t bitrev(size_t n, size_t len)
    {
        return bitrev32_fast(n) >> (32 - len);
    }

    // 利用查找表DP递推
    template <typename It>
    inline void binary_reverse_swap1(It begin, It end)
    {
        const size_t len = end - begin;
        std::vector<size_t> rev(len);
        rev[0] = 0;
        int log_n = hint_log2(len);
        for (size_t i = 0; i < len; ++i)
        {
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (log_n - 1)); // 动态规划求rev交换数组
            // 通过i / 2 的逆序数来计算i的逆序数
            // 如 i = 10110, i >> 1 = 01011, i >> 1 的逆序数 rev[i / 2] = 11010
            // i = (i >> 1) << 1, i 的逆序数为 i >> 1 的逆序数右移一位
            // 当i为奇数时, rev[i]的最高位为1, 需要加上 1 << (log_n - 1)
            size_t rev_index = rev[i];
            if (i < rev_index) // 防止重复交换
            {
                std::swap(begin[i], begin[rev_index]);
            }
        }
    }

    // 利用位运算模拟逆序进位加法递推
    template <typename It>
    inline void binary_reverse_swap2(It begin, It end)
    {
        const size_t len = end - begin;
        auto get_next_bitrev = [=](size_t last_rev)
        {
            // 假设i的逆序数为last_rev
            // 进行反方向的加1进位，计算i + 1的逆序数
            // 例如i = 10110, last_rev = 01101
            // i + 1 = 10111
            // last_rev加1, 从高位向低位进位
            // 先异或k = len / 2, 向最高位加1 得11101, 此时k <= last_rev，得到i + 1的逆序数11101
            // 由于大于0的偶数的逆序数最高位为0，所以异或之后最高位变为1，一定满足k <= last_rev, 不需要再进位
            // 对于奇数，last_rev的最高位为1，异或k之后，最高位变为0，因此k > last_rev，并且存在进位
            // 因此k右移，继续异或，实现进位，当进位是从0到1时，k <= last_rev，说明不会再进位，得到i + 1的逆序数
            size_t k = len / 2;
            last_rev ^= k;
            while (k > last_rev)
            {
                k >>= 1;
                last_rev ^= k;
            };
            return last_rev;
        };
        size_t rev = len / 2; // 1的逆序数
        int log_n = hint_log2(len);
        // 从1开始，到len - 2结束，因为0和len - 1的逆序数为自身
        for (size_t i = 1; i < len - 1; ++i)
        {
            if (i < rev) // 防止重复交换
            {
                std::swap(begin[i], begin[rev]);
            }
            rev = get_next_bitrev(rev); // 根据rev计算下一个逆序数
        }
    }

    // 减少循环次数
    template <typename It>
    inline void binary_reverse_swap(It begin, It end)
    {
        const size_t len = end - begin;
        // 左下标小于右下标时交换,防止重复交换
        auto smaller_swap = [=](It it_left, It it_right)
        {
            if (it_left < it_right)
            {
                std::swap(it_left[0], it_right[0]);
            }
        };
        // 若i的逆序数的迭代器为last,则返回i+1的逆序数的迭代器
        auto get_next_bitrev = [=](It last)
        {
            size_t k = len / 2, indx = last - begin;
            indx ^= k;
            while (k > indx)
            {
                k >>= 1;
                indx ^= k;
            };
            return begin + indx;
        };
        // 长度较短的普通逆序
        if (len <= 16)
        {
            for (auto i = begin + 1, j = begin + len / 2; i < end - 1; i++)
            {
                smaller_swap(i, j);
                j = get_next_bitrev(j);
            }
            return;
        }
        // 该算法位上面的优化版
        // i的逆序为rev，则i + len / 2的逆序数为rev + 1
        // 如0101的逆序数为1010，则0101 + 1000 = 1101的逆序数为1011 = 1010 + 1
        // 同理i + len / 4的逆序数为rev + 2, i + len / 8 的逆序数为rev + 4
        // 对这几种情况进行组合可以得到 j 到 j + 7 的逆序数
        const size_t len_8 = len / 8;
        const auto last = begin + len_8;
        auto i0 = begin + 1, i1 = i0 + len / 2, i2 = i0 + len / 4, i3 = i1 + len / 4;
        for (auto j = begin + len / 2; i0 < last; i0++, i1++, i2++, i3++)
        {
            smaller_swap(i0, j);
            smaller_swap(i1, j + 1);
            smaller_swap(i2, j + 2);
            smaller_swap(i3, j + 3);
            smaller_swap(i0 + len_8, j + 4);
            smaller_swap(i1 + len_8, j + 5);
            smaller_swap(i2 + len_8, j + 6);
            smaller_swap(i3 + len_8, j + 7);
            j = get_next_bitrev(j);
        }
    }

    // 减少循环次数
    template <typename It>
    inline void binary_reverse_swap3(It begin, It end)
    {
        const size_t len = end - begin;
        const int shift = 32 - hint_log2(len);
        // 左下标小于右下标时交换,防止重复交换
        auto smaller_swap = [=](It it_left, It it_right)
        {
            if (it_left < it_right)
            {
                std::swap(it_left[0], it_right[0]);
            }
        };
        // 若i的逆序数的迭代器为last,则返回i+1的逆序数的迭代器
        auto get_next_bitrev = [=](It last)
        {
            size_t k = len / 2, indx = last - begin;
            indx ^= k;
            while (k > indx)
            {
                k >>= 1;
                indx ^= k;
            };
            return begin + indx;
        };
        // 长度较短的普通逆序
        if (len <= 16)
        {
            for (auto i = begin + 1, j = begin + len / 2; i < end - 1; i++)
            {
                smaller_swap(i, j);
                j = get_next_bitrev(j);
            }
            return;
        }
        // 该算法位上面的优化版
        // i的逆序为rev，则i + len / 2的逆序数为rev + 1
        // 如0101的逆序数为1010，则0101 + 1000 = 1101的逆序数为1011 = 1010 + 1
        // 同理i + len / 4的逆序数为rev + 2, i + len / 8 的逆序数为rev + 4
        // 对这几种情况进行组合可以得到 j 到 j + 7 的逆序数
        const size_t len_8 = len / 8;
        const auto last = begin + len_8;
        auto i0 = begin + 1, i1 = i0 + len / 2, i2 = i0 + len / 4, i3 = i1 + len / 4;
        for (auto j = begin + len / 2; i0 < last; i0++, i1++, i2++, i3++)
        {
            j = (bitrev32_fast(i0 - begin) >> shift) + begin;
            smaller_swap(i0, j);
            smaller_swap(i0 + len_8, j + 4);
            smaller_swap(i2, j + 2);
            smaller_swap(i2 + len_8, j + 6);
            smaller_swap(i1, j + 1);
            smaller_swap(i1 + len_8, j + 5);
            smaller_swap(i3, j + 3);
            smaller_swap(i3 + len_8, j + 7);
            // j = get_next_bitrev(j);
        }
    }

    template <typename It>
    void cobra(It begin, It end)
    {
        using Type = typename std::iterator_traits<It>::value_type;
        constexpr size_t LOG_BLOCK_WIDTH = 8;
        const size_t len = end - begin;
        int log_n = hint_log2(len);
        const int num_b_bits = log_n - 2 * LOG_BLOCK_WIDTH;
        const size_t b_size = 1 << num_b_bits;
        const size_t BLOCK_WIDTH = 1 << LOG_BLOCK_WIDTH;

        Type buffer[BLOCK_WIDTH * BLOCK_WIDTH]{};
        for (size_t b = 0; b < b_size; ++b)
        {
            size_t b_rev = bitrev32_fast(b) >> __builtin_clz(b_size - 1);

            // Copy block to buffer
            for (size_t a = 0; a < BLOCK_WIDTH; ++a)
            {
                size_t a_rev = bitrev32_fast(a) >> __builtin_clz(BLOCK_WIDTH - 1);
                for (size_t c = 0; c < BLOCK_WIDTH; ++c)
                {
                    buffer[(a_rev << LOG_BLOCK_WIDTH) | c] =
                        begin[(a << num_b_bits << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c];
                }
            }

            for (size_t c = 0; c < BLOCK_WIDTH; ++c)
            {
                size_t c_rev = bitrev32_fast(c) >> __builtin_clz(BLOCK_WIDTH - 1);

                for (size_t a_rev = 0; a_rev < BLOCK_WIDTH; ++a_rev)
                {
                    size_t a = bitrev32_fast(a_rev) >> __builtin_clz(BLOCK_WIDTH - 1);

                    // To guarantee each value is swapped only one time:
                    // index < reversed_index <-->
                    // a b c < c' b' a' <-->
                    // a < c' ||
                    // a <= c' && b < b' ||
                    // a <= c' && b <= b' && a' < c
                    bool index_less_than_reverse = a < c_rev || (a == c_rev && b < b_rev) || (a == c_rev && b == b_rev && a_rev < c);

                    if (index_less_than_reverse)
                    {
                        int v_idx = (c_rev << num_b_bits << LOG_BLOCK_WIDTH) | (b_rev << LOG_BLOCK_WIDTH) | a_rev;
                        int b_idx = (a_rev << LOG_BLOCK_WIDTH) | c;
                        std::swap(begin[v_idx], buffer[b_idx]);
                    }
                }
            }

            // Copy changes that were swapped into buffer above:
            for (size_t a = 0; a < BLOCK_WIDTH; ++a)
            {
                size_t a_rev = bitrev32_fast(a) >> __builtin_clz(BLOCK_WIDTH - 1);
                for (size_t c = 0; c < BLOCK_WIDTH; ++c)
                {
                    size_t c_rev = bitrev32_fast(c) >> __builtin_clz(BLOCK_WIDTH - 1);
                    bool index_less_than_reverse = a < c_rev || (a == c_rev && b < b_rev) || (a == c_rev && b == b_rev && a_rev < c);

                    if (index_less_than_reverse)
                    {
                        size_t v_idx = (a << num_b_bits << LOG_BLOCK_WIDTH) | (b << LOG_BLOCK_WIDTH) | c;
                        size_t b_idx = (a_rev << LOG_BLOCK_WIDTH) | c;
                        std::swap(begin[v_idx], buffer[b_idx]);
                    }
                }
            }
        }
    }
}

void check_binary_reverse_swap()
{
    using namespace fft;
    int log_n = 23;
    size_t len = 1 << log_n;
    std::vector<size_t> vec(len);
    std::iota(vec.begin(), vec.end(), 0); // 生成0到len - 1的序列
    auto t1 = std::chrono::high_resolution_clock::now();
    binary_reverse_swap1(vec.begin(), vec.end());
    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < len; ++i)
    {
        assert(vec[i] == bitrev(i, log_n));
    }
    std::iota(vec.begin(), vec.end(), 0);
    auto t3 = std::chrono::high_resolution_clock::now();
    binary_reverse_swap2(vec.begin(), vec.end());
    auto t4 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < len; ++i)
    {
        assert(vec[i] == bitrev(i, log_n));
    }
    std::iota(vec.begin(), vec.end(), 0);
    auto t5 = std::chrono::high_resolution_clock::now();
    binary_reverse_swap(vec.begin(), vec.end());
    auto t6 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < len; ++i)
    {
        assert(vec[i] == bitrev(i, log_n));
    }
    std::cout << "binary_reverse_swap1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
    std::cout << "binary_reverse_swap2: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << "us" << std::endl;
    std::cout << "binary_reverse_swap: " << std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count() << "us" << std::endl;
}

void test_bitrev32()
{
    size_t sum1 = 0, sum2 = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < size_t(1e9); ++i)
    {
        sum1 += fft::bitrev32(i);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < size_t(1e9); ++i)
    {
        sum2 += fft::bitrev32_fast(i);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "bitrev32: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us" << std::endl;
    std::cout << "bitrev32_fast: " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << "us" << std::endl;
    std::cout << "sum1: " << sum1 << std::endl;
    std::cout << "sum2: " << sum2 << std::endl;
}

int main()
{
    check_binary_reverse_swap();
    test_bitrev32();
}