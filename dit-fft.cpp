#include <complex>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <chrono>

namespace fft
{
    using Complex = std::complex<double>;
    constexpr double PI = 3.14159265358979323846;
    constexpr double TWO_PI = 2 * PI;

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

    // 2点FFT
    template <typename T>
    inline void transform2(T &sum, T &diff)
    {
        T temp0 = sum, temp1 = diff;
        sum = temp0 + temp1;
        diff = temp0 - temp1;
    }

    static std::vector<std::vector<Complex>> twiddle_factors(32);

    inline void init_twiddle_factors(int fft_log_len)
    {
        if (twiddle_factors[fft_log_len].size() > 0)
        {
            return;
        }
        size_t fft_len = (size_t(1) << fft_log_len), fft_len_half = fft_len / 2;
        twiddle_factors[fft_log_len].resize(fft_len_half);
        // 计算长度为2^fft_log_len的FFT所需的旋转因子，有fft_len / 2个
        for (size_t k = 0; k < fft_len_half; ++k)
        {
            twiddle_factors[fft_log_len][k] = std::polar(1.0, -TWO_PI * k / fft_len);
        }
    }

    inline void init_all_twiddle_factors(int max_fft_log_len)
    {
        for (int fft_log_len = 0; fft_log_len <= max_fft_log_len; ++fft_log_len)
        {
            init_twiddle_factors(fft_log_len);
        }
    }

    inline void dit(Complex inout[], size_t fft_len)
    {
        if (fft_len <= 1)
        {
            return;
        }
        if (fft_len == 2)
        {
            transform2(inout[0], inout[1]);
            return;
        }
        const size_t stride = fft_len / 2;
        // 递归调用，对前半部分和后半部分进行FFT计算
        dit(inout, stride);
        dit(inout + stride, stride);
        int fft_log_len = hint_log2(fft_len);
        init_twiddle_factors(fft_log_len);
        // 合并FFT结果
        for (size_t k = 0; k < stride; ++k)
        {
            inout[stride + k] *= twiddle_factors[fft_log_len][k];
            transform2(inout[k], inout[stride + k]);
        }
    }

    // 二进制逆序
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

    inline void fft_dit(Complex inout[], size_t fft_len)
    {
        binary_reverse_swap(inout, inout + fft_len);
        dit(inout, fft_len);
    }

    // Naive FFT, O(N^2)
    inline void dft(Complex inout[], size_t fft_len)
    {
        std::vector<Complex> res(fft_len);
        for (size_t i = 0; i < fft_len; ++i)
        {
            Complex sum = 0;
            for (size_t j = 0; j < fft_len; ++j)
            {
                sum += inout[j] * std::polar(1.0, -TWO_PI * i * j / fft_len);
            }
            res[i] = sum;
        }
        std::copy(res.begin(), res.end(), inout);
    }
}

inline void check_fft()
{
    using namespace fft;
    size_t fft_len = 1 << 10;
    std::vector<Complex> a(fft_len);
    std::vector<Complex> b(fft_len);
    for (size_t i = 0; i < fft_len; ++i)
    {
        a[i] = b[i] = rand();
    }
    // 验证FFT算法的正确性
    fft_dit(a.data(), fft_len);
    dft(b.data(), fft_len);
    for (size_t i = 0; i < fft_len; ++i)
    {
        if (std::abs(a[i] - b[i]) > 1e-6)
        {
            std::cout << "Error!" << std::endl;
            return;
        }
    }
    std::cout << "Success!" << std::endl;
}
inline void perform_fft()
{
    using namespace fft;
    size_t fft_len = 1 << 18;
    std::vector<Complex> a(fft_len);
    for (size_t i = 0; i < fft_len; ++i)
    {
        a[i] = rand();
    }
    // 测试FFT算法的性能
    auto t1 = std::chrono::high_resolution_clock::now();
    fft_dit(a.data(), fft_len);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "FFT time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
}

int main()
{
    using namespace fft;
    init_all_twiddle_factors(20);
    check_fft();
    perform_fft();
}