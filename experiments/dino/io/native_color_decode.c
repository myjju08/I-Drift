#include <stddef.h>
#include <stdint.h>
#include <emmintrin.h>

void idrift_color_xy_encode(const uint8_t *src, uint8_t *dst, size_t h, size_t w) {
    size_t plane=h*w;
    for (size_t y=0; y<h; ++y) {
        uint8_t carry[3]={0,0,0};
        size_t x=0;
        for (; x+16<=w; x+=16) {
            __m128i d[3];
            for (size_t c=0; c<3; ++c) {
                size_t off=c*plane+y*w+x;
                __m128i v=_mm_loadu_si128((const __m128i *)(src+off));
                if (y) v=_mm_sub_epi8(v,_mm_loadu_si128((const __m128i *)(src+off-w)));
                __m128i left=_mm_or_si128(_mm_slli_si128(v,1),_mm_cvtsi32_si128(carry[c]));
                d[c]=_mm_sub_epi8(v,left);
                carry[c]=(uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(v,15));
            }
            _mm_storeu_si128((__m128i *)(dst+y*w+x),d[1]);
            _mm_storeu_si128((__m128i *)(dst+plane+y*w+x),_mm_sub_epi8(d[0],d[1]));
            _mm_storeu_si128((__m128i *)(dst+2*plane+y*w+x),_mm_sub_epi8(d[2],d[1]));
        }
        for (; x<w; ++x) {
            uint8_t d[3];
            for (size_t c=0;c<3;++c) {
                size_t off=c*plane+y*w+x;
                uint8_t v=(uint8_t)(src[off]-(y?src[off-w]:0));
                d[c]=(uint8_t)(v-carry[c]); carry[c]=v;
            }
            dst[y*w+x]=d[1]; dst[plane+y*w+x]=(uint8_t)(d[0]-d[1]);
            dst[2*plane+y*w+x]=(uint8_t)(d[2]-d[1]);
        }
    }
}

/* Exact modulo-256 inverse of vertical delta, horizontal delta, and
 * (G,R-G,B-G). No floating point, allocation, RNG or external state. */
void idrift_color_xy_decode(const uint8_t *src, uint8_t *dst, size_t h, size_t w) {
    size_t plane = h * w;
    for (size_t c = 0; c < 3; ++c) {
        for (size_t y = 0; y < h; ++y) {
            size_t off = c * plane + y * w;
            size_t x = 0;
            for (; x + 16 <= w; x += 16) {
                __m128i a = _mm_loadu_si128((const __m128i *)(src + off + x));
                if (y) a = _mm_add_epi8(a, _mm_loadu_si128((const __m128i *)(dst + off + x - w)));
                _mm_storeu_si128((__m128i *)(dst + off + x), a);
            }
            for (; x < w; ++x) dst[off+x] = (uint8_t)(src[off+x] + (y ? dst[off+x-w] : 0));
        }
        for (size_t y = 0; y < h; ++y) {
            uint8_t *row = dst + c * plane + y * w;
            uint8_t carry = 0;
            size_t x = 0;
            for (; x + 16 <= w; x += 16) {
                __m128i a = _mm_loadu_si128((const __m128i *)(row + x));
                a = _mm_add_epi8(a, _mm_slli_si128(a, 1));
                a = _mm_add_epi8(a, _mm_slli_si128(a, 2));
                a = _mm_add_epi8(a, _mm_slli_si128(a, 4));
                a = _mm_add_epi8(a, _mm_slli_si128(a, 8));
                a = _mm_add_epi8(a, _mm_set1_epi8((char)carry));
                _mm_storeu_si128((__m128i *)(row+x), a);
                carry = (uint8_t)_mm_cvtsi128_si32(_mm_srli_si128(a, 15));
            }
            for (; x < w; ++x) { carry = (uint8_t)(carry + row[x]); row[x] = carry; }
        }
    }
    size_t i = 0;
    for (; i + 16 <= plane; i += 16) {
        __m128i g = _mm_loadu_si128((const __m128i *)(dst+i));
        __m128i r = _mm_add_epi8(_mm_loadu_si128((const __m128i *)(dst+plane+i)), g);
        __m128i b = _mm_add_epi8(_mm_loadu_si128((const __m128i *)(dst+2*plane+i)), g);
        _mm_storeu_si128((__m128i *)(dst+i), r);
        _mm_storeu_si128((__m128i *)(dst+plane+i), g);
        _mm_storeu_si128((__m128i *)(dst+2*plane+i), b);
    }
    for (; i < plane; ++i) {
        uint8_t g=dst[i]; dst[i]=(uint8_t)(dst[plane+i]+g);
        dst[plane+i]=g; dst[2*plane+i]=(uint8_t)(dst[2*plane+i]+g);
    }
}
