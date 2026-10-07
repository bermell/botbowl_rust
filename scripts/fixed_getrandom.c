/* LD_PRELOAD shim for scripts/perf_search_bench.sh: `getrandom` returns a fixed stream (seed from
 * FIXRAND_SEED), so std's RandomState keys, and every hash-map iteration order that depends on
 * them, are the same on every run. Benchmarks only. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

static uint64_t ctr;
static int init;

ssize_t getrandom(void *buf, size_t len, unsigned int flags) {
    (void)flags;
    unsigned char *b = buf;
    if (!init) {
        const char *e = getenv("FIXRAND_SEED");
        ctr = e ? strtoull(e, 0, 10) : 0x9E3779B97F4A7C15ull;
        init = 1;
    }
    for (size_t i = 0; i < len; i++) {
        ctr = ctr * 6364136223846793005ull + 1442695040888963407ull;
        b[i] = (unsigned char)(ctr >> 56);
    }
    return (ssize_t)len;
}
