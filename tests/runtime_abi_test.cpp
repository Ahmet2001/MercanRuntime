#include "mercan.h"
#include <cstdio>

int main() {
    if (mercan_abi_version() != MERCAN_ABI_VERSION ||
        mercan_model_params_size() != sizeof(mercan_model_params) ||
        mercan_context_params_size() != sizeof(mercan_context_params)) {
        std::fputs("Runtime ABI does not match installed header\n", stderr);
        return 1;
    }
    return 0;
}
