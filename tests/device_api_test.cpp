#include "mercan.h"
#include <cstring>
#include <iostream>

static bool rejected(const char * device, int layers, const char * message) {
    auto p = mercan_model_default_params();
    p.n_gpu_layers = layers;
    auto * model = mercan_model_load_device_v1("/missing-model.mercan", p, device);
    if (model) { mercan_model_free(model); return false; }
    return std::strstr(mercan_last_error(), message) != nullptr;
}
int main() {
    mercan_backend_init();
    if (mercan_device_api_version() != 1) return 1;
    const char * json = mercan_runtime_devices_json_v1();
    if (!json || !std::strstr(json, "\"cuda_compiled\":") ||
        !std::strstr(json, "\"devices\":[")) return 2;
    const bool available = mercan_cuda_available_v1();
    if ((std::strstr(json, "\"cuda_available\":true") != nullptr) != available) return 3;
    if (!rejected("invalid", 0, "device must be")) return 4;
    if (!rejected("legacy", 0, "device must be")) return 10;
    if (!rejected("cpu", -1, "conflicts")) return 5;
    if (!rejected("cuda", 0, "conflicts")) return 6;
    if (!rejected("auto", -2, "gpu_layers")) return 7;
    if (!available) {
        const bool compiled = std::strstr(json, "\"cuda_compiled\":true") != nullptr;
        if (!rejected("cuda", -1, compiled ? "no usable CUDA device" : "built without CUDA")) return 8;
        if (!rejected("auto", -1, "explicitly requested")) return 9;
    }
    std::cout << json << '\n';
    mercan_backend_free();
}
