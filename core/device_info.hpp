#pragma once
#include "ggml-backend.h"
#include <cstring>
#include <string>
#include <vector>

namespace mercan_devices {
inline std::string quote(const std::string & value) {
    std::string out = "\"";
    const char * hex = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += c;
    }
    return out + '"';
}
inline const char * backend(ggml_backend_dev_t dev) {
    return dev ? ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)) : "unknown";
}
inline bool gpu(ggml_backend_dev_t dev) {
    return dev && (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
                   ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU);
}
inline std::vector<ggml_backend_dev_t> cuda() {
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (gpu(dev) && std::strcmp(backend(dev), "CUDA") == 0) devices.push_back(dev);
    }
    return devices;
}
inline ggml_backend_dev_t first_gpu() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (gpu(dev)) return dev;
    }
    return nullptr;
}
inline const char * kind(ggml_backend_dev_t dev) {
    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accelerator";
        default: return "other";
    }
}
}
