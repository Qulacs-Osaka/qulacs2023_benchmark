#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    auto device = sycl::device(sycl::cpu_selector_v);
    std::cout << "Device Name: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Device Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
}