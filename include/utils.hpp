#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>

// Utility to read raw floats from disk
std::vector<float> read_tensor_data(const std::string &filename,
                                    size_t offset,
                                    size_t size);

#endif // UTILS_HPP
