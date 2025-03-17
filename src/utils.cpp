#include "utils.hpp"
#include <fstream>
#include <iostream>

std::vector<float> read_tensor_data(const std::string &filename,
                                    size_t offset,
                                    size_t size)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error opening tensor file: " << filename << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    if (offset + size * sizeof(float) > fileSize)
    {
        std::cerr << "Error: Requested tensor size exceeds file size" << std::endl;
        return {};
    }

    file.seekg(offset, std::ios::beg);
    std::vector<float> tensor_data(size);
    file.read(reinterpret_cast<char *>(tensor_data.data()), size * sizeof(float));

    if (!file)
    {
        std::cerr << "Error reading tensor data" << std::endl;
        return {};
    }
    return tensor_data;
}
