#pragma once
#include <string>

namespace MGXConvert {
bool convertRawToOnnx(const std::string& modelPath,
                      const std::string& outPath);
}
