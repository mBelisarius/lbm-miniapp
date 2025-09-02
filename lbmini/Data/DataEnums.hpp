#ifndef LBM_MINIAPP_DATA_DATAENUMS_HPP_
#define LBM_MINIAPP_DATA_DATAENUMS_HPP_

#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>

namespace lbmini {
enum class TargetEnum {
  CPU,
  GPU,
};

enum class BackendEnum {
  Plain,
  OpenMP,
};
} // namespace lbmini

namespace YAML {
// Specialization for TargetEnum
template<>
struct convert<lbmini::TargetEnum> {
  static Node encode(const lbmini::TargetEnum& target) {
    Node node;
    switch (target) {
      case lbmini::TargetEnum::CPU:
        node = "CPU";
        break;
      case lbmini::TargetEnum::GPU:
        node = "GPU";
        break;
      default:
        throw std::runtime_error("Invalid TargetEnum value for encoding.");
    }
    return node;
  }

  static bool decode(const Node& node, lbmini::TargetEnum& target) {
    if (!node.IsScalar()) {
      return false;
    }

    const auto value = node.as<std::string>();
    if (value == "CPU") {
      target = lbmini::TargetEnum::CPU;
    } else if (value == "GPU") {
      target = lbmini::TargetEnum::GPU;
    } else {
      return false; // Invalid value
    }
    return true;
  }
};

// Specialization for BackendEnum
template<>
struct convert<lbmini::BackendEnum> {
  static Node encode(const lbmini::BackendEnum& backend) {
    Node node;
    switch (backend) {
      case lbmini::BackendEnum::Plain:
        node = "Plain";
        break;
      case lbmini::BackendEnum::OpenMP:
        node = "OpenMP";
        break;
      default:
        throw std::runtime_error("Invalid BackendEnum value for encoding.");
    }
    return node;
  }

  static bool decode(const Node& node, lbmini::BackendEnum& backend) {
    if (!node.IsScalar()) {
      return false;
    }

    const auto value = node.as<std::string>();
    if (value == "Plain") {
      backend = lbmini::BackendEnum::Plain;
    } else if (value == "OpenMP") {
      backend = lbmini::BackendEnum::OpenMP;
    } else {
      return false; // Invalid value
    }
    return true;
  }
};
} // namespace YAML

#endif // LBM_MINIAPP_DATA_DATAENUMS_HPP_
