#ifndef LBMINI_OPENMP_CPU_COUNTCORES_HPP_
#define LBMINI_OPENMP_CPU_COUNTCORES_HPP_

#include <cmath>
#include <fstream>
#include <string>
#include <thread>

namespace lbmini::openmp::cpu {
/**
 * @brief Count physical "performance" CPU cores available to the process.
 *
 * On Intel hybrid CPUs (Alder Lake / Raptor Lake / Meteor Lake and later) the
 * Linux kernel exposes per-PMU CPU masks under
 *   `/sys/devices/cpu_core/cpus`  (P-cores)
 *   `/sys/devices/cpu_atom/cpus`  (E-cores)
 * so we can opt out of E-cores, which are typically the slow path for tight,
 * memory-bound LBM kernels (smaller caches, no AVX-512, different frequency
 * ceiling) and degrade scalability when mixed with P-cores under `schedule(static)`.
 *
 * Returns the number of P-cores parsed from `cpu_core/cpus`. If the file does
 * not exist (non-hybrid CPU, non-Linux) or cannot be parsed, falls back to
 * `std::thread::hardware_concurrency()` (which is what OpenMP's default would
 * pick anyway). Returns 0 only if both sources fail, letting the caller keep
 * OpenMP's own default.
 */
inline int CountPerformanceCores() {
  auto parseCpuMask = [](const std::string& path) -> int {
    std::ifstream in(path);
    if (!in.is_open())
      return 0;

    std::string s;
    std::getline(in, s);
    if (s.empty())
      return 0;

    int count = 0;
    std::size_t i = 0;
    while (i < s.size()) {
      std::size_t j = i;
      while (j < s.size() && s[j] != ',')
        ++j;
      if (const std::string tok = s.substr(i, j - i); !tok.empty()) {
        if (const std::size_t dash = tok.find('-'); dash == std::string::npos) {
          count += 1;
        } else {
          const int a = std::stoi(tok.substr(0, dash));
          const int b = std::stoi(tok.substr(dash + 1));
          if (b >= a)
            count += (b - a + 1);
        }
      }
      i = j + 1;
    }
    return count;
  };

  if (const int pCores = parseCpuMask("/sys/devices/cpu_core/cpus"); pCores > 0)
    return pCores;

  return static_cast<int>(std::thread::hardware_concurrency());
}
}

#endif // LBMINI_OPENMP_CPU_COUNTCORES_HPP_
