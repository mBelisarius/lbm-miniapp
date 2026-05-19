#ifndef LBMINI_ASYNC_WRITER_HPP_
#define LBMINI_ASYNC_WRITER_HPP_

#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>
#include "Data/FluidData.hpp"
#include "Data/MeshData.hpp"

namespace lbmini {

template<typename Scalar>
struct OutputSnapshot {
  Eigen::Index step;
  std::string outDir;
  double elapsed_seconds;
  Eigen::Tensor<Scalar, 2, Eigen::RowMajor> rho;
  Eigen::Tensor<Scalar, 2, Eigen::RowMajor> p;
  Eigen::Tensor<Scalar, 3, Eigen::RowMajor> u;
};

template<typename Scalar>
class AsyncWriter {
public:
  using Index = Eigen::Index;

  AsyncWriter(
    const FluidData<Scalar>& fluid,
    const MeshData<Scalar, 2>& mesh,
    size_t maxQueueSize = 8
  ) : fluid_(fluid), mesh_(mesh), maxQueueSize_(maxQueueSize), stopping_(false), inFlight_(0) {
    PrecomputeCoordinates();
    workerThread_ = std::thread(&AsyncWriter::WorkerLoop, this);
  }

  ~AsyncWriter() {
    Stop();
  }

  AsyncWriter(const AsyncWriter&) = delete;
  AsyncWriter& operator=(const AsyncWriter&) = delete;

  void Enqueue(OutputSnapshot<Scalar>&& snapshot) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // Apply backpressure if queue is full to bound memory usage
      cvFlush_.wait(lock, [this]() {
        return queue_.size() < maxQueueSize_ || stopping_;
      });

      if (stopping_) {
        return;
      }

      queue_.push(std::move(snapshot));
      ++inFlight_;
    }
    cvTask_.notify_one();
  }

  void WaitAll() {
    std::unique_lock<std::mutex> lock(mutex_);
    cvFlush_.wait(lock, [this]() {
      return queue_.empty() && inFlight_ == 0;
    });
  }

  void Stop() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (stopping_) {
        return;
      }
      stopping_ = true;
    }
    cvTask_.notify_all();
    cvFlush_.notify_all();

    if (workerThread_.joinable()) {
      workerThread_.join();
    }
  }

private:
  void PrecomputeCoordinates() {
    dx_ = mesh_.lx() / static_cast<Scalar>(mesh_.nx());
    dy_ = (mesh_.ny() > 0) ? mesh_.ly() / static_cast<Scalar>(mesh_.ny()) : mesh_.ly();

    constexpr Scalar cs2 = static_cast<Scalar>(1.0) / static_cast<Scalar>(3.0);
    const Scalar uPhysRef = std::sqrt(fluid_.gamma * fluid_.pressureL / fluid_.densityL);
    const Scalar uLuRef = std::sqrt(fluid_.gamma * cs2);
    velocityScale_ = uPhysRef / uLuRef;
    pressureScale_ = static_cast<Scalar>(1.0) / cs2;

    const Index nx = mesh_.nx();
    const Index ny = mesh_.ny();

    xCoords_.resize(nx);
    xCoordStrings_.resize(nx);
    for (Index i = 0; i < nx; ++i) {
      xCoords_[i] = 0.5 * dx_ + dx_ * i;
      char buf[32];
      auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), xCoords_[i]);
      xCoordStrings_[i].assign(buf, ptr - buf);
    }

    yCoords_.resize(ny);
    yCoordStrings_.resize(ny);
    for (Index j = 0; j < ny; ++j) {
      yCoords_[j] = 0.5 * dy_ + dy_ * j;
      char buf[32];
      auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), yCoords_[j]);
      yCoordStrings_[j].assign(buf, ptr - buf);
    }
  }

  void WorkerLoop() {
    // Reusable write buffer (64KB chunks to optimize file I/O)
    constexpr size_t kBufferSize = 1024 * 1024; // 1 MB chunk buffer
    std::vector<char> buffer(kBufferSize);

    while (true) {
      OutputSnapshot<Scalar> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cvTask_.wait(lock, [this]() {
          return !queue_.empty() || stopping_;
        });

        if (queue_.empty() && stopping_) {
          break;
        }

        task = std::move(queue_.front());
        queue_.pop();
      }

      WriteSnapshotToDisk(task, buffer);

      {
        std::unique_lock<std::mutex> lock(mutex_);
        --inFlight_;
        cvFlush_.notify_all();
      }
    }
  }

  void WriteSnapshotToDisk(const OutputSnapshot<Scalar>& task, std::vector<char>& chunkBuffer) {
    std::filesystem::path dir(task.outDir);
    if (!std::filesystem::exists(dir)) {
      std::filesystem::create_directories(dir);
    }
    std::filesystem::path filePath = dir / ("data_" + std::to_string(task.step) + ".csv");

    #if defined(_WIN32)
    FILE* fp = nullptr;
    _wfopen_s(&fp, filePath.wstring().c_str(), L"wb");
    #else
    FILE* fp = fopen(filePath.c_str(), "wb");
    #endif

    if (!fp) {
      std::cerr << "Failed to open output file for writing: " << filePath << std::endl;
      return;
    }

    // Use large internal C stream buffer
    constexpr size_t kStreamBufferSize = 512 * 1024;
    std::vector<char> streamBuf(kStreamBufferSize);
    setvbuf(fp, streamBuf.data(), _IOFBF, kStreamBufferSize);

    // Format runtime string once
    char runtimeBuf[32];
    auto [runtimePtr, runtimeEc] = std::to_chars(runtimeBuf, runtimeBuf + sizeof(runtimeBuf), task.elapsed_seconds);
    const std::string_view runtimeStr(runtimeBuf, runtimePtr - runtimeBuf);

    // Write CSV header
    static constexpr std::string_view kHeader = "runtime,x,y,ux,uy,density,pressure\n";
    fwrite(kHeader.data(), 1, kHeader.size(), fp);

    size_t bufPos = 0;
    const size_t bufCap = chunkBuffer.size();
    char* bufData = chunkBuffer.data();

    auto flushChunk = [&]() {
      if (bufPos > 0) {
        fwrite(bufData, 1, bufPos, fp);
        bufPos = 0;
      }
    };

    const Index nx = mesh_.nx();
    const Index ny = mesh_.ny();

    for (Index i = 0; i < nx; ++i) {
      const std::string_view xStr = xCoordStrings_[i];

      for (Index j = 0; j < ny; ++j) {
        const std::string_view yStr = yCoordStrings_[j];

        // Format row into buffer; flush if near capacity
        // Worst-case row length is ~128 bytes
        if (bufPos + 256 >= bufCap) {
          flushChunk();
        }

        // runtime,x,y
        std::memcpy(bufData + bufPos, runtimeStr.data(), runtimeStr.size());
        bufPos += runtimeStr.size();
        bufData[bufPos++] = ',';

        std::memcpy(bufData + bufPos, xStr.data(), xStr.size());
        bufPos += xStr.size();
        bufData[bufPos++] = ',';

        std::memcpy(bufData + bufPos, yStr.data(), yStr.size());
        bufPos += yStr.size();
        bufData[bufPos++] = ',';

        // ux
        const Scalar uxVal = task.u(i, j, 0) * velocityScale_;
        auto [uxPtr, uxEc] = std::to_chars(bufData + bufPos, bufData + bufPos + 32, uxVal);
        bufPos = uxPtr - bufData;
        bufData[bufPos++] = ',';

        // uy
        const Scalar uyVal = task.u(i, j, 1) * velocityScale_;
        auto [uyPtr, uyEc] = std::to_chars(bufData + bufPos, bufData + bufPos + 32, uyVal);
        bufPos = uyPtr - bufData;
        bufData[bufPos++] = ',';

        // density
        const Scalar rhoVal = task.rho(i, j);
        auto [rhoPtr, rhoEc] = std::to_chars(bufData + bufPos, bufData + bufPos + 32, rhoVal);
        bufPos = rhoPtr - bufData;
        bufData[bufPos++] = ',';

        // pressure
        const Scalar pVal = task.p(i, j) * pressureScale_;
        auto [pPtr, pEc] = std::to_chars(bufData + bufPos, bufData + bufPos + 32, pVal);
        bufPos = pPtr - bufData;
        bufData[bufPos++] = '\n';
      }
    }

    flushChunk();
    fclose(fp);
  }

  FluidData<Scalar> fluid_;
  MeshData<Scalar, 2> mesh_;
  size_t maxQueueSize_;

  Scalar dx_{0};
  Scalar dy_{0};
  Scalar velocityScale_{0};
  Scalar pressureScale_{0};

  std::vector<Scalar> xCoords_;
  std::vector<std::string> xCoordStrings_;
  std::vector<Scalar> yCoords_;
  std::vector<std::string> yCoordStrings_;

  std::queue<OutputSnapshot<Scalar>> queue_;
  std::mutex mutex_;
  std::condition_variable cvTask_;
  std::condition_variable cvFlush_;
  std::thread workerThread_;
  bool stopping_;
  size_t inFlight_;
};

} // namespace lbmini

#endif // LBMINI_ASYNC_WRITER_HPP_
