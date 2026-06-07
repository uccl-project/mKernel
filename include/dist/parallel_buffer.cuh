#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/utils/pybind.h>

#include "../comm/vmm.cuh"
#include "../comm/ipc.cuh"
#include "../comm/local_broker.cuh"

namespace dist {

/**
 * @brief Distributed tensor wrapper for multi-GPU IPC sharing and multicast.
 *        Can be later used for easy dbuf creation right before a kernel call.
 *        Meant to be used as a single object per thread/process.
 */
struct ParallelBuffer {
    inline static std::map<std::pair<int, int>, comm::LocalBroker> brokers_; // lazily initialized

    at::Tensor data_; // for direct access from PyTorch
    std::vector<int64_t> shape_;
    at::ScalarType dtype_;

    std::vector<void *> raw_ptrs_;
    size_t allocated_size_;
    // Keep the original VMM allocation handle for the local rank (VMM flavor only).
    // Some CUDA multicast paths require binding the original allocation handle,
    // not a handle re-retained from the mapped address.
    std::optional<comm::vmm::Handle> local_vmm_handle_;

    int local_rank_; // identical to device index
    int local_world_size_;

    bool multicast_;
    void *multicast_ptr_;
    size_t multicast_allocated_size_;

    comm::ipc::Flavor ipc_flavor_;

    __host__ inline ParallelBuffer(
        const at::Tensor &tensor,
        int local_rank,
        int local_world_size,
        bool multicast
    ) : data_(tensor),
        shape_(tensor.sizes().vec()),
        dtype_(tensor.scalar_type()),
        raw_ptrs_(local_world_size, nullptr),
        allocated_size_(tensor.nbytes()),
        local_vmm_handle_(std::nullopt),
        local_rank_(local_rank),
        local_world_size_(local_world_size),
        multicast_(multicast),
        multicast_ptr_(nullptr),
        multicast_allocated_size_(0),
        ipc_flavor_(comm::ipc::Flavor::kLegacy) {

        TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
        TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
        TORCH_CHECK(tensor.dim() <= 4, "Only tensors with dim <= 4 are supported for ParallelBuffer");
        TORCH_CHECK(tensor.device().index() == local_rank_, "Tensor device index must match local_rank");
        TORCH_CHECK(local_rank_ >= 0, "local_rank must be non-negative");
        TORCH_CHECK(local_rank_ < local_world_size_, "local_rank must be less than local_world_size");
        TORCH_CHECK(!multicast, "Multicast is not supported for pre-allocated tensors");

        brokers_.try_emplace(
            {local_rank_, local_world_size_},
            local_rank_, local_world_size_
        );

        if (brokers_.size() > 1)
            std::cerr << "WARNING: 2 LocalBroker instances created in the same process. This is not safe." << std::endl;

        c10::cuda::CUDAGuard device_guard(local_rank_);
        exchange_ipc_handles<comm::ipc::Flavor::kLegacy>();
    }

    __host__ inline ParallelBuffer(
        const std::vector<int64_t> &shape,
        const at::ScalarType dtype,
        int local_rank,
        int local_world_size,
        bool multicast,
        const std::string &backing = "vmm"
    ) : shape_(shape),
        dtype_(dtype),
        raw_ptrs_(local_world_size, nullptr),
        allocated_size_(0),
        local_vmm_handle_(std::nullopt),
        local_rank_(local_rank),
        local_world_size_(local_world_size),
        multicast_(multicast),
        multicast_ptr_(nullptr),
        multicast_allocated_size_(0),
        ipc_flavor_(comm::ipc::Flavor::kVmmFd) {

        TORCH_CHECK(local_rank_ >= 0, "local_rank must be non-negative");
        TORCH_CHECK(local_rank_ < local_world_size_, "local_rank must be less than local_world_size");

        brokers_.try_emplace(
            {local_rank_, local_world_size_},
            local_rank_, local_world_size_
        );

        if (brokers_.size() > 1)
            std::cerr << "WARNING: 2 LocalBroker instances created in the same process. This is not safe." << std::endl;

        TORCH_CHECK(backing == "vmm" || backing == "cuda_malloc",
            "Unsupported ParallelBuffer backing '", backing,
            "'. Expected 'vmm' or 'cuda_malloc'.");

        c10::cuda::CUDAGuard device_guard(local_rank_);
        if (backing == "vmm") {
            ipc_flavor_ = comm::ipc::Flavor::kVmmFd;
            create_shareable_cuda_tensor();
            exchange_ipc_handles<comm::ipc::Flavor::kVmmFd>();

            if (multicast_)
                initialize_multicast();
        } else if (backing == "cuda_malloc") {
            TORCH_CHECK(!multicast_,
                "ParallelBuffer backing='cuda_malloc' uses legacy CUDA IPC and "
                "does not support CUDA multicast. Use backing='vmm' for multicast buffers.");
            ipc_flavor_ = comm::ipc::Flavor::kLegacy;
            create_cuda_malloc_tensor();
            exchange_ipc_handles<comm::ipc::Flavor::kLegacy>();
        } else {
            TORCH_CHECK(false, "Unreachable ParallelBuffer backing: ", backing);
        }
    }

    ParallelBuffer(const ParallelBuffer&) = delete;
    ParallelBuffer& operator=(const ParallelBuffer&) = delete;
    ParallelBuffer& operator=(ParallelBuffer&& other) = delete;

    __host__ inline ParallelBuffer(ParallelBuffer&& other) :
        data_(std::move(other.data_)),
        shape_(std::move(other.shape_)),
        dtype_(std::move(other.dtype_)),
        raw_ptrs_(std::move(other.raw_ptrs_)),
        allocated_size_(other.allocated_size_),
        local_rank_(other.local_rank_),
        local_world_size_(other.local_world_size_),
        multicast_(other.multicast_),
        multicast_ptr_(other.multicast_ptr_),
        multicast_allocated_size_(other.multicast_allocated_size_),
        ipc_flavor_(other.ipc_flavor_) {
        local_vmm_handle_ = std::move(other.local_vmm_handle_);
        other.data_ = at::Tensor();
        other.shape_.clear();
        other.dtype_ = at::ScalarType::Undefined;
        other.raw_ptrs_.clear();
        other.allocated_size_ = 0;
        other.local_vmm_handle_.reset();
        other.local_rank_ = -1;
        other.local_world_size_ = -1;
        other.multicast_ = false;
        other.multicast_ptr_ = nullptr;
        other.multicast_allocated_size_ = 0;
    }

    __host__ inline ~ParallelBuffer() {
        destroy();
    }

    __host__ inline at::Tensor data() const {
        return data_;
    }

    __host__ inline void create_shareable_cuda_tensor() {
        c10::cuda::CUDAGuard device_guard(local_rank_);

        TORCH_CHECK(!shape_.empty(), "Shape must be non-empty");
        TORCH_CHECK(shape_.size() <= 4, "Shape must have at most 4 dimensions for ParallelBuffer");
        size_t size = c10::elementSize(dtype_);
        for (auto dim : shape_) {
            TORCH_CHECK(dim > 0, "Size dimensions must be positive");
            size *= static_cast<size_t>(dim);
        }

        void *raw_ptr;
        // Allocate + map + set access, but keep the original allocation handle.
        // We release it during destroy().
        comm::vmm::Handle handle;
        comm::vmm::alloc(&handle, &allocated_size_, size, local_rank_);
        comm::vmm::map(&raw_ptr, handle, allocated_size_);
        comm::vmm::set_access(raw_ptr, allocated_size_, local_world_size_);
        local_vmm_handle_ = handle;

        int local_rank = local_rank_;
        size_t allocated_size = allocated_size_;

        auto deleter = [local_rank, raw_ptr, allocated_size](void* p) mutable {
            if (!p) return;
            c10::cuda::CUDAGuard device_guard(local_rank);
            auto stream = c10::cuda::getCurrentCUDAStream().stream();
            MKERNEL_CUDACHECK(cudaStreamSynchronize(stream));
            comm::vmm::unmap(raw_ptr, allocated_size);
        };

        at::TensorOptions options = at::TensorOptions()
            .dtype(dtype_)
            .device(at::kCUDA, local_rank_);

        data_ = at::from_blob(raw_ptr, shape_, std::move(deleter), options);
    }

    __host__ inline void create_cuda_malloc_tensor() {
        c10::cuda::CUDAGuard device_guard(local_rank_);

        TORCH_CHECK(!shape_.empty(), "Shape must be non-empty");
        TORCH_CHECK(shape_.size() <= 4, "Shape must have at most 4 dimensions for ParallelBuffer");
        size_t size = c10::elementSize(dtype_);
        for (auto dim : shape_) {
            TORCH_CHECK(dim > 0, "Size dimensions must be positive");
            size *= static_cast<size_t>(dim);
        }

        void *raw_ptr = nullptr;
        MKERNEL_CUDACHECK(cudaMalloc(&raw_ptr, size));
        allocated_size_ = size;

        int local_rank = local_rank_;
        auto deleter = [local_rank](void* p) mutable {
            if (!p) return;
            c10::cuda::CUDAGuard device_guard(local_rank);
            MKERNEL_CUDACHECK(cudaFree(p));
        };

        at::TensorOptions options = at::TensorOptions()
            .dtype(dtype_)
            .device(at::kCUDA, local_rank_);

        data_ = at::from_blob(raw_ptr, shape_, std::move(deleter), options);
    }

    template <comm::ipc::Flavor IPC_FLAVOR>
    __host__ inline void exchange_ipc_handles() {
        using handle_t = comm::ipc::Handle<IPC_FLAVOR>;

        comm::ipc::check_support(local_rank_);
        void *raw_ptr = reinterpret_cast<void *>(data_.data_ptr());
        handle_t ipc_handle;
        comm::ipc::export_handle(&ipc_handle, raw_ptr);

        std::vector<handle_t> all_ipc_handles(local_world_size_);
        if constexpr (IPC_FLAVOR == comm::ipc::Flavor::kLegacy) {
            brokers_.at({local_rank_, local_world_size_}).exchange_data(
                reinterpret_cast<void *>(all_ipc_handles.data()),
                reinterpret_cast<void *>(&ipc_handle),
                sizeof(handle_t)
            );
        } else if constexpr (IPC_FLAVOR == comm::ipc::Flavor::kVmmFd) {
            brokers_.at({local_rank_, local_world_size_}).exchange_fds(
                reinterpret_cast<int *>(all_ipc_handles.data()),
                ipc_handle.value
            );
        } else {
            throw std::runtime_error("Invalid IPC flavor");
        }

        for (int i = 0; i < local_world_size_; i++) {
            if (i == local_rank_)
                raw_ptrs_[i] = raw_ptr;
            else
                comm::ipc::import_handle(&raw_ptrs_[i], all_ipc_handles[i], allocated_size_, local_world_size_);
        }
    }

    __host__ inline void initialize_multicast() {
        using handle_t = comm::ipc::Handle<comm::ipc::Flavor::kVmmFd>;

        comm::vmm::multicast_check(local_rank_);
        comm::ipc::check_support(local_rank_);
        comm::vmm::Handle multicast_handle;

        if (local_rank_ == 0) {
            comm::vmm::multicast_create_handle(
                &multicast_handle,
                &multicast_allocated_size_,
                allocated_size_,
                local_world_size_
            );

            if (allocated_size_ != multicast_allocated_size_)
                throw std::runtime_error("Multicast allocated size does not match memory allocated size");

            handle_t ipc_handle;
            comm::ipc::export_handle(&ipc_handle, multicast_handle);
            brokers_.at({local_rank_, local_world_size_}).broadcast_fd(nullptr, ipc_handle.value, 0);
        } else {
            handle_t ipc_handle;
            brokers_.at({local_rank_, local_world_size_}).broadcast_fd(&ipc_handle.value, -1, 0);
            multicast_allocated_size_ = allocated_size_;
            comm::ipc::import_handle(&multicast_handle, ipc_handle);
        }

        comm::vmm::multicast_bind_device(multicast_handle, local_rank_);
        brokers_.at({local_rank_, local_world_size_}).sync(); // must ensure all devices are added

        TORCH_CHECK(local_vmm_handle_.has_value(), "Missing local VMM allocation handle for multicast binding");
        const bool bind_retained_handle = []() {
            const char *v = std::getenv("MKERNEL_BIND_RETAINED_HANDLE");
            return v && std::string(v) == "1";
        }();
        if (bind_retained_handle) {
            CUmemGenericAllocationHandle retained_handle;
            comm::vmm::retain_handle(&retained_handle, reinterpret_cast<void *>(data_.data_ptr()));
            comm::vmm::multicast_bind_memory(multicast_handle, retained_handle, allocated_size_);
            comm::vmm::release(retained_handle);
        } else {
            comm::vmm::multicast_bind_memory(multicast_handle, *local_vmm_handle_, allocated_size_);
        }
        brokers_.at({local_rank_, local_world_size_}).sync();

        std::vector<uint64_t> mc_addrs(local_world_size_, 0);
        if (local_rank_ == 0) {
            comm::vmm::map(&multicast_ptr_, multicast_handle, multicast_allocated_size_);
            mc_addrs[0] = reinterpret_cast<uint64_t>(multicast_ptr_);
        }
        brokers_.at({local_rank_, local_world_size_}).sync();
        brokers_.at({local_rank_, local_world_size_}).exchange_data(mc_addrs.data(), mc_addrs.data(), sizeof(uint64_t));
        if (local_rank_ != 0) {
            comm::vmm::map_at(&multicast_ptr_, reinterpret_cast<void *>(mc_addrs[0]), multicast_handle, multicast_allocated_size_);
        }
        comm::vmm::set_access(multicast_ptr_, multicast_allocated_size_, local_world_size_);

        comm::vmm::release(multicast_handle);
    }

    __host__ inline void destroy() {
        if (local_rank_ < 0 || local_world_size_ <= 0) {
            return;
        }

        if (multicast_ && multicast_ptr_) {
            brokers_.at({local_rank_, local_world_size_}).sync();
            comm::vmm::Handle multicast_handle;
            comm::vmm::retain_handle(&multicast_handle, multicast_ptr_);
            comm::vmm::unmap(multicast_ptr_, multicast_allocated_size_);
            comm::vmm::multicast_unbind_device(multicast_handle, multicast_allocated_size_, local_rank_);
            brokers_.at({local_rank_, local_world_size_}).sync();
            comm::vmm::release(multicast_handle);
        }

        for (int i = 0; i < local_world_size_; i++) {
            if (i != local_rank_ && i < raw_ptrs_.size()) {
                if (ipc_flavor_ == comm::ipc::Flavor::kLegacy) {
                    comm::ipc::free_imported_mapping<comm::ipc::Flavor::kLegacy>(raw_ptrs_[i], allocated_size_);
                } else if (ipc_flavor_ == comm::ipc::Flavor::kVmmFd) {
                    comm::ipc::free_imported_mapping<comm::ipc::Flavor::kVmmFd>(raw_ptrs_[i], allocated_size_);
                } else {
                    throw std::runtime_error("Invalid IPC flavor");
                }
            }
        }
        brokers_.at({local_rank_, local_world_size_}).sync(); // must sync before destroying the tensor

        if (data_.defined())
            data_.reset(); // properly decreases the ref count

        if (local_vmm_handle_.has_value()) {
            comm::vmm::release(*local_vmm_handle_);
            local_vmm_handle_.reset();
        }

        shape_.clear();
        dtype_ = at::ScalarType::Undefined;
        raw_ptrs_.clear();
        allocated_size_ = 0;
        local_vmm_handle_.reset();
        local_rank_ = -1;
        local_world_size_ = -1;
        multicast_ = false;
        multicast_ptr_ = nullptr;
        multicast_allocated_size_ = 0;
    }
};

} // namespace dist

#define BIND_DIST_PARALLEL_BUFFER(m) \
    pybind11::class_<dist::ParallelBuffer>(m, "DistBuffer", pybind11::module_local()) \
        .def(pybind11::init<const at::Tensor&, int, int, bool>(), \
             pybind11::arg("tensor"), \
             pybind11::arg("local_rank"), \
             pybind11::arg("local_world_size"), \
             pybind11::arg("multicast") = false) \
        .def(pybind11::init<const std::vector<int64_t>&, const at::ScalarType&, int, int, bool, const std::string&>(), \
             pybind11::arg("shape"), \
             pybind11::arg("dtype"), \
             pybind11::arg("local_rank"), \
             pybind11::arg("local_world_size"), \
             pybind11::arg("multicast") = false, \
             pybind11::arg("backing") = "vmm") \
        .def("data", &dist::ParallelBuffer::data) \
        .def_readonly("data_", &dist::ParallelBuffer::data_) \
        .def_property_readonly("multicast_ptr_u64", [](const dist::ParallelBuffer &t) { \
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(t.multicast_ptr_)); \
        }) \
        .def_property_readonly("multicast_enabled", [](const dist::ParallelBuffer &t) { \
            return t.multicast_; \
        }) \
        .def_readonly("local_rank_", &dist::ParallelBuffer::local_rank_) \
        .def_readonly("local_world_size_", &dist::ParallelBuffer::local_world_size_)
