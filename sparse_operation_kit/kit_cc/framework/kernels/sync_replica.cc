#include "facade.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;

class SyncReplicaOp : public OpKernel {
 public:
  explicit SyncReplicaOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    try {
      SparseOperationKit::Facade::instance()->sync_replica();
    } catch (const std::exception& error) {
      ctx->SetStatus(errors::Aborted(error.what()));
      return;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("SyncReplica").Device(DEVICE_CPU), SyncReplicaOp);

}  // namespace tensorflow