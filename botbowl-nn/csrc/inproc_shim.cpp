// In-process libtorch inference for botbowl-nn (plan 066), behind the `torch` cargo feature.
//
// A thin `extern "C"` layer over the libtorch C++ API that the Python sidecar
// (`scripts/nn_server.py`) drives through pybind: the same TorchScript module, the same
// `optimize_for_inference` folds, the same per-bucket CUDA graph capture (three warm-up calls on a
// side stream, then capture on a dedicated capture stream into a private memory pool), the same
// single-copy pinned staging buffer and the same replay sequence (H2D, replay, D2H of the value
// vector, D2H of the policy rows only when asked). Each step below names the sidecar line it
// mirrors; keep them in step, or plan 066's bit-parity gate (§4 A) will say so.
//
// Every entry point catches C++ exceptions and returns an error code; `bbt_last_error` holds the
// message (thread-local). Nothing here is called concurrently on one object: a runner belongs to
// one batcher thread, and model-level calls (load, warm-up, capture) are serialised by the model's
// mutex.

#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

thread_local std::string g_err;

int set_err(const std::string& msg) {
  g_err = msg;
  return -1;
}

#define BBT_TRY try {
#define BBT_CATCH(ret)                                   \
  }                                                      \
  catch (const c10::Error& e) {                          \
    set_err(e.what_without_backtrace());                 \
    return ret;                                          \
  }                                                      \
  catch (const std::exception& e) {                      \
    set_err(e.what());                                   \
    return ret;                                          \
  }                                                      \
  catch (...) {                                          \
    set_err("unknown C++ exception");                    \
    return ret;                                          \
  }

struct Model {
  torch::jit::Module module;
  c10::Device device;
  int64_t c, f, a;
  std::mutex mu;
  // One replay stream per batcher lane, created on first use.
  std::vector<std::optional<c10::cuda::CUDAStream>> lanes;
  // `torch.cuda.graph`'s class-level default capture stream: one per process in the sidecar, one
  // per model here (only ever used under `mu`).
  std::optional<c10::cuda::CUDAStream> capture_stream;
  Model(torch::jit::Module m, c10::Device d, int64_t c_, int64_t f_, int64_t a_)
      : module(std::move(m)), device(d), c(c_), f(f_), a(a_) {}

  c10::cuda::CUDAStream lane(int i) {
    if ((size_t)i >= lanes.size()) lanes.resize(i + 1);
    if (!lanes[i]) lanes[i] = c10::cuda::getStreamFromPool(false, device.index());
    return *lanes[i];
  }
};

std::vector<torch::jit::IValue> inputs3(const at::Tensor& s, const at::Tensor& g, const at::Tensor& m) {
  return {s, g, m};
}

// (policy, value) out of the masked module's tuple.
std::pair<at::Tensor, at::Tensor> outputs2(const torch::jit::IValue& out) {
  auto t = out.toTuple();
  return {t->elements()[0].toTensor(), t->elements()[1].toTensor()};
}

struct Runner {
  Model* model;
  int64_t bucket, h, w;
  int64_t n_s, n_g, n_m;
  at::Tensor dev;     // device input: spatial | global | mask
  at::Tensor host;    // pinned staging, same layout
  at::Tensor host_v;  // pinned (bucket,)
  at::Tensor host_p;  // pinned (bucket, A*h*w)
  at::Tensor value_flat, policy_flat;  // graph outputs, views
  std::unique_ptr<at::cuda::CUDAGraph> graph;
  c10::cuda::CUDAStream stream;
  // `torch.cuda.Event(enable_timing=True)` / `(enable_timing=True, blocking=True)`.
  at::cuda::CUDAEvent start{cudaEventDefault};
  at::cuda::CUDAEvent done{cudaEventBlockingSync};
  Runner(Model* m, int64_t b, int64_t h_, int64_t w_, c10::cuda::CUDAStream s)
      : model(m), bucket(b), h(h_), w(w_), stream(s) {}
};

}  // namespace

extern "C" {

const char* bbt_last_error() { return g_err.c_str(); }

int bbt_cuda_device_count() {
  BBT_TRY
  return (int)c10::cuda::device_count();
  BBT_CATCH(-1)
}

// `torch.backends.cudnn.benchmark` / `.deterministic` — process-wide, as in Python. The sidecar
// sets benchmark=True in `main` before loading anything.
int bbt_set_cudnn(int benchmark, int deterministic) {
  BBT_TRY
  at::globalContext().setBenchmarkCuDNN(benchmark != 0);
  at::globalContext().setDeterministicCuDNN(deterministic != 0);
  return 0;
  BBT_CATCH(-1)
}

// Load a TorchScript `MaskedNet` (`python -m bbnn.export_ts`) onto `device` ("cuda", "cuda:1",
// "cpu"), put it in eval mode and, if `optimize`, apply `torch::jit::optimize_for_inference` — the
// C++ twin of the `torch.jit.optimize_for_inference` call in the sidecar's `maybe_trace_masked`.
// Returns NULL on error.
void* bbt_model_load(const char* path, const char* device, int optimize, int64_t c, int64_t f, int64_t a) {
  BBT_TRY
  c10::Device dev(device);
  if (dev.is_cuda()) {
    int n = (int)c10::cuda::device_count();
    if (n <= 0) {
      set_err("no CUDA device visible to libtorch (device_count() == 0)");
      return nullptr;
    }
    if (dev.has_index() && dev.index() >= n) {
      set_err("CUDA device index out of range");
      return nullptr;
    }
    if (!dev.has_index()) dev = c10::Device(c10::kCUDA, 0);
  }
  torch::jit::Module m = torch::jit::load(path, dev);
  m.eval();
  if (optimize) {
    m = torch::jit::optimize_for_inference(m);
  }
  return new Model(std::move(m), dev, c, f, a);
  BBT_CATCH(nullptr)
}

void bbt_model_free(void* model) { delete static_cast<Model*>(model); }

// The optimised forward graph as text, for diffing against the sidecar's (`traced.graph`).
// Writes at most `cap` bytes (NUL-terminated) and returns the full length, or -1.
int64_t bbt_model_graph(void* model, char* buf, int64_t cap) {
  BBT_TRY
  auto* m = static_cast<Model*>(model);
  std::string s = m->module.get_method("forward").graph()->toString();
  if (buf && cap > 0) {
    int64_t n = std::min<int64_t>(cap - 1, (int64_t)s.size());
    std::memcpy(buf, s.data(), n);
    buf[n] = 0;
  }
  return (int64_t)s.size();
  BBT_CATCH(-1)
}

// One uncaptured forward under `no_grad` on the model's device: `input` is the staging layout
// (`n` rows of spatial C*h*w, then n*F globals, then n*h*w mask), outputs are written to
// `out_policy` (n*A*h*w, may be NULL) and `out_value` (n). Used for the sidecar's
// trace-validation calls (which also advance the TorchScript profiling executor exactly as the
// sidecar's do), and by tests.
int bbt_model_forward(void* model, const float* input, int64_t n, int64_t h, int64_t w, float* out_policy,
                      float* out_value) {
  BBT_TRY
  auto* m = static_cast<Model*>(model);
  std::lock_guard<std::mutex> lock(m->mu);
  c10::DeviceGuard dg(m->device);
  const int64_t n_s = n * m->c * h * w, n_g = n * m->f, n_m = n * h * w;
  auto host = torch::from_blob(const_cast<float*>(input), {n_s + n_g + n_m}, torch::kFloat32);
  auto dev = host.to(m->device);
  auto s = dev.slice(0, 0, n_s).view({n, m->c, h, w});
  auto g = dev.slice(0, n_s, n_s + n_g).view({n, m->f});
  auto mk = dev.slice(0, n_s + n_g).view({n, 1, h, w});
  torch::NoGradGuard ng;
  auto [p, v] = outputs2(m->module.forward(inputs3(s, g, mk)));
  auto vc = v.reshape({-1}).to(torch::kCPU).contiguous();
  std::memcpy(out_value, vc.data_ptr<float>(), sizeof(float) * n);
  if (out_policy) {
    auto pc = p.reshape({n, -1}).to(torch::kCPU).contiguous();
    std::memcpy(out_policy, pc.data_ptr<float>(), sizeof(float) * pc.numel());
  }
  return 0;
  BBT_CATCH(-1)
}

// `GraphRunner.__init__`: device + pinned buffers for one `(bucket, h, w)`, three warm-up calls
// on a side stream, then the capture. `lane` picks the replay stream (one per batcher thread).
// Returns NULL on error. CUDA only.
void* bbt_runner_new(void* model, int64_t bucket, int64_t h, int64_t w, int lane) {
  BBT_TRY
  auto* m = static_cast<Model*>(model);
  if (!m->device.is_cuda()) {
    set_err("graph runners need a CUDA device");
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(m->mu);
  c10::cuda::CUDAGuard dg(m->device.index());
  auto r = std::make_unique<Runner>(m, bucket, h, w, m->lane(lane));
  r->n_s = bucket * m->c * h * w;
  r->n_g = bucket * m->f;
  r->n_m = bucket * h * w;
  const int64_t total = r->n_s + r->n_g + r->n_m;
  // self.dev = torch.zeros(n_s + n_g + n_m, device=device)
  r->dev = torch::zeros({total}, torch::TensorOptions().dtype(torch::kFloat32).device(m->device));
  auto dev_s = r->dev.slice(0, 0, r->n_s).view({bucket, m->c, h, w});
  auto dev_g = r->dev.slice(0, r->n_s, r->n_s + r->n_g).view({bucket, m->f});
  auto dev_m = r->dev.slice(0, r->n_s + r->n_g).view({bucket, 1, h, w});
  // Slot buffers: torch.zeros(...).pin_memory(). One slot per runner here: a runner is owned by
  // one batcher thread that launches and waits strictly in turn, and the pipeline's depth comes
  // from the batcher threads instead of from slots.
  auto pinned = torch::TensorOptions().dtype(torch::kFloat32).pinned_memory(true);
  r->host = torch::zeros({total}, pinned);
  r->host_v = torch::zeros({bucket}, pinned);
  r->host_p = torch::zeros({bucket, m->a * h * w}, pinned);

  auto cur = c10::cuda::getCurrentCUDAStream(m->device.index());
  {
    // stream = torch.cuda.Stream(); stream.wait_stream(current)
    auto side = c10::cuda::getStreamFromPool(false, m->device.index());
    at::cuda::CUDAEvent ev;
    ev.record(cur);
    ev.block(side);
    {
      c10::cuda::CUDAStreamGuard sg(side);
      torch::NoGradGuard ng;
      for (int i = 0; i < 3; i++) m->module.forward(inputs3(dev_s, dev_g, dev_m));
    }
    // torch.cuda.current_stream().wait_stream(stream)
    at::cuda::CUDAEvent ev2;
    ev2.record(side);
    ev2.block(cur);
  }
  // torch.cuda.graph.__enter__: synchronize, empty_cache, enter the capture stream, capture_begin.
  c10::cuda::device_synchronize();
  c10::cuda::CUDACachingAllocator::emptyCache();
  if (!m->capture_stream) m->capture_stream = c10::cuda::getStreamFromPool(false, m->device.index());
  r->graph = std::make_unique<at::cuda::CUDAGraph>();
  {
    c10::cuda::CUDAStreamGuard sg(*m->capture_stream);
    // Thread-local capture mode, not the sidecar's "global": the sidecar captures from its only
    // thread, but here another net's batchers may be replaying (and syncing) on other threads
    // while this one captures (the eval phase loads its second net while the first serves), and
    // global mode turns their calls into capture errors. The mode only scopes error checking; the
    // captured kernels are the same.
    r->graph->capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    {
      torch::NoGradGuard ng;
      auto [p, v] = outputs2(m->module.forward(inputs3(dev_s, dev_g, dev_m)));
      r->value_flat = v.reshape({-1});
      r->policy_flat = p.reshape({bucket, -1});
    }
    r->graph->capture_end();
  }
  return r.release();
  BBT_CATCH(nullptr)
}

void bbt_runner_free(void* runner) { delete static_cast<Runner*>(runner); }

float* bbt_runner_host_in(void* runner) { return static_cast<Runner*>(runner)->host.data_ptr<float>(); }
float* bbt_runner_host_value(void* runner) { return static_cast<Runner*>(runner)->host_v.data_ptr<float>(); }
float* bbt_runner_host_policy(void* runner) { return static_cast<Runner*>(runner)->host_p.data_ptr<float>(); }

// `GraphRunner.launch`: record start, one H2D of the whole staging buffer, replay, D2H of the value
// vector (all `bucket` floats — cheaper than slicing), D2H of the first `n` policy rows only if
// asked, record done. Non-blocking; all on the runner's lane stream.
int bbt_runner_launch(void* runner, int64_t n, int want_policy) {
  BBT_TRY
  auto* r = static_cast<Runner*>(runner);
  c10::cuda::CUDAGuard dg(r->model->device.index());
  c10::cuda::CUDAStreamGuard sg(r->stream);
  r->start.record(r->stream);
  r->dev.copy_(r->host, /*non_blocking=*/true);
  r->graph->replay();
  r->host_v.copy_(r->value_flat, true);
  if (want_policy) r->host_p.slice(0, 0, n).copy_(r->policy_flat.slice(0, 0, n), true);
  r->done.record(r->stream);
  return 0;
  BBT_CATCH(-1)
}

// `GraphRunner.wait`: sleep on the done event (blocking sync), return device ms in `*device_ms`.
int bbt_runner_wait(void* runner, float* device_ms) {
  BBT_TRY
  auto* r = static_cast<Runner*>(runner);
  c10::cuda::CUDAGuard dg(r->model->device.index());
  r->done.synchronize();
  if (device_ms) *device_ms = r->start.elapsed_time(r->done);
  return 0;
  BBT_CATCH(-1)
}

}  // extern "C"
