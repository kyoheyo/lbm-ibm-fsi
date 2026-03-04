/// @file plugins/plugin_registry.cpp
/// @brief PluginRegistry 单例的 C 兼容回调适配器。
///
/// 本翻译单元提供具体的 `IBoundaryPlugin` / `IMeshPlugin` /
/// `IMotionPlugin` / `IFlexibleSolverPlugin` 实现，
/// 将普通 C 函数指针包装为 C++ 虚函数。
/// 这些适配器允许 Rust（及任何 C 消费者）通过 C ABI 注册插件回调，
/// 而无需将 C++ 虚函数派发细节暴露到 ABI 边界之外。
///
/// 四种适配器类型由 `lbm_set_plugins()`（定义于 `lbm_capi.cpp`）
/// 实例化，并存储在静态槽中，使其生命周期覆盖整个仿真过程。

#include "plugins/boundary_plugin.hpp"
#include "plugins/mesh_plugin.hpp"
#include "plugins/motion_plugin.hpp"
#include "plugins/flexible_plugin.hpp"
#include "plugins/plugin_registry.hpp"

#include <cstddef> // nullptr

// ---------------------------------------------------------------------------
// C 函数指针回调类型（必须与 lbm_capi.cpp 中的声明一致）
// Rust 对应类型: ffi::BoundaryFn / MeshFn / MotionFn / FlexibleFn — bindings/src/lib.rs
// ---------------------------------------------------------------------------
extern "C" {

typedef void (*lbm_boundary_fn) (void* grid, int step,
                                  void* userdata);
// Rust 对应: ffi::BoundaryFn
typedef void (*lbm_mesh_adapt_fn)(void* grid, int step,
                                  void* userdata);
// Rust 对应: ffi::MeshFn
typedef void (*lbm_motion_fn)   (void* grid, void* markers,
                                  double dt, int step,
                                  void* userdata);
// Rust 对应: ffi::MotionFn
typedef void (*lbm_flexible_fn) (void* markers,
                                  double dt, int step,
                                  void* userdata);
// Rust 对应: ffi::FlexibleFn

} // extern "C"

// ---------------------------------------------------------------------------
// 适配器：将 C 回调包装为 IBoundaryPlugin
// ---------------------------------------------------------------------------
namespace {

class BoundaryAdapter final : public lbm::IBoundaryPlugin {
public:
    lbm_boundary_fn fn   = nullptr;
    void*           data = nullptr;

    void apply(lbm::LatticeGrid& grid, int step) override {
        if (fn) fn(static_cast<void*>(&grid), step, data);
    }
    const char* name() const override { return "c_callback_boundary"; }
};

// ---------------------------------------------------------------------------
// 适配器：将 C 回调包装为 IMeshPlugin
// ---------------------------------------------------------------------------
class MeshAdapter final : public lbm::IMeshPlugin {
public:
    lbm_mesh_adapt_fn fn   = nullptr;
    void*             data = nullptr;

    void initialize(lbm::LatticeGrid& /*grid*/,
                    const void* /*cfg*/) override {}  // 不通过 C ABI 暴露

    void adapt(lbm::LatticeGrid& grid, int step) override {
        if (fn) fn(static_cast<void*>(&grid), step, data);
    }
    const char* name() const override { return "c_callback_mesh"; }
};

// ---------------------------------------------------------------------------
// 适配器：将 C 回调包装为 IMotionPlugin
// ---------------------------------------------------------------------------
class MotionAdapter final : public lbm::IMotionPlugin {
public:
    lbm_motion_fn fn   = nullptr;
    void*         data = nullptr;

    void update(lbm::LatticeGrid& grid,
                ibm::MarkerSet*  markers,
                double dt, int step) override {
        if (fn) fn(static_cast<void*>(&grid),
                   static_cast<void*>(markers),
                   dt, step, data);
    }
    const char* name() const override { return "c_callback_motion"; }
};

// ---------------------------------------------------------------------------
// 适配器：将 C 回调包装为 IFlexibleSolverPlugin
// ---------------------------------------------------------------------------
class FlexibleAdapter final : public lbm::IFlexibleSolverPlugin {
public:
    lbm_flexible_fn fn   = nullptr;
    void*           data = nullptr;

    void step(ibm::MarkerSet& markers, double dt, int step_idx) override {
        if (fn) fn(static_cast<void*>(&markers), dt, step_idx, data);
    }
    const char* name() const override { return "c_callback_flexible"; }
};

// ---------------------------------------------------------------------------
// 静态适配器槽 — 生命周期覆盖整个程序运行期
// ---------------------------------------------------------------------------
static BoundaryAdapter g_boundary_adapter;
static MeshAdapter     g_mesh_adapter;
static MotionAdapter   g_motion_adapter;
static FlexibleAdapter g_flexible_adapter;

} // anonymous namespace

// ---------------------------------------------------------------------------
// 公开 C ABI — 由 lbm_capi.cpp 调用
// ---------------------------------------------------------------------------
extern "C" {

/// 一次性注册全部四个可选插件回调。
/// Rust 封装: register_plugins() — bindings/src/lib.rs
///
/// 对不需要激活的回调传入 `nullptr`。
/// 第二次调用本函数会替换所有之前的注册。
///
/// @param boundary_fn   每步标准边界条件后调用。
/// @param boundary_data 转发给 boundary_fn 的不透明用户数据指针。
/// @param mesh_fn       每步流式迁移/宏观量更新后调用。
/// @param mesh_data     转发给 mesh_fn 的不透明用户数据指针。
/// @param motion_fn     碰撞前调用（位置更新）。
/// @param motion_data   转发给 motion_fn 的不透明用户数据指针。
/// @param flexible_fn   代替/补充 BeamSolver::step() 调用。
/// @param flexible_data 转发给 flexible_fn 的不透明用户数据指针。
void lbm_set_plugins(
    lbm_boundary_fn  boundary_fn,  void* boundary_data,
    lbm_mesh_adapt_fn mesh_fn,     void* mesh_data,
    lbm_motion_fn    motion_fn,    void* motion_data,
    lbm_flexible_fn  flexible_fn,  void* flexible_data)
{
    auto& reg = lbm::PluginRegistry::instance();

    if (boundary_fn) {
        g_boundary_adapter.fn   = boundary_fn;
        g_boundary_adapter.data = boundary_data;
        reg.set_boundary_plugin(&g_boundary_adapter);
    } else {
        reg.set_boundary_plugin(nullptr);
    }

    if (mesh_fn) {
        g_mesh_adapter.fn   = mesh_fn;
        g_mesh_adapter.data = mesh_data;
        reg.set_mesh_plugin(&g_mesh_adapter);
    } else {
        reg.set_mesh_plugin(nullptr);
    }

    if (motion_fn) {
        g_motion_adapter.fn   = motion_fn;
        g_motion_adapter.data = motion_data;
        reg.set_motion_plugin(&g_motion_adapter);
    } else {
        reg.set_motion_plugin(nullptr);
    }

    if (flexible_fn) {
        g_flexible_adapter.fn   = flexible_fn;
        g_flexible_adapter.data = flexible_data;
        reg.set_flexible_plugin(&g_flexible_adapter);
    } else {
        reg.set_flexible_plugin(nullptr);
    }
}

} // extern "C"
