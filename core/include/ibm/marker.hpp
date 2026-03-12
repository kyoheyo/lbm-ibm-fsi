#pragma once
#include <vector>
#include <string>

namespace ibm {

// ---------------------------------------------------------------------------
// 浸入边界上的一个拉格朗日标记点
// ---------------------------------------------------------------------------
struct Marker {
    double x, y, z;            ///< 当前位置
    double x0, y0, z0;         ///< 参考（未变形）位置
    double fx, fy, fz;         ///< 该标记点处的 IBM 力密度
    double ux, uy, uz;         ///< 标记点处插值得到的流体速度
    double ds;                 ///< 弧长元素（三维中为面积元素）
};

// ---------------------------------------------------------------------------
// 构成一个浸入边界的标记点集合
// ---------------------------------------------------------------------------
struct MarkerSet {
    std::vector<Marker> markers;

    /// 标记点总数
    [[nodiscard]] int size() const {
        return static_cast<int>(markers.size());
    }

    /// 在二维中生成均匀分布的圆形标记点环
    static MarkerSet make_circle(double cx, double cy,
                                 double radius, int n_markers);

    /// 在二维中生成沿 x 轴均匀分布的直线丝状体标记点
    static MarkerSet make_filament(double x0, double y0,
                                   double length, int n_markers);

    /// 从 CSV 文件加载标记点（第三方网格接口）。
    ///
    /// 文件格式（每行一个标记点，以逗号分隔）：
    ///   x, y [, z [, ds]]
    ///   - x, y：标记点坐标（必需）
    ///   - z：z 坐标（可选，缺省 0.0）
    ///   - ds：弧长/面积元素（可选；缺省值为相邻标记点间距的平均值）
    ///
    /// 忽略空行和以 '#' 开头的注释行。
    ///
    /// 若文件无法打开或格式错误，抛出 std::runtime_error。
    ///
    /// @param filename  CSV 文件路径
    static MarkerSet make_from_file(const std::string& filename);
};

} // namespace ibm
