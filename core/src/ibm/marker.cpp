#include "ibm/marker.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace ibm {

static constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// 生成均匀分布在圆周上的标记点
// ---------------------------------------------------------------------------
MarkerSet MarkerSet::make_circle(double cx, double cy,
                                 double radius, int n_markers)
{
    MarkerSet ms;
    ms.markers.resize(n_markers);
    const double dtheta = 2.0 * PI / n_markers;  // 相邻标记点的角度间距
    const double ds = radius * dtheta;            // 弧长元素

    for (int k = 0; k < n_markers; ++k) {
        const double theta = k * dtheta;
        auto& m = ms.markers[k];
        m.x  = m.x0 = cx + radius * std::cos(theta);
        m.y  = m.y0 = cy + radius * std::sin(theta);
        m.z  = m.z0 = 0.0;
        m.fx = m.fy = m.fz = 0.0;   // 力初始化为零
        m.ux = m.uy = m.uz = 0.0;   // 速度初始化为零
        m.ds = ds;
    }
    return ms;
}

// ---------------------------------------------------------------------------
// 生成沿 x 轴均匀分布的直线丝状体标记点
// ---------------------------------------------------------------------------
MarkerSet MarkerSet::make_filament(double x0, double y0,
                                   double length, int n_markers)
{
    MarkerSet ms;
    ms.markers.resize(n_markers);
    const double dl = length / (n_markers - 1);   // 相邻标记点的间距

    for (int k = 0; k < n_markers; ++k) {
        auto& m = ms.markers[k];
        m.x  = m.x0 = x0 + k * dl;
        m.y  = m.y0 = y0;
        m.z  = m.z0 = 0.0;
        m.fx = m.fy = m.fz = 0.0;
        m.ux = m.uy = m.uz = 0.0;
        m.ds = dl;
    }
    return ms;
}

// ---------------------------------------------------------------------------
// 从 CSV 文件加载标记点（第三方网格接口）
//
// 文件格式（每行一个标记点，以逗号分隔）：
//   x, y [, z [, ds]]
//   - x, y：标记点坐标（必需）
//   - z：z 坐标（可选，缺省 0.0）
//   - ds：弧长/面积元素（可选）
//
// 忽略空行和以 '#' 开头的注释行。
// 若 ds 列缺失，则在加载完所有坐标后用相邻标记点间距的平均值填充。
// ---------------------------------------------------------------------------
MarkerSet MarkerSet::make_from_file(const std::string& filename)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error("MarkerSet::make_from_file: cannot open file: " + filename);
    }

    MarkerSet ms;
    bool has_ds = true;  // 假设有 ds 列，若第一行没有则设 false

    std::string line;
    int line_no = 0;
    bool first_data_line = true;

    while (std::getline(ifs, line)) {
        ++line_no;
        // 去掉前后空白
        const auto trim_pos = line.find_first_not_of(" \t\r\n");
        if (trim_pos == std::string::npos) continue;  // 空行
        if (line[trim_pos] == '#') continue;           // 注释行

        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);

        double x = 0.0, y = 0.0, z = 0.0, ds = 0.0;
        if (!(iss >> x >> y)) {
            throw std::runtime_error(
                "MarkerSet::make_from_file: parse error at line " +
                std::to_string(line_no) + " in " + filename);
        }
        // z 列（可选）
        if (!(iss >> z)) z = 0.0;
        // ds 列（可选）
        bool parsed_ds = static_cast<bool>(iss >> ds);

        if (first_data_line) {
            has_ds = parsed_ds;
            first_data_line = false;
        }

        Marker m{};
        m.x = m.x0 = x;
        m.y = m.y0 = y;
        m.z = m.z0 = z;
        m.fx = m.fy = m.fz = 0.0;
        m.ux = m.uy = m.uz = 0.0;
        m.ds = has_ds ? ds : 0.0;  // 暂存；无 ds 时稍后填充
        ms.markers.push_back(m);
    }

    if (ms.markers.empty()) {
        throw std::runtime_error(
            "MarkerSet::make_from_file: no markers loaded from " + filename);
    }

    // 若文件中无 ds 列，用相邻标记点间距的平均值填充
    if (!has_ds) {
        const int nm = static_cast<int>(ms.markers.size());
        if (nm == 1) {
            ms.markers[0].ds = 1.0;
        } else {
            // 计算所有相邻点对的间距均值
            double total_len = 0.0;
            for (int k = 0; k < nm - 1; ++k) {
                const double ddx = ms.markers[k+1].x - ms.markers[k].x;
                const double ddy = ms.markers[k+1].y - ms.markers[k].y;
                const double ddz = ms.markers[k+1].z - ms.markers[k].z;
                total_len += std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
            }
            const double avg_ds = total_len / (nm - 1);
            for (auto& mk : ms.markers) mk.ds = avg_ds;
        }
    }

    return ms;
}

} // namespace ibm
