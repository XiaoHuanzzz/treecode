// Mesh-only utility: reuse AFEPack geometry/adaptation, but never evaluate
// the density, a potential, an interaction list, or a numerical error.
#define TREECODE_NO_MAIN
#include "../compare_exact.cpp"
#include "mesh_context.hpp"

#include <filesystem>
#include <sstream>
#include <system_error>

namespace gd_localized {
namespace fs = std::filesystem;

constexpr std::size_t maximum_active_elements = 300000;
constexpr std::array<double, 3> source_center{{-2.0, -2.0, 0.0}};

struct Options {
  std::string mesh;
  std::string output;
  double tolerance = 0.0002;
  int passes = 4;
  double width = 1.2;
  double floor = 0.0001;
};

void usage(std::ostream& out) {
  out << "Usage: generate_localized_mesh --mesh FILE --output PREFIX [options]\n"
      << "\nGenerate an actual adaptive OpenDX mesh and mesh-only JSON metadata.\n"
      << "No potential or error estimate is computed. Existing outputs are refused.\n"
      << "\nOptions:\n"
      << "  --tol VALUE     Positive adaptation tolerance (default 0.0002)\n"
      << "  --passes N      Adaptation passes, 0..20 (default 4)\n"
      << "  --width VALUE   Positive radial width (default 1.2)\n"
      << "  --floor VALUE   Nonnegative background monitor (default 0.0001)\n"
      << "  --help          Print this help and exit\n"
      << "\nMonitor: eta_K = integral_K [floor + (1 + |x-c|^2/width^2)^(-3)] dx,\n"
      << "c = (-2,-2,0), order-6 quadrature, one refinement step per pass.\n"
      << "This prescribed mesh monitor is not a proven error estimator.\n"
      << "The active-element cap is 300000; outputs are PREFIX.dx and PREFIX.json.\n";
}

double parse_real(const std::string& value, const std::string& name) {
  std::size_t consumed = 0;
  double result = 0.0;
  try {
    result = std::stod(value, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("Invalid numeric value for " + name);
  }
  if (consumed != value.size() || !std::isfinite(result)) {
    throw std::invalid_argument("Expected a finite numeric value for " + name);
  }
  return result;
}

int parse_passes(const std::string& value) {
  std::size_t consumed = 0;
  int result = 0;
  try {
    result = std::stoi(value, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("Expected an integer for --passes");
  }
  if (consumed != value.size() || result < 0 || result > 20) {
    throw std::invalid_argument("--passes must be an integer in 0..20");
  }
  return result;
}

bool path_present(const fs::path& path) {
  // symlink_status also detects dangling symlinks, which must not be clobbered.
  return fs::symlink_status(path).type() != fs::file_type::not_found;
}

void refuse_existing_outputs(const Options& options) {
  for (const char* suffix : {".dx", ".json"}) {
    const fs::path path = options.output + suffix;
    if (path_present(path)) {
      throw std::runtime_error("Refusing to overwrite existing output " + path.string());
    }
  }
}

Options parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument != "--mesh" && argument != "--output"
        && argument != "--tol" && argument != "--passes"
        && argument != "--width" && argument != "--floor") {
      throw std::invalid_argument("Unknown argument " + argument);
    }
    if (i + 1 == argc) {
      throw std::invalid_argument("Missing value for " + argument);
    }
    const std::string value = argv[++i];
    if (argument == "--mesh") options.mesh = value;
    else if (argument == "--output") options.output = value;
    else if (argument == "--tol") options.tolerance = parse_real(value, argument);
    else if (argument == "--passes") options.passes = parse_passes(value);
    else if (argument == "--width") options.width = parse_real(value, argument);
    else if (argument == "--floor") options.floor = parse_real(value, argument);
  }
  if (options.mesh.empty() || options.output.empty()) {
    throw std::invalid_argument("Required: --mesh FILE --output PREFIX (see --help)");
  }
  if (!(options.tolerance > 0.0) || !(options.width > 0.0)
      || options.floor < 0.0) {
    throw std::invalid_argument("--tol and --width must be positive; --floor must be nonnegative");
  }
  if (!fs::is_regular_file(options.mesh)) {
    throw std::invalid_argument("Mesh is not a readable regular file: " + options.mesh);
  }
  std::ifstream input(options.mesh);
  if (!input) throw std::runtime_error("Cannot read mesh " + options.mesh);
  if (fs::path(options.output).filename().empty()) {
    throw std::invalid_argument("--output must be a file prefix, not a directory path");
  }
  refuse_existing_outputs(options);
  return options;
}

std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned int>(c) << std::dec;
        } else {
          out << c;
        }
    }
  }
  out << '"';
  return out.str();
}

struct MeshStatistics {
  std::size_t tetrahedra = 0;
  double volume = 0.0;
  std::array<double, 3> minimum{{INFINITY, INFINITY, INFINITY}};
  std::array<double, 3> maximum{{-INFINITY, -INFINITY, -INFINITY}};
};

MeshStatistics mesh_statistics(gd::MeshContext& mesh) {
  MeshStatistics stats;
  const auto tetrahedra = mesh.tetrahedra();
  stats.tetrahedra = tetrahedra.size();
  long double volume = 0.0L;
  for (const auto& tet : tetrahedra) {
    for (const auto& point : tet) {
      for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(point[d])) {
          throw std::runtime_error("Mesh has a nonfinite coordinate");
        }
        stats.minimum[d] = std::min(stats.minimum[d], point[d]);
        stats.maximum[d] = std::max(stats.maximum[d], point[d]);
      }
    }
    const double cell_volume = tetrahedronVolume(tet[0], tet[1], tet[2], tet[3]);
    if (!(cell_volume > 0.0) || !std::isfinite(cell_volume)) {
      throw std::runtime_error("Mesh has an invalid tetrahedron volume");
    }
    volume += cell_volume;
  }
  stats.volume = static_cast<double>(volume);
  if (!(stats.volume > 0.0) || !std::isfinite(stats.volume)) {
    throw std::runtime_error("Mesh has an invalid total domain volume");
  }
  return stats;
}

// Staging beside the requested output keeps hard-link publication on the
// same filesystem. create_hard_link refuses even a concurrently created
// destination, unlike a truncating stream or a replacing rename.
class StagedOutputs {
public:
  fs::path directory;

  explicit StagedOutputs(const fs::path& parent) {
    std::string pattern = (parent / ".localized-mesh-XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char* created = ::mkdtemp(writable.data());
    if (!created) throw std::runtime_error("Cannot create mesh output staging directory");
    directory = created;
  }

  ~StagedOutputs() {
    std::error_code ignored;
    fs::remove(directory / "mesh.dx", ignored);
    fs::remove(directory / "mesh.json", ignored);
    fs::remove(directory, ignored);
  }

  StagedOutputs(const StagedOutputs&) = delete;
  StagedOutputs& operator=(const StagedOutputs&) = delete;
};

void write_metadata(const fs::path& path, const Options& options,
                    const MeshStatistics& initial, const MeshStatistics& final,
                    std::size_t initial_count, std::size_t final_count,
                    std::size_t root_count, double mesh_seconds) {
  std::ofstream out(path);
  out << std::setprecision(17)
      << "{\n  \"mode\": \"mesh_only\",\n"
      << "  \"mesh_kind\": \"adaptive_localized\",\n"
      << "  \"input_mesh\": " << json_string(options.mesh) << ",\n"
      << "  \"mesh_file\": " << json_string(options.output + ".dx") << ",\n"
      << "  \"source_center\": [-2, -2, 0],\n"
      << "  \"monitor\": \"integral_K [floor + (1 + |x-c|^2/width^2)^(-3)] dx\",\n"
      << "  \"monitor_kind\": \"prescribed smooth localized algebraic profile\",\n"
      << "  \"monitor_is_error_estimator\": false,\n"
      << "  \"potential_computed\": false,\n"
      << "  \"error_computed\": false,\n"
      << "  \"parameters\": {\"tolerance\": " << options.tolerance
      << ", \"passes\": " << options.passes
      << ", \"width\": " << options.width
      << ", \"floor\": " << options.floor
      << ", \"refine_steps\": 1, \"global_refinements\": 0, \"quadrature_order\": 6},\n"
      << "  \"max_active_elements\": " << maximum_active_elements << ",\n"
      << "  \"N\": " << final_count << ",\n"
      << "  \"initial_N\": " << initial_count << ",\n"
      << "  \"root_count\": " << root_count << ",\n"
      << "  \"tetrahedra\": " << final.tetrahedra << ",\n"
      << "  \"domain_volume\": " << final.volume << ",\n"
      << "  \"initial_domain_volume\": " << initial.volume << ",\n"
      << "  \"volume_change\": " << final.volume - initial.volume << ",\n"
      << "  \"bounding_box\": {\"min\": [" << final.minimum[0] << ", "
      << final.minimum[1] << ", " << final.minimum[2] << "], \"max\": ["
      << final.maximum[0] << ", " << final.maximum[1] << ", "
      << final.maximum[2] << "]},\n"
      << "  \"mesh_s\": " << mesh_seconds << "\n}\n";
  out.close();
  if (!out) throw std::runtime_error("Cannot write " + path.string());
}

int run(const Options& options) {
  const double start = omp_get_wtime();
  gd::MeshContext mesh(options.mesh, 0, maximum_active_elements);
  const std::size_t initial_count = mesh.fem.n_element();
  const MeshStatistics initial = mesh_statistics(mesh);
  // Bound the largest possible integrated indicator before invoking AFEPack.
  if (options.floor > std::numeric_limits<double>::max()
                          / std::max(1.0, initial.volume) - 1.0) {
    throw std::invalid_argument("--floor is too large for finite mesh indicators");
  }
  const auto monitor = [&options](double x, double y, double z) {
    // Divide by width before squaring so tiny/large finite widths cannot
    // cause 0/0 through a rounded width^2. Overflow to +infinity gives the
    // correct limiting profile value zero far from the center.
    const double dx = (x - source_center[0]) / options.width;
    const double dy = (y - source_center[1]) / options.width;
    const double dz = (z - source_center[2]) / options.width;
    const double inverse = 1.0 / (1.0 + dx * dx + dy * dy + dz * dz);
    return options.floor + inverse * inverse * inverse;
  };
  mesh.adapt_with_monitor(options.tolerance, options.passes, 1, monitor);
  const MeshStatistics final = mesh_statistics(mesh);
  if (std::abs(final.volume - initial.volume)
      > 1.0e-10 * std::max(1.0, initial.volume)) {
    throw std::runtime_error("Adaptation changed the represented domain volume");
  }
  const std::size_t root_count = mesh.roots().size();
  const double mesh_seconds = omp_get_wtime() - start;
  if (!std::isfinite(mesh_seconds)) throw std::runtime_error("Nonfinite mesh timing");

  refuse_existing_outputs(options);
  fs::path parent = fs::path(options.output).parent_path();
  if (parent.empty()) parent = ".";
  fs::create_directories(parent);
  StagedOutputs staged(parent);
  mesh.export_dx((staged.directory / "mesh.dx").string());
  if (!fs::is_regular_file(staged.directory / "mesh.dx")
      || fs::file_size(staged.directory / "mesh.dx") == 0) {
    throw std::runtime_error("AFEPack did not produce a nonempty OpenDX mesh");
  }
  write_metadata(staged.directory / "mesh.json", options, initial, final,
                 initial_count, mesh.fem.n_element(), root_count, mesh_seconds);
  fs::create_hard_link(staged.directory / "mesh.dx", options.output + ".dx");
  try {
    fs::create_hard_link(staged.directory / "mesh.json", options.output + ".json");
  } catch (...) {
    // Remove only the link just published by this invocation.
    std::error_code ignored;
    if (fs::equivalent(staged.directory / "mesh.dx", options.output + ".dx", ignored)) {
      fs::remove(options.output + ".dx", ignored);
    }
    throw;
  }
  std::cout << "Mesh-only generation complete: N=" << mesh.fem.n_element()
            << ", tetrahedra=" << final.tetrahedra
            << ", volume=" << final.volume << "\n"
            << "Wrote " << options.output << ".dx and " << options.output << ".json\n";
  return 0;
}
} // namespace gd_localized

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--help") {
        gd_localized::usage(std::cout);
        return 0;
      }
    }
    std::cout << std::setprecision(15);
    return gd_localized::run(gd_localized::parse(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "LOCALIZED-MESH ERROR: " << error.what() << '\n';
    return 1;
  }
}
