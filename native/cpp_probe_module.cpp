#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

template <typename T, int ExtraFlags>
py::buffer_info checked(py::array_t<T, ExtraFlags> array, int ndim, const char* name) {
    py::buffer_info info = array.request();
    if (info.ndim != ndim) {
        throw std::runtime_error(std::string(name) + " has the wrong rank");
    }
    return info;
}

inline float clamp01(float value) {
    return std::min(1.0f, std::max(0.0f, value));
}

inline float sq(float value) {
    return value * value;
}

struct Wave {
    int kind_code = 0;
    std::string kind;
    float color[3] = {0.0f, 0.0f, 0.0f};
    int color_u8[3] = {0, 0, 0};
    float intensity = 0.0f;
    float origin[3] = {0.0f, 0.0f, 0.0f};
    float speed = 0.0f;
    float decay_rate = 0.95f;
    float spread = 0.1f;
    float age = 0.0f;
    float radius_limit = 1.0f;
    float color_velocity = 0.0f;
    int direction_hint = 0;
    float width = 1.0f;
    int pulse_count = 1;
    float phase = 0.0f;
    bool secondary = false;
};

int kind_code_for(const std::string& kind) {
    if (kind == "flash" || kind == "explosion") return 1;
    if (kind == "impact_pulse") return 2;
    if (kind == "shockwave") return 3;
    if (kind == "directional_sweep" || kind == "scene_wipe") return 4;
    if (kind == "energy_trail") return 5;
    if (kind == "flame_shimmer") return 6;
    if (kind == "underwater") return 7;
    if (kind == "portal_vortex") return 8;
    if (kind == "color_bloom") return 9;
    if (kind == "lightning") return 10;
    if (kind == "ember_particles") return 11;
    if (kind == "negative_wave") return 12;
    return 0;
}

int priority_for_kind(const std::string& kind) {
    if (kind == "lightning") return 100;
    if (kind == "explosion") return 95;
    if (kind == "shockwave") return 90;
    if (kind == "impact_pulse") return 85;
    if (kind == "negative_wave") return 80;
    if (kind == "scene_wipe") return 70;
    if (kind == "directional_sweep") return 65;
    if (kind == "portal_vortex") return 60;
    if (kind == "camera_pan") return 55;
    if (kind == "energy_trail") return 45;
    if (kind == "top_color_exit") return 40;
    if (kind == "spill") return 35;
    if (kind == "flame_shimmer") return 25;
    if (kind == "underwater") return 20;
    if (kind == "color_bloom") return 15;
    return 0;
}

void render_waves_into(
    float* leds,
    int total_leds,
    const int32_t* spatial_idx,
    int spatial_count,
    const float* positions,
    const float* angles,
    const int32_t* kind_codes,
    const float* colors,
    const float* intensities,
    const float* origins,
    const float* speeds,
    const float* ages,
    const float* radius_limits,
    const float* spreads,
    const float* widths,
    const int32_t* direction_hints,
    const float* phases,
    int wave_count,
    float room_diagonal,
    float room_width,
    float room_depth
) {
    std::fill(leds, leds + total_leds * 3, 0.0f);

    for (int wave_index = 0; wave_index < wave_count; ++wave_index) {
        const int kind = kind_codes[wave_index];
        const float intensity = intensities[wave_index];
        const float origin_x = origins[wave_index * 3 + 0];
        const float origin_y = origins[wave_index * 3 + 1];
        const float origin_z = origins[wave_index * 3 + 2];
        const float travelled = speeds[wave_index] * ages[wave_index];
        const float radius_limit = radius_limits[wave_index];
        const float spread = spreads[wave_index];
        const float width = widths[wave_index];
        const int direction_hint = direction_hints[wave_index];
        const float phase = phases[wave_index];
        const float travel_fade = std::max(0.0f, 1.0f - travelled / std::max(0.1f, radius_limit));
        const float contribution_scale = intensity * (0.2f + 0.8f * travel_fade);
        const float red = colors[wave_index * 3 + 0] * contribution_scale;
        const float green = colors[wave_index * 3 + 1] * contribution_scale;
        const float blue = colors[wave_index * 3 + 2] * contribution_scale;

        for (int local_index = 0; local_index < spatial_count; ++local_index) {
            const float px = positions[local_index * 3 + 0];
            const float py = positions[local_index * 3 + 1];
            const float pz = positions[local_index * 3 + 2];
            const float dx = px - origin_x;
            const float dy = py - origin_y;
            const float dz = pz - origin_z;
            const float distances_sq = dx * dx + dy * dy + dz * dz;
            float envelope = 0.0f;

            if (kind == 1) {
                const float fill_radius = std::max(spread, travelled + spread);
                envelope = std::exp(-distances_sq / (2.0f * fill_radius * fill_radius));
            } else if (kind == 2) {
                const float fill_radius = std::max(room_diagonal, travelled + spread);
                const float local = std::exp(-distances_sq / (2.0f * fill_radius * fill_radius));
                const float collapse = std::max(0.0f, 1.0f - ages[wave_index] / std::max(0.2f, width));
                envelope = std::max(local, 0.45f * collapse);
            } else if (kind == 3) {
                const float distance = std::sqrt(distances_sq);
                const float ring_width = std::max(spread * 0.45f, width * 0.08f);
                float ring = std::exp(-sq(distance - travelled) / (2.0f * ring_width * ring_width));
                float room_flash = std::max(0.0f, 1.0f - travelled / std::max(0.1f, radius_limit)) * 0.32f;
                if (direction_hint != 0) {
                    const float sign = direction_hint > 0 ? 1.0f : -1.0f;
                    const float projection = dx * sign;
                    const float forward = clamp01((projection / std::max(distance, 0.001f) + 1.0f) * 0.5f);
                    const float gate = clamp01(forward * 1.35f);
                    ring *= gate;
                    room_flash *= 0.35f + 0.65f * gate;
                }
                envelope = std::max(ring, room_flash);
            } else if (kind == 4) {
                const float axis = px;
                const float span = std::max(std::max(room_width, room_depth), 0.1f);
                float front = (travelled / std::max(0.1f, radius_limit)) * span;
                if (direction_hint < 0) {
                    front = span - front;
                }
                const float sweep_width = std::max(0.12f, width * 0.25f);
                envelope = std::exp(-sq(axis - front) / (2.0f * sweep_width * sweep_width));
            } else if (kind == 5) {
                const float distance = std::sqrt(distances_sq);
                const float head = std::exp(-sq(distance - travelled) / (2.0f * spread * spread));
                const float tail_distance = distance - std::max(0.0f, travelled - spread * 3.0f);
                const float tail_spread = spread * 2.2f;
                const float tail = std::exp(-(tail_distance * tail_distance) / (2.0f * tail_spread * tail_spread)) * 0.45f;
                envelope = std::max(head, tail);
            } else if (kind == 6) {
                const float noise = 0.65f + 0.35f * std::sin(px * 7.0f + pz * 5.0f + ages[wave_index] * 18.0f + phase * 6.28f);
                const float local_spread = std::max(spread * 2.4f, 0.1f);
                envelope = std::exp(-distances_sq / (2.0f * local_spread * local_spread)) * noise;
            } else if (kind == 7) {
                const float distance = std::sqrt(distances_sq);
                const float ripple_base = std::sin(distance * 6.0f - ages[wave_index] * 4.0f + phase * 6.28f);
                const float ripple = 0.45f + 0.55f * ripple_base * ripple_base;
                const float local_spread = std::max(spread * 5.0f, 0.1f);
                envelope = std::exp(-distances_sq / (2.0f * local_spread * local_spread)) * ripple;
            } else if (kind == 8) {
                const float distance = std::sqrt(distances_sq);
                const int turns = std::max(1, direction_hint);
                const float spiral = 0.5f + 0.5f * std::sin(angles[local_index] * 3.0f * turns + distance * 4.0f - ages[wave_index] * 6.0f);
                const float vortex_spread = spread * 2.5f;
                const float drift = distance - travelled * 0.45f;
                envelope = std::exp(-(drift * drift) / (2.0f * vortex_spread * vortex_spread)) * spiral;
            } else if (kind == 9 || kind == 10 || kind == 11) {
                const float bloom_spread = kind == 9 ? spread * 4.0f : spread;
                const float local_spread = std::max(bloom_spread, 0.1f);
                envelope = std::exp(-distances_sq / (2.0f * local_spread * local_spread));
                if (kind == 10 && intensity >= 0.72f) {
                    const float strobe = (static_cast<int>(ages[wave_index] * 28.0f + phase * 7.0f) % 2) == 0 ? 1.0f : 0.0f;
                    envelope = std::max(envelope, strobe * intensity * 0.55f);
                }
            } else {
                const float distance = std::sqrt(distances_sq);
                envelope = std::exp(-sq(distance - travelled) / (2.0f * spread * spread));
            }

            const int led_index = spatial_idx[local_index];
            if (led_index < 0 || led_index >= total_leds) {
                continue;
            }
            float* led = leds + led_index * 3;
            if (kind == 12) {
                const float dim = envelope * intensity * 0.65f;
                led[0] -= dim;
                led[1] -= dim;
                led[2] -= dim;
            } else {
                led[0] += envelope * red;
                led[1] += envelope * green;
                led[2] += envelope * blue;
            }
        }
    }
}

void rgb_to_hsv(float r, float g, float b, float& h, float& s, float& v) {
    const float max_v = std::max(r, std::max(g, b));
    const float min_v = std::min(r, std::min(g, b));
    const float delta = max_v - min_v;
    v = max_v;
    s = max_v <= 0.0f ? 0.0f : delta / max_v;
    if (delta <= 0.0f) {
        h = 0.0f;
    } else if (max_v == r) {
        h = std::fmod((g - b) / delta, 6.0f);
    } else if (max_v == g) {
        h = ((b - r) / delta) + 2.0f;
    } else {
        h = ((r - g) / delta) + 4.0f;
    }
    h /= 6.0f;
    if (h < 0.0f) {
        h += 1.0f;
    }
}

void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
    const float c = v * s;
    const float hh = h * 6.0f;
    const float x = c * (1.0f - std::fabs(std::fmod(hh, 2.0f) - 1.0f));
    const float m = v - c;
    float rp = 0.0f;
    float gp = 0.0f;
    float bp = 0.0f;
    if (hh < 1.0f) {
        rp = c; gp = x;
    } else if (hh < 2.0f) {
        rp = x; gp = c;
    } else if (hh < 3.0f) {
        gp = c; bp = x;
    } else if (hh < 4.0f) {
        gp = x; bp = c;
    } else if (hh < 5.0f) {
        rp = x; bp = c;
    } else {
        rp = c; bp = x;
    }
    r = rp + m;
    g = gp + m;
    b = bp + m;
}

py::array_t<uint8_t> post_process_impl(
    py::array_t<float, py::array::c_style | py::array::forcecast> leds,
    py::array_t<float, py::array::c_style | py::array::forcecast> previous_leds,
    float color_smoothing,
    float saturation,
    py::array_t<float, py::array::c_style | py::array::forcecast> white_balance,
    float brightness,
    float gamma
) {
    py::buffer_info leds_info = checked(leds, 2, "leds");
    py::buffer_info previous_info = checked(previous_leds, 2, "previous_leds");
    py::buffer_info wb_info = checked(white_balance, 1, "white_balance");
    if (leds_info.shape[1] != 3 || previous_info.shape[1] != 3 || leds_info.shape[0] != previous_info.shape[0] || wb_info.shape[0] != 3) {
        throw std::runtime_error("post-process arrays have incompatible shapes");
    }

    const int led_count = static_cast<int>(leds_info.shape[0]);
    auto out = py::array_t<uint8_t>(std::vector<py::ssize_t>{led_count, 3});
    py::buffer_info out_info = out.request();
    const float* src = static_cast<const float*>(leds_info.ptr);
    const float* previous = static_cast<const float*>(previous_info.ptr);
    const float* wb = static_cast<const float*>(wb_info.ptr);
    uint8_t* dst = static_cast<uint8_t*>(out_info.ptr);
    const float alpha = clamp01(color_smoothing);
    const float inv_alpha = 1.0f - alpha;
    const float gamma_value = std::max(0.1f, gamma);
    const float inv_gamma = 1.0f / gamma_value;
    const bool apply_saturation = std::fabs(saturation - 1.0f) > 0.001f;
    const bool apply_gamma = std::fabs(gamma_value - 1.0f) > 0.001f;

    for (int i = 0; i < led_count; ++i) {
        float r = clamp01(src[i * 3 + 0]) * inv_alpha + previous[i * 3 + 0] * alpha;
        float g = clamp01(src[i * 3 + 1]) * inv_alpha + previous[i * 3 + 1] * alpha;
        float b = clamp01(src[i * 3 + 2]) * inv_alpha + previous[i * 3 + 2] * alpha;

        if (apply_saturation) {
            float h, s, v;
            rgb_to_hsv(r, g, b, h, s, v);
            hsv_to_rgb(h, clamp01(s * saturation), v, r, g, b);
        }

        r = clamp01(r * wb[0] * brightness);
        g = clamp01(g * wb[1] * brightness);
        b = clamp01(b * wb[2] * brightness);
        if (apply_gamma) {
            r = std::pow(r, inv_gamma);
            g = std::pow(g, inv_gamma);
            b = std::pow(b, inv_gamma);
        }
        dst[i * 3 + 0] = static_cast<uint8_t>(clamp01(r) * 255.0f);
        dst[i * 3 + 1] = static_cast<uint8_t>(clamp01(g) * 255.0f);
        dst[i * 3 + 2] = static_cast<uint8_t>(clamp01(b) * 255.0f);
    }
    return out;
}

py::array_t<float> render_waves_probe(
    int total_leds,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> spatial_idx,
    py::array_t<float, py::array::c_style | py::array::forcecast> positions,
    py::array_t<float, py::array::c_style | py::array::forcecast> angles,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> kind_codes,
    py::array_t<float, py::array::c_style | py::array::forcecast> colors,
    py::array_t<float, py::array::c_style | py::array::forcecast> intensities,
    py::array_t<float, py::array::c_style | py::array::forcecast> origins,
    py::array_t<float, py::array::c_style | py::array::forcecast> speeds,
    py::array_t<float, py::array::c_style | py::array::forcecast> ages,
    py::array_t<float, py::array::c_style | py::array::forcecast> radius_limits,
    py::array_t<float, py::array::c_style | py::array::forcecast> spreads,
    py::array_t<float, py::array::c_style | py::array::forcecast> widths,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> direction_hints,
    py::array_t<float, py::array::c_style | py::array::forcecast> phases,
    float room_diagonal,
    float room_width,
    float room_depth
) {
    py::buffer_info spatial_info = checked(spatial_idx, 1, "spatial_idx");
    py::buffer_info positions_info = checked(positions, 2, "positions");
    py::buffer_info angles_info = checked(angles, 1, "angles");
    py::buffer_info kind_info = checked(kind_codes, 1, "kind_codes");
    py::buffer_info colors_info = checked(colors, 2, "colors");
    py::buffer_info intensities_info = checked(intensities, 1, "intensities");
    py::buffer_info origins_info = checked(origins, 2, "origins");
    py::buffer_info speeds_info = checked(speeds, 1, "speeds");
    py::buffer_info ages_info = checked(ages, 1, "ages");
    py::buffer_info radius_info = checked(radius_limits, 1, "radius_limits");
    py::buffer_info spreads_info = checked(spreads, 1, "spreads");
    py::buffer_info widths_info = checked(widths, 1, "widths");
    py::buffer_info hints_info = checked(direction_hints, 1, "direction_hints");
    py::buffer_info phases_info = checked(phases, 1, "phases");

    const int spatial_count = static_cast<int>(spatial_info.shape[0]);
    const int wave_count = static_cast<int>(kind_info.shape[0]);
    if (positions_info.shape[0] != spatial_count || positions_info.shape[1] != 3 || angles_info.shape[0] != spatial_count) {
        throw std::runtime_error("spatial arrays have incompatible shapes");
    }
    if (colors_info.shape[0] != wave_count || colors_info.shape[1] != 3 || origins_info.shape[0] != wave_count || origins_info.shape[1] != 3 ||
        intensities_info.shape[0] != wave_count || speeds_info.shape[0] != wave_count || ages_info.shape[0] != wave_count ||
        radius_info.shape[0] != wave_count || spreads_info.shape[0] != wave_count || widths_info.shape[0] != wave_count ||
        hints_info.shape[0] != wave_count || phases_info.shape[0] != wave_count) {
        throw std::runtime_error("wave arrays have incompatible shapes");
    }

    auto out = py::array_t<float>(std::vector<py::ssize_t>{total_leds, 3});
    py::buffer_info out_info = out.request();
    render_waves_into(
        static_cast<float*>(out_info.ptr),
        total_leds,
        static_cast<const int32_t*>(spatial_info.ptr),
        spatial_count,
        static_cast<const float*>(positions_info.ptr),
        static_cast<const float*>(angles_info.ptr),
        static_cast<const int32_t*>(kind_info.ptr),
        static_cast<const float*>(colors_info.ptr),
        static_cast<const float*>(intensities_info.ptr),
        static_cast<const float*>(origins_info.ptr),
        static_cast<const float*>(speeds_info.ptr),
        static_cast<const float*>(ages_info.ptr),
        static_cast<const float*>(radius_info.ptr),
        static_cast<const float*>(spreads_info.ptr),
        static_cast<const float*>(widths_info.ptr),
        static_cast<const int32_t*>(hints_info.ptr),
        static_cast<const float*>(phases_info.ptr),
        wave_count,
        room_diagonal,
        room_width,
        room_depth
    );
    return out;
}

py::array_t<uint8_t> full_probe(
    int total_leds,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> spatial_idx,
    py::array_t<float, py::array::c_style | py::array::forcecast> positions,
    py::array_t<float, py::array::c_style | py::array::forcecast> angles,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> kind_codes,
    py::array_t<float, py::array::c_style | py::array::forcecast> colors,
    py::array_t<float, py::array::c_style | py::array::forcecast> intensities,
    py::array_t<float, py::array::c_style | py::array::forcecast> origins,
    py::array_t<float, py::array::c_style | py::array::forcecast> speeds,
    py::array_t<float, py::array::c_style | py::array::forcecast> ages,
    py::array_t<float, py::array::c_style | py::array::forcecast> radius_limits,
    py::array_t<float, py::array::c_style | py::array::forcecast> spreads,
    py::array_t<float, py::array::c_style | py::array::forcecast> widths,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> direction_hints,
    py::array_t<float, py::array::c_style | py::array::forcecast> phases,
    float room_diagonal,
    float room_width,
    float room_depth,
    py::array_t<float, py::array::c_style | py::array::forcecast> previous_leds,
    float color_smoothing,
    float saturation,
    py::array_t<float, py::array::c_style | py::array::forcecast> white_balance,
    float brightness,
    float gamma
) {
    py::array_t<float> rendered = render_waves_probe(
        total_leds, spatial_idx, positions, angles, kind_codes, colors, intensities, origins, speeds, ages,
        radius_limits, spreads, widths, direction_hints, phases, room_diagonal, room_width, room_depth
    );
    return post_process_impl(rendered, previous_leds, color_smoothing, saturation, white_balance, brightness, gamma);
}

py::tuple full_renderer_step(py::dict d) {
    int total_leds = d["total_leds"].cast<int>();
    auto spatial_idx = d["spatial_idx"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto positions = d["positions"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto angles = d["angles"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto kind_codes = d["kind_codes"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto colors = d["colors"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto intensities = d["intensities"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto origins = d["origins"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto speeds = d["speeds"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto ages = d["ages"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto radius_limits = d["radius_limits"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto spreads = d["spreads"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto widths = d["widths"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto direction_hints = d["direction_hints"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto phases = d["phases"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    float room_diagonal = d["room_diagonal"].cast<float>();
    float room_width = d["room_width"].cast<float>();
    float room_depth = d["room_depth"].cast<float>();
    float wave_min_intensity = d["wave_min_intensity"].cast<float>();
    auto sync_codes = d["sync_codes"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto blend = d["blend"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto tv_sample_kind = d["tv_sample_kind"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto tv_sample_a = d["tv_sample_a"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto tv_sample_b = d["tv_sample_b"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto tv_frame_bgr = d["tv_frame_bgr"].cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
    auto front_indices = d["front_indices"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto front_strip_colors = d["front_strip_colors"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto front_color = d["front_color"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    float front_intensity = d["front_intensity"].cast<float>();
    auto ambient_ext_indices = d["ambient_ext_indices"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto ambient_ext_parent = d["ambient_ext_parent"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    auto ambient_ext_fade = d["ambient_ext_fade"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto ambient_ext_strength = d["ambient_ext_strength"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    float ambient_base_intensity = d["ambient_base_intensity"].cast<float>();
    float ambient_boost = d["ambient_boost"].cast<float>();
    auto ambient_scene_color = d["ambient_scene_color"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    bool ambient_side_spill_enabled = d["ambient_side_spill_enabled"].cast<bool>();
    auto previous_leds = d["previous_leds"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    float color_smoothing = d["color_smoothing"].cast<float>();
    float saturation = d["saturation"].cast<float>();
    auto white_balance = d["white_balance"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    float brightness = d["brightness"].cast<float>();
    float gamma = d["gamma"].cast<float>();
    py::buffer_info wave_info = checked(kind_codes, 1, "kind_codes");
    int wave_count = static_cast<int>(wave_info.shape[0]);

    py::buffer_info intensities_info = checked(intensities, 1, "intensities");
    py::buffer_info ages_info = checked(ages, 1, "ages");
    py::buffer_info speeds_info = checked(speeds, 1, "speeds");
    py::buffer_info radius_info = checked(radius_limits, 1, "radius_limits");
    py::buffer_info spreads_info = checked(spreads, 1, "spreads");
    py::buffer_info sync_info = checked(sync_codes, 1, "sync_codes");
    py::buffer_info blend_info = checked(blend, 1, "blend");
    py::buffer_info tv_kind_info = checked(tv_sample_kind, 1, "tv_sample_kind");
    py::buffer_info tv_a_info = checked(tv_sample_a, 1, "tv_sample_a");
    py::buffer_info tv_b_info = checked(tv_sample_b, 1, "tv_sample_b");
    py::buffer_info frame_info = tv_frame_bgr.request();
    py::buffer_info front_idx_info = checked(front_indices, 1, "front_indices");
    py::buffer_info front_strip_info = front_strip_colors.request();
    py::buffer_info front_color_info = checked(front_color, 1, "front_color");
    py::buffer_info ext_idx_info = checked(ambient_ext_indices, 1, "ambient_ext_indices");
    py::buffer_info ext_parent_info = checked(ambient_ext_parent, 1, "ambient_ext_parent");
    py::buffer_info ext_fade_info = checked(ambient_ext_fade, 1, "ambient_ext_fade");
    py::buffer_info ext_strength_info = checked(ambient_ext_strength, 1, "ambient_ext_strength");
    py::buffer_info scene_color_info = checked(ambient_scene_color, 1, "ambient_scene_color");

    if (sync_info.shape[0] != total_leds || blend_info.shape[0] != total_leds || tv_kind_info.shape[0] != total_leds ||
        tv_a_info.shape[0] != total_leds || tv_b_info.shape[0] != total_leds) {
        throw std::runtime_error("static LED plans have incompatible shapes");
    }

    auto working = py::array_t<float>(std::vector<py::ssize_t>{total_leds, 3});
    py::buffer_info working_info = working.request();
    float* leds = static_cast<float*>(working_info.ptr);
    render_waves_into(
        leds,
        total_leds,
        static_cast<const int32_t*>(checked(spatial_idx, 1, "spatial_idx").ptr),
        static_cast<int>(checked(spatial_idx, 1, "spatial_idx").shape[0]),
        static_cast<const float*>(checked(positions, 2, "positions").ptr),
        static_cast<const float*>(checked(angles, 1, "angles").ptr),
        static_cast<const int32_t*>(wave_info.ptr),
        static_cast<const float*>(checked(colors, 2, "colors").ptr),
        static_cast<const float*>(intensities_info.ptr),
        static_cast<const float*>(checked(origins, 2, "origins").ptr),
        static_cast<const float*>(speeds_info.ptr),
        static_cast<const float*>(ages_info.ptr),
        static_cast<const float*>(radius_info.ptr),
        static_cast<const float*>(spreads_info.ptr),
        static_cast<const float*>(checked(widths, 1, "widths").ptr),
        static_cast<const int32_t*>(checked(direction_hints, 1, "direction_hints").ptr),
        static_cast<const float*>(checked(phases, 1, "phases").ptr),
        wave_count,
        room_diagonal,
        room_width,
        room_depth
    );

    const int32_t* sync = static_cast<const int32_t*>(sync_info.ptr);
    const float* blend_ptr = static_cast<const float*>(blend_info.ptr);
    const int32_t* tv_kind = static_cast<const int32_t*>(tv_kind_info.ptr);
    const float* tv_a = static_cast<const float*>(tv_a_info.ptr);
    const float* tv_b = static_cast<const float*>(tv_b_info.ptr);
    if (frame_info.ndim == 3 && frame_info.shape[0] > 0 && frame_info.shape[1] > 0 && frame_info.shape[2] == 3) {
        const int height = static_cast<int>(frame_info.shape[0]);
        const int width = static_cast<int>(frame_info.shape[1]);
        const uint8_t* frame = static_cast<const uint8_t*>(frame_info.ptr);
        const int edge_h = std::max(1, static_cast<int>(std::round(height * 0.04f)));
        const int edge_w = std::max(1, static_cast<int>(std::round(width * 0.04f)));
        for (int i = 0; i < total_leds; ++i) {
            if (sync[i] != 1 && sync[i] != 2) {
                continue;
            }
            float r = 0.0f, g = 0.0f, b = 0.0f;
            const int kind = tv_kind[i];
            if (kind == 1 || kind == 2) {
                const int x = std::min(width - 1, std::max(0, static_cast<int>(std::round(tv_a[i] * (width - 1)))));
                const int y0 = kind == 1 ? 0 : height - edge_h;
                for (int y = 0; y < edge_h; ++y) {
                    const uint8_t* px = frame + ((y0 + y) * width + x) * 3;
                    b += px[0] / 255.0f; g += px[1] / 255.0f; r += px[2] / 255.0f;
                }
                r /= edge_h; g /= edge_h; b /= edge_h;
            } else if (kind == 3 || kind == 4) {
                const int y = std::min(height - 1, std::max(0, static_cast<int>(std::round((1.0f - tv_a[i]) * (height - 1)))));
                const int x0 = kind == 3 ? 0 : width - edge_w;
                for (int x = 0; x < edge_w; ++x) {
                    const uint8_t* px = frame + (y * width + x0 + x) * 3;
                    b += px[0] / 255.0f; g += px[1] / 255.0f; r += px[2] / 255.0f;
                }
                r /= edge_w; g /= edge_w; b /= edge_w;
            } else if (kind == 5) {
                const int x = std::min(width - 1, std::max(0, static_cast<int>(std::round(tv_a[i] * (width - 1)))));
                const int y = std::min(height - 1, std::max(0, static_cast<int>(std::round((1.0f - tv_b[i]) * (height - 1)))));
                const uint8_t* px = frame + (y * width + x) * 3;
                b = px[0] / 255.0f; g = px[1] / 255.0f; r = px[2] / 255.0f;
            }
            if (sync[i] == 1) {
                leds[i * 3 + 0] = r; leds[i * 3 + 1] = g; leds[i * 3 + 2] = b;
            } else {
                const float mix = clamp01(blend_ptr[i]);
                leds[i * 3 + 0] = leds[i * 3 + 0] * (1.0f - mix) + r * mix;
                leds[i * 3 + 1] = leds[i * 3 + 1] * (1.0f - mix) + g * mix;
                leds[i * 3 + 2] = leds[i * 3 + 2] * (1.0f - mix) + b * mix;
            }
        }
    }

    auto ambient = py::array_t<float>(std::vector<py::ssize_t>{total_leds, 3});
    py::buffer_info ambient_info = ambient.request();
    float* amb = static_cast<float*>(ambient_info.ptr);
    std::fill(amb, amb + total_leds * 3, 0.0f);
    const int32_t* front_idx = static_cast<const int32_t*>(front_idx_info.ptr);
    const float* front_color_ptr = static_cast<const float*>(front_color_info.ptr);
    int front_count = static_cast<int>(front_idx_info.shape[0]);
    if (front_intensity > 0.0f && front_count > 0) {
        const bool use_strip = front_strip_info.ndim == 2 && front_strip_info.shape[0] > 0 && front_strip_info.shape[1] == 3;
        const float* strip = use_strip ? static_cast<const float*>(front_strip_info.ptr) : nullptr;
        const int strip_count = use_strip ? static_cast<int>(front_strip_info.shape[0]) : 0;
        for (int j = 0; j < front_count; ++j) {
            const int led_index = front_idx[j];
            if (led_index < 0 || led_index >= total_leds || sync[led_index] == 1) {
                continue;
            }
            float r = front_color_ptr[0], g = front_color_ptr[1], b = front_color_ptr[2];
            if (use_strip) {
                const int src = front_count <= 1 ? 0 : std::min(strip_count - 1, std::max(0, static_cast<int>(std::round((static_cast<float>(j) / (front_count - 1)) * (strip_count - 1)))));
                r = strip[src * 3 + 0]; g = strip[src * 3 + 1]; b = strip[src * 3 + 2];
            }
            amb[led_index * 3 + 0] = r * front_intensity;
            amb[led_index * 3 + 1] = g * front_intensity;
            amb[led_index * 3 + 2] = b * front_intensity;
            leds[led_index * 3 + 0] += amb[led_index * 3 + 0];
            leds[led_index * 3 + 1] += amb[led_index * 3 + 1];
            leds[led_index * 3 + 2] += amb[led_index * 3 + 2];
        }
    }

    if (ambient_side_spill_enabled) {
        const int32_t* ext_idx = static_cast<const int32_t*>(ext_idx_info.ptr);
        const int32_t* ext_parent = static_cast<const int32_t*>(ext_parent_info.ptr);
        const float* ext_fade = static_cast<const float*>(ext_fade_info.ptr);
        const float* ext_strength = static_cast<const float*>(ext_strength_info.ptr);
        const float* scene = static_cast<const float*>(scene_color_info.ptr);
        const int ext_count = static_cast<int>(ext_idx_info.shape[0]);
        const float strength_boost = clamp01(ambient_boost) * std::max(0.0f, ambient_base_intensity);
        const float tint_blend = std::min(0.90f, clamp01(ambient_boost) * 1.05f);
        const bool tint = ambient_boost > 0.05f && std::max(scene[0], std::max(scene[1], scene[2])) > 0.001f;
        for (int e = 0; e < ext_count; ++e) {
            const int led_index = ext_idx[e];
            const int parent = ext_parent[e];
            if (led_index < 0 || led_index >= total_leds || parent < 0 || parent >= total_leds) {
                continue;
            }
            float r = amb[parent * 3 + 0];
            float g = amb[parent * 3 + 1];
            float b = amb[parent * 3 + 2];
            if (tint && (r > 0.0f || g > 0.0f || b > 0.0f)) {
                const float level = std::max(r, std::max(g, b));
                r = r * (1.0f - tint_blend) + scene[0] * std::max(level, 0.08f) * tint_blend;
                g = g * (1.0f - tint_blend) + scene[1] * std::max(level, 0.08f) * tint_blend;
                b = b * (1.0f - tint_blend) + scene[2] * std::max(level, 0.08f) * tint_blend;
            }
            const float strength = clamp01(ext_strength[e] * (1.0f + strength_boost * 0.85f));
            const float cap = clamp01(ambient_base_intensity * (1.0f + strength_boost * 0.85f));
            r = std::min(cap, r * ext_fade[e] * strength * ambient_base_intensity);
            g = std::min(cap, g * ext_fade[e] * strength * ambient_base_intensity);
            b = std::min(cap, b * ext_fade[e] * strength * ambient_base_intensity);
            leds[led_index * 3 + 0] = std::max(leds[led_index * 3 + 0], r);
            leds[led_index * 3 + 1] = std::max(leds[led_index * 3 + 1], g);
            leds[led_index * 3 + 2] = std::max(leds[led_index * 3 + 2], b);
        }
    }

    py::array_t<uint8_t> out = post_process_impl(working, previous_leds, color_smoothing, saturation, white_balance, brightness, gamma);

    const float* intensity_ptr = static_cast<const float*>(intensities_info.ptr);
    const float* age_ptr = static_cast<const float*>(ages_info.ptr);
    const float* speed_ptr = static_cast<const float*>(speeds_info.ptr);
    const float* radius_ptr = static_cast<const float*>(radius_info.ptr);
    const float* spread_ptr = static_cast<const float*>(spreads_info.ptr);
    py::list active_indices;
    py::list new_intensities;
    py::list new_ages;
    for (int i = 0; i < wave_count; ++i) {
        const float travelled = speed_ptr[i] * age_ptr[i];
        if (intensity_ptr[i] >= wave_min_intensity && travelled <= radius_ptr[i] + spread_ptr[i] * 3.0f) {
            active_indices.append(i);
            new_intensities.append(intensity_ptr[i]);
            new_ages.append(age_ptr[i]);
        }
    }
    return py::make_tuple(out, working, active_indices, new_intensities, new_ages);
}

class Renderer {
public:
    Renderer(py::dict d) {
        total_leds = d["total_leds"].cast<int>();
        max_active_waves = d["max_active_waves"].cast<int>();
        room_diagonal = d["room_diagonal"].cast<float>();
        room_width = d["room_width"].cast<float>();
        room_depth = d["room_depth"].cast<float>();
        wave_speed = d["wave_speed"].cast<float>();
        wave_decay = d["wave_decay"].cast<float>();
        wave_spread = d["wave_spread"].cast<float>();
        wave_min_intensity = d["wave_min_intensity"].cast<float>();
        variable_event_intensity = d["variable_event_intensity"].cast<bool>();
        color_velocity_speed_boost = d["color_velocity_speed_boost"].cast<float>();
        color_velocity_decay_boost = d["color_velocity_decay_boost"].cast<float>();
        room_fill_threshold = d["room_fill_threshold"].cast<float>();
        level1_threshold = d["level1_threshold"].cast<float>();
        level2_threshold = d["level2_threshold"].cast<float>();
        color_smoothing = d["color_smoothing"].cast<float>();
        saturation = d["saturation"].cast<float>();
        brightness = d["brightness"].cast<float>();
        gamma = d["gamma"].cast<float>();
        ambient_base_intensity = d["ambient_base_intensity"].cast<float>();
        ambient_side_spill_enabled = d["ambient_side_spill_enabled"].cast<bool>();
        auto wb = d["white_balance"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        py::buffer_info wb_info = checked(wb, 1, "white_balance");
        const float* wb_ptr = static_cast<const float*>(wb_info.ptr);
        white_balance = py::array_t<float>(std::vector<py::ssize_t>{3});
        std::copy(wb_ptr, wb_ptr + 3, static_cast<float*>(white_balance.request().ptr));

        spatial_idx = d["spatial_idx"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        positions = d["positions"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        angles = d["angles"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        if (d.contains("spatial_band")) {
            spatial_band = d["spatial_band"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            spatial_band_count = d["spatial_band_count"].cast<int>();
        } else {
            py::buffer_info spatial_info = checked(spatial_idx, 1, "spatial_idx");
            spatial_band = py::array_t<int32_t>(std::vector<py::ssize_t>{spatial_info.shape[0]});
            int32_t* band = static_cast<int32_t*>(spatial_band.request().ptr);
            std::fill(band, band + spatial_info.shape[0], 0);
            spatial_band_count = 1;
        }
        sync_codes = d["sync_codes"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        blend = d["blend"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        tv_sample_kind = d["tv_sample_kind"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        tv_sample_a = d["tv_sample_a"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        tv_sample_b = d["tv_sample_b"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        front_indices = d["front_indices"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        ambient_ext_indices = d["ambient_ext_indices"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        ambient_ext_parent = d["ambient_ext_parent"].cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        ambient_ext_fade = d["ambient_ext_fade"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        ambient_ext_strength = d["ambient_ext_strength"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        previous_leds = py::array_t<float>(std::vector<py::ssize_t>{total_leds, 3});
        float* previous_ptr = static_cast<float*>(previous_leds.request().ptr);
        std::fill(previous_ptr, previous_ptr + total_leds * 3, 0.0f);
        working_leds.resize(static_cast<size_t>(total_leds) * 3);
        output_leds = py::array_t<uint8_t>(std::vector<py::ssize_t>{total_leds, 3});
        tv_frame = py::array_t<uint8_t>(std::vector<py::ssize_t>{0, 0, 3});
        front_strip_colors = py::array_t<float>(std::vector<py::ssize_t>{0, 3});
        waves.reserve(std::max(1, max_active_waves));
        wave_kind.reserve(std::max(1, max_active_waves));
        wave_colors.reserve(static_cast<size_t>(std::max(1, max_active_waves)) * 3);
        wave_intensities.reserve(std::max(1, max_active_waves));
        wave_origins.reserve(static_cast<size_t>(std::max(1, max_active_waves)) * 3);
        wave_speeds.reserve(std::max(1, max_active_waves));
        wave_ages.reserve(std::max(1, max_active_waves));
        wave_radius.reserve(std::max(1, max_active_waves));
        wave_spreads.reserve(std::max(1, max_active_waves));
        wave_widths.reserve(std::max(1, max_active_waves));
        wave_hints.reserve(std::max(1, max_active_waves));
        wave_phases.reserve(std::max(1, max_active_waves));
    }

    void add_events(py::list events) {
        for (py::handle item : events) {
            py::dict e = py::cast<py::dict>(item);
            std::string kind = e["kind"].cast<std::string>();
            float intensity = e["intensity"].cast<float>();
            int r = e["r"].cast<int>();
            int g = e["g"].cast<int>();
            int b = e["b"].cast<int>();
            float origin_x = e["origin_x"].cast<float>();
            float origin_y = e["origin_y"].cast<float>();
            float origin_z = e["origin_z"].cast<float>();
            float color_velocity = clamp01(e["color_velocity"].cast<float>());
            int direction_hint = e["direction_hint"].cast<int>();
            float width = std::max(0.05f, e["width"].cast<float>());
            int pulse_count = std::max(1, e["pulse_count"].cast<int>());
            float phase = e["phase"].cast<float>();
            bool secondary = e["secondary"].cast<bool>();

            if (kind == "front_ambient") {
                float color[3] = {r / 255.0f, g / 255.0f, b / 255.0f};
                front_color[0] = front_color[0] * 0.85f + color[0] * 0.15f;
                front_color[1] = front_color[1] * 0.85f + color[1] * 0.15f;
                front_color[2] = front_color[2] * 0.85f + color[2] * 0.15f;
                front_intensity = std::max(front_intensity * 0.8f, event_brightness_gain(kind, intensity));
                continue;
            }

            float speed = room_diagonal * 0.18f * wave_speed * (0.55f + intensity * 1.35f);
            speed *= 1.0f + color_velocity * color_velocity_speed_boost;
            float spread = std::max(0.05f, room_diagonal * 0.025f * (0.65f + intensity) * wave_spread / 9.0f);
            float decay = std::min(std::max(wave_decay - color_velocity * color_velocity_decay_boost, 0.50f), 0.999f);
            if (kind == "flash") {
                speed *= 2.2f; spread *= 2.0f; decay *= 0.90f;
            } else if (kind == "shockwave" || kind == "scene_wipe" || kind == "directional_sweep") {
                speed *= 1.55f; spread *= std::max(0.7f, width); decay *= 0.95f;
            } else if (kind == "color_bloom" || kind == "underwater") {
                speed *= 0.25f; spread *= 3.0f; decay = std::min(std::max(decay, 0.955f), 0.975f);
            } else if (kind == "lightning" || kind == "impact_pulse") {
                speed *= 2.5f; spread *= 2.5f; decay *= 0.78f;
            } else if (kind == "energy_trail") {
                speed *= 1.7f; spread *= 0.75f; decay *= 0.90f;
            } else if (kind == "ember_particles") {
                speed *= 0.55f; spread = std::max(spread * 0.75f, room_diagonal * 0.06f); decay *= 0.88f;
            }
            float radius = radius_limit(intensity, kind);
            int count = std::max(1, (kind == "ember_particles" || kind == "lightning") ? pulse_count : 1);
            for (int pulse = 0; pulse < count; ++pulse) {
                float pulse_phase = phase + static_cast<float>(pulse) / std::max(1, count);
                Wave wave;
                wave.kind = kind;
                wave.kind_code = kind_code_for(kind);
                wave.color_u8[0] = r; wave.color_u8[1] = g; wave.color_u8[2] = b;
                wave.color[0] = r / 255.0f; wave.color[1] = g / 255.0f; wave.color[2] = b / 255.0f;
                wave.intensity = event_brightness_gain(kind, intensity) * (1.0f - pulse * 0.035f);
                wave.origin[0] = origin_x; wave.origin[1] = origin_y; wave.origin[2] = origin_z;
                if (kind == "ember_particles" || kind == "lightning") {
                    float angle = pulse_phase * 6.2831853f;
                    wave.origin[0] += std::cos(angle) * room_diagonal * 0.12f;
                    wave.origin[1] += 0.25f * std::sin(angle * 1.7f) * room_diagonal * 0.12f;
                    wave.origin[2] += std::sin(angle) * room_diagonal * 0.12f;
                }
                wave.speed = speed * (0.85f + pulse_phase * 0.35f);
                wave.decay_rate = decay;
                wave.spread = spread;
                wave.radius_limit = radius;
                wave.color_velocity = color_velocity;
                wave.direction_hint = direction_hint;
                wave.width = width;
                wave.pulse_count = count;
                wave.phase = pulse_phase;
                wave.secondary = secondary;
                waves.push_back(wave);
            }
        }
        trim_waves();
    }

    void set_waves(py::list items) {
        waves.clear();
        for (py::handle item : items) {
            py::dict w = py::cast<py::dict>(item);
            Wave wave;
            wave.kind = w["kind"].cast<std::string>();
            wave.kind_code = kind_code_for(wave.kind);
            wave.color_u8[0] = w["r"].cast<int>();
            wave.color_u8[1] = w["g"].cast<int>();
            wave.color_u8[2] = w["b"].cast<int>();
            wave.color[0] = wave.color_u8[0] / 255.0f;
            wave.color[1] = wave.color_u8[1] / 255.0f;
            wave.color[2] = wave.color_u8[2] / 255.0f;
            wave.intensity = w["intensity"].cast<float>();
            wave.origin[0] = w["origin_x"].cast<float>();
            wave.origin[1] = w["origin_y"].cast<float>();
            wave.origin[2] = w["origin_z"].cast<float>();
            wave.speed = w["speed"].cast<float>();
            wave.decay_rate = w["decay_rate"].cast<float>();
            wave.spread = w["spread"].cast<float>();
            wave.age = w["age"].cast<float>();
            wave.radius_limit = w["radius_limit"].cast<float>();
            wave.color_velocity = w["color_velocity"].cast<float>();
            wave.direction_hint = w["direction_hint"].cast<int>();
            wave.width = w["width"].cast<float>();
            wave.pulse_count = w["pulse_count"].cast<int>();
            wave.phase = w["phase"].cast<float>();
            wave.secondary = w["secondary"].cast<bool>();
            waves.push_back(wave);
        }
        trim_waves();
    }

    py::list get_waves() const {
        py::list result;
        for (const Wave& wave : waves) {
            py::dict w;
            w["kind"] = wave.kind;
            w["r"] = wave.color_u8[0]; w["g"] = wave.color_u8[1]; w["b"] = wave.color_u8[2];
            w["intensity"] = wave.intensity;
            w["origin_x"] = wave.origin[0]; w["origin_y"] = wave.origin[1]; w["origin_z"] = wave.origin[2];
            w["speed"] = wave.speed; w["decay_rate"] = wave.decay_rate; w["spread"] = wave.spread; w["age"] = wave.age;
            w["radius_limit"] = wave.radius_limit; w["color_velocity"] = wave.color_velocity; w["direction_hint"] = wave.direction_hint;
            w["width"] = wave.width; w["pulse_count"] = wave.pulse_count; w["phase"] = wave.phase; w["secondary"] = wave.secondary;
            result.append(w);
        }
        return result;
    }

    void duck_lower_priority_waves(const std::string& primary_kind, float factor) {
        int primary = priority_for_kind(primary_kind);
        for (Wave& wave : waves) {
            if (wave.kind != primary_kind && !(wave.kind == "ember_particles" && (primary_kind == "explosion" || primary_kind == "lightning" || primary_kind == "flame_shimmer")) && priority_for_kind(wave.kind) < primary) {
                wave.intensity *= factor;
            }
        }
    }

    void set_tv_frame(py::object frame) {
        if (frame.is_none()) {
            tv_frame = py::array_t<uint8_t>(std::vector<py::ssize_t>{0, 0, 3});
        } else {
            tv_frame = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(frame);
        }
    }

    void set_front_ambient_strip(py::array_t<float, py::array::c_style | py::array::forcecast> colors, float intensity) {
        py::buffer_info info = colors.request();
        if (info.ndim != 2 || info.shape[0] <= 0 || info.shape[1] != 3) return;
        const float* src = static_cast<const float*>(info.ptr);
        int count = static_cast<int>(info.shape[0]);
        float luma_sum = 0.0f;
        float color_sum[3] = {0.0f, 0.0f, 0.0f};
        float chroma_std_proxy = 0.0f;
        for (int i = 0; i < count; ++i) {
            const float r = clamp01(src[i * 3 + 0]);
            const float g = clamp01(src[i * 3 + 1]);
            const float b = clamp01(src[i * 3 + 2]);
            luma_sum += r * 0.2126f + g * 0.7152f + b * 0.0722f;
            color_sum[0] += r; color_sum[1] += g; color_sum[2] += b;
            const float level = std::max(r, std::max(g, b));
            if (level > 0.001f) {
                const float cr = r / level, cg = g / level, cb = b / level;
                const float mean = (cr + cg + cb) / 3.0f;
                chroma_std_proxy += (std::fabs(cr - mean) + std::fabs(cg - mean) + std::fabs(cb - mean)) / 3.0f;
            }
        }
        const float mean_luma = (luma_sum / std::max(1, count)) * clamp01(intensity);
        const float previous_luma = front_scene_luma;
        front_scene_luma = previous_luma * 0.78f + mean_luma * 0.22f;
        if (mean_luma <= 0.015f) {
            ambient_boost *= 0.82f;
        } else {
            const float color_level = std::max(color_sum[0], std::max(color_sum[1], color_sum[2])) / std::max(1, count);
            if (color_level > 0.001f) {
                const float color_min = std::min(color_sum[0], std::min(color_sum[1], color_sum[2])) / std::max(1, count);
                const float saturation_proxy = (color_level - color_min) / std::max(color_level, 0.001f);
                const float uniformity = 1.0f - chroma_std_proxy / std::max(1, count);
                const float rising = std::max(0.0f, mean_luma - previous_luma);
                const float uniform_factor = clamp01((uniformity - 0.55f) / 0.45f);
                const float rise_factor = clamp01(rising / 0.045f);
                const float brightness_factor = clamp01((mean_luma - 0.015f) / 0.45f);
                const float color_factor = clamp01(0.35f + saturation_proxy * 0.65f);
                ambient_boost = std::max(ambient_boost * 0.82f, uniform_factor * rise_factor * brightness_factor * color_factor);
            } else {
                ambient_boost *= 0.82f;
            }
        }
        if (front_strip_colors.request().ndim != 2 || front_strip_colors.request().shape[0] != count) {
            front_strip_colors = py::array_t<float>(std::vector<py::ssize_t>{count, 3});
            std::copy(src, src + count * 3, static_cast<float*>(front_strip_colors.request().ptr));
        } else {
            float* dst = static_cast<float*>(front_strip_colors.request().ptr);
            for (int i = 0; i < count * 3; ++i) dst[i] = dst[i] * 0.70f + clamp01(src[i]) * 0.30f;
        }
        float* strip = static_cast<float*>(front_strip_colors.request().ptr);
        front_color[0] = front_color[1] = front_color[2] = 0.0f;
        for (int i = 0; i < count; ++i) {
            front_color[0] += strip[i * 3 + 0]; front_color[1] += strip[i * 3 + 1]; front_color[2] += strip[i * 3 + 2];
        }
        front_color[0] /= count; front_color[1] /= count; front_color[2] /= count;
        front_intensity = std::max(front_intensity * 0.8f, clamp01(intensity));
    }

    void set_ambient_spill_scene_boost(float score, py::tuple color) {
        score = clamp01(score);
        float target[3] = {color[0].cast<float>() / 255.0f, color[1].cast<float>() / 255.0f, color[2].cast<float>() / 255.0f};
        if (score <= 0.0f || std::max(target[0], std::max(target[1], target[2])) <= 0.001f) {
            ambient_boost *= 0.92f;
            return;
        }
        ambient_boost = std::max(ambient_boost * 0.88f, score);
        if (std::max(scene_color[0], std::max(scene_color[1], scene_color[2])) <= 0.001f) {
            scene_color[0] = target[0]; scene_color[1] = target[1]; scene_color[2] = target[2];
        } else {
            scene_color[0] = scene_color[0] * 0.65f + target[0] * 0.35f;
            scene_color[1] = scene_color[1] * 0.65f + target[1] * 0.35f;
            scene_color[2] = scene_color[2] * 0.65f + target[2] * 0.35f;
        }
    }

    void set_spatial_priority_budget(py::object budget) {
        if (budget.is_none()) priority_budget = -1;
        else priority_budget = budget.cast<int>();
    }

    py::array_t<uint8_t> step(float dt) {
        update_active_waves(dt);
        float* leds = working_leds.data();
        render_native_waves(leds);
        apply_tv_sync(leds);
        apply_front_ambient(leds, dt);
        py::array_t<uint8_t> out = post_process_and_store_previous(leds);
        return out;
    }

private:
    int total_leds = 0;
    int max_active_waves = 50;
    int priority_budget = -1;
    float room_diagonal = 1.0f, room_width = 1.0f, room_depth = 1.0f;
    float wave_speed = 1.0f, wave_decay = 0.95f, wave_spread = 9.0f, wave_min_intensity = 0.01f;
    bool variable_event_intensity = true;
    float color_velocity_speed_boost = 0.85f, color_velocity_decay_boost = 0.10f;
    float room_fill_threshold = 0.88f, level1_threshold = 0.30f, level2_threshold = 0.70f;
    float color_smoothing = 0.35f, saturation = 1.0f, brightness = 0.5f, gamma = 2.2f;
    float ambient_base_intensity = 0.45f, ambient_boost = 0.0f;
    bool ambient_side_spill_enabled = true;
    float front_color[3] = {0.0f, 0.0f, 0.0f};
    float scene_color[3] = {0.0f, 0.0f, 0.0f};
    float front_scene_luma = 0.0f;
    float front_intensity = 0.0f;
    int spatial_band_count = 1;
    std::vector<Wave> waves;
    std::vector<float> working_leds;
    std::vector<int32_t> wave_kind, wave_hints;
    std::vector<float> wave_colors, wave_intensities, wave_origins, wave_speeds, wave_ages, wave_radius, wave_spreads, wave_widths, wave_phases;
    py::array_t<int32_t> spatial_idx, spatial_band, sync_codes, tv_sample_kind, front_indices, ambient_ext_indices, ambient_ext_parent;
    py::array_t<float> positions, angles, blend, tv_sample_a, tv_sample_b, ambient_ext_fade, ambient_ext_strength, white_balance, previous_leds, front_strip_colors;
    py::array_t<uint8_t> tv_frame, output_leds;

    float event_brightness_gain(const std::string& kind, float intensity) const {
        intensity = clamp01(intensity);
        if (kind == "front_ambient" || kind == "color_bloom" || kind == "underwater") return intensity;
        if (!variable_event_intensity) return 1.0f;
        return clamp01(std::pow(intensity, 0.7f));
    }

    float radius_limit(float intensity, const std::string& kind) const {
        if (kind == "flash" || kind == "explosion" || kind == "impact_pulse" || kind == "shockwave" || kind == "color_bloom" || kind == "underwater" || kind == "portal_vortex" || kind == "negative_wave" || kind == "ember_particles" || intensity >= room_fill_threshold) return room_diagonal;
        if (kind == "directional_sweep" || kind == "scene_wipe") return room_diagonal * 0.9f;
        if (intensity >= level2_threshold) return room_diagonal * 0.65f;
        if (intensity >= level1_threshold) return room_diagonal * 0.35f;
        return room_diagonal * 0.2f;
    }

    void trim_waves() {
        if (static_cast<int>(waves.size()) <= max_active_waves) return;
        std::sort(waves.begin(), waves.end(), [](const Wave& a, const Wave& b) { return a.intensity > b.intensity; });
        waves.resize(max_active_waves);
    }

    void update_active_waves(float dt) {
        size_t write = 0;
        for (size_t read = 0; read < waves.size(); ++read) {
            Wave& wave = waves[read];
            wave.age += dt;
            wave.intensity *= std::pow(wave.decay_rate, dt * 30.0f);
            float travelled = wave.speed * wave.age;
            if (wave.intensity >= wave_min_intensity && travelled <= wave.radius_limit + wave.spread * 3.0f) {
                if (write != read) {
                    waves[write] = std::move(wave);
                }
                ++write;
            }
        }
        waves.resize(write);
    }

    void render_native_waves(float* leds) {
        std::fill(leds, leds + total_leds * 3, 0.0f);
        if (waves.empty()) return;
        py::buffer_info spatial_info = checked(spatial_idx, 1, "spatial_idx");
        py::buffer_info positions_info = checked(positions, 2, "positions");
        py::buffer_info angles_info = checked(angles, 1, "angles");
        py::buffer_info band_info = checked(spatial_band, 1, "spatial_band");
        const int spatial_count = static_cast<int>(spatial_info.shape[0]);
        const int32_t* idx = static_cast<const int32_t*>(spatial_info.ptr);
        const int32_t* bands = static_cast<const int32_t*>(band_info.ptr);
        const float* pos = static_cast<const float*>(positions_info.ptr);
        const float* angle = static_cast<const float*>(angles_info.ptr);
        const int max_band = priority_budget < 0 ? spatial_band_count : std::max(1, std::min(priority_budget, spatial_band_count));
        if (max_band >= spatial_band_count) {
            fill_wave_vectors();
            render_waves_into(
                leds,
                total_leds,
                idx,
                spatial_count,
                pos,
                angle,
                wave_kind.data(),
                wave_colors.data(),
                wave_intensities.data(),
                wave_origins.data(),
                wave_speeds.data(),
                wave_ages.data(),
                wave_radius.data(),
                wave_spreads.data(),
                wave_widths.data(),
                wave_hints.data(),
                wave_phases.data(),
                static_cast<int>(waves.size()),
                room_diagonal,
                room_width,
                room_depth
            );
            return;
        }

        for (const Wave& wave : waves) {
            const int kind = wave.kind_code;
            const float travelled = wave.speed * wave.age;
            const float travel_fade = std::max(0.0f, 1.0f - travelled / std::max(0.1f, wave.radius_limit));
            const float contribution_scale = wave.intensity * (0.2f + 0.8f * travel_fade);
            const float red = wave.color[0] * contribution_scale;
            const float green = wave.color[1] * contribution_scale;
            const float blue = wave.color[2] * contribution_scale;

            for (int local_index = 0; local_index < spatial_count; ++local_index) {
                if (bands[local_index] >= max_band) continue;
                const float px = pos[local_index * 3 + 0];
                const float py = pos[local_index * 3 + 1];
                const float pz = pos[local_index * 3 + 2];
                const float dx = px - wave.origin[0];
                const float dy = py - wave.origin[1];
                const float dz = pz - wave.origin[2];
                const float distances_sq = dx * dx + dy * dy + dz * dz;
                float envelope = 0.0f;

                if (kind == 1) {
                    const float fill_radius = std::max(wave.spread, travelled + wave.spread);
                    envelope = std::exp(-distances_sq / (2.0f * fill_radius * fill_radius));
                } else if (kind == 2) {
                    const float fill_radius = std::max(room_diagonal, travelled + wave.spread);
                    const float local = std::exp(-distances_sq / (2.0f * fill_radius * fill_radius));
                    const float collapse = std::max(0.0f, 1.0f - wave.age / std::max(0.2f, wave.width));
                    envelope = std::max(local, 0.45f * collapse);
                } else if (kind == 3) {
                    const float distance = std::sqrt(distances_sq);
                    const float ring_width = std::max(wave.spread * 0.45f, wave.width * 0.08f);
                    float ring = std::exp(-sq(distance - travelled) / (2.0f * ring_width * ring_width));
                    float room_flash = std::max(0.0f, 1.0f - travelled / std::max(0.1f, wave.radius_limit)) * 0.32f;
                    if (wave.direction_hint != 0) {
                        const float sign = wave.direction_hint > 0 ? 1.0f : -1.0f;
                        const float projection = dx * sign;
                        const float forward = clamp01((projection / std::max(distance, 0.001f) + 1.0f) * 0.5f);
                        const float gate = clamp01(forward * 1.35f);
                        ring *= gate;
                        room_flash *= 0.35f + 0.65f * gate;
                    }
                    envelope = std::max(ring, room_flash);
                } else if (kind == 4) {
                    const float axis = px;
                    const float span = std::max(std::max(room_width, room_depth), 0.1f);
                    float front = (travelled / std::max(0.1f, wave.radius_limit)) * span;
                    if (wave.direction_hint < 0) front = span - front;
                    const float sweep_width = std::max(0.12f, wave.width * 0.25f);
                    envelope = std::exp(-sq(axis - front) / (2.0f * sweep_width * sweep_width));
                } else if (kind == 5) {
                    const float distance = std::sqrt(distances_sq);
                    const float head = std::exp(-sq(distance - travelled) / (2.0f * wave.spread * wave.spread));
                    const float tail_distance = distance - std::max(0.0f, travelled - wave.spread * 3.0f);
                    const float tail_spread = wave.spread * 2.2f;
                    const float tail = std::exp(-(tail_distance * tail_distance) / (2.0f * tail_spread * tail_spread)) * 0.45f;
                    envelope = std::max(head, tail);
                } else if (kind == 6) {
                    const float noise = 0.65f + 0.35f * std::sin(px * 7.0f + pz * 5.0f + wave.age * 18.0f + wave.phase * 6.28f);
                    const float local_spread = std::max(wave.spread * 2.4f, 0.1f);
                    envelope = std::exp(-distances_sq / (2.0f * local_spread * local_spread)) * noise;
                } else if (kind == 7) {
                    const float distance = std::sqrt(distances_sq);
                    const float ripple_base = std::sin(distance * 6.0f - wave.age * 4.0f + wave.phase * 6.28f);
                    const float ripple = 0.45f + 0.55f * ripple_base * ripple_base;
                    const float local_spread = std::max(wave.spread * 5.0f, 0.1f);
                    envelope = std::exp(-distances_sq / (2.0f * local_spread * local_spread)) * ripple;
                } else if (kind == 8) {
                    const float distance = std::sqrt(distances_sq);
                    const int turns = std::max(1, wave.direction_hint);
                    const float spiral = 0.5f + 0.5f * std::sin(angle[local_index] * 3.0f * turns + distance * 4.0f - wave.age * 6.0f);
                    const float vortex_spread = wave.spread * 2.5f;
                    const float drift = distance - travelled * 0.45f;
                    envelope = std::exp(-(drift * drift) / (2.0f * vortex_spread * vortex_spread)) * spiral;
                } else if (kind == 9 || kind == 10 || kind == 11) {
                    const float bloom_spread = kind == 9 ? wave.spread * 4.0f : wave.spread;
                    const float local_spread = std::max(bloom_spread, 0.1f);
                    envelope = std::exp(-distances_sq / (2.0f * local_spread * local_spread));
                    if (kind == 10 && wave.intensity >= 0.72f) {
                        const float strobe = (static_cast<int>(wave.age * 28.0f + wave.phase * 7.0f) % 2) == 0 ? 1.0f : 0.0f;
                        envelope = std::max(envelope, strobe * wave.intensity * 0.55f);
                    }
                } else {
                    const float distance = std::sqrt(distances_sq);
                    envelope = std::exp(-sq(distance - travelled) / (2.0f * wave.spread * wave.spread));
                }

                const int led_index = idx[local_index];
                if (led_index < 0 || led_index >= total_leds) continue;
                float* led = leds + led_index * 3;
                if (kind == 12) {
                    const float dim = envelope * wave.intensity * 0.65f;
                    led[0] -= dim; led[1] -= dim; led[2] -= dim;
                } else {
                    led[0] += envelope * red;
                    led[1] += envelope * green;
                    led[2] += envelope * blue;
                }
            }
        }
    }

    void fill_wave_vectors() {
        const size_t count = waves.size();
        wave_kind.resize(count);
        wave_colors.resize(count * 3);
        wave_intensities.resize(count);
        wave_origins.resize(count * 3);
        wave_speeds.resize(count);
        wave_ages.resize(count);
        wave_radius.resize(count);
        wave_spreads.resize(count);
        wave_widths.resize(count);
        wave_hints.resize(count);
        wave_phases.resize(count);
        for (size_t i = 0; i < count; ++i) {
            const Wave& wave = waves[i];
            wave_kind[i] = wave.kind_code;
            wave_colors[i * 3 + 0] = wave.color[0];
            wave_colors[i * 3 + 1] = wave.color[1];
            wave_colors[i * 3 + 2] = wave.color[2];
            wave_intensities[i] = wave.intensity;
            wave_origins[i * 3 + 0] = wave.origin[0];
            wave_origins[i * 3 + 1] = wave.origin[1];
            wave_origins[i * 3 + 2] = wave.origin[2];
            wave_speeds[i] = wave.speed;
            wave_ages[i] = wave.age;
            wave_radius[i] = wave.radius_limit;
            wave_spreads[i] = wave.spread;
            wave_widths[i] = wave.width;
            wave_hints[i] = wave.direction_hint;
            wave_phases[i] = wave.phase;
        }
    }

    void apply_tv_sync(float* leds) {
        py::buffer_info frame_info = tv_frame.request();
        if (!(frame_info.ndim == 3 && frame_info.shape[0] > 0 && frame_info.shape[1] > 0 && frame_info.shape[2] == 3)) return;
        py::buffer_info sync_info = checked(sync_codes, 1, "sync_codes");
        py::buffer_info blend_info = checked(blend, 1, "blend");
        py::buffer_info tv_kind_info = checked(tv_sample_kind, 1, "tv_sample_kind");
        py::buffer_info tv_a_info = checked(tv_sample_a, 1, "tv_sample_a");
        py::buffer_info tv_b_info = checked(tv_sample_b, 1, "tv_sample_b");
        const int32_t* sync = static_cast<const int32_t*>(sync_info.ptr);
        const float* blend_ptr = static_cast<const float*>(blend_info.ptr);
        const int32_t* tv_kind = static_cast<const int32_t*>(tv_kind_info.ptr);
        const float* tv_a = static_cast<const float*>(tv_a_info.ptr);
        const float* tv_b = static_cast<const float*>(tv_b_info.ptr);
        const int height = static_cast<int>(frame_info.shape[0]);
        const int width = static_cast<int>(frame_info.shape[1]);
        const uint8_t* frame = static_cast<const uint8_t*>(frame_info.ptr);
        const int edge_h = std::max(1, static_cast<int>(std::round(height * 0.04f)));
        const int edge_w = std::max(1, static_cast<int>(std::round(width * 0.04f)));
        for (int i = 0; i < total_leds; ++i) {
            if (sync[i] != 1 && sync[i] != 2) continue;
            float r = 0.0f, g = 0.0f, b = 0.0f;
            const int kind = tv_kind[i];
            if (kind == 1 || kind == 2) {
                const int x = std::min(width - 1, std::max(0, static_cast<int>(std::round(tv_a[i] * (width - 1)))));
                const int y0 = kind == 1 ? 0 : height - edge_h;
                for (int y = 0; y < edge_h; ++y) {
                    const uint8_t* px = frame + ((y0 + y) * width + x) * 3;
                    b += px[0] / 255.0f; g += px[1] / 255.0f; r += px[2] / 255.0f;
                }
                r /= edge_h; g /= edge_h; b /= edge_h;
            } else if (kind == 3 || kind == 4) {
                const int y = std::min(height - 1, std::max(0, static_cast<int>(std::round((1.0f - tv_a[i]) * (height - 1)))));
                const int x0 = kind == 3 ? 0 : width - edge_w;
                for (int x = 0; x < edge_w; ++x) {
                    const uint8_t* px = frame + (y * width + x0 + x) * 3;
                    b += px[0] / 255.0f; g += px[1] / 255.0f; r += px[2] / 255.0f;
                }
                r /= edge_w; g /= edge_w; b /= edge_w;
            } else if (kind == 5) {
                const int x = std::min(width - 1, std::max(0, static_cast<int>(std::round(tv_a[i] * (width - 1)))));
                const int y = std::min(height - 1, std::max(0, static_cast<int>(std::round((1.0f - tv_b[i]) * (height - 1)))));
                const uint8_t* px = frame + (y * width + x) * 3;
                b = px[0] / 255.0f; g = px[1] / 255.0f; r = px[2] / 255.0f;
            }
            if (sync[i] == 1) {
                leds[i * 3 + 0] = r; leds[i * 3 + 1] = g; leds[i * 3 + 2] = b;
            } else {
                const float mix = clamp01(blend_ptr[i]);
                leds[i * 3 + 0] = leds[i * 3 + 0] * (1.0f - mix) + r * mix;
                leds[i * 3 + 1] = leds[i * 3 + 1] * (1.0f - mix) + g * mix;
                leds[i * 3 + 2] = leds[i * 3 + 2] * (1.0f - mix) + b * mix;
            }
        }
    }

    void apply_front_ambient(float* leds, float dt) {
        if (front_intensity <= wave_min_intensity) return;
        py::buffer_info front_idx_info = checked(front_indices, 1, "front_indices");
        py::buffer_info sync_info = checked(sync_codes, 1, "sync_codes");
        py::buffer_info front_strip_info = front_strip_colors.request();
        py::buffer_info ext_idx_info = checked(ambient_ext_indices, 1, "ambient_ext_indices");
        py::buffer_info ext_parent_info = checked(ambient_ext_parent, 1, "ambient_ext_parent");
        py::buffer_info ext_fade_info = checked(ambient_ext_fade, 1, "ambient_ext_fade");
        py::buffer_info ext_strength_info = checked(ambient_ext_strength, 1, "ambient_ext_strength");
        std::vector<float> ambient(static_cast<size_t>(total_leds) * 3, 0.0f);
        const int32_t* front_idx = static_cast<const int32_t*>(front_idx_info.ptr);
        const int32_t* sync = static_cast<const int32_t*>(sync_info.ptr);
        const int front_count = static_cast<int>(front_idx_info.shape[0]);
        const bool use_strip = front_strip_info.ndim == 2 && front_strip_info.shape[0] > 0 && front_strip_info.shape[1] == 3;
        const float* strip = use_strip ? static_cast<const float*>(front_strip_info.ptr) : nullptr;
        const int strip_count = use_strip ? static_cast<int>(front_strip_info.shape[0]) : 0;
        for (int j = 0; j < front_count; ++j) {
            const int led_index = front_idx[j];
            if (led_index < 0 || led_index >= total_leds || sync[led_index] == 1) continue;
            float r = front_color[0], g = front_color[1], b = front_color[2];
            if (use_strip) {
                const int src = front_count <= 1 ? 0 : std::min(strip_count - 1, std::max(0, static_cast<int>(std::round((static_cast<float>(j) / (front_count - 1)) * (strip_count - 1)))));
                r = strip[src * 3 + 0]; g = strip[src * 3 + 1]; b = strip[src * 3 + 2];
            }
            ambient[led_index * 3 + 0] = r * front_intensity;
            ambient[led_index * 3 + 1] = g * front_intensity;
            ambient[led_index * 3 + 2] = b * front_intensity;
            leds[led_index * 3 + 0] += ambient[led_index * 3 + 0];
            leds[led_index * 3 + 1] += ambient[led_index * 3 + 1];
            leds[led_index * 3 + 2] += ambient[led_index * 3 + 2];
        }
        if (ambient_side_spill_enabled) {
            const int32_t* ext_idx = static_cast<const int32_t*>(ext_idx_info.ptr);
            const int32_t* ext_parent = static_cast<const int32_t*>(ext_parent_info.ptr);
            const float* ext_fade = static_cast<const float*>(ext_fade_info.ptr);
            const float* ext_strength = static_cast<const float*>(ext_strength_info.ptr);
            const int ext_count = static_cast<int>(ext_idx_info.shape[0]);
            const float strength_boost = clamp01(ambient_boost) * std::max(0.0f, ambient_base_intensity);
            const float tint_blend = std::min(0.90f, clamp01(ambient_boost) * 1.05f);
            const bool tint = ambient_boost > 0.05f && std::max(scene_color[0], std::max(scene_color[1], scene_color[2])) > 0.001f;
            for (int e = 0; e < ext_count; ++e) {
                const int led_index = ext_idx[e];
                const int parent = ext_parent[e];
                if (led_index < 0 || led_index >= total_leds || parent < 0 || parent >= total_leds) continue;
                float r = ambient[parent * 3 + 0];
                float g = ambient[parent * 3 + 1];
                float b = ambient[parent * 3 + 2];
                if (tint && (r > 0.0f || g > 0.0f || b > 0.0f)) {
                    const float level = std::max(r, std::max(g, b));
                    r = r * (1.0f - tint_blend) + scene_color[0] * std::max(level, 0.08f) * tint_blend;
                    g = g * (1.0f - tint_blend) + scene_color[1] * std::max(level, 0.08f) * tint_blend;
                    b = b * (1.0f - tint_blend) + scene_color[2] * std::max(level, 0.08f) * tint_blend;
                }
                const float strength = clamp01(ext_strength[e] * (1.0f + strength_boost * 0.85f));
                const float cap = clamp01(ambient_base_intensity * (1.0f + strength_boost * 0.85f));
                r = std::min(cap, r * ext_fade[e] * strength * ambient_base_intensity);
                g = std::min(cap, g * ext_fade[e] * strength * ambient_base_intensity);
                b = std::min(cap, b * ext_fade[e] * strength * ambient_base_intensity);
                leds[led_index * 3 + 0] = std::max(leds[led_index * 3 + 0], r);
                leds[led_index * 3 + 1] = std::max(leds[led_index * 3 + 1], g);
                leds[led_index * 3 + 2] = std::max(leds[led_index * 3 + 2], b);
            }
            ambient_boost *= std::pow(0.94f, dt * 30.0f);
        }
        front_intensity *= std::pow(0.92f, dt * 30.0f);
    }

    py::array_t<uint8_t> post_process_and_store_previous(float* leds) {
        uint8_t* dst = static_cast<uint8_t*>(output_leds.request().ptr);
        float* previous = static_cast<float*>(previous_leds.request().ptr);
        const float* wb = static_cast<const float*>(white_balance.request().ptr);
        const float alpha = clamp01(color_smoothing);
        const float inv_alpha = 1.0f - alpha;
        const float gamma_value = std::max(0.1f, gamma);
        const float inv_gamma = 1.0f / gamma_value;
        const bool apply_saturation = std::fabs(saturation - 1.0f) > 0.001f;
        const bool apply_gamma = std::fabs(gamma_value - 1.0f) > 0.001f;
        for (int i = 0; i < total_leds; ++i) {
            float r = clamp01(leds[i * 3 + 0]) * inv_alpha + previous[i * 3 + 0] * alpha;
            float g = clamp01(leds[i * 3 + 1]) * inv_alpha + previous[i * 3 + 1] * alpha;
            float b = clamp01(leds[i * 3 + 2]) * inv_alpha + previous[i * 3 + 2] * alpha;
            previous[i * 3 + 0] = r; previous[i * 3 + 1] = g; previous[i * 3 + 2] = b;
            if (apply_saturation) {
                float h, s, v;
                rgb_to_hsv(r, g, b, h, s, v);
                hsv_to_rgb(h, clamp01(s * saturation), v, r, g, b);
            }
            r = clamp01(r * wb[0] * brightness);
            g = clamp01(g * wb[1] * brightness);
            b = clamp01(b * wb[2] * brightness);
            if (apply_gamma) {
                r = std::pow(r, inv_gamma);
                g = std::pow(g, inv_gamma);
                b = std::pow(b, inv_gamma);
            }
            dst[i * 3 + 0] = static_cast<uint8_t>(clamp01(r) * 255.0f);
            dst[i * 3 + 1] = static_cast<uint8_t>(clamp01(g) * 255.0f);
            dst[i * 3 + 2] = static_cast<uint8_t>(clamp01(b) * 255.0f);
        }
        return output_leds;
    }

    void fill_wave_arrays(py::array_t<int32_t>& kind, py::array_t<float>& colors, py::array_t<float>& intensities, py::array_t<float>& origins, py::array_t<float>& speeds, py::array_t<float>& ages, py::array_t<float>& radius, py::array_t<float>& spreads, py::array_t<float>& widths, py::array_t<int32_t>& hints, py::array_t<float>& phases) {
        int32_t* k = static_cast<int32_t*>(kind.request().ptr);
        float* c = static_cast<float*>(colors.request().ptr);
        float* in = static_cast<float*>(intensities.request().ptr);
        float* o = static_cast<float*>(origins.request().ptr);
        float* sp = static_cast<float*>(speeds.request().ptr);
        float* ag = static_cast<float*>(ages.request().ptr);
        float* ra = static_cast<float*>(radius.request().ptr);
        float* spr = static_cast<float*>(spreads.request().ptr);
        float* wi = static_cast<float*>(widths.request().ptr);
        int32_t* hi = static_cast<int32_t*>(hints.request().ptr);
        float* ph = static_cast<float*>(phases.request().ptr);
        for (size_t i = 0; i < waves.size(); ++i) {
            const Wave& w = waves[i];
            k[i] = w.kind_code;
            c[i * 3 + 0] = w.color[0]; c[i * 3 + 1] = w.color[1]; c[i * 3 + 2] = w.color[2];
            in[i] = w.intensity;
            o[i * 3 + 0] = w.origin[0]; o[i * 3 + 1] = w.origin[1]; o[i * 3 + 2] = w.origin[2];
            sp[i] = w.speed; ag[i] = w.age; ra[i] = w.radius_limit; spr[i] = w.spread; wi[i] = w.width; hi[i] = w.direction_hint; ph[i] = w.phase;
        }
    }

    void update_previous(py::array_t<float>& working) {
        float* prev = static_cast<float*>(previous_leds.request().ptr);
        float* src = static_cast<float*>(working.request().ptr);
        float alpha = clamp01(color_smoothing);
        for (int i = 0; i < total_leds * 3; ++i) {
            float value = clamp01(src[i]);
            prev[i] = value * (1.0f - alpha) + prev[i] * alpha;
        }
    }
};

}  // namespace

PYBIND11_MODULE(_spatial_native, m) {
    m.doc() = "Native spatial renderer acceleration";
    m.def("render_waves_probe", &render_waves_probe);
    m.def("post_process_probe", &post_process_impl);
    m.def("full_probe", &full_probe);
    m.def("full_renderer_step", &full_renderer_step);
    py::class_<Renderer>(m, "Renderer")
        .def(py::init<py::dict>())
        .def("add_events", &Renderer::add_events)
        .def("set_waves", &Renderer::set_waves)
        .def("get_waves", &Renderer::get_waves)
        .def("duck_lower_priority_waves", &Renderer::duck_lower_priority_waves)
        .def("set_tv_frame", &Renderer::set_tv_frame)
        .def("set_front_ambient_strip", &Renderer::set_front_ambient_strip)
        .def("set_ambient_spill_scene_boost", &Renderer::set_ambient_spill_scene_boost)
        .def("set_spatial_priority_budget", &Renderer::set_spatial_priority_budget)
        .def("step", &Renderer::step);
}
