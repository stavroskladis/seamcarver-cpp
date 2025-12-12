/*  SeamCarver - Content-aware image resizing
    Author: Stavros Kladis <stavroskladis@hotmail.com>
    Copyright © 2025 Stavros Kladis
*/
#define IMGUI_IMPL_OPENGL_LOADER_GLAD
// clang-format off
// GLAD must be included before glfw_utils.hpp (which includes imgui.h and GLFW/glfw3.h)
#include <glad/glad.h>
#include "glfw_utils.hpp"
// clang-format on

#include "image_processing.hpp"

#include <spdlog/spdlog.h>
#include <string>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// Shared window size state for dynamic resizing of the Image Window
static float g_image_window_width = 1200.0f;
static float g_image_window_height = 800.0f;

void glfw_error_callback(int error, const char *description) {
    spdlog::error("GLFW Error {}: {}", error, description);
}

unsigned char *load_image(const std::string &path, int &width, int &height, int &channels) {

    unsigned char *pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb);

    if (!pixels) {
        spdlog::error("Failed to load image: {}", path.c_str());
    }

    return pixels;
}

void initLogging(const spdlog::level::level_enum &level) {
    spdlog::set_level(level);
    spdlog::set_pattern("[%H:%M:%S] [%^%L%$] %v");
}

GLFWwindow *initWindow(int w, int h, const std::string &title, std::string &glsl_version) {
    // Register the error handler callback
    glfwSetErrorCallback(glfw_error_callback);

    // Setup window
    if (!glfwInit()) {
        spdlog::error("Failed to initialize GLFW");
        return nullptr;
    }

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100 (Mobile/Embedded devices)
    glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);           // Required on Mac
#else
    // GL 3.0 + GLSL 130 (Windows/Linux Desktops)
    glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+
    // only glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 3.0+ only
#endif

    GLFWwindow *window = glfwCreateWindow(w, h, title.c_str(), nullptr, nullptr);
    if (!window) {
        spdlog::error("Failed to create window");
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Platform/Renderer backends
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        spdlog::error("Failed to initialize GLAD");
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    return window;
}

void initImGui(GLFWwindow *window, const std::string &glsl_version) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_ViewportsEnable;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    // ImGui::StyleColorsLight();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform
    // windows can look identical to regular ones.
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGuiStyle &style = ImGui::GetStyle();
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // ImGui Backend Setup (Connect ImGui to GLFW and OpenGL)
    ImGui_ImplGlfw_InitForOpenGL(window, true);   // Connect ImGui to GLFW for input handling
    ImGui_ImplOpenGL3_Init(glsl_version.c_str()); // Connect ImGui to OpenGL for rendering
}

void BeginGuiFrame() {

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame(); // Prepares OpenGL for ImGui rendering & sets up the OpenGL
                                  // context
    ImGui_ImplGlfw_NewFrame();    // Preparing ImGui for a new frame by processing input from GLFW
    ImGui::NewFrame();            // Starts a new ImGui frame, clears previous frame's data

    ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode);
}

void DrawSettingsWindow(bool &show_demo_window) {

    ImGui::Begin("Settings");

    ImGui::Text("Configure the app below.");

    ImGui::Checkbox("Demo Window", &show_demo_window);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate);

    ImGui::Separator();
    ImGui::Text("Image Window Size");
    ImGui::Text("Resize the window containing all three images");

    // Sliders to control the Image Window size dynamically
    ImGui::SliderFloat("Width", &g_image_window_width, 400.0f, 2560.0f, "%.0f px");
    ImGui::SliderFloat("Height", &g_image_window_height, 300.0f, 1440.0f, "%.0f px");

    // Display current size
    ImGui::Text("Current: %.0f x %.0f", g_image_window_width, g_image_window_height);

    ImGui::End();
}

// Upload an already-decoded pixel buffer to an OpenGL texture
bool loadTextureFromPixels(const unsigned char *pixels, int width, int height, int channels,
                           GLuint *out_texture) {

    // Sanity check: ensure we have data and valid dimensions
    if (!pixels || width <= 0 || height <= 0) {
        return false;
    }

    // Figure out the appropriate OpenGL format based on channel count.
    //  We support 3-channel (RGB) and 4-channel (RGBA) images; anything
    //  else falls back to a single-channel RED texture.
    GLenum format = (channels == 4) ? GL_RGBA : (channels == 3) ? GL_RGB : GL_RED;
    if (format != GL_RGB) {
        spdlog::error("We only support 3-channel RGB images!");
        return false;
    }

    GLuint tex_id;

    // Generate an OpenGL texture ID using glGenTextures.
    glGenTextures(1, &tex_id);

    // Bind the texture
    glBindTexture(GL_TEXTURE_2D, tex_id);

    // Set texture parameters (e.g., glTexParameteri for filtering and wrapping).
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Copy the pixel buffer from CPU to GPU memory
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, pixels);

    *out_texture = tex_id;

    return true;
}

// Image window helper
void ShowImageWindow() {

    // Set window size dynamically (updates when sliders in Settings window change)
    ImGui::SetNextWindowSize(ImVec2(g_image_window_width, g_image_window_height), ImGuiCond_Always);

    ImGui::Begin("Image Window");

    const std::string img_path = ASSET_PATH "/schmetterling_mid.jpg";

    // 1. Load image in an 1-dimensional array that represents a 2-dimensional image through
    // row-major ordering see:
    // https://icarus.cs.weber.edu/~dab/cs1410/textbook/7.Arrays/row_major.html (We store 1 pixel in
    // RGB using 1byte (8bits: 0-2^8 - 1) per each channel in a row-major order)
    int width, height, channels;
    unsigned char *pixels = load_image(img_path, width, height, channels);
    if (!pixels) {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Failed to load image!");
        ImGui::End();
        return;
    }

    // 2. Display original image
    ShowOriginalImage(pixels, width, height, channels);

    // 4. add a simple imgui slider here to scale image from 0 to 100%
    static float target_scale_perc = 100.0f;
    static int target_width = width;     // Initialize to original width (no scaling)
    static bool needs_recompute = false; // Start with false, only process when needed
    // Add some spacing to ensure proper slider positioning
    ImGui::Spacing();

    auto slider =
        ImGui::SliderFloat(fmt::format("Scale Image By ({:.1f}%)", target_scale_perc).c_str(),
                           &target_scale_perc, 10.0f, 100.0f);
    if (slider) {

        // Compute the target width (do not change the height)
        // Horizontal-only scaling (wider/narrower, height unchanged)
        target_width = static_cast<int>(width * (target_scale_perc / 100.0f));
        needs_recompute = (target_width < width); // Only recompute if scaling is needed
    }

    // Show seam carved image
    ShowSeamCarvedImage(pixels, width, height, channels, target_width, needs_recompute);

    // Show primitive resized image
    ShowPrimitiveResizedImage(pixels, width, height, channels, target_width, needs_recompute);

    // Reset flag after both algorithms have run
    needs_recompute = false; // it is critical that we reset this flag since it is a static var!

    ImGui::End();

    // Clean up the frame's resources
    stbi_image_free(pixels);
}

// Show the original image
void ShowOriginalImage(const unsigned char *pixels, int width, int height, int channels) {
    // Cache the original texture to avoid creating a new one every frame (memory leak fix)
    static GLuint original_tex_id = 0;
    static int cached_width = 0;
    static int cached_height = 0;

    // Only create texture if it doesn't exist or dimensions changed
    if (original_tex_id == 0 || cached_width != width || cached_height != height) {
        GLuint new_tex_id = 0;
        if (loadTextureFromPixels(pixels, width, height, channels, &new_tex_id)) {
            // Delete old texture if it exists
            if (original_tex_id != 0) {
                glDeleteTextures(1, &original_tex_id);
            }
            original_tex_id = new_tex_id;
            cached_width = width;
            cached_height = height;
        }
    }

    // Display original image
    // (https://github.com/ocornut/imgui/wiki/Image-Loading-and-Displaying-Examples)
    ImGui::Text("Original");
    if (original_tex_id != 0) {
        ImGui::Image((ImTextureID)(intptr_t)original_tex_id, ImVec2(width, height));
    }
}

// Show the seam carved image
void ShowSeamCarvedImage(const unsigned char *pixels, int width, int height, int channels,
                         int target_width, bool needs_recompute) {
    // We keep the seam-carved result in a persistent OpenGL texture so the picture stays on-screen
    // every frame. The texture is (re)built the first time we enter the window and each time the
    // slider requests a new target width. Between those events we simply reuse the cached ID.
    // No extra CPU/GPU work is done and no flicker takes place.
    static GLuint seam_tex_id = 0; // cached GPU texture for seam image
    static int seam_width = 0;     // cached width  of that texture
    static int seam_height = 0;    // cached height of that texture

    // label (kept outside so UI order is stable)
    ImGui::Text("Processed (Seam Carved)");

    // (re)build only if we don't have a texture yet (first time) OR the slider moved
    if (seam_tex_id == 0 || needs_recompute) {

        // Remove seams iteratively to reduce image size
        // Size in bytes of the original image
        const size_t pixel_count = static_cast<size_t>(width) * height * channels;

        // Allocate working buffer (consider a static work buffer if the recompute happens
        // frequent!)
        std::unique_ptr<unsigned char[]> work_buffer =
            std::make_unique<unsigned char[]>(pixel_count);

        // Copy the whole frame into the buffer
        memcpy(work_buffer.get(), pixels, pixel_count);

        // Initialize working dimensions
        int current_width = width; // will shrink
        int current_height = height;
        bool seams_removal_ok = true;
        std::vector<int> seam(current_height); // avoid reallocating the seam vector inside the loop

        // Hybrid approach: OpenCV for initial energy map, manual updates for subsequent iterations
        bool first_iteration = true;
        std::vector<int> energy_map;

#ifdef DEBUG
        auto seam_carving_start = std::chrono::high_resolution_clock::now();
        int iterations = 0;
        int opencv_time = 0;
        int custom_time = 0;
        long long total_seam_time = 0; // Track total time for seam operations.
        // We know that the total time for now includes both energyMapCompute() and
        // energyMapComputeCustom(), but this is included in all of our measurements and its
        // negligible for our purposes.
#endif

        while (current_width > target_width) {

            if (first_iteration) {
                // Use OpenCV for the first, full energy map computation
                // Tongle implementations to compare times. OpenCV is much faster because it uses
                // SIMD instructions.

#ifdef DEBUG
                // OpenCV time
                auto opencv_start = std::chrono::high_resolution_clock::now();
                auto energy_opencv =
                    energyMapCompute(work_buffer.get(), current_width, current_height, channels);
                auto opencv_end = std::chrono::high_resolution_clock::now();
                opencv_time =
                    std::chrono::duration_cast<std::chrono::microseconds>(opencv_end - opencv_start)
                        .count();

                // Custom implementation time of the energy map computation (for fun and testing)
                // Hint: [12:35:09] [I] Energy Map Comparison: OpenCV 20128 μs vs Custom 52234 μs
                // (2.6x slower)
                auto custom_start = std::chrono::high_resolution_clock::now();
                auto energy_custom = energyMapComputeCustom(work_buffer.get(), current_width,
                                                            current_height, channels);
                auto custom_end = std::chrono::high_resolution_clock::now();
                custom_time =
                    std::chrono::duration_cast<std::chrono::microseconds>(custom_end - custom_start)
                        .count();

                // Use the custom implementation for the actual processing (just a proof that it
                // also works)
                auto energy = std::move(energy_custom);
#else
                // If not in debug mode, call only the opencv implementation since it is much faster
                auto energy =
                    energyMapCompute(work_buffer.get(), current_width, current_height, channels);
#endif

                if (!energy) {
                    ImGui::TextColored(ImVec4(1, 0, 0, 1), "Energy map failed!");
                    seams_removal_ok = false;
                    break;
                }
                energy_map = std::move(*energy);
                first_iteration = false;
            }
            else {
                // Use manual incremental update for subsequent iterations (not the first iteration)
                updateEnergyMapAroundSeam(energy_map, seam, current_width, current_height,
                                          work_buffer.get(), channels);
            }

#ifdef DEBUG
            auto seam_operation_start = std::chrono::high_resolution_clock::now();
#endif

            // Find the lowest energy seam in the energy map
            findLowestEnergySeam(std::make_unique<std::vector<int>>(energy_map), seam,
                                 current_width, current_height);
            // findLowestEnergySeamDP(std::make_unique<std::vector<int>>(energy_map), seam,
            // current_width, current_height);

            // Remove the seam from the working buffer
            if (!removeSeam(work_buffer.get(), current_width, current_height, seam)) {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "Seam removal failed!");
                seams_removal_ok = false;
                break;
            }

#ifdef DEBUG
            auto seam_operation_end = std::chrono::high_resolution_clock::now();
            auto seam_operation_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                seam_operation_end - seam_operation_start);
            total_seam_time += seam_operation_duration.count();
#endif

#ifdef DEBUG
            iterations++;
#endif
        }

#ifdef DEBUG
        auto seam_carving_end = std::chrono::high_resolution_clock::now();
        auto seam_carving_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            seam_carving_end - seam_carving_start);
        spdlog::info("Seam Carving: {} iterations, {}x{} -> {}x{} took {} ms", iterations, width,
                     height, current_width, current_height, seam_carving_duration.count());
        if (iterations > 0) {
            double avg_seam_time = static_cast<double>(total_seam_time) / iterations;
            spdlog::info("Average time per seam: {:.2f} μs ({:.2f} ms)", avg_seam_time,
                         avg_seam_time / 1000.0);
        }
        spdlog::info("Energy Map Comparison: OpenCV {} μs vs Custom {} μs ({:.1f}x slower)",
                     opencv_time, custom_time, static_cast<double>(custom_time) / opencv_time);
#endif

        if (seams_removal_ok) {
            GLuint new_tex = 0;
            loadTextureFromPixels(work_buffer.get(), current_width, current_height, channels,
                                  &new_tex);

            if (new_tex != 0) {
                // free previous GPU memory (prevent leaks)
                if (seam_tex_id != 0) {
                    glDeleteTextures(1, &seam_tex_id);
                }

                seam_tex_id = new_tex;
                seam_width = current_width; // remember size for ImGui::Image()
                seam_height = current_height;
            }
        }
    }

    // Draw cached seam-carved image (if a texture exists)
    if (seam_tex_id != 0) {
        ImGui::Image((ImTextureID)(intptr_t)seam_tex_id, ImVec2(seam_width, seam_height));
    }
}

// Show the primitive resized image
void ShowPrimitiveResizedImage(const unsigned char *pixels, int width, int height, int channels,
                               int target_width, bool needs_recompute) {
    // Similar to seam_tex_id, we cache the bilinear-resized texture so it is drawn every frame
    // without re-uploading unless the slider value changes. The primitive resize is cheap but there
    // is still no reason to redo it 60 times/sec when the image is static.
    static GLuint primitive_tex_id = 0; // cached GPU texture for primitive image

    // Build only if texture doesn't exist yet OR slider requested new width
    if (primitive_tex_id == 0 || needs_recompute) {

#ifdef DEBUG
        auto primitive_start = std::chrono::high_resolution_clock::now();
#endif

        auto buffer = buildPrimitiveResizeTexture(pixels, width, height, channels, target_width);

#ifdef DEBUG
        auto primitive_end = std::chrono::high_resolution_clock::now();
        auto primitive_duration =
            std::chrono::duration_cast<std::chrono::microseconds>(primitive_end - primitive_start);
        int pixels_removed = width - target_width;
        if (pixels_removed > 0) {
            double avg_per_pixel = static_cast<double>(primitive_duration.count()) / pixels_removed;
            double avg_per_seam = static_cast<double>(primitive_duration.count()) / pixels_removed;
            spdlog::info("Primitive Resize: {}x{} -> {}x{} took {} μs ({} ms)", width, height,
                         target_width, height, primitive_duration.count(),
                         primitive_duration.count() / 1000.0);
            spdlog::info("Average time per pixel removed: {:.2f} μs", avg_per_pixel);
            spdlog::info("Equivalent time per seam: {:.2f} μs ({:.3f} ms) for {} seams",
                         avg_per_seam, avg_per_seam / 1000.0, pixels_removed);
        }
#endif

        if (buffer) {
            GLuint new_tex_id = 0;
            loadTextureFromPixels(buffer.get(), target_width, height, channels, &new_tex_id);

            if (new_tex_id != 0) {
                // Replace previous texture (prevent VRAM leak, avoid flicker)
                if (primitive_tex_id != 0)
                    glDeleteTextures(1, &primitive_tex_id);

                primitive_tex_id = new_tex_id;
            }
        }
    }

    // always draw if we have something to show
    if (primitive_tex_id != 0) {
        ImGui::Text("Primitive Resized");
        ImGui::Image((ImTextureID)(intptr_t)primitive_tex_id, ImVec2(target_width, height));
    }
}

void RenderGuiFrame(GLFWwindow *window, const ImVec4 &clear_color) {
    // Primary viewport
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Update and Render additional Platform Windows
    // (Platform functions may change the current OpenGL context, so we
    // save/restore it to make it easier to paste this code elsewhere.
    //  For this specific demo app we could also call
    //  glfwMakeContextCurrent(window) directly)
    ImGuiIO &io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        GLFWwindow *backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
}
