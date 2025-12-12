/*  SeamCarver - Content-aware image resizing
    Author: Stavros Kladis <stavroskladis@hotmail.com>
    Copyright © 2025 Stavros Kladis
*/
#define IMGUI_IMPL_OPENGL_LOADER_GLAD
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <spdlog/spdlog.h>

#include "glfw_utils.hpp"
#include "image_processing.hpp"

int main(int, char **) {

    // Initialize the logger with debug level output (usually controlled by a config file)
    initLogging(spdlog::level::debug);

    // Initialize the window with a 1280x720 resolution and a title
    std::string glsl_version;
    GLFWwindow *window = initWindow(1280, 720, "SeamCarver", glsl_version);
    if (!window) {
        spdlog::error("Failed to initialize window");
        return 1;
    }

    // Initialize the ImGui context and connect it to GLFW/OpenGL
    initImGui(window, glsl_version);

    // Our state
    static bool show_demo_window = false;
    ImVec4 clear_color = ImVec4(0.168f, 0.394f, 0.534f, 1.00f);

    // Main loop
    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();

        BeginGuiFrame();

        // 1. Show the big demo window
        if (show_demo_window) {
            ImGui::ShowDemoWindow(&show_demo_window);
        }

        // 2. Simple window to let the user show settings and other info
        DrawSettingsWindow(show_demo_window);

        // 3. Show image window (seam-carved and primitive resize are called from this function)
        ShowImageWindow();

        // Rendering
        ImGui::Render();
        RenderGuiFrame(window, clear_color);
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate(); // clean up GLFW's resources.

    return 0;
}