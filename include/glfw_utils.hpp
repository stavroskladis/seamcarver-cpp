/*  SeamCarver - Content-aware image resizing
    Author: Stavros Kladis <stavroskladis@hotmail.com>
    Copyright © 2025 Stavros Kladis
*/
#pragma once

#include <string>

#include "imgui.h"
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

// Load an image from disk and return its pixel data
unsigned char *load_image(const std::string &path, int &width, int &height, int &channels);

// GLFW error callback that logs the error description
void glfw_error_callback(int error, const char *description);

// Initialize a GLFW window with an OpenGL context and return it
GLFWwindow *initWindow(int w, int h, const std::string &title, std::string &glsl_version);

// Set up Dear ImGui context and connect it to GLFW/OpenGL
void initImGui(GLFWwindow *window, const std::string &glsl_version);

// Upload a CPU pixel buffer to GPU and return the resulting OpenGL texture ID
bool loadTextureFromPixels(const unsigned char *pixels, int width, int height, int channels,
                           GLuint *out_texture);

// Configure spdlog logging formatting and set the desired log level (default is info)
void initLogging(const spdlog::level::level_enum &level = spdlog::level::info);

// Begin a new ImGui frame and create a dock space
void BeginGuiFrame();

// Draw the application settings window (FPS, demo toggle, etc.)
void DrawSettingsWindow(bool &show_demo_window);

// Draw the main image preview and processing window
void ShowImageWindow();

// Modular image display functions called from ShowImageWindow
void ShowOriginalImage(const unsigned char *pixels, int width, int height, int channels);
void ShowSeamCarvedImage(const unsigned char *pixels, int width, int height, int channels,
                         int target_width, bool needs_recompute);
void ShowPrimitiveResizedImage(const unsigned char *pixels, int width, int height, int channels,
                               int target_width, bool needs_recompute);

// Render the ImGui frame and handle multi-viewport rendering
void RenderGuiFrame(GLFWwindow *window, const ImVec4 &clear_color);