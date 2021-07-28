#pragma once
#include "Main.hpp"

extern std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> inputCloud;

static double yaw = 0.0, pitch = 0.0, lastX = 0.0, lastY = 0.0;
static int ml = 0;
static double tx = 0.0, ty = 0.0, tz = 0.0;

class Window {
 private:
  GLFWwindow *const window;

 public:
  Window(int width = 1280, int height = 960, const char *title = "main")
      : window(glfwCreateWindow(width, height, title, NULL, NULL)) {
    if (window == NULL) {
      std::cerr << "Can't create GLFW window." << std::endl;
      exit(1);
    }
    // set current window to control taget
    glfwMakeContextCurrent(window);

    // wait for vertical sync
    glfwSwapInterval(1);

    // window resize
    glfwSetWindowSizeCallback(window, resize);
    resize(window, width, height);
  }

  virtual ~Window() { glfwDestroyWindow(window); }

  void windowSetting(std::string setting) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60, (float)640 / 480, 0.01f, 50.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt(-1, 0, 1, 0, 0, 0, 0, 0, 1);
    glTranslatef(0, 0, +0.5f);
    glRotated(pitch, 0, 1, 0);
    glRotated(yaw, 0, 0, 1);
    glTranslatef(0, 0, -0.5f);

    glTranslatef(tx, ty, tz);

    // We will render our depth data as a set of points in 3D space
    glPointSize(2);
    glEnable(GL_DEPTH_TEST);
  }

  static void on_mouse_button(GLFWwindow *win, int button, int action,
                              int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) ml = action == GLFW_PRESS;
  }
  static double clamp(double val, double lo, double hi) {
    return val < lo ? lo : val > hi ? hi : val;
  }
  static void on_cursor_pos(GLFWwindow *win, double x, double y) {
    if (ml)

    {
      yaw = clamp(yaw - (x - lastX), -180, 180);
      pitch = clamp(pitch + (y - lastY), -180, 180);
    }
    lastX = x;
    lastY = y;
  }

  static void on_key(GLFWwindow *win, int key, int scancode, int action, int mods)
  {
    
  }

  void makeContextCurrent() { glfwMakeContextCurrent(window); }

  int shouldClose() const {
    return glfwWindowShouldClose(window) || glfwGetKey(window, GLFW_KEY_ESCAPE);
  }

  // Process when D & D a file into window
  static void drop_callback(GLFWwindow *window, int count, const char **paths) {
    std::cout << paths[0] << std::endl;
    std::string path = std::string(paths[0]);
    int n = path.size();
    if (path.substr(n-3, 3) == "xml") {
      loadXMLFile(path);
    }
    else if (path.substr(n-3, 3) == "txt"){
      loadTXTFile(path);
    }
    else{
      loadPCDFile(path);
    }
  }

  void swapBuffers() {
    // swap color buffer
    glfwSwapBuffers(window);

    glfwPollEvents();

    glfwSetDropCallback(window, drop_callback);
    glfwSetCursorPosCallback(window, on_cursor_pos);
    glfwSetMouseButtonCallback(window, on_mouse_button);
    glfwSetKeyCallback(window, on_key);

    int front_state = glfwGetKey(window, GLFW_KEY_W);
    int back_state = glfwGetKey(window, GLFW_KEY_S);
    int left_state = glfwGetKey(window, GLFW_KEY_A);
    int right_state = glfwGetKey(window, GLFW_KEY_D);
    int up_state = glfwGetKey(window, GLFW_KEY_R);
    int down_state = glfwGetKey(window, GLFW_KEY_F);

    if (front_state == GLFW_PRESS)
    {
      tx -= 0.1f;
    }
    else if (back_state == GLFW_PRESS)
    {
      tx += 0.1f;
    }
    else if (left_state == GLFW_PRESS)
    {
      ty -= 0.1f;
    }
    else if (right_state == GLFW_PRESS)
    {
      ty += 0.1f;
    }
    else if (up_state == GLFW_PRESS)
    {
      tz -= 0.1f;
    }
    else if (down_state == GLFW_PRESS)
    {
      tz += 0.1f;
    }
  }

  static void resize(GLFWwindow *const window, int width, int height) {
    // set the entire window as a viewport
    glViewport(0, 0, width, height);
  }
};
