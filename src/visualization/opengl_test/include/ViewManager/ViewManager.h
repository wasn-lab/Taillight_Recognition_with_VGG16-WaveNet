#ifndef _VIEWMANAGER_H_
#define _VIEWMANAGER_H_

#include "Common.h"

/**
 * @brief The ViewManager class
 * The ViewManager class provides viewing manipulation related functionalities.
 *
 * To use the ViewManager class, call mousePressEvent(),
 * mouseReleaseEvent(), mouseMoveEvent(), wheelEvent() in your event handlers
 * with the event objects.
 *
 * The viewing manipulation will be done for you in the class. When you are ready
 * to render something, call GetModelMatrix(), GetViewMatrix(), GetProjectionMatrix()
 * and their composite versions to get the MVP matrices which encode current viewing
 * properties.
 */

class ViewManager
{
public:
    ViewManager();

	void mouseEvents(int button, int state, int x, int y);
	void mousePressEvent(int button, int x, int y);
	void mouseReleaseEvent(int button, int x, int y);
    void mouseMoveEvent(int x,int y);
	void keyEvents(unsigned char key);
    void wheelEvent(int direction);
	void Translate(glm::vec3 vec);

	glm::mat4 GetModelMatrix();
	glm::mat4 GetViewMatrix();
	glm::mat4 GetProjectionMatrix();
	glm::mat4 GetProjectionMatrix(float aspect);
	glm::mat4 GetViewProjectionMatrix(float aspect);
	glm::mat4 GetModelViewProjectionMatrix(float aspect);
	glm::vec3 GetEyePosition() {return eyePosition;}
	glm::vec3 GetViewVector() {return viewVector;}
	float GetZoom() {return zoom;}

	glm::vec3 GetWorldEyePosition();
	glm::vec3 GetWorldViewVector();

    bool IsOrthoProjection() { return ortho; }

	void SetZoom(float value);
    void SetRotation(float theta, float phi);
    void SetRotation(float x, float y, float z);
	void SetWindowSize(int viewport_width, int viewport_height);
    void SetWindowSize(int ld_corner_x, int ld_corner_y, int viewport_width, int viewport_height, int full_window_width, int full_window_height);
    //
    void SwitchGLViewPortAndCleanDraw();
    //

    // Set the camera pose
    void SetCameraModel(glm::mat4 camera_model_in);
    void SetInvCameraModel(glm::mat4 camera_model_inv_in);

    bool ToggleOrtho() { return ortho = !ortho; }
    void Zoom(float distance);
    void Reset();

    // Mouse
    bool is_mouse_out_of_bound(int x_cv, int y_cv);
    void convert_global_cv_coor_to_local_cv_coor(int x_cv_g, int y_cv_g, int &x_cv_l, int & y_cv_l);

    // camera pose on view (translationMatrix only)
    glm::vec3 get_trans_world_at_camera();



    //------------------------------------------------------------------//
    bool assign_cal_viewport(
        bool (*cal_viewport_in)(
            int full_window_width, int full_window_height,
            int &ld_corner_x, int &ld_corner_y, int &viewport_width, int &viewport_height
        )
    )
    {
        _cal_viewport = cal_viewport_in;
    }
    //------------------------------------------------------------------//

private:
	float aspect;					///< 儲存目前視窗的長寬比。
    bool ortho;						///< 是否使用正交視角。
    float zoom;
	float moveSpeed;				///< 相機的移動速度。
	float orthoScale;

	glm::mat4 translationMatrix;	///< 紀錄Translate動作的Matrix。
	glm::mat4 rotationMatrix;		///< 紀錄Rotation動作的Matrix。
    glm::mat4 camera_model_inv;     // test, camera_model^-1
    glm::mat4 tansformMatrix;       // test, T*R
	glm::mat4 viewMatrix;			///< 紀錄ViewMatrix。
	glm::mat4 projMatrix;			///< 紀錄projMatrix。
	glm::vec3 viewVector;			///< 紀錄相機看往焦點看的向量。
	glm::vec3 rotateXAxis;			///< 紀錄相機的X軸旋轉。
	glm::vec3 rotateYAxis;			///< 紀錄相機的Y軸旋轉。
	glm::vec3 eyePosition;			///< 紀錄相機的位置。
	glm::vec3 eyeLookPosition;		///< 紀錄相機的所看的位置。

    bool lmbDown;					///< 紀錄滑鼠左鍵是否被按住。
    bool midDown;					///< 紀錄滑鼠中鍵是否被按住。
    bool rmbDown;					///< 紀錄滑鼠左鍵是否被按住。
	glm::vec2 lmbDownCoord;			///< 紀錄滑鼠左鍵點擊時的座標。
	glm::vec2 midDownCoord;			///< 紀錄滑鼠中鍵點擊時的座標。
    glm::vec2 rmbDownCoord;			///< 紀錄滑鼠左鍵點擊時的座標。

    // The window
    int w_width;					// The width of the window
	int w_height;					// The height of the window
    // The viewport
    int v_ld_corner_x;              // The left-down corner of the current viewport
    int v_ld_corner_y;              // The left-down corner of the current viewport
    int v_width;                    // The viewport width
    int v_height;                   // The viewport height


	float wheel_val;				///< 紀錄滾輪的值。




    // Function pointer for _copy_func
    bool (*_cal_viewport)(
        int full_window_width, int full_window_height,
        int &ld_corner_x, int &ld_corner_y, int &viewport_width, int &viewport_height
    );
    // Note: static members are belong to class itself not the object
    static bool _default_cal_viewport(
        int full_window_width, int full_window_height,
        int &ld_corner_x, int &ld_corner_y, int &viewport_width, int &viewport_height
    ){
        ld_corner_x = 0;
        ld_corner_y = 0;
        viewport_width = full_window_width;
        viewport_height = full_window_height;
        return true;
    }
    //
    
};

#endif // _VIEWMANAGER_H_
