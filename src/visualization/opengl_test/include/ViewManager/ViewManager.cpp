#include "ViewManager.h"

using namespace glm;

/**
* 建立相機，並進行初始化。
*/
ViewManager::ViewManager():
    _cal_viewport(&_default_cal_viewport)
{
    // The window
    w_width       = 10;
    w_height      = 10;
    // The viewport
    v_ld_corner_x = 0;
    v_ld_corner_y = 0;
    v_width       = w_width;
    v_height      = w_height;


    // Flags
	ortho = false;
	zoom = 1.0f;
	moveSpeed = 10.0f;
	orthoScale = 1.5f;
	lmbDown = false;
	midDown = false;

	eyePosition = vec3(0.0f, 0.0f, 12.0f);
	eyeLookPosition = vec3(0.0f, 0.0f, 0.0f);
	vec3 up = vec3(0, 1, 0);
    /*
	eyePosition = vec3(-5.0f, 0.0f, 5.0f);
	eyeLookPosition = vec3(10.0f, 0.0f, 0.0f);
	vec3 up = vec3(1, 0, 0);
    */
	viewMatrix = lookAt(eyePosition, eyeLookPosition, up);
	viewVector = eyePosition - eyeLookPosition;
	viewVector = normalize(viewVector);
	Reset();
}

/**
* 取得Model Matrix。
*/
mat4 ViewManager::GetModelMatrix() {
	// return translationMatrix * rotationMatrix;
    // test
    // return tansformMatrix;
    return tansformMatrix*camera_model_inv;
}

/**
* 取得View Matrix。
*/
mat4 ViewManager::GetViewMatrix()
{
	return viewMatrix;
}

/**
* 設定並取得Projection Matrix。
* @param aspect 畫面的長寬比。
*/
mat4 ViewManager::GetProjectionMatrix(float aspect)
{
	float nearVal;
	float farVal;
	nearVal = 0.1f;
	farVal = 5000.0f; // 500.0f;
	if (ortho) {
		float size = orthoScale * zoom;
		projMatrix = glm::ortho(-aspect * size, aspect * size, -size, size, nearVal, farVal);
	}
	else {
		projMatrix = perspective(radians(30.0f * zoom), aspect, nearVal, farVal);
	}
	return projMatrix;
}

/**
* 取得Projection Matrix。
*/
mat4 ViewManager::GetProjectionMatrix()
{
	return GetProjectionMatrix(aspect);
}

/**
* 取得 V * P 的矩陣。
* @param aspect 畫面的長寬比。
*/
mat4 ViewManager::GetViewProjectionMatrix(float aspect)
{
	return GetProjectionMatrix(aspect) * viewMatrix;
}

/**
* 取得 M * V * P 的矩陣。
* @param aspect 畫面的長寬比。
*/
mat4 ViewManager::GetModelViewProjectionMatrix(float aspect)
{
	return GetViewProjectionMatrix(aspect) * GetModelMatrix();
}

/**
* 取得目前相機在世界座標的位置。
*/
vec3 ViewManager::GetWorldEyePosition() {
	vec4 wordEyePosition = vec4(eyePosition, 0) * GetModelMatrix();
	return wordEyePosition.xyz();
}

/**
* 取得目前相機在世界座標的所看的點位置。
*/
vec3 ViewManager::GetWorldViewVector() {
	vec4 wordLookVector = vec4(-viewVector, 0) * GetModelMatrix();
	return wordLookVector.xyz();
}

/**
* 處理當有按鍵輸入時，相機的動作。
* @param key 輸入的按鍵。
*/
void ViewManager::keyEvents(unsigned char key) {
	switch (key)
	{
		//向前移動。
	case 'w':
	case 'W':
		if (ortho) orthoScale += 0.1;
		else Translate(vec3(0, 0, 0.02));
		break;

		//向左移動。
	case 'a':
	case 'A':
		Translate(vec3(moveSpeed, 0, 0));
		break;

		//向後移動。
	case 's':
	case 'S':
		if (ortho) orthoScale -= 0.1;
		else Translate(vec3(0, 0, -0.02));
		break;

		//向右移動。
	case 'd':
	case 'D':
		Translate(vec3(-moveSpeed, 0, 0));
		break;

		//向上移動。
	case 'r':
	case 'R':
		Translate(vec3(0, moveSpeed, 0));
		break;

		//向下移動。
	case 'f':
	case 'F':
		Translate(vec3(0, -moveSpeed, 0));
		break;

		//放大。
	case '+':
		wheelEvent(-moveSpeed);
		break;

		//縮小。
	case '-':
		wheelEvent(moveSpeed);
		break;

		//重設相機旋轉。
	case 'z':
	case 'Z':
		Reset();
		break;

	default:
		break;
	}
}

/**
* 處理當有滑鼠事件時，相機的動作。
* @param button 輸入的按鍵。
* @param state 按鍵的狀態,點下,彈起。
* @param x 輸入的時, 滑鼠在畫面的x座標值。
* @param y 輸入的時, 滑鼠在畫面的y座標值。
*/
void ViewManager::mouseEvents(int button, int state, int x_cv_g, int y_cv_g) {

	if (state == GLUT_UP)
	{
		mouseReleaseEvent(button, x_cv_g, y_cv_g);
	}
	else if (state == GLUT_DOWN)
	{
		mousePressEvent(button, x_cv_g, y_cv_g);
	}

	//處理滑鼠中鍵向上滾動時
	if (button == 4)
	{
		wheelEvent(1);
	}
	//處理滑鼠中鍵向下滾動時
	else if (button == 3)
	{
		wheelEvent(-1);
	}
}

/**
* 處理當滑鼠按鍵點下時的動作。
* @param button 滑鼠的按鍵。
* @param x 輸入的時, 滑鼠在畫面的x座標值。
* @param y 輸入的時, 滑鼠在畫面的y座標值。
*/
void ViewManager::mousePressEvent(int button, int x_cv_g, int y_cv_g)
{
    // std::cout << "(x_cv_g, y_cv_g) = " << x_cv_g << ", " << y_cv_g << "\n";
    int x_cv_l, y_cv_l;
    convert_global_cv_coor_to_local_cv_coor(x_cv_g, y_cv_g, x_cv_l, y_cv_l);
    // Note: the (x_cv_l,y_cv_l) in mouse is using the "image-coordinate", not the "opengl-coordinate"
    if (is_mouse_out_of_bound(x_cv_l, y_cv_l)){
        return;
    }

	if (button == GLUT_LEFT_BUTTON)
	{
		//紀錄現在左鍵被按住
		lmbDown = true;
		lmbDownCoord = vec2(x_cv_l, y_cv_l);
		// mat4 invrtRot = inverse(rotationMatrix);
        // test
        // mat4 invrtRot = inverse(tansformMatrix);
		// rotateYAxis = (invrtRot * vec4(0, 1, 0, 0)).xyz();
		// rotateXAxis = (invrtRot * vec4(1, 0, 0, 0)).xyz();
	}
	if (button == GLUT_MIDDLE_BUTTON)
	{
		//紀錄現在中鍵被按住
		midDown = true;
		midDownCoord = vec2(x_cv_l, y_cv_l);
	}
    if (button == GLUT_RIGHT_BUTTON)
	{
		//紀錄現在左鍵被按住
		rmbDown = true;
		rmbDownCoord = vec2(x_cv_l, y_cv_l);
	}
}

/**
* 處理當滑鼠按鍵彈起時的動作。
* @param button 滑鼠的按鍵。
* @param x 輸入的時, 滑鼠在畫面的x座標值。
* @param y 輸入的時, 滑鼠在畫面的y座標值。
*/
void ViewManager::mouseReleaseEvent(int button, int x_cv_g, int y_cv_g)
{
    /*
    int x_cv_l, y_cv_l;
    convert_global_cv_coor_to_local_cv_coor(x_cv_g, y_cv_g, x_cv_l, y_cv_l);
    */

	if (button == GLUT_LEFT_BUTTON)
	{
		lmbDown = false;
	}
    if (button == GLUT_MIDDLE_BUTTON || button == 3 || button == 4) {
		midDown = false;
	}
    if (button == GLUT_RIGHT_BUTTON)
	{
		rmbDown = false;
	}
}

/**
* 處理當滑鼠移動時的動作。
* @param x 滑鼠在畫面的x座標值。
* @param y 滑鼠在畫面的y座標值。
*/
void ViewManager::mouseMoveEvent(int x_cv_g, int y_cv_g)
{
    int x_cv_l, y_cv_l;
    convert_global_cv_coor_to_local_cv_coor(x_cv_g, y_cv_g, x_cv_l, y_cv_l);

	if (lmbDown)
	{
		/*
		* 當滑鼠左鍵按住,進行拖曳時的時候
		* 計算移動的向量,進行相機的旋轉
		*/
		vec2 coord = vec2(x_cv_l, y_cv_l);
		vec2 diff = coord - lmbDownCoord;
		float factor = 0.002f;
		// vec3 rotateAxis = (diff.x_cv_l * factor)*rotateYAxis + (diff.y * factor)*rotateXAxis;
        vec3 rotateAxis = (diff.x * factor)*vec3(0,1,0) + (diff.y * factor)*vec3(1,0,0);
        double _n = l2Norm(rotateAxis);
		// rotateAxis = normalize(rotateAxis); // <-- no need to do this, since the rotate will ignore the norm of the axis
		// rotationMatrix = rotate(rotationMatrix, float(1.0f*_n), rotateAxis);
        // test
        glm::mat4 _delta_rot(1.0);
        _delta_rot = rotate(_delta_rot, float(1.0f*_n), rotateAxis);
        tansformMatrix = _delta_rot*tansformMatrix;
        //
		lmbDownCoord = coord;
	}
	else if (midDown)
	{
		vec2 coord = vec2(x_cv_l, y_cv_l);
		vec2 diff = coord - midDownCoord;

		vec4 up = vec4(0, 1, 0, 0);
		vec4 right = vec4(1, 0, 0, 0);

        //
		vec3 diffUp = up.xyz() * diff.y / (float)v_height;
		vec3 diffRight = right.xyz() * diff.x / (float)v_width;

        // Method 1: Fix delta_trans
        // glm::vec3 delta_trans = (-diffUp + diffRight) * zoom * 3.0f;
        // Method 2: Variable delta_trans
        glm::vec3 trans_world_at_cam = get_trans_world_at_camera();
        glm::vec3 delta_trans = (-diffUp + diffRight) * zoom * 3.0f * ( abs( trans_world_at_cam[2] - 12.0f ) + 0.2f)*0.3f;

		// translationMatrix = translate(translationMatrix, (-diffUp + diffRight) * zoom * 3.0f);
        // test
        glm::mat4 _delta_trans_M = translate(glm::mat4(1.0), delta_trans);
        tansformMatrix = _delta_trans_M*tansformMatrix;
        //
		midDownCoord = coord;
	}else if (rmbDown){
        vec2 coord = vec2(x_cv_l, y_cv_l);
        // std::cout << "coord = " << coord.x << ", " << coord.y << "\n";
		vec2 diff = coord - rmbDownCoord;
        //
        vec2 _central = vec2(v_width/2, v_height/2) + glm::vec2(v_ld_corner_x,v_ld_corner_y);
        float factor = 0.002f;
        //
        vec3 _axis = cross(vec3(diff,0)*factor, vec3(coord - _central,0)*factor);
        double _n = l2Norm(_axis);
        if (_n < 0.000001){
            _axis = vec3(0,0,1);
            // std::cout << "The axis is too short.\n";
        }
        // test
        glm::mat4 _delta_rot(1.0);
        _delta_rot = rotate(_delta_rot, float(1.0f*_n), _axis);
        tansformMatrix = _delta_rot*tansformMatrix;
        //
		rmbDownCoord = coord;
    }
}

/**
* 根據中鍵的滾動方向處理事件。
* @param direction 前滾,後滾。
*/
void ViewManager::wheelEvent(int direction)
{
	wheel_val = direction * 15.0f;
	// Zoom(wheel_val / 120.0f);

    // Method 1: Fix delta trans
    // glm::vec3 delta_trans = (-wheel_val/120.0f)*vec3(0,0,1);

    // Method 2: variable delta_trans accroding to distance between world and camera (farther get faster)
    glm::vec3 trans_world_at_cam = get_trans_world_at_camera();
    glm::vec3 delta_trans = (-wheel_val/120.0f)*vec3(0,0,1) * ( abs( trans_world_at_cam[2] - 12.0f ) + 0.2f)*0.3f;
    // test
    glm::mat4 _delta_trans_M = translate(glm::mat4(1.0), delta_trans);
    tansformMatrix = _delta_trans_M*tansformMatrix;
}

/**
* 根據輸入的值,調整目前的縮放值。
* @param distance 增加的值。
*/
void ViewManager::Zoom(float distance)
{
	zoom *= (1.0f + 0.05f * distance);
	zoom = clamp(0.1f, zoom, 3.0f);
	// zoom = clamp(0.001f, zoom, 30.0f);
}

/**
* 告訴相機現在的螢幕大小。
* @param width 螢幕的寬。
* @param height 螢幕的高。
*/
void ViewManager::SetWindowSize(int width, int height) {
    /*
	v_width = width;
	v_height = height;
    */
    w_width = width;
    w_height = height;
    _cal_viewport(width, height, v_ld_corner_x, v_ld_corner_y, v_width, v_height);

    aspect = v_width * 1.0/v_height;
	projMatrix = GetProjectionMatrix();
}
void ViewManager::SetWindowSize(int ld_corner_x, int ld_corner_y, int viewport_width, int viewport_height, int full_window_width, int full_window_height){
    v_ld_corner_x = ld_corner_x;
    v_ld_corner_y = ld_corner_y;
    v_width = viewport_width;
	v_height = viewport_height;
    w_width = full_window_width;
    w_height = full_window_height;
    aspect = v_width * 1.0/v_height;
	projMatrix = GetProjectionMatrix();
}
void ViewManager::SwitchGLViewPortAndCleanDraw(){
    // Switch viewport
    glViewport(v_ld_corner_x, v_ld_corner_y, v_width, v_height);
    // Clean draws

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

}

/**
* 設定相機的縮放。
* @param value 縮放。
*/
void ViewManager::SetZoom(float value)
{
	zoom = value;
}

/**
* 以尤拉角設定相機的旋轉。
* @param theta 尤拉角的theta旋轉。
* @param phi 尤拉角的phi旋轉。
*/
void ViewManager::SetRotation(float theta, float phi)
{
	rotationMatrix = mat4(1.0);
	rotationMatrix = rotate(rotationMatrix, theta, vec3(0, 1, 0));
	rotationMatrix = rotate(rotationMatrix, phi, vec3(1, 0, 0));
    // test
    tansformMatrix = rotationMatrix;
}

/**
* 以尤拉角設定相機的旋轉。
* @param x 尤拉角的x值。
* @param y 尤拉角的y值。
* @param z 尤拉角的z值。
*/
void ViewManager::SetRotation(float x, float y, float z)
{
	vec3 v(x, y, z);
	v = normalize(v);
	vec3 o(0, 0, 1);
	double angle = acos(dot(v, o));

    // Pre-set rotation
	rotationMatrix = mat4(1.0);
	rotationMatrix = rotate(rotationMatrix, (float)angle, cross(o, v));
    //
    // test
    glm::mat4 _delta_rot(1.0);
    _delta_rot = rotate(_delta_rot, (float)angle, cross(o, v));
    tansformMatrix = _delta_rot*tansformMatrix;
}


void ViewManager::SetCameraModel(glm::mat4 camera_model_in){
    camera_model_inv = inverse(camera_model_in);
}
void ViewManager::SetInvCameraModel(glm::mat4 camera_model_inv_in){
    camera_model_inv = camera_model_inv_in;
}

/**
* 重設相機的設定。
*/
void ViewManager::Reset()
{
	wheel_val = 0.0f;
	zoom = 3.0f;
	translationMatrix = mat4(1.0);
	rotationMatrix = mat4(1.0);
    rotationMatrix = rotate(rotationMatrix, deg2rad(90.0f), vec3(0.0f, 0.0f, 1.0f)); // z-axis
    rotationMatrix = rotate(rotationMatrix, deg2rad(75.0f), vec3(0.0f, 1.0f, 0.0f)); // y-axis
    // test
    // tansformMatrix = mat4(1.0);
    tansformMatrix = rotationMatrix*mat4(1.0);
    //
    camera_model_inv = mat4(1.0);
    // test
    // camera_model_inv = inverse(translate(glm::mat4(1.0), glm::vec3(69,-1396,6.67)));
}

/**
* 使相機移動。
* @param vec 使相機移動vec單位。
*/
void ViewManager::Translate(vec3 vec) {
	vec3 diff = vec;

	vec4 up = vec4(0, 1, 0, 0);
	vec4 right = vec4(1, 0, 0, 0);
	vec4 forward = vec4(0, 0, 1, 0);

	vec3 diffUp = up.xyz() * diff.y / (float)v_height;
	vec3 diffRight = right.xyz() * diff.x / (float)v_width;
	vec3 diffForward = forward.xyz() * diff.z;

	translationMatrix = translate(translationMatrix, (-diffUp + diffRight + diffForward) * zoom * 3.0f);
    // test
    glm::mat4 _delta_trans_M(1.0);
    _delta_trans_M = translate(_delta_trans_M, (-diffUp + diffRight + diffForward) * zoom * 3.0f);
    tansformMatrix = _delta_trans_M*tansformMatrix;
}




bool ViewManager::is_mouse_out_of_bound(int x_cv_l, int y_cv_l){
    int x_gl = x_cv_l;
    int y_gl = v_height - y_cv_l;
    if (x_gl < 0 || y_gl < 0 || x_gl >= v_width || y_gl >= v_height){
        // std::cout << "Out of bound.\n";
        return true;
    }
    return false;
}
void ViewManager::convert_global_cv_coor_to_local_cv_coor(int x_cv_g, int y_cv_g, int &x_cv_l, int & y_cv_l){
    x_cv_l = x_cv_g - v_ld_corner_x;
    y_cv_l = y_cv_g - (w_height - (v_ld_corner_y + v_height));
}

// camera pose on view
glm::vec3 ViewManager::get_trans_world_at_camera(){
    return tansformMatrix[3].xyz();
}
