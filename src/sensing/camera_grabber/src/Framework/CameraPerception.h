#ifndef __CameraPerception__
#define __CameraPerception__

#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "../Framework/PerceptionOutput.h"

namespace SensingSubSystem {
    class CameraPerception {
        public:

            CameraPerception();
            CameraPerception(const std::string moduleName);
            virtual ~CameraPerception();

            virtual void initialize();
            virtual void run(const std::vector<void *>& raws_ptr);
            virtual void run(std::vector<cv::Mat>& images);
            virtual void display(std::vector<cv::Mat>& images);
            virtual void release();

            //info from imu, vehicle
            static float v_speed;
            static float v_steeringAngle;
            static float imu_angularZ;
            static float imu_orientationX;
            static float imu_orientationY;
            static float imu_orientationZ;

        protected:
            bool DEBUG_SHOW = true;
            std::string moduleName_;




    };

}

#endif

