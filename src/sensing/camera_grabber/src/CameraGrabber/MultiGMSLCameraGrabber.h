#ifndef __MULTIGMSLCAMERAGRABBER__
#define __MULTIGMSLCAMERAGRABBER__

#include <memory>
#include <vector>
#include <string>

#define MAX_PORTS_COUNT 4
#define MAX_CAMS 16

namespace SensingSubSystem
{
    class MultiGMSLCameraGrabber
    {
        private:
            class PIML_GRABBER;
            std::unique_ptr<PIML_GRABBER> piml;
        public:
            MultiGMSLCameraGrabber(const std::string& selected, const std::string& slave = "0");
            ~MultiGMSLCameraGrabber();

            void initializeCameras();
            void retrieveNextFrame();
            void returnCameraFrame();

            //Expose pointers to get current frame data (modify these pointer will cause undefined behavior)
            const void* getCurrentFrameData(uint id);
            
            static const uint32_t W  = 1920;
            static const uint32_t H = 1208;
            static const uint32_t S = 4;  //RGBA uint8
            static const uint32_t ImageSize = 1920 * 1208 * 4 * sizeof(uint8_t);  //RGBA uint8
    };
}

#endif
