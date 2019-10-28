#include <gflags/gflags.h>
#include <gflags/gflags_gflags.h>
#include <glog/logging.h>
#include "TegraAGrabber.h"
#include "TegraBGrabber.h"
#include "Util/ProgramArguments.hpp"
#include "grabber_args_parser.h"

int main(int argc, char** argv)
{

	gflags::ParseCommandLineFlags(&argc, &argv, false);
	cv::setNumThreads(0);

	// mode selection
	auto mode = SensingSubSystem::get_mode();

	if (mode == "a")
	{
		ros::init(argc, argv, "camera_a_grabber");
		printf("Running Camera a grabber\n");
		SensingSubSystem::TegraAGrabber app;
		app.initializeModules();
		return app.runPerception();
	}
	else if (mode == "b")
	{
		ros::init(argc, argv, "camera_b_grabber");
		printf("Running Camera b grabber\n");
		SensingSubSystem::TegraBGrabber app;
		app.initializeModules();
		return app.runPerception();
	}
	else
	{
		LOG(WARNING) << "Unknown or unsupported mode: " << mode;
		return 0;
	}
}
