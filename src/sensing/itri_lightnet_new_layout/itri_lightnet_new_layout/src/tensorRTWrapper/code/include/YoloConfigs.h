#ifndef _YOLO_CONFIGS_H_
#define _YOLO_CONFIGS_H_


namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.5f;
    static constexpr int CLASS_NUM = 7;

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };
	//YOLO 608*384 DivU
     YoloKernel yolo1 = {
         19,
         12,
         {116,90,  156,198,  373,326}
     };
     YoloKernel yolo2 = {
         38,
         24,
         {30,61,  62,45,  59,119}
     };
     YoloKernel yolo3 = {
         76,
         48,
         {10,13,  16,30,  33,23}
     };

    //YOLO 608
    //YoloKernel yolo1 = {
    //    19,
    //    19,
    //    {116,90,  156,198,  373,326}
    //};
    //YoloKernel yolo2 = {
    //    38,
     //   38,
    //    {30,61,  62,45,  59,119}
    //};
    //YoloKernel yolo3 = {
    //    76,
    //    76,
     //   {10,13,  16,30,  33,23}
    //};

    //YOLO 416
    // YoloKernel yolo1 = {
    //     13,
    //     13,
    //     {116,90,  156,198,  373,326}
    // };
    // YoloKernel yolo2 = {
    //     26,
    //     26,
    //     {30,61,  62,45,  59,119}
    // };
    // YoloKernel yolo3 = {
    //     52,
    //     52,
    //     {10,13,  16,30,  33,23}
    // };
/*
    //YOLO 384 * 608
     YoloKernel yolo1 = {
         19,
         12,
         {116,90,  156,198,  373,326}
     };
     YoloKernel yolo2 = {
         38,
         24,
         {30,61,  62,45,  59,119}
     };
     YoloKernel yolo3 = {
         76,
         48,
         {10,13,  16,30,  33,23}
     }; 
*/
    //YOLO 352 * 704
    // YoloKernel yolo1 = {
    //     22,
    //     11,
    //     {116,90,  156,198,  373,326}
    // };
    // YoloKernel yolo2 = {
    //     44,
    //     22,
    //     {30,61,  62,45,  59,119}
    // };
    // YoloKernel yolo3 = {
    //     88,
    //     44,
    //     {10,13,  16,30,  33,23}
    // }; 
}

#endif
