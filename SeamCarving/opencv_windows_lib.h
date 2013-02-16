#ifndef opencv_windows_lib_h
#define opencv_windows_lib_h

#ifdef _DEBUG
#pragma comment( lib, "opencv_legacy242d.lib" )
#pragma comment( lib, "opencv_ml242d.lib" )
#pragma comment( lib, "opencv_core242d.lib" )
#pragma comment( lib, "opencv_features2d242d.lib" )
#pragma comment( lib, "opencv_objdetect242d.lib" )
#pragma comment( lib, "opencv_flann242d.lib" )
#pragma comment( lib, "opencv_stitching242d.lib" )
#pragma comment( lib, "opencv_ts242d.lib" )
#pragma comment( lib, "opencv_highgui242d.lib" )
#pragma comment( lib, "opencv_video242d.lib" )
#pragma comment( lib, "opencv_imgproc242d.lib" )
#pragma comment( lib, "opencv_videostab242d.lib" )
#else
#pragma comment( lib, "opencv_ml242.lib" )
#pragma comment( lib, "opencv_contrib242.lib" )
#pragma comment( lib, "opencv_nonfree242.lib" )
#pragma comment( lib, "opencv_core242.lib" )
#pragma comment( lib, "opencv_objdetect242.lib" )
#pragma comment( lib, "opencv_features2d242.lib" )
#pragma comment( lib, "opencv_photo242.lib" )
#pragma comment( lib, "opencv_flann242.lib" )
#pragma comment( lib, "opencv_stitching242.lib" )
#pragma comment( lib, "opencv_ts242.lib" )
#pragma comment( lib, "opencv_highgui242.lib" )
#pragma comment( lib, "opencv_video242.lib" )
#pragma comment( lib, "opencv_imgproc242.lib" )
#pragma comment( lib, "opencv_videostab242.lib" )
#pragma comment( lib, "opencv_legacy242.lib" )
#endif

#endif