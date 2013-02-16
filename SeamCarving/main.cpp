#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <iostream>
#include <string>
#include <vector>

#include "opencv_windows_lib.h"

#include<algorithm>
#include<fstream>

using namespace std;


void drawCarvePath( cv::Mat& src, vector<int>& paths, int cur_itr );

//戻り値はcur_itr+1
//入力と出力は同じ
int carve( cv::Mat& src, vector<int>& paths, int cur_itr, cv::Mat& path_pix ){
	const int width = src.cols;
	const int height = src.rows;

	uchar* path_pix_data = path_pix.ptr(cur_itr);

	for( int i=0; i<height; i++ ){
		uchar* src_data = src.ptr(i);
		int cur_path = paths[cur_itr*height+i];
		memmove( src_data+3*cur_path, src_data+3*cur_path+3, 3*(width-cur_path));
		/*
		if( src_data[3*cur_path] != path_pix_data[3*i] ){
		cout<< "failed" <<endl;
		}
		*/
	}
	cout<< "width,height:" << width << "," << height <<endl;
	src = cv::Mat(src, cv::Rect(0,0, width-1, height));

	return cur_itr+1;
}

//戻り値はcur_itr-1
int uncarve( cv::Mat& src, vector<int>& paths, cv::Mat& path_pix, int cur_itr, cv::Mat& dst ){
	const int width = src.cols;
	const int height = src.rows;

	dst.create(src.rows, src.cols+1, CV_8UC3);

	uchar* path_data = path_pix.ptr(cur_itr);
	uchar test_data[3] = {255,0,0};
	//ofstream ofs("uncarve.txt");
	cout<< "cur_itr:" << cur_itr << ", path_pix_height:" << path_pix.rows << endl; 
	for( int i=0; i<height; i++ ){
		uchar* src_data = src.ptr(i);
		uchar* dst_data = dst.ptr(i);
		int cur_path = paths[cur_itr*height+i];
		//ofs<< "["<<i<<"]" << cur_path<<endl;
		memcpy( dst_data, src_data, 3*cur_path);
		//cout<< "["<<i<<"]start:" << cur_path << ":" <<flush;
		memcpy( dst_data+3*cur_path+3, src_data+3*cur_path, 3*(width-cur_path));
		//cout<< "end" <<endl;
		memcpy( dst_data+3*cur_path, path_data+3*i, 3);
		//memcpy( dst_data+3*cur_path, test_data, 3);
	}
	return cur_itr-1;
}

//入力に必要なもの：画像、削るピクセル数
//出力すべきもの：パス、パスに対応するピクセル値
void genCarvingData( const cv::Mat& src, int itr_num, std::vector<int>& paths, cv::Mat& path_pix )
{
	const int src_width = src.cols;
	const int src_height = src.rows;

	paths.clear();
	paths.resize(itr_num*src_height);
	path_pix.create( itr_num, src_height, CV_8UC3);//値の連続性を考えてcolsにheightを設定
	//cv::Mat path_pix( itr_num, src_height, CV_8UC3);//値の連続性を考えてcolsにheightを設定

	cv::Mat orig_image = src.clone();
	cv::Mat gray,edge;

	for( int n=0; n<itr_num; n++){
		cout<< "current iteration times is " << n <<endl;

		cv::cvtColor(orig_image, gray, CV_BGR2GRAY );
		cv::Laplacian(gray, edge, 0);
		//cv::Sobel(gray,edge,CV_8UC1,1,0,-1);

		cv::imshow("edge",edge);
		cv::waitKey(5);

		const int width = orig_image.cols;
		const int height = orig_image.rows;
		vector<int> table(height*width, 0);
		vector<int> path(height*width, 0);

		uchar* edge_data = edge.ptr(0);
		for( int i=0; i<width; i++ )
			table[i] = edge_data[i];

		cv::Mat path_image(height, width,CV_8UC3);

		for( int i=1; i<height; i++){
			uchar* edge_data0 = edge.ptr(i-1);
			uchar* edge_data1 = edge.ptr(i);

			uchar* path_data = path_image.ptr(i);

			for( int h=0; h<width; h++ ){
				int id = -1;
				int min = INT_MAX;

				const int dir[3] = {-1,1,0};
				for( int g=0; g<3; g++){
					int x = h + dir[g];

					if(x< 0 || width <= x) continue;
					//*
					if( edge_data0[x] <= min ){
						min = edge_data0[x];
						id = dir[g];
					}
					/*/
					if( table[i*height+x] < min ){
					min = table[i*+x];
					id = g;
					}
					*/
				}
				const uchar r_pix[3] = {0,0,255};
				const uchar g_pix[3] = {0,255,0};
				const uchar b_pix[3] = {255,0,0};
				if( id == -1 ) memcpy(path_data+3*h, r_pix, 3);
				if( id == 0 ) memcpy(path_data+3*h, g_pix, 3);
				if( id == 1 ) memcpy(path_data+3*h, b_pix, 3);

				path[width*i+h] = id;
				table[width*i+h] = table[width*(i-1)+(h+id)] + edge_data1[h];
			}

		}
		cv::imshow( "path", path_image);
		cv::waitKey(5);

		int x_id = 0;
		int min_val = INT_MAX;
		for( int x=0; x<width; x++ ){
			int cur_val = table[(height-1)*width+x];
			if( min_val > cur_val ){
				min_val = cur_val;
				x_id = x;
			}
		}

		paths[(n*height)+height-1] = x_id;
		uchar *path_pix_data = path_pix.ptr(n);
		memcpy( path_pix_data+3*(height-1), orig_image.ptr(height-1)+3*x_id, 3);
		//cout<< "[" << x_id << "," << height-1 << "]" <<flush;
		for( int i=height-2; i>=0; i--){
			x_id = x_id + path[i*width+x_id];
			paths[(n*height)+i] = x_id;
			memcpy( path_pix_data+3*(i), orig_image.ptr(i)+3*x_id, 3);
			//cout<< "[" << x_id << "," << i-1 << "]" <<flush;
		}

		/*
		drawCarvePath( orig_image, paths, n );
		cv::imshow( "DEBUG:Carve path", orig_image );
		*/
		carve( orig_image, paths, n, path_pix);
		/*
		cv::imshow("pathpix", path_pix);
		cv::imshow( "DEBUG:Carving", orig_image );
		cv::waitKey(10);
		*/
	}
}

void drawCarvePath( cv::Mat& src, vector<int>& paths, int cur_itr )
{	
	const int width = src.cols;
	const int height = src.rows;

	uchar replace_pix[3] = {0,0,255};
	//ofstream ofs("draw.txt");
	//cout<<endl;
	for( int i=0; i<height; i++ ){
		uchar* src_data = src.ptr(i);
		int cur_path = paths[cur_itr*height+i];
		//cout<< "["<< cur_path << "," << i<<"]" <<flush;
		memcpy(src_data+3*cur_path, replace_pix,3);
	}
	//cout<<"aa" <<endl;
}

//srcは元画像
void genCarvingMap(const cv::Mat& src, vector<int>&paths, int itr_num, cv::Mat& path_pix, cv::Mat& dst )
{

	cv::Mat orig_image = src.clone();
	for( int i=0; i<itr_num; i++ ){
		carve( orig_image, paths, i, path_pix );
	}
	cv::Mat carve_map(orig_image.size(), CV_32SC1);
	carve_map.setTo(INT_MAX);
	const int height = carve_map.rows;
	for( int n=itr_num-1; n>=0; n-- ){

		uchar* path_data = path_pix.ptr(n);
		uchar test_data[3] = {255,0,0};
		//ofstream ofs("uncarve.txt");
		const int width = carve_map.cols;
		cv::Mat tmp_map( height, width+1, CV_32SC1 );
		tmp_map.setTo(INT_MAX);
		for( int i=0; i<height; i++ ){
			int* src_data = (int*)carve_map.ptr(i);
			int* dst_data = (int*)tmp_map.ptr(i);
			int cur_path = paths[n*height+i];

			memcpy( dst_data, src_data, sizeof(int)*cur_path);
			memcpy( dst_data+cur_path+1, src_data+cur_path, sizeof(int)*(width-cur_path));
			dst_data[cur_path] = n;

			//memcpy( dst_data+3*cur_path, test_data, 3);
		}
		carve_map = tmp_map;
		cv::imshow( "carve_map", carve_map );
		cv::waitKey(10);
	}

	dst = carve_map;
}

void carveFromMap( const cv::Mat& src, const cv::Mat& map, int red_level, cv::Mat& dst )
{

	assert( map.type() == CV_32SC1 );
	const int width = src.cols;
	const int height = src.rows;
	dst.create( height, width, CV_8UC3 );
	int carved_width = 0;
	for( int i=0; i<height; i++ ){
		const int* map_data = (int*)map.ptr(i);
		const uchar* src_data = src.ptr(i);
		uchar* dst_data = dst.ptr(i);
		int red_count = 0;
		for( int h=0; h<width; h++ ){
			int carve_val = map_data[h];
			if( carve_val == UCHAR_MAX || carve_val >= red_level ){
				memcpy( dst_data+3*(h-red_count), src_data+3*h, 3 );
			}
			else if( carve_val < red_level ){
				red_count++;
			}
		}
		if( carved_width == 0){
			carved_width = width - red_count;
		}
		else {
			assert( carved_width == width - red_count);
		}
		
	}

	dst = cv::Mat( dst, cv::Rect(0,0,carved_width,height));
}

void matInt2Char( const cv::Mat& src, cv::Mat& dst )
{
	const int width = src.cols;
	const int height = src.rows;

	assert( src.type() == CV_32SC1 );
	cv::Mat tmp = src.clone();
	dst.create( src.size(), CV_8UC1 );
	for( int i=0; i<height; i++ ){
		int* src_data = (int*)tmp.ptr(i);
		uchar* dst_data = dst.ptr(i);
		for( int h=0; h<width; h++ ){
			int* src_pix = src_data + h;
			uchar* dst_pix = dst_data + h;

			if( src_pix[0] == INT_MAX ) dst_pix[0] = 255;
			else dst_pix[0] = (uchar)src_pix[0]%255;
		}
	}
}

struct IndexVal {
	long long int index;
	int val;
	IndexVal():index(0),val(0){}
	IndexVal( int index ):index(index){}
	bool operator<(const IndexVal& i){
		return val < i.val;
	}
};

void sortCarveMap( const cv::Mat&map, int itr_num, cv::Mat& sorted_map )
{
	const int width = map.cols;
	const int height = map.rows;

	assert( map.type() == CV_32SC1 );

	vector<IndexVal> pos;//Craveするタイミング
	for( int i=0; i<itr_num; i++ ){
		pos.push_back(IndexVal(i));
	}
	for( int i=0; i<height; i++ ){
		const int* map_data = (int*)map.ptr(i);
		for( int h=0; h<width; h++){
			const int* pix = map_data + h;

			if( pix[0] == INT_MAX ) continue;

			pos[pix[0]].val += h;
		}
	}

	for( int i=0; i<itr_num; i++ ){
		pos[i].val /= height;
	}

	sort( pos.begin(), pos.end());

	vector<int> transition;//元インデックス→遷移先インデックス
	transition.resize(itr_num);
	for( int i=0; i<itr_num; i++ ){
		transition[pos[i].index] = i;
		cout<< pos[i].index << "->" << i <<endl;
	}

	sorted_map = map.clone();
	for( int i=0; i<height; i++ ){
		const int* map_data = (int*)map.ptr(i);
		int* sorted_data = (int*)sorted_map.ptr(i);

		for( int h=0; h<width; h++){
			const int* map_pix = map_data + h;
			int* sorted_pix = sorted_data + h;

			if( map_pix[0] == INT_MAX ) continue;

			sorted_pix[0] = transition[map_pix[0]];
		}
	}

	cv::imshow("original map", map);
	cv::imshow("sorted map", sorted_map);
	cv::waitKey();

}

void mergeMapLevel( const cv::Mat& map, int itr_num, int level, cv::Mat& merged )
{
	assert( map.type() == CV_32SC1 );

	const int width = map.cols;
	const int height = map.rows;

	merged.create(map.size(), CV_32SC1);

	int unit = ((double)itr_num / level)+0.5;
	vector<int> transition;
	transition.resize(itr_num);
	for( int i=0; i<itr_num; i++ ){
		transition[i] = i/unit;
	}

	for( int i=0; i<height; i++ ){
		const int* map_data = (int*)map.ptr(i);
		int* merged_data = (int*)merged.ptr(i);

		for( int h=0; h<width; h++ ){
			const int* map_pix = map_data + h;
			int* merged_pix = merged_data + h;

			if( map_pix[0] == INT_MAX ) merged_pix[0] = INT_MAX;
			else {
				merged_pix[0] = transition[map_pix[0]];
			}
		}
	}
}


int main( int argc, char** argv ){
	//string fname = "C:\\Users\\u-ta\\Desktop\\neko.jpg";
	//string fname = "C:\\Users\\u-ta\\Desktop\\program_user_1_0.jpg";
	//string fname = "C:\\Users\\u-ta\\Desktop\\tuba.jpg";
	string fname = "C:\\Users\\u-ta\\Desktop\\DSC03239-2.jpg";
	//string fname = "C:\\Users\\u-ta\\Desktop\\DSC03239-2-short.jpg";
	int itr_num = 1000;


	vector<int> paths;
	const cv::Mat src = cv::imread(fname);


	cv::Mat path_pix;

	genCarvingData( src, itr_num, paths, path_pix);

	cv::imshow("path_pix", path_pix);
	cv::waitKey();


	cv::imshow("original", src);
	cv::waitKey();

	cv::Mat orig_image = src.clone();
	int wait_time = 10;
	for( int i=0; i<itr_num; i++ ){
		drawCarvePath(orig_image, paths,i);
		cv::imshow( "carve path", orig_image );

		carve( orig_image, paths, i, path_pix);
		cv::imshow( "carved", orig_image );
		int ikey = cv::waitKey(wait_time);
		if( ikey == 0x1b ) break;
	}
	cv::Mat dst;
	//wait_time = 0;

	for( int i=itr_num-1; i>=0; i-- ){
		//for( int i=0; i<itr_num; i++ ){
		cv::Mat path_image = orig_image.clone();
		drawCarvePath( path_image, paths, i );
		cv::imshow( "carve path", path_image );
		uncarve( orig_image, paths, path_pix, i, dst);
		cv::imshow( "uncarve", dst );
		cout<< "uncarve : " << i <<endl;
		int ikey = cv::waitKey(wait_time);
		if( ikey == 0x1b ) break;
		orig_image = dst.clone();
	}

	//検証
	for( int i=0; i<dst.rows; i++ ){
		uchar* dst_p = dst.ptr(i);
		const uchar* src_p = src.ptr(i);
		for( int h=0; h<dst.cols; h++){
			uchar* dst_pix = dst_p + 3*h;
			const uchar* src_pix = src_p + 3*h;
			for( int ch=0; ch<3; ch++ )
				assert( dst_pix[ch] == src_pix[ch] );
		}
	}
	cout<<"end" <<endl;
	cv::waitKey();

	cv::Mat carve_map;
	genCarvingMap( src, paths, itr_num, path_pix, carve_map );
	cv::imwrite("carving_map.png", carve_map );

	cv::Mat carve_from_map_image;

	cv::Mat tmp_map;
	//matInt2Char( carve_map, tmp_map );
	tmp_map = carve_map;
	for( int i=0; i<20; i++ ){
		carveFromMap( src, tmp_map, rand()%itr_num, carve_from_map_image );
		cv::imshow( "carve from map", carve_from_map_image );
		cv::waitKey(wait_time);
	}
	
	int merged_level = 254;
	cv::Mat merged_map;
	mergeMapLevel(carve_map, itr_num, merged_level, merged_map);
	matInt2Char( merged_map, tmp_map );
	cv::imwrite( "merged_map.png", tmp_map );
	for( int i=0; i<merged_level; i++ ){
		carveFromMap( src, merged_map, i, carve_from_map_image );
		cv::imshow( "carve from map", carve_from_map_image );
		cv::waitKey(wait_time);
	}

	cv::Mat sorted_map;
	sortCarveMap( carve_map, itr_num, sorted_map);
	tmp_map = sorted_map;
	cv::destroyAllWindows();

	int ikey;
	do {
		for( int i=0; i<itr_num; i++ ){
			carveFromMap( src, tmp_map, i, carve_from_map_image );
			cv::imshow( "carve from map", carve_from_map_image );
			cv::waitKey(wait_time);
		}
		for( int i=itr_num-1; i>=0; i-- ){
			carveFromMap( src, tmp_map, i, carve_from_map_image );
			cv::imshow( "carve from map", carve_from_map_image );
			cv::waitKey(wait_time);
		}
		ikey = cv::waitKey();
	}while(ikey != 0x1b );

	//cv::Mat merged_map;
	mergeMapLevel( sorted_map, itr_num, merged_level, merged_map );
	tmp_map = merged_map;
	//matInt2Char( merged_map, tmp_map );
	for( int i=0; i<merged_map.rows; i++ ){
		int* data = (int*)tmp_map.ptr(i);
		for( int h=0; h<merged_map.cols; h++ ){
			if( data[h] == INT_MAX ) continue;
			assert( data[h] % merged_level == data[h] );
		}
	}
	do {
		for( int i=0; i<merged_level; i++ ){
			carveFromMap( src, tmp_map, i, carve_from_map_image );
			cv::imshow( "carve from map", carve_from_map_image );
			cv::waitKey(wait_time);
		}
		for( int i=merged_level-1; i>=0; i-- ){
			carveFromMap( src, tmp_map, i, carve_from_map_image );
			cv::imshow( "carve from map", carve_from_map_image );
			cv::waitKey(wait_time);
		}
		ikey = cv::waitKey();
	}while(ikey != 0x1b );

	matInt2Char( merged_map, tmp_map );
	cv::imwrite( "sorted_merged_map.png", tmp_map );
}