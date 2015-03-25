#include<cv.h>
#include<iostream>
using namespace std;
using namespace cv;
Vec3b get_color(float res)
{
   float red;
   float blue;
   float green ;
   if(res<-90)
   {
      red=0;
      green=0;
      blue=0;    
   }
   else if(res<0.01)
   {
      red=255;
      green=0;
      blue=0;
   }
   else if(res<0.2)
   {
     red=172;
     green=103;
     blue=33;
   }
   else if(res<0.5)
   {
      red=0;
      green=0;
      blue=0; 
   }
   else
   {
     red=0;
     green=0;
     blue=0;
   }
//   float mil=1600000;
//   float val=res*mil;
//   float red = (((float)(-res) *100)+(200*(-res)) ;
//   float blue = (((float)(mil-val)/mil) *100)+(200*val/mil);
//   float green = (((float)(val)/mil) *70)+(200*val/mil);
   return Vec3b(blue,green,red);
}
void draw(Mat classified_image,vector<Point>superpixel,float response,float response_back)
{
   Vec3b temp=get_color(response);
//   cout<<response-response_back<<endl;
//   if(response<=(float)-90)
//   {
//     temp=Vec3b(0,0,0);
//   }
//   else if(response<0.2)
//   {
//    temp=Vec3b(0,0,255); 
//   }  
//   else 
//   {
//     temp=Vec3b(0,255,0); 
//   }
   for(int  xx=0; xx<superpixel.size();xx++)
   {
       int x=superpixel[xx].x;
       int y=superpixel[xx].y;
       classified_image.at<Vec3b>(x,y)=temp;
   }
}
