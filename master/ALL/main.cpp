//@Author:NIRWAN DOGRA
//#include "stdafx.h"
#include<queue>
#include "GCoptimization.h"
#include "drawing.h"
#include "cv.h"
#include "highgui.h"
#include "ml.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include<cv.h>
#include<conio.h>
#include<iostream>
#include <opencv/cv.h>
#include "Conditional_Random_Field.h"
//#include "svm.h"
#define pi pair<int,int>//this is like1 a structure struct node{int x,y;}
#define pii pair<int, pi>//this is like a structure struct ndoe{int x,y,z;}
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>
#define N 4
//#define Super_rows 45
//#define Super_cols 45
#define Dictionary_size 400
#define Number_vocabulary_images 100
#define Number_training_images 50
#define inf 111111
using namespace std;
using namespace cv;
#include "slic.h"

//VARIABLES USED
static int dxc[4]= {0,0,-1,1};//this is the direcction array in x direction
static int dyc[4]= {1,-1,0,0};//this is the direcction array in y direction
TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
static int flag=0;int len[2222][2222];
static Mat img;
static vector<KeyPoint>keypoints;
static Mat descriptor;
static Mat features_unclustered;
static SiftDescriptorExtractor detector;
static Mat segment_image;
char* path_People="C:\\Users\\NIRWAN\\Desktop\\people\\";
char* test_image_path="C:\\Users\\NIRWAN\\Desktop\\people\\person_005.image.png";
char* path_Bikes="C:\\Users\\NIRWAN\\Desktop\\bikes\\";
char* path_Cars="C:\\Users\\NIRWAN\\Desktop\\cars\\";
static float prob[2222];
static Ptr<DescriptorMatcher>matcher(new FlannBasedMatcher);
static Ptr<DescriptorExtractor>extractor(new SiftDescriptorExtractor);

static BOWImgDescriptorExtractor bowde(extractor,matcher);
static BOWKMeansTrainer bowtrainer(Dictionary_size);

static vector<float>label;
static vector<float>label_back;
static Mat labels;
static Mat labels_back;
static Mat training_data(0,Dictionary_size,CV_32F);
static Mat training_data_back(0,Dictionary_size,CV_32F);
static vector<KeyPoint>keypoint1;
static Mat bowdescriptor1;
static Mat groundTruth(0,1,CV_32F);
static Mat evalData(0,Dictionary_size,CV_32F);
static int k=0;
static vector<KeyPoint>keypoint2;
static Mat bowdescriptor2;
static Mat results(0,1,CV_32F);
static CvSVM svm;
static CvSVM svm_back;
static int GR[2222][2222];
static int differ[2222][2222];
class ConditionalRandomField
{
public :
   struct ForDataFn
   {
      int numLab;
      int *data;
   };
//   int smoothFn(int p1, int p2, int l1, int l2)
//   {
//      int length=(float)len[p1][p2];
//      cout<<" Length "<<length<<endl;
//      int diff=1+differ[p1][p2];
//      cout<<" Diff "<<diff<<endl;
//      float xx=((float)length/(float)diff)*100;
//      cout<<xx<<" Smoothfun "<<endl;
//      return xx;
//   }
   vector<int> GeneralGraph_DArraySArray(int width,int height,int num_pixels,int num_labels)
   {
     // int *result = new int[num_pixels];   // stores result of optimization
      cout<<width<<" "<<height<<" "<<num_pixels<<" "<<num_labels <<endl;
      vector<int>result(num_pixels);
      // first set up the array for data costs
      int *data = new int[num_pixels*num_labels];
      for ( int i = 0; i < num_pixels; i++ )
      {
         for (int l = 0; l < num_labels; l++ )
         {
            int temp=prob[i]*1000;
            if(l%2==0)
            {
              data[i*num_labels+l] =temp;
            }
            else
            {
              data[i*num_labels+l] =-temp; 
            }
         }
      }
      // next set up the array for smooth costs
    int *smooth = new int[num_labels*num_labels];
	  smooth[0]=0;smooth[1]=0;smooth[2]=0;smooth[3]=0;
	  for(int p1=0;p1<num_pixels;p1++)
	  {
		  for(int p2=0;p2<num_pixels;p2++)
		   {
	         for ( int l1 = 0; l1 < num_labels; l1++ )
             {
              for (int l2 = 0; l2 < num_labels; l2++ )
               {
                  int length=len[p1][p2];
                  int differ=1+abs(prob[p1]-prob[p2]);
                  float val=((float)length/(float)differ);
                  smooth[l1+l2*num_labels] =smooth[l1+l2*num_labels]+val;
               }
		     }
		   }
	  }
      try
      {
         GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
         gc->setDataCost(data);
         gc->setSmoothCost(smooth);

//          now set up a grid neighborhood system
//          first set up horizontal neighbors
         for (int x = 0; x < num_pixels; x++ )
            {
              for (int  y = 0; y < num_pixels; y++ )
               {
                  if(GR[x][y]==1 && x!=y)
                  {
                      gc->setNeighbors(x,y);
                  } 
               }
            }
         printf("\nBefore optimization energy is %d",gc->compute_energy());
         gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
         printf("\nAfter optimization energy is %d",gc->compute_energy());

         for ( int  i = 0; i < num_pixels; i++ )
            result[i] = gc->whatLabel(i);

         delete gc;
         return result;
      }
      catch (GCException e)
      {
         e.Report();
      }
      delete [] smooth;
      delete [] data;
	  vector<int>xx;
	  return xx;
   }
};
static ConditionalRandomField crf;

static int get_class(Mat mask,int x1,int y1,int x2,int y2)
{
   if(x2>mask.rows || y2>mask.cols || x1>mask.rows || y1>mask.cols)
   {
      return 0;
   }
   int red=0;
   int black=0;
   for(int  i=x1; i<x2; i++)
   {
      for(int  j=y1; j<y2; j++)
      {
         Vec3b temp=mask.at<Vec3b>(i,j);
         int temp2=temp.val[0]+temp.val[2];
         if(temp2!=255)
         {
            black++;
         }
         else
         {
            red++;
         }
      }
   }
   int ret=0;
   Vec3b temp;
   if(red>=black)
   {
      ret=1;
      temp=Vec3b(0,0,255);
   }
   else
   {
      ret=0;
      temp=Vec3b(0,0,0);
   }
   return ret;
}
static void floyd_warshall(int nodes)
{
   cout<<" Computing Shortest Paths "<<endl;
   for(int  k=0; k<nodes; k++)
   {
      for(int  i=0; i<nodes; i++)
      {
	     if(GR[i][k]==111111 || GR[i][k]>N)
			 {
			   continue;
			 }
         for(int  j=0; j<nodes; j++)
         {
            GR[i][j]=min(GR[i][j],GR[i][k]+GR[k][j]);
         }
      }
   }
 cout<<"Shortest Paths Built "<<endl;
}
vector<Mat> make_descriptors(vector< vector<Point> > superpixel,Mat img)
{
  cout<<" Making Bowdescriptors "<<endl;
   vector<Mat> ret;
   vector<KeyPoint>keypoint1;
   Mat bowdescriptor1;
   //imshow("sf" , img);
   //while(waitKey()!=27);
   for(int  k=0; k<superpixel.size(); k++)
   {
      int x1=superpixel[k][0].x;
      int y1=superpixel[k][0].y;
      int x2=superpixel[k][1].x;
      int y2=superpixel[k][1].y;
      Mat newimg=Mat(x2-x1+1,y2-y1+1,0,Scalar(255,255,255));
      for(int l=2; l<superpixel[k].size(); l++)
      {
         int x=superpixel[k][l].x;
         int y=superpixel[k][l].y;
         newimg.at<uchar>(x-x1,y-y1)=img.at<uchar>(x,y);
      }
      //keypoint1.clear();
      detector.detect(newimg,keypoint1);
      bowde.compute(newimg,keypoint1,bowdescriptor1);
      // cout<<k<<" "<<endl;
      ret.push_back(bowdescriptor1);
   }
   for(int i=0;i<superpixel.size();i++)
	   {
	      int cnt=1;
	      for(int j=0;j<superpixel.size();j++)
			  {
			     if(i==j)
					 {
					   continue;
					 }
				 if(GR[i][j]<=N && ret[j].rows!=0 && ret[i].rows!=0)
					 {
					    if(ret[i].rows==0)
						{
							  ret[i]=ret[j];
						      continue;
						}
						ret[i]=ret[i]+ret[j];
						cnt++;
					 }
			  }
		  ret[i]=ret[i]/cnt;
	   }
   cout<<" GRAPH "<<endl;
   for(int i=0;i<40;i++)
	   {
	     for(int j=0;j<40;j++)
			 { 
			    cout<<GR[i][j]<<" ";
			 }
		 cout<<endl;
	   }
   cout<<endl;
   cout<<" LEAVING bowdescriptors "<<endl;
   return ret;
}
static vector<vector<Point> >  make_superpixels(Mat original)
{
   IplImage *image = new IplImage(original);
   IplImage *lab_image = cvCloneImage(image);
   cout<<" yoman "<<endl;
   //cvCvtColor(image, lab_image, CV_BGR2Lab);
   cout<<" yoman2 "<<endl;
   int w = image->width, h = image->height;
   int nr_superpixels = atoi("2000");
   int nc = atoi("60");
   double step = sqrt((w * h) / (double) nr_superpixels);
   Slic slic;
   slic.generate_superpixels(lab_image, step, nc);
   slic.create_connectivity(lab_image);
   Mat ret;
   /*cout<<" size "<<slic.centers.size()<<endl;*/
   vector<vector<Point> > contours(slic.centers.size());
   for(int  i=0; i<slic.centers.size()+5; i++)
   {
      for(int  j=0; j<slic.centers.size()+5; j++)
      {
         len[i][j]=0;
         differ[i][j]=0;
         if(i==j)
         {
            GR[i][j]=0;
            continue;
         }
         GR[i][j]=inf;
      }
   }
   cout<<" Graph Initialized "<<slic.clusters.size()<<endl;
   /*freopen("out.txt","w",stdout);
   for(int i=0; i<slic.clusters.size(); i++)
   {
      for(int j=0; j<slic.clusters[i].size(); j++)
      {
        cout<<slic.clusters[i][j]<<" ";
     }
     cout<<endl;
   }
   cout<<endl;*/
   for(int i=0; i<slic.clusters.size(); i++)
   {
      for(int j=0; j<slic.clusters[i].size(); j++)
      {
         int cluster=slic.clusters[i][j];
         if(cluster<0)
         {
            cluster=0;
         }
         if(contours[cluster].size()==0)
         {
            contours[cluster].push_back(Point(11111,11111));
            contours[cluster].push_back(Point(-1,-1));
         }
         // if(i==637){cout<<cluster<<" cluster leave "<<endl;}
         int xc=i;
         int yc=j;
         //if(i==637)cout<<" YES 1"<<endl;
         contours[cluster].push_back(Point(yc,xc));
         contours[cluster][0]=Point(min(contours[cluster][0].x,yc),min(contours[cluster][0].y,xc));
         contours[cluster][1]=Point(max(contours[cluster][1].x,yc),max(contours[cluster][1].y,xc));
         //if(i==637)cout<<" YES 2"<<endl;
         for(int  k=0; k<4; k++)
         {
            int newxc=xc+dxc[k];
            int newyc=yc+dyc[k];
            if(newxc<0 ||  newyc<0 || newxc>=slic.clusters.size() || newyc>=slic.clusters[i].size())
            {
               continue;
            }
            int newcluster=slic.clusters[newxc][newyc];
            if(newcluster<0)
            {
              newcluster=0;
            }
            GR[cluster][newcluster]=1;
            GR[newcluster][cluster]=1;
            len[cluster][newcluster]=len[cluster][newcluster]+1;
            //len[newcluster][cluster]=len[newcluster][cluster]+1;
         }
      }
      // cout<<endl;
   }
   floyd_warshall(slic.centers.size());
   cout<<" LEAVING "<<endl;
   return contours;
}
static void make_vocabulary()
{
   if(flag==1)
   {
      return ;
   }
   cout<<" MAKING VOCABULARY...."<<endl;
   for(int i=1; i<=20; i++)
   {
      cout<<" Reading File "<<i<<endl;
      stringstream ss;
      ss << path_People << "person_"<<setfill('0') << setw(3) << i <<".image.png";
      cout<<ss.str()<<endl;
      img=imread(ss.str(),0);
      Mat tempp=imread(ss.str(),1);
      //vector< vector<Point > > superpixel=make_superpixels(tempp);
      //cout<<superpixel.size()<<" Superpixel size "<<endl;
      for(int  k=0; k<1; k++)
      {
         /*   int x1=superpixel[k][0].x;
            int y1=superpixel[k][0].y;
            int x2=superpixel[k][1].x;
            int y2=superpixel[k][1].y;
            Mat newimg=Mat(x2-x1+1,y2-y1+1,0,Scalar(255,255,255));
            for(int l=2; l<superpixel[k].size(); l++)
            {
               int x=superpixel[k][l].x;
               int y=superpixel[k][l].y;
               newimg.at<uchar>(x-x1,y-y1)=img.at<uchar>(x,y);
            }*/
         keypoints.clear();
         detector.detect(img,keypoints);
         detector.compute(img,keypoints,descriptor);
         features_unclustered.push_back(descriptor);
      }
   }
   cout<<"VOCABULARY BUILT...."<<endl;
   cout<<endl;
}
static void cluster_vocabulary()
{
   if(flag==1)
   {
      Mat dictionary;
      FileStorage fs("vocabulary.xml", FileStorage::READ);
      fs["vocabulary"] >> dictionary;
      fs.release();
      Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
      //create Sift feature point extracter
      Ptr<FeatureDetector> detector(new SiftFeatureDetector());
//      create Sift descriptor extractor
      Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
//      create BoF (or BoW) descriptor extractor
      //    BOWImgDescriptorExtractor bowDE(extractor,matcher);
//      Set the dictionary with the vocabulary we created in the first step
      bowde.setVocabulary(dictionary);
      cout<<dictionary.rows<<" Dictionary Rows "<<endl;
      cout<<"Dictionary Loaded From Earlier Data ...."<<endl;
      return ;
   }
   cout<<" CLUSTERING VOCABULARY...."<<endl;
   Mat dictionary=bowtrainer.cluster(features_unclustered);
   bowde.setVocabulary(dictionary);
   cout<<" VOCABULARY CLUSTERED...."<<endl;
   cout<<" Vocabulary Saved To dictionary.yml"<<endl;
   FileStorage fs("dictionary.yml", FileStorage::WRITE);
   fs << "vocabulary" << dictionary;
   fs.release();
   cout<<endl;
}

static void compute_training_data()
{
   if(flag==1)
   {
      return ;
   }
   cout<<" COMPUTING THE TRAINING DATA"<<endl;
   for(int i=30; i<=50; i++)
   {
      cout<<" Reading File "<<i<<" In Training"<<endl;
      stringstream s2;
      s2 << path_People << "person_"<<setfill('0') << setw(3) << i <<".mask.0.png";
      Mat mask_image=imread(s2.str(),CV_LOAD_IMAGE_COLOR);
//      int xxx=1;
//      while(1)
//      {
//        stringstream s3;
//        s3 << path_Cars << "carsgraz_"<<setfill('0') << setw(3) << xxx <<".mask."<<xxx<<".png";
//        Mat t=imread(s3.str(),1);
//        //cout<<s3.str()<<endl;
//        if(!t.data)
//        {
//          break;
//        }
//        xxx++;
//        //cout<<"  inside " <<endl;
//        mask_image=mask_image+t;
//      }
     // imshow("fs",mask_image);
      //while(waitKey()!=27);
      stringstream ss;
      ss << path_People << "person_"<<setfill('0') << setw(3) << i <<".image.png";
      cout<<ss.str()<<endl;
      img=imread(ss.str(),0);
      Mat tempp=imread(ss.str(),1);
      //cout<<" CLassifij"<<endl;
      Mat classifiedimage=Mat::zeros(tempp.size(),tempp.type());
      vector< vector<Point > > superpixel=make_superpixels(tempp);
      vector<Mat> desc=make_descriptors(superpixel,img);
	    cout<<superpixel.size()<<" Superpixel size "<<endl;
      for(int  k=0; k<superpixel.size(); k++)
      {
         if(superpixel[k].size()<2)
         {
           continue;
         }
         int x1=superpixel[k][0].x;
         int y1=superpixel[k][0].y;
         int x2=superpixel[k][1].x;
         int y2=superpixel[k][1].y;
         Mat newimg=Mat(x2-x1+1,y2-y1+1,0,Scalar(255,255,255));
         for(int l=2; l<superpixel[k].size(); l++)
         {
            int x=superpixel[k][l].x;
            int y=superpixel[k][l].y;
            newimg.at<uchar>(x-x1,y-y1)=img.at<uchar>(x,y);
         }
         detector.detect(newimg,keypoint1);
         bowde.compute(newimg,keypoint1,bowdescriptor1);
         bowdescriptor1=desc[k];
		 if(bowdescriptor1.rows==0)
         {
            continue;
         }
         int clas=get_class(mask_image,x1,y1,x2,y2);
		     training_data.push_back(bowdescriptor1);
         training_data_back.push_back(bowdescriptor1);
         label.push_back((float)clas);
         label_back.push_back((float)(1-clas));
//		 if(clas==1)draw(classifiedimage,superpixel[k],Vec3b(0,0,255));
         //	 if(clas==0)draw(classifiedimage,superpixel[k],Vec3b(0,255,0));
      }
      //  imshow("yes",classifiedimage);
      //while(waitKey()!=27);
   }
   cout<<" SIZE OF TRAINING DATA AND LABELS -:"<<training_data.rows<<" "<<label.size()<<" "<<endl;
   Mat(label).copyTo(labels);
   Mat(label_back).copyTo(labels_back);
   cout<<"TRAINING DATA COMPUTED "<<endl;
   cout<<endl;
}
static void train_usingdata()
{
   if(flag==1)
   {
      svm.load("my_svm.yml");
      svm_back.load("my_svm.yml");
      return ;
   }
   cout<<"TRAINING USING THE TRAINING DATA "<<endl;
   CvSVMParams params;
   params.kernel_type=CvSVM::RBF;
   params.svm_type=CvSVM::C_SVC;
   params.gamma=0.5062;
   params.C=312.503;
   params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,1);
   bool res=svm.train(training_data,labels,Mat(),Mat(),params);
   bool res2=svm_back.train(training_data_back,labels_back,Mat(),Mat(),params);
   svm.save( "my_svm.yml" );
   svm_back.save( "my_svm_back.yml" );
   cout<<" TRAINED "<<endl;
   //svm.save( "my_svm.xml" );
   cout<<" TRAINING DATA SAVED TO my_svm.xml"<<endl;
   cout<<endl;
}
static void test_image(Mat tempp)
{
   cout<<" TESTING IMAGE.......... "<<endl;
   //img=imread(test_image_path,0);
   Mat img=tempp.clone();
   //cvtColor(tempp,img,CV_RGB2GRAY);
   Mat classified_image=Mat::zeros(img.rows+1,img.cols+1,CV_8UC3);
   vector< vector<Point> > superpixel=make_superpixels(tempp);
   vector< Mat >desc=make_descriptors(superpixel,img);
   cout<<superpixel.size()<<" Superpixel size "<<endl;
   int cnt=0;
   for(int  k=0; k<superpixel.size(); k++)
   {
     prob[k]=0;
     if(superpixel[k].size()<2 )
         {
           continue;
         }
      int x1=superpixel[k][0].x;
      int y1=superpixel[k][0].y;
      int x2=superpixel[k][1].x;
      int y2=superpixel[k][1].y;
      Mat newimg=Mat(x2-x1+1,y2-y1+1,img.type());
      for(int l=2; l<superpixel[k].size(); l++)
      {
         int x=superpixel[k][l].x;
         int y=superpixel[k][l].y;
         newimg.at<Vec3b>(x-x1,y-y1)=img.at<Vec3b>(x,y);
      }
      detector.detect(newimg,keypoint2);
      bowde.compute(newimg,keypoint2,bowdescriptor2);
	    bowdescriptor2=desc[k];
      if(bowdescriptor2.rows==0)
      {
         draw(classified_image,superpixel[k],-100,100);
         continue;
      }
      cout<<" PREDICTING "<<newimg.rows<<" "<<newimg.cols<<endl;
      float response=svm.predict(bowdescriptor2,1);
      float response_back=svm_back.predict(bowdescriptor2,1);
      prob[k]=response;
      cout<<"RESPONSE ="<<response<<" "<<response_back<<" "<<response-response_back<<endl;

      Vec3b color=get_color(1+response-response_back);
      draw(classified_image,superpixel[k],response,response_back);
      cnt++;
   }
   int width=640,height=480,num_pixels=superpixel.size(),num_labels=2;
   vector<int>ans=crf.GeneralGraph_DArraySArray(width,height,num_pixels,num_labels);
   Mat crf_image=Mat(classified_image.size(),classified_image.type());
   for(int  i=0; i<superpixel.size(); i++)
   {
       cout<<ans[i]<<" ";
	   if(ans[i]==1)
		   {
		     draw(crf_image,superpixel[i],-100,0);
		   }
	   else
		   {
		    draw(crf_image,superpixel[i],-1,0);
		   }
   }
   cout<<endl;
   for(int i=0;i<10;i++)
	   {
	   for(int j=0;j<10;j++)
		   {
		      cout<<len[i][j]<<" ";
		   }
	   cout<<endl;
	   }
   cout<<endl;
   cout<<cnt<<endl;
   imshow(" IMAGE CLASSIFIED ",classified_image);
   imshow("Original ",tempp);
   imshow("CRF  ",crf_image);
   while(waitKey()!=27);
}
int main()
{ 
   flag=1;
   make_vocabulary();
   cluster_vocabulary();
   flag=1;
   compute_training_data();
   train_usingdata();
   for(int i=100; i<320; i++)
   {
      stringstream ss;
      ss << path_Bikes << "bike_"<<setfill('0') << setw(3) << i <<".image.png";
      cout<<ss.str()<<endl;
      Mat send=imread(ss.str(),1);
      test_image(send);
   }
   return 0;
}
/////////////////////////////////////////////////////////////////////////////
// Example illustrating the use of GCoptimization.cpp
//
/////////////////////////////////////////////////////////////////////////////
//
//  Optimization problem:
//  is a set of sites (pixels) of width 10 and hight 5. Thus number of pixels is 50
//  grid neighborhood: each pixel has its left, right, up, and bottom pixels as neighbors
//  7 labels
//  Data costs: D(pixel,label) = 0 if pixel < 25 and label = 0
//            : D(pixel,label) = 10 if pixel < 25 and label is not  0
//            : D(pixel,label) = 0 if pixel >= 25 and label = 5
//            : D(pixel,label) = 10 if pixel >= 25 and label is not  5
// Smoothness costs: V(p1,p2,l1,l2) = min( (l1-l2)*(l1-l2) , 4 )
// Below in the main program, we illustrate different ways of setting data and smoothness costs
// that our interface allow and solve this optimizaiton problem

// For most of the examples, we use no spatially varying pixel dependent terms.
// For some examples, to demonstrate spatially varying terms we use
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1


#include <iostream>

using namespace std;

int main()
{
    cout << "Hello world!" << endl;
    return 0;
}
